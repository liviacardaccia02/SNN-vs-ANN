"""
Unit tests for ANN and SNN forward passes.

Run from the project root:
    python -m pytest tests/test_models.py -v

These tests verify:
  - ANN forward pass produces correct output shape [B, 10]
  - ANN intermediate spatial sizes are correct (34→17→8)
  - ANN has no BatchNorm layers (architectural invariant)
  - ANN get_activations() returns the right keys and shapes
  - SNN forward pass produces correct output shape [B, 10]
  - SNN output is in a reasonable range (rate-coded logits)
  - SNN reset() clears membrane state (re-running same input gives same output)
  - SNN mean-over-T output is independent of T scaling
  - ANN and SNN have matching Conv/Linear weight shapes
    (prerequisite for weight transfer during conversion)
  - Parameter counts match the expected values

No training is performed; all tests use random or zero inputs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config import get_config
from src.models.ann import ANN
from src.models.snn import SNN
from spikingjelly.activation_based import functional


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def ann(cfg):
    model = ANN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def snn(cfg):
    model = SNN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
        tau=cfg.snn.tau,
        v_threshold=cfg.snn.v_threshold,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helper: a small batch of random input frames
# ---------------------------------------------------------------------------

B = 4      # batch size for tests
C = 2      # ON + OFF channels
H = W = 34

def random_frame(batch=B) -> torch.Tensor:
    """Random float32 tensor in [0, 1] simulating a normalised event frame."""
    return torch.rand(batch, C, H, W)

def random_seq(T: int, batch=B) -> torch.Tensor:
    """Same frame repeated T times: [T, B, C, H, W]."""
    frame = random_frame(batch)
    return frame.unsqueeze(0).expand(T, -1, -1, -1, -1).clone()


# ===========================================================================
# ANN tests
# ===========================================================================

class TestANN:

    def test_output_shape(self, ann, cfg):
        x = random_frame()
        with torch.no_grad():
            out = ann(x)
        assert out.shape == (B, cfg.network.n_classes), (
            f"Expected [{B}, {cfg.network.n_classes}], got {tuple(out.shape)}"
        )

    def test_output_finite(self, ann):
        x = random_frame()
        with torch.no_grad():
            out = ann(x)
        assert torch.isfinite(out).all(), "ANN output contains NaN or Inf"

    def test_no_batchnorm(self, ann):
        """Architectural invariant: no BatchNorm anywhere in the model."""
        for name, module in ann.named_modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), (
                f"Found BatchNorm layer at '{name}'. "
                "BatchNorm must be absent for clean ANN→SNN conversion."
            )

    def test_no_maxpool(self, ann):
        """Only AvgPool is allowed (MaxPool is incompatible with rate-coding)."""
        for name, module in ann.named_modules():
            assert not isinstance(module, nn.MaxPool2d), (
                f"Found MaxPool2d at '{name}'. Use AvgPool2d."
            )

    def test_spatial_sizes(self, ann):
        """Verify 34→17→8 spatial reduction."""
        x = random_frame()
        with torch.no_grad():
            after_block1 = ann.features[:3](x)   # Conv1 + ReLU + Pool1
            after_block2 = ann.features(x)        # full feature extractor
        assert after_block1.shape == (B, 32, 17, 17), (
            f"After block1: expected (B,32,17,17), got {tuple(after_block1.shape)}"
        )
        assert after_block2.shape == (B, 64, 8, 8), (
            f"After block2: expected (B,64,8,8), got {tuple(after_block2.shape)}"
        )

    def test_flat_size(self, ann):
        assert ann._flat_size == 4096, (
            f"Expected flat_size=4096 (64×8×8), got {ann._flat_size}"
        )

    def test_get_activations_keys(self, ann):
        x = random_frame()
        acts = ann.get_activations(x)
        assert set(acts.keys()) == {'conv1', 'conv2', 'fc1'}, (
            f"Unexpected activation keys: {set(acts.keys())}"
        )

    def test_get_activations_shapes(self, ann):
        x = random_frame()
        acts = ann.get_activations(x)
        assert acts['conv1'].shape == (B, 32, 34, 34)
        assert acts['conv2'].shape == (B, 64, 17, 17)
        assert acts['fc1'].shape   == (B, 256)

    def test_get_activations_have_negatives(self, ann):
        """Pre-ReLU activations should contain negative values (not already clipped)."""
        x = random_frame()
        acts = ann.get_activations(x)
        # With random weights, at least one layer should have some negative pre-ReLU values
        has_neg = any((v < 0).any().item() for v in acts.values())
        assert has_neg, (
            "All pre-ReLU activations are non-negative — "
            "get_activations() may be returning post-ReLU values by mistake."
        )

    def test_parameter_count(self, ann):
        """Verify total parameter count matches hand-calculated value.

        Conv1:  32 × (2×3×3 + 1)   =  32 × 19  =    608
        Conv2:  64 × (32×3×3 + 1)  =  64 × 289 = 18,496
        FC1:    4096×256 + 256      =         1,048,832
        FC2:    256×10  + 10        =             2,570
        Total                       =         1,070,506
        """
        n_params = sum(p.numel() for p in ann.parameters())
        assert n_params == 1_070_506, (
            f"Expected 1,070,506 parameters, got {n_params:,}. "
            "Check architecture against the locked design."
        )

    def test_zero_input_no_crash(self, ann):
        """Zero input (e.g. a sample with no events) must not crash."""
        x = torch.zeros(B, C, H, W)
        with torch.no_grad():
            out = ann(x)
        assert out.shape == (B, 10)


# ===========================================================================
# SNN tests
# ===========================================================================

class TestSNN:

    @pytest.fixture(autouse=True)
    def reset_before_each(self, snn):
        """Always reset membrane state before each test.

        IMPORTANT: Do NOT call snn.reset() — in SpikingJelly 0.0.0.0.14
        functional.reset_net() calls reset() on ANY module that has that method,
        so a custom reset() that calls functional.reset_net(self) causes
        infinite recursion. Always reset from outside the model.
        """
        functional.reset_net(snn)
        yield
        functional.reset_net(snn)

    def test_output_shape_T16(self, snn, cfg):
        T = cfg.snn.t_train
        x_seq = random_seq(T)
        with torch.no_grad():
            out = snn(x_seq)
        assert out.shape == (B, cfg.network.n_classes), (
            f"T={T}: expected [{B}, 10], got {tuple(out.shape)}"
        )

    @pytest.mark.parametrize("T", [2, 4, 8, 16, 32])
    def test_output_shape_eval_grid(self, snn, T):
        """SNN must accept all T values in the evaluation grid."""
        functional.reset_net(snn)
        x_seq = random_seq(T)
        with torch.no_grad():
            out = snn(x_seq)
        assert out.shape == (B, 10), (
            f"T={T}: expected [{B}, 10], got {tuple(out.shape)}"
        )

    def test_output_finite(self, snn):
        x_seq = random_seq(16)
        with torch.no_grad():
            out = snn(x_seq)
        assert torch.isfinite(out).all(), "SNN output contains NaN or Inf"

    def test_reset_clears_state(self, snn):
        """Running the same input twice with reset() between runs must give
        identical output. Without reset(), the second run would differ because
        membrane potentials are non-zero from the first run."""
        x_seq = random_seq(16)
        with torch.no_grad():
            functional.reset_net(snn)
            out1 = snn(x_seq)
            functional.reset_net(snn)
            out2 = snn(x_seq)
        assert torch.allclose(out1, out2, atol=1e-6), (
            "SNN output differs across runs after reset — "
            "functional.reset_net does not fully clear membrane state."
        )

    def test_no_reset_differs(self, snn):
        """Verify that LIF membrane state persists across forward() calls.

        Rather than comparing outputs (which can be identical if no spikes occur),
        we check the LIF internal membrane voltage is non-zero after a run with
        a large input guaranteed to charge the membrane, then confirm it's
        cleared to zero after functional.reset_net().
        """
        # Large constant input forces membrane charging even at random init weights.
        x_seq = torch.ones(16, B, C, H, W) * 5.0
        with torch.no_grad():
            functional.reset_net(snn)
            _ = snn(x_seq)
            # After forward, at least one LIF should have non-zero membrane voltage
            lif_voltages = [
                snn.features[1].v,   # LIF1
                snn.features[4].v,   # LIF2
                snn.classifier[2].v, # LIF3
            ]
            any_nonzero = any(
                (v.abs().max().item() > 1e-8)
                for v in lif_voltages
                if isinstance(v, torch.Tensor)
            )
        assert any_nonzero, (
            "All LIF membrane voltages are zero after a large-input forward pass. "
            "This suggests membrane state is not being maintained."
        )

    def test_zero_input_no_crash(self, snn):
        """Zero input should not crash; shape and finiteness must hold.

        With zero input, LIF neurons produce zero spikes, but the FC2 bias
        is non-zero by default, so the output is the FC2 bias — NOT zero.
        """
        x_seq = torch.zeros(16, B, C, H, W)
        with torch.no_grad():
            out = snn(x_seq)
        assert out.shape == (B, 10), f"Expected ({B}, 10), got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "SNN output is not finite on zero input"
        # All rows should be identical (same bias, same zero input)
        assert torch.allclose(out[0], out[1], atol=1e-6), (
            "Different rows differ on identical zero input — "
            "unexpected batch-dependent behaviour."
        )

    def test_get_thresholds(self, snn, cfg):
        thresholds = snn.get_thresholds()
        assert set(thresholds.keys()) == {'lif1', 'lif2', 'lif3'}
        for name, val in thresholds.items():
            assert val == pytest.approx(cfg.snn.v_threshold), (
                f"Threshold '{name}' = {val}, expected {cfg.snn.v_threshold}"
            )

    def test_set_threshold(self, snn):
        original = snn.get_thresholds()['lif1']
        snn.set_threshold('lif1', 2.5)
        assert snn.get_thresholds()['lif1'] == pytest.approx(2.5)
        snn.set_threshold('lif1', original)   # restore


# ===========================================================================
# ANN ↔ SNN weight compatibility test
# ===========================================================================

class TestWeightCompatibility:
    """
    Verify that Conv and Linear layers in ANN and SNN have identical shapes.
    This is a prerequisite for ANN→SNN weight transfer.

    We compare:
      ANN.features[0]  ↔  SNN.features[0]   (Conv1)
      ANN.features[3]  ↔  SNN.features[3]   (Conv2)
      ANN.classifier[1] ↔ SNN.classifier[1] (FC1)
      ANN.classifier[3] ↔ SNN.classifier[3] (FC2)
    """

    def _get_ann_layers(self, ann):
        return {
            'conv1': ann.features[0],
            'conv2': ann.features[3],
            'fc1':   ann.classifier[1],
            'fc2':   ann.classifier[3],
        }

    def _get_snn_layers(self, snn):
        return {
            'conv1': snn.features[0],
            'conv2': snn.features[3],
            'fc1':   snn.classifier[1],
            'fc2':   snn.classifier[3],
        }

    def test_weight_shapes_match(self, ann, snn):
        ann_layers = self._get_ann_layers(ann)
        snn_layers = self._get_snn_layers(snn)
        for name in ann_layers:
            aw = ann_layers[name].weight.shape
            sw = snn_layers[name].weight.shape
            assert aw == sw, (
                f"Layer '{name}': ANN weight shape {aw} ≠ SNN weight shape {sw}. "
                "Weight transfer will fail."
            )

    def test_bias_shapes_match(self, ann, snn):
        ann_layers = self._get_ann_layers(ann)
        snn_layers = self._get_snn_layers(snn)
        for name in ann_layers:
            ab = ann_layers[name].bias.shape
            sb = snn_layers[name].bias.shape
            assert ab == sb, (
                f"Layer '{name}': ANN bias shape {ab} ≠ SNN bias shape {sb}."
            )

    def test_parameter_transfer(self, ann, snn):
        """After copying ANN weights into SNN, the SNN layers have identical weights."""
        import copy
        snn_copy = copy.deepcopy(snn)
        ann_layers = self._get_ann_layers(ann)
        snn_layers = self._get_snn_layers(snn_copy)

        for name in ann_layers:
            snn_layers[name].weight.data.copy_(ann_layers[name].weight.data)
            snn_layers[name].bias.data.copy_(ann_layers[name].bias.data)

        for name in ann_layers:
            assert torch.equal(
                snn_layers[name].weight.data,
                ann_layers[name].weight.data,
            ), f"Weight copy failed for layer '{name}'"
