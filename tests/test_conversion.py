"""
Unit tests for ANN-to-SNN conversion (src/conversion.py).

Run from the project root:
    python -m pytest tests/test_conversion.py -v

These tests verify:
  - copy_weights() gives SNN layers the same weights as ANN
  - copy_weights() is a deep copy (modifying ANN after copy doesn't affect SNN)
  - calibrate_thresholds() produces positive thresholds for all three LIF layers
  - Thresholds are within a plausible range (not NaN, not 0, not huge)
  - calibrate_thresholds() with percentile=100 gives the maximum activation
  - calibrate_thresholds() raises ValueError on bad percentile
  - convert() gives identical results to copy_weights() + calibrate_thresholds()
  - Converted SNN forward pass produces correct shape and finite output
  - Converted SNN output is deterministic (reset + same input → same output)

Uses a tiny synthetic DataLoader (single batch of 8 samples) so tests run
on CPU in <10 s without the N-MNIST data files.
"""

from __future__ import annotations

import copy
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import functional

from src.config import get_config
from src.models.ann import ANN
from src.models.snn import SNN
from src.conversion import copy_weights, calibrate_thresholds, convert


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def ann(cfg):
    """Trained-ish ANN: random weights (not trained, but sufficient for tests)."""
    model = ANN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
    )
    # Use a fixed seed for reproducibility
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.1)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fresh_snn(cfg):
    """SNN with default (uncalibrated) thresholds."""
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


@pytest.fixture(scope="module")
def calib_loader():
    """Tiny synthetic DataLoader: 16 random frames, batch_size=8.
    Values in [0, 1] to match normalised N-MNIST frames.
    """
    torch.manual_seed(42)
    x = torch.rand(16, 2, 34, 34)
    y = torch.randint(0, 10, (16,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_fresh_snn(cfg) -> SNN:
    """Return a new SNN with default thresholds (avoids mutating fixtures)."""
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
# copy_weights tests
# ---------------------------------------------------------------------------

class TestCopyWeights:

    def test_conv1_weight_matches(self, ann, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        assert torch.equal(
            ann.features[0].weight.data,
            snn.features[0].weight.data,
        ), "Conv1 weights were not copied correctly."

    def test_conv1_bias_matches(self, ann, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        assert torch.equal(
            ann.features[0].bias.data,
            snn.features[0].bias.data,
        ), "Conv1 bias was not copied correctly."

    def test_conv2_weight_matches(self, ann, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        assert torch.equal(
            ann.features[3].weight.data,
            snn.features[3].weight.data,
        )

    def test_fc1_weight_matches(self, ann, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        assert torch.equal(
            ann.classifier[1].weight.data,
            snn.classifier[1].weight.data,
        )

    def test_fc2_weight_matches(self, ann, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        assert torch.equal(
            ann.classifier[3].weight.data,
            snn.classifier[3].weight.data,
        )

    def test_copy_is_independent(self, ann, cfg):
        """Modifying ANN weights after copy must NOT affect SNN weights."""
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        original = snn.features[0].weight.data.clone()
        # Mutate ANN weight
        ann.features[0].weight.data.fill_(999.0)
        assert torch.equal(snn.features[0].weight.data, original), (
            "SNN weights changed when ANN weights were mutated — "
            "copy_weights() is sharing tensors instead of copying."
        )
        # Restore ANN (fixture is module-scoped, must not stay mutated)
        torch.nn.init.normal_(ann.features[0].weight, mean=0.0, std=0.1)


# ---------------------------------------------------------------------------
# calibrate_thresholds tests
# ---------------------------------------------------------------------------

class TestCalibrateThresholds:

    def test_returns_three_keys(self, ann, calib_loader, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        thresholds = calibrate_thresholds(ann, snn, calib_loader)
        assert set(thresholds.keys()) == {'lif1', 'lif2', 'lif3'}, (
            f"Unexpected threshold keys: {set(thresholds.keys())}"
        )

    def test_all_thresholds_positive(self, ann, calib_loader, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        thresholds = calibrate_thresholds(ann, snn, calib_loader)
        for name, val in thresholds.items():
            assert val > 0, f"Threshold '{name}' = {val} is not positive."

    def test_thresholds_are_finite(self, ann, calib_loader, cfg):
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        thresholds = calibrate_thresholds(ann, snn, calib_loader)
        for name, val in thresholds.items():
            assert torch.isfinite(torch.tensor(val)), (
                f"Threshold '{name}' = {val} is not finite."
            )

    def test_thresholds_applied_to_snn(self, ann, calib_loader, cfg):
        """Verify get_thresholds() reflects the calibrated values."""
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        thresholds = calibrate_thresholds(ann, snn, calib_loader)
        snn_thresholds = snn.get_thresholds()
        for name in ('lif1', 'lif2', 'lif3'):
            assert snn_thresholds[name] == pytest.approx(thresholds[name], rel=1e-5), (
                f"SNN threshold '{name}' ({snn_thresholds[name]:.6f}) "
                f"doesn't match returned value ({thresholds[name]:.6f})."
            )

    def test_percentile_100_gives_max(self, ann, calib_loader, cfg):
        """percentile=100 should give the maximum activation, not the 99th."""
        snn_99  = make_fresh_snn(cfg)
        snn_100 = make_fresh_snn(cfg)
        copy_weights(ann, snn_99);  copy_weights(ann, snn_100)
        t99  = calibrate_thresholds(ann, snn_99,  calib_loader, percentile=99.0)
        t100 = calibrate_thresholds(ann, snn_100, calib_loader, percentile=100.0)
        for name in ('lif1', 'lif2', 'lif3'):
            assert t100[name] >= t99[name], (
                f"percentile=100 threshold ({t100[name]:.4f}) is smaller than "
                f"percentile=99 threshold ({t99[name]:.4f}) for '{name}'. "
                "Max should be ≥ 99th percentile."
            )

    def test_bad_percentile_raises(self, ann, calib_loader, cfg):
        snn = make_fresh_snn(cfg)
        with pytest.raises(ValueError, match="percentile"):
            calibrate_thresholds(ann, snn, calib_loader, percentile=0.0)
        with pytest.raises(ValueError, match="percentile"):
            calibrate_thresholds(ann, snn, calib_loader, percentile=101.0)

    def test_n_batches_limits_data(self, ann, calib_loader, cfg):
        """n_batches=1 and n_batches=2 should both succeed and give positive thresholds."""
        snn = make_fresh_snn(cfg)
        copy_weights(ann, snn)
        thresholds = calibrate_thresholds(ann, snn, calib_loader, n_batches=1)
        for name, val in thresholds.items():
            assert val > 0, f"Threshold '{name}' not positive with n_batches=1."


# ---------------------------------------------------------------------------
# convert() tests
# ---------------------------------------------------------------------------

class TestConvert:

    def test_convert_matches_manual(self, ann, calib_loader, cfg):
        """convert() must give the same result as copy_weights + calibrate_thresholds."""
        snn_manual  = make_fresh_snn(cfg)
        snn_convert = make_fresh_snn(cfg)

        # Manual pipeline
        copy_weights(ann, snn_manual)
        t_manual = calibrate_thresholds(ann, snn_manual, calib_loader)

        # convert() pipeline
        _, t_convert = convert(ann, snn_convert, calib_loader)

        for name in ('lif1', 'lif2', 'lif3'):
            assert t_manual[name] == pytest.approx(t_convert[name], rel=1e-5), (
                f"convert() threshold for '{name}' differs from manual pipeline."
            )
            assert (
                snn_manual.get_thresholds()[name]
                == pytest.approx(snn_convert.get_thresholds()[name], rel=1e-5)
            )

    def test_converted_snn_output_shape(self, ann, calib_loader, cfg):
        snn = make_fresh_snn(cfg)
        snn, _ = convert(ann, snn, calib_loader)
        T, B = 16, 4
        x_seq = torch.rand(T, B, 2, 34, 34)
        functional.reset_net(snn)
        with torch.no_grad():
            out = snn(x_seq)
        assert out.shape == (B, 10), f"Expected ({B}, 10), got {tuple(out.shape)}"

    def test_converted_snn_output_finite(self, ann, calib_loader, cfg):
        snn = make_fresh_snn(cfg)
        snn, _ = convert(ann, snn, calib_loader)
        T, B = 16, 4
        x_seq = torch.rand(T, B, 2, 34, 34)
        functional.reset_net(snn)
        with torch.no_grad():
            out = snn(x_seq)
        assert torch.isfinite(out).all(), "Converted SNN output contains NaN or Inf."

    def test_converted_snn_deterministic(self, ann, calib_loader, cfg):
        """After reset, same input → same output."""
        snn = make_fresh_snn(cfg)
        snn, _ = convert(ann, snn, calib_loader)
        T, B = 16, 4
        x_seq = torch.rand(T, B, 2, 34, 34)
        with torch.no_grad():
            functional.reset_net(snn)
            out1 = snn(x_seq)
            functional.reset_net(snn)
            out2 = snn(x_seq)
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Converted SNN gives different outputs for the same input after reset."
        )
