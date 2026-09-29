"""
ANN-to-SNN conversion utilities.

Pipeline (protocol v1.0):
  1. copy_weights(ann, snn)          — copy Conv/Linear parameters in-place
  2. calibrate_thresholds(...)       — set per-layer LIF thresholds using the
                                       99th-percentile activation method
  3. convert(ann, snn, ...)          — convenience wrapper: step 1 + step 2

Threshold calibration method (Rathi et al. 2020 / Rueckauer et al. 2017):
  For each layer i, collect the post-ReLU activations of the ANN on a
  calibration subset of training data.  The LIF threshold is set to the
  p-th percentile (default: 99th) of those activations.

  Intuition: with a constant input c fed over T time steps and a LIF with
  threshold θ, the average firing rate ≈ c / θ.  If θ = 99th-percentile(c),
  then ~99 % of inputs produce firing rates ≤ 1 spike/step, which matches
  the rate expected from the normalised ANN activation.

  We use post-ReLU values (clamp(z, min=0)) because:
    - Negative pre-ReLU values are suppressed by ReLU in the ANN.
    - The LIF neuron only fires for positive membrane potential, so negative
      inputs are already suppressed; calibrating on positives gives a more
      accurate threshold.

No BatchNorm → no BN-folding step needed.
Weights are copied exactly (no additional weight scaling).

Layer mapping:
  ANN                 ←→  SNN
  ─────────────────────────────────────────
  features[0]  Conv1      features[0]  Conv1
  (features[1] ReLU)  →   features[1]  LIF1   (threshold calibrated)
  features[2]  Pool1      features[2]  Pool1
  features[3]  Conv2      features[3]  Conv2
  (features[4] ReLU)  →   features[4]  LIF2
  features[5]  Pool2      features[5]  Pool2
  classifier[1] FC1       classifier[1] FC1
  (classifier[2] ReLU) →  classifier[2] LIF3
  classifier[3] FC2       classifier[3] FC2

Note on FC2 (output layer):
  FC2 is kept as a regular Linear layer in both models.  No LIF is placed
  after it; the output is the time-averaged membrane potential (logits).
  This is standard practice in ANN-SNN conversion pipelines.

References:
  Rueckauer et al. (2017). Conversion of Continuous-Valued Deep Networks
    to Efficient Event-Driven Networks for Image Classification. Front. Neurosci.
  Rathi et al. (2020). Enabling Deep Spiking Neural Networks with Hybrid
    Conversion and Spike Timing Dependent Backpropagation. arXiv:2005.01807.
"""

from __future__ import annotations

import copy
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from src.models.ann import ANN
from src.models.snn import SNN


# ---------------------------------------------------------------------------
# Layer mapping — keeps the index constants in one place so changes to
# the architecture only need to be made here.
# ---------------------------------------------------------------------------

# (ANN index, SNN index, activation key from get_activations(), LIF name)
_LAYER_MAP: list[tuple[int, str, str, str, str]] = [
    # container   ann_idx  snn_idx  act_key   lif_name
    ('features',    0,       0,    'conv1',   'lif1'),
    ('features',    3,       3,    'conv2',   'lif2'),
    ('classifier',  1,       1,    'fc1',     'lif3'),
]
# Output layer (FC2) — weights copied but no LIF calibration needed.
_OUTPUT_LAYER_MAP: list[tuple[str, int, int]] = [
    ('classifier', 3, 3),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def copy_weights(ann: ANN, snn: SNN) -> None:
    """
    Copy Conv and Linear parameters from a trained ANN into the SNN in-place.

    Only Conv/Linear weights and biases are copied; no thresholds are changed.
    Call calibrate_thresholds() afterwards to set appropriate LIF thresholds.

    Args:
        ann: Trained ANN instance.
        snn: SNN instance with matching architecture (same channel widths, FC sizes).

    Raises:
        ValueError: If any weight tensor shapes do not match.
    """
    ann.eval()
    snn.eval()

    def _get_container(model, container_name: str) -> nn.Module:
        return getattr(model, container_name)

    with torch.no_grad():
        # Copy hidden layers (conv1, conv2, fc1)
        for container, ann_idx, snn_idx, _, _ in _LAYER_MAP:
            ann_layer = _get_container(ann, container)[ann_idx]
            snn_layer = _get_container(snn, container)[snn_idx]

            _check_shapes(ann_layer, snn_layer, f"{container}[{ann_idx}]")
            snn_layer.weight.data.copy_(ann_layer.weight.data)
            if ann_layer.bias is not None and snn_layer.bias is not None:
                snn_layer.bias.data.copy_(ann_layer.bias.data)

        # Copy output layer (fc2)
        for container, ann_idx, snn_idx in _OUTPUT_LAYER_MAP:
            ann_layer = _get_container(ann, container)[ann_idx]
            snn_layer = _get_container(snn, container)[snn_idx]
            _check_shapes(ann_layer, snn_layer, f"{container}[{ann_idx}] (output)")
            snn_layer.weight.data.copy_(ann_layer.weight.data)
            if ann_layer.bias is not None and snn_layer.bias is not None:
                snn_layer.bias.data.copy_(ann_layer.bias.data)


def calibrate_thresholds(
    ann: ANN,
    snn: SNN,
    calibration_loader: DataLoader,
    percentile: float = 99.0,
    device: torch.device = torch.device('cpu'),
    n_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Set per-layer LIF thresholds using the percentile-of-activation method.

    The threshold for LIF layer i is set to the `percentile`-th percentile
    of the post-ReLU activations of the corresponding ANN layer, measured
    over `n_batches` batches of calibration data.

    "Post-ReLU" here means clamp(pre_relu_activation, min=0) — negative
    values are excluded because they are suppressed by the ANN's ReLU and
    would never contribute to LIF firing.

    Args:
        ann:                Trained ANN (used for activation statistics only).
        snn:                SNN whose LIF thresholds will be modified in-place.
        calibration_loader: DataLoader over the training subset.
                            Using ~1,000–5,000 samples is sufficient.
        percentile:         Activation percentile used as the threshold.
                            Default 99.0 (≈ 1% of inputs exceed threshold).
        device:             Device to run ANN inference on.
        n_batches:          Number of batches to use for calibration.
                            None means use the full loader (may be slow on CPU).

    Returns:
        Dict mapping layer name → calibrated threshold value.
        Keys: 'lif1', 'lif2', 'lif3'.
    """
    if not (0.0 < percentile <= 100.0):
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")

    ann = ann.to(device).eval()

    # Collect post-ReLU activations from each layer
    # Using lists of 1-D flattened tensors to avoid OOM on large datasets
    accumulated: dict[str, list[torch.Tensor]] = {
        'conv1': [],
        'conv2': [],
        'fc1':   [],
    }
    n_samples = 0

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(calibration_loader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            x = x.to(device)
            acts = ann.get_activations(x)
            for key in accumulated:
                # clamp(min=0) → post-ReLU; flatten to 1-D
                pos_acts = acts[key].clamp(min=0).flatten().cpu()
                accumulated[key].append(pos_acts)
            n_samples += x.shape[0]

    if n_samples == 0:
        raise RuntimeError("calibration_loader provided zero samples.")

    # Compute percentile threshold per layer
    calibrated: dict[str, float] = {}
    lif_key_map = {'conv1': 'lif1', 'conv2': 'lif2', 'fc1': 'lif3'}

    for act_key, lif_key in lif_key_map.items():
        all_vals = torch.cat(accumulated[act_key])

        if percentile == 100.0:
            thresh = float(all_vals.max())
        else:
            thresh = float(torch.quantile(all_vals, percentile / 100.0))

        if thresh <= 0.0:
            warnings.warn(
                f"Calibrated threshold for '{lif_key}' is ≤ 0 ({thresh:.6f}). "
                "This may indicate no activations exceeded zero (dead layer). "
                "Defaulting to 1.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            thresh = 1.0

        calibrated[lif_key] = thresh
        snn.set_threshold(lif_key, thresh)

    return calibrated


def convert(
    ann: ANN,
    snn: SNN,
    calibration_loader: DataLoader,
    percentile: float = 99.0,
    device: torch.device = torch.device('cpu'),
    n_batches: Optional[int] = None,
) -> tuple[SNN, dict[str, float]]:
    """
    Full ANN-to-SNN conversion: copy weights then calibrate thresholds.

    This is a convenience wrapper around copy_weights() and
    calibrate_thresholds().  The SNN is modified in-place AND returned
    for chaining.

    Args:
        ann:                Trained ANN.
        snn:                Uninitialised (or freshly constructed) SNN.
        calibration_loader: DataLoader for activation statistics.
        percentile:         Activation percentile for threshold setting (default 99).
        device:             Inference device.
        n_batches:          Batches to use for calibration (None = all).

    Returns:
        (snn, thresholds) — the modified SNN and a dict of calibrated thresholds.

    Example:
        ann = ANN(); train(ann, ...)
        snn = SNN(tau=2.0)
        snn, thresholds = convert(ann, snn, calib_loader, n_batches=10)
        functional.reset_net(snn)
        logits = snn(x_seq)
    """
    copy_weights(ann, snn)
    thresholds = calibrate_thresholds(
        ann, snn, calibration_loader,
        percentile=percentile, device=device, n_batches=n_batches,
    )
    return snn, thresholds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_shapes(ann_layer: nn.Module, snn_layer: nn.Module, label: str) -> None:
    """Raise ValueError if weight shapes don't match."""
    if not hasattr(ann_layer, 'weight') or not hasattr(snn_layer, 'weight'):
        raise ValueError(
            f"Layer '{label}': one or both sides have no 'weight' attribute. "
            f"ANN: {type(ann_layer).__name__}, SNN: {type(snn_layer).__name__}"
        )
    aw, sw = ann_layer.weight.shape, snn_layer.weight.shape
    if aw != sw:
        raise ValueError(
            f"Layer '{label}': weight shape mismatch. "
            f"ANN {aw} ≠ SNN {sw}. "
            "Check that both models use identical architecture settings."
        )
