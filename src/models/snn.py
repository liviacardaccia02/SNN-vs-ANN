"""
SNN model for the SNN-vs-ANN experiment.

This SNN is the structural counterpart of ANN (src/models/ann.py):
every ReLU is replaced by a LIF neuron, and every AvgPool2d is kept
as-is (pooling is compatible with rate coding).

Architecture (locked, protocol v1.0):
    Input:  [T, B, 2, 34, 34]  — event frame repeated T times
    Conv1:  Conv2d(2→32, k=3, pad=1)            → [T, B, 32, 34, 34]
    LIF1:   LIFNode (soft reset, ATan surrogate) → [T, B, 32, 34, 34]  (spike)
    Pool1:  AvgPool2d(2)                         → [T, B, 32, 17, 17]
    Conv2:  Conv2d(32→64, k=3, pad=1)            → [T, B, 64, 17, 17]
    LIF2:   LIFNode                              → [T, B, 64, 17, 17]  (spike)
    Pool2:  AvgPool2d(2)                         → [T, B, 64,  8,  8]
    Flat:   Flatten (spatial dims only)          → [T, B, 4096]
    FC1:    Linear(4096→256)                     → [T, B, 256]
    LIF3:   LIFNode                              → [T, B, 256]         (spike)
    FC2:    Linear(256→10)                       → [T, B, 10]
    Mean:   Mean over T dimension                → [B, 10]             (logits)

Step mode: SpikingJelly multi-step ('m').
  The [T, B, ...] tensor is fed to the network as-is; SpikingJelly
  iterates the recurrent membrane equation internally across T.
  reset() must be called at the start of every forward pass to clear
  membrane potentials from the previous sample.

Soft reset (v_reset=None):
  After a spike: v ← v − v_threshold   (residual potential retained)
  This is preferred over hard reset (v ← 0) because it reduces the
  information lost per spike and improves accuracy at low T.

ATan surrogate gradient:
  The derivative of the Heaviside step function is approximated by
  the derivative of (1/π)·arctan(αx) + 0.5, which is smooth and
  bounded. α=2.0 (SpikingJelly default).

Conversion compatibility:
  Weights are copied directly from the trained ANN (same Conv/Linear
  layers). Thresholds are calibrated per layer using the 99th-percentile
  of ANN activations on a calibration subset.

References:
  Rathi et al. 2020 (arXiv:2005.01807) — hybrid pipeline
  Neftci et al. 2019 — surrogate gradients
  Fang et al. 2023 (Science Advances) — SpikingJelly
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from spikingjelly.activation_based import neuron, functional, surrogate, layer


class SNN(nn.Module):
    """
    Spiking Neural Network counterpart to ANN.

    Args:
        in_channels:  Input channels (2).
        conv1_out:    Conv1 output channels (32).
        conv2_out:    Conv2 output channels (64).
        fc1_out:      FC1 hidden size (256).
        n_classes:    Output classes (10).
        tau:          LIF time constant. Decay factor β = 1 − 1/τ.
        v_threshold:  Initial spike threshold (calibrated per layer
                      during ANN→SNN conversion).
        surrogate_fn: Surrogate gradient. Default: ATan.

    Usage — direct surrogate-gradient training:
        model = SNN()
        functional.set_step_mode(model, 'm')   # set once at construction
        # x_seq: [T, B, 2, 34, 34]
        functional.reset_net(model)            # clear state before each sample
        logits = model(x_seq)                  # [B, 10]

    Usage — inference after ANN→SNN conversion:
        Same as above; thresholds will differ per layer.
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv1_out: int = 32,
        conv2_out: int = 64,
        fc1_out: int = 256,
        n_classes: int = 10,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate_fn=None,
    ) -> None:
        super().__init__()

        if surrogate_fn is None:
            surrogate_fn = surrogate.ATan()

        self.tau = tau
        self.v_threshold = v_threshold

        # LIF constructor arguments — same for every neuron layer
        lif_kwargs = dict(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=None,           # soft reset: v ← v − v_threshold
            surrogate_function=surrogate_fn,
            detach_reset=True,      # detach reset graph — standard practice
            step_mode='m',          # multi-step: input is [T, B, ...]
        )

        # ---- Feature extractor ----
        # SpikingJelly's layer.Conv2d / layer.Linear wrappers support
        # multi-step mode natively (they reshape [T*B, ...] → forward → reshape back).
        self.features = nn.Sequential(
            layer.Conv2d(in_channels, conv1_out, kernel_size=3, padding=1, bias=True, step_mode='m'),
            neuron.LIFNode(**lif_kwargs),
            layer.AvgPool2d(kernel_size=2, stride=2, step_mode='m'),   # 34→17
            layer.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1, bias=True, step_mode='m'),
            neuron.LIFNode(**lif_kwargs),
            layer.AvgPool2d(kernel_size=2, stride=2, step_mode='m'),   # 17→8
        )

        self._flat_size = conv2_out * 8 * 8   # 4096

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            layer.Flatten(start_dim=1, step_mode='m'),  # [B,64,8,8] → [B,4096] per step

            layer.Linear(self._flat_size, fc1_out, bias=True, step_mode='m'),
            neuron.LIFNode(**lif_kwargs),
            layer.Linear(fc1_out, n_classes, bias=True, step_mode='m'),
        )

    # ------------------------------------------------------------------

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Float tensor [T, B, C, H, W].
                   Typically the same event frame repeated T times:
                   x_seq = x.unsqueeze(0).expand(T, -1, -1, -1, -1)

        Returns:
            Logits [B, n_classes] — mean firing rate over T steps.
            Mean (not sum) so that the output scale is independent of T,
            which makes the CrossEntropyLoss comparable across T values.

        IMPORTANT — call functional.reset_net(model) before every forward pass
        (every new batch), otherwise membrane state leaks from the previous batch:

            functional.reset_net(snn)
            logits = snn(x_seq)
        """
        out = self.features(x_seq)    # [T, B, conv2_out, 8, 8]
        out = self.classifier(out)    # [T, B, n_classes]
        return out.mean(dim=0)        # [B, n_classes]  — average over time

    # ------------------------------------------------------------------
    # NOTE: Do NOT define a reset() method on this class.
    # In SpikingJelly 0.0.0.0.14, functional.reset_net() calls reset() on
    # every module that has such a method (not just MemoryModule subclasses).
    # A custom reset() that calls functional.reset_net(self) would create
    # infinite recursion.  Always reset from outside the model:
    #
    #     from spikingjelly.activation_based import functional
    #     functional.reset_net(snn)     # correct
    #
    # ------------------------------------------------------------------


    @torch.no_grad()
    def set_threshold(self, layer_name: str, value: float) -> None:
        """
        Set the spike threshold of a named LIF layer.

        Used during ANN→SNN conversion after per-layer 99th-percentile
        calibration. layer_name options: 'lif1', 'lif2', 'lif3'
        (corresponding to features[1], features[4], classifier[2]).
        """
        lif_map = {
            'lif1': self.features[1],
            'lif2': self.features[4],
            'lif3': self.classifier[2],
        }
        if layer_name not in lif_map:
            raise KeyError(f"Unknown LIF layer '{layer_name}'. Choose from {list(lif_map)}")
        lif_map[layer_name].v_threshold = value

    # ------------------------------------------------------------------

    def get_thresholds(self) -> dict[str, float]:
        """Return current spike thresholds for all LIF layers."""
        return {
            'lif1': float(self.features[1].v_threshold),
            'lif2': float(self.features[4].v_threshold),
            'lif3': float(self.classifier[2].v_threshold),
        }
