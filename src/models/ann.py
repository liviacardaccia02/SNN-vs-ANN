"""
ANN model for the SNN-vs-ANN experiment.

Architecture (locked, protocol v1.0):
    Input:  [B, 2, 34, 34]  — ON/OFF event-rate frame
    Conv1:  Conv2d(2→32, k=3, pad=1) + ReLU   → [B, 32, 34, 34]
    Pool1:  AvgPool2d(2)                        → [B, 32, 17, 17]
    Conv2:  Conv2d(32→64, k=3, pad=1) + ReLU   → [B, 64, 17, 17]
    Pool2:  AvgPool2d(2, floor division)        → [B, 64,  8,  8]
    Flat:   Flatten                             → [B, 4096]
    FC1:    Linear(4096→256) + ReLU             → [B, 256]
    FC2:    Linear(256→10)                      → [B, 10]   (logits)

Design decisions:
  - No BatchNorm — avoids BN-folding step during ANN→SNN conversion.
  - AvgPool2d — MaxPool is incompatible with rate-coding conversion;
    average pooling transfers directly to the SNN.
  - ReLU — corresponds to the firing-rate approximation in the SNN.
    Each ReLU neuron maps to one LIF neuron during conversion.
  - floor(17/2) = 8 → flattened size = 64×8×8 = 4096.
  - No softmax in forward() — use CrossEntropyLoss (which applies
    log-softmax internally) during training.

Reference: Rathi et al. (2020), Rueckauer et al. (2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class ANN(nn.Module):
    """
    Convolutional ANN for N-MNIST digit classification.

    Args:
        in_channels: Number of input channels (2 for ON+OFF events).
        conv1_out:   Output channels of first conv layer (32).
        conv2_out:   Output channels of second conv layer (64).
        fc1_out:     Hidden size of first fully-connected layer (256).
        n_classes:   Number of output classes (10).

    Usage:
        model = ANN()
        logits = model(x)          # x: [B, 2, 34, 34]
        probs  = logits.softmax(-1)
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv1_out: int = 32,
        conv2_out: int = 64,
        fc1_out: int = 256,
        n_classes: int = 10,
    ) -> None:
        super().__init__()

        # ---- Feature extractor ----
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, conv1_out, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),   # 34→17
            # Block 2
            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),   # 17→8 (floor)
        )

        # Spatial size after two AvgPool2d(2):  34 → 17 → 8  (floor division)
        # Flattened: conv2_out × 8 × 8
        self._flat_size = conv2_out * 8 * 8   # 64 × 8 × 8 = 4096

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, fc1_out, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc1_out, n_classes, bias=True),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape [B, C, H, W] — normalised event frame.
               Expected: C=2, H=W=34, values in [0, 1].

        Returns:
            Logits of shape [B, n_classes]. NOT softmax-ed.
            Pass to nn.CrossEntropyLoss directly.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    # ------------------------------------------------------------------

    def get_activations(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Return pre-ReLU activations at each layer for threshold calibration
        during ANN→SNN conversion (99th-percentile normalization).

        Keys: 'conv1', 'conv2', 'fc1'
        Values: activations BEFORE ReLU, shape varies.
        """
        acts: dict[str, torch.Tensor] = {}

        # Conv1
        z1 = self.features[0](x)     # Conv2d
        acts['conv1'] = z1.detach()
        h1 = self.features[1](z1)    # ReLU
        h1 = self.features[2](h1)    # AvgPool

        # Conv2
        z2 = self.features[3](h1)    # Conv2d
        acts['conv2'] = z2.detach()
        h2 = self.features[4](z2)    # ReLU
        h2 = self.features[5](h2)    # AvgPool

        # FC1
        flat = self.classifier[0](h2)           # Flatten
        z3   = self.classifier[1](flat)          # Linear
        acts['fc1'] = z3.detach()

        return acts
