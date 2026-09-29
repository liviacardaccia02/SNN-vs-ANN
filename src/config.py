"""
Central configuration for the SNN-vs-ANN experiment.

All hyperparameters live here. Do not hard-code values anywhere else.
Import with:  from src.config import get_config

Protocol version: v1.0  (2026-09-28)
Locked decisions:
  - Input encoding:  single aggregated frame, direct coding [2, 34, 34]
  - T_train:         16   (fixed training horizon for Design Option 2)
  - T_eval_grid:     [2, 4, 8, 16, 32]  (inference time-step grid)
  - Neuron model:    LIF, soft reset (v_reset=None), ATan surrogate
  - Threshold basis: test-based — threshold_s = ANN_test_acc_s - 0.02
  - Architecture:    Conv(2→32)→AvgPool→Conv(32→64)→AvgPool→FC(4096→256)→FC(256→10)
  - No BatchNorm (avoids conversion complications)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_root: str = "./data"
    """Where Tonic will download the raw N-MNIST binary files."""

    cache_root: str = "./cache"
    """Where preprocessed frame tensors are cached for fast DataLoader access."""

    train_val_split: float = 0.8
    """Fraction of the official 60k training set used for training.
    Remainder (0.2 → 12,000 samples) becomes the validation set."""

    seed_split: int = 42
    """Fixed RNG seed for the train/val split.
    This is DIFFERENT from the experiment seeds so that the split is
    identical across all experimental seeds."""


@dataclass
class PreprocessConfig:
    sensor_size: tuple = (34, 34, 2)
    """N-MNIST sensor size: (width=34, height=34, n_polarities=2).
    Matches tonic.datasets.NMNIST.sensor_size — verified in unit tests."""

    n_time_bins: int = 1
    """Number of temporal bins for event-to-frame conversion.
    n_time_bins=1 → all events accumulated into a single frame per polarity.
    This is the rate-coding encoding compatible with ANN-to-SNN conversion.
    Temporal ordering of events within a recording is discarded (by design)."""


@dataclass
class NetworkConfig:
    in_channels: int = 2
    """Input channels: ON polarity (ch 0) and OFF polarity (ch 1)."""

    conv1_out: int = 32
    conv2_out: int = 64
    fc1_out: int = 256
    n_classes: int = 10

    # Spatial dimensions after two AvgPool2d(kernel=2, stride=2):
    #   34 → floor(34/2)=17 → floor(17/2)=8
    # Flattened size after conv2: 64 * 8 * 8 = 4096
    _flat_size: int = 4096  # Derived; do not change independently.


@dataclass
class SNNConfig:
    tau: float = 2.0
    """LIF membrane time constant (in simulation time-steps).
    Decay factor: beta = 1 - 1/tau = 0.5 for tau=2.
    (SpikingJelly uses 1/tau as the decay rate parameter.)"""

    v_threshold: float = 1.0
    """Initial spike threshold. Will be calibrated per-layer during ANN→SNN conversion
    using 99th-percentile activation normalization (Rathi et al. 2020)."""

    v_reset: Optional[float] = None
    """Soft reset: v_reset=None means membrane potential decreases by v_threshold
    after a spike (v_t+1 = v_t - v_threshold). This retains the residual potential
    and is preferred over hard reset (v_reset=0) for conversion quality at low T."""

    surrogate: str = "atan"
    """Surrogate gradient function for backpropagation through spikes.
    'atan' → arc-tangent approximation (SpikingJelly default, Neftci et al. 2019)."""

    t_train: int = 16
    """Number of SNN simulation time-steps during TRAINING.
    Design Option 2: fixed training horizon; models are evaluated at multiple
    inference T values without retraining. All SNN training uses this value."""

    t_eval_grid: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    """Inference time-step grid. The primary outcome (minimum qualifying T)
    is the smallest value in this list where test accuracy ≥ threshold.
    Provisional — will be confirmed or revised after the pilot."""


@dataclass
class TrainConfig:
    # ---- ANN training ----
    ann_epochs: int = 30
    ann_lr: float = 1e-3
    ann_weight_decay: float = 1e-4
    ann_batch_size: int = 128

    # ---- Hybrid fine-tuning (ANN→SNN then SG fine-tune) ----
    snn_ft_epochs: int = 10
    snn_ft_lr: float = 1e-4       # lower LR for fine-tuning pre-trained weights
    snn_ft_weight_decay: float = 1e-4
    snn_ft_batch_size: int = 128

    # ---- Direct surrogate-gradient training from scratch ----
    snn_sg_epochs: int = 30
    snn_sg_lr: float = 1e-3
    snn_sg_weight_decay: float = 1e-4
    snn_sg_batch_size: int = 128

    # ---- Early stopping (applied to all three training phases) ----
    early_stop_patience: int = 7
    """Stop if validation accuracy does not improve for this many consecutive epochs.
    Applies independently to ANN training, SNN-SG training, and SNN-FT fine-tuning.
    Pilot value: 7 epochs. May be revised after observing convergence curves."""

    # ---- Gradient clipping (SNN only) ----
    max_grad_norm: float = 1.0
    """Maximum L2-norm of the gradient vector for SNN training.
    Clip to this value before each optimizer step to prevent exploding gradients
    through the surrogate gradient path. ANN training is not clipped (None)."""

    # ---- Threshold calibration (Method B only) ----
    calib_batches: int = 10
    """Number of training batches to use for per-layer activation statistics
    during ANN→SNN threshold calibration.  At batch_size=128 this covers
    128*10=1,280 samples, which is sufficient for 99th-percentile estimation."""

    calib_percentile: float = 99.0
    """Percentile of post-ReLU activations used as LIF threshold per layer.
    99.0 means ~1 % of inputs exceed the threshold, producing ≤1 spike/step."""

    # ---- LR scheduler ----
    lr_scheduler: str = "cosine"
    """Learning rate schedule. Currently only 'cosine' (CosineAnnealingLR)
    is supported. T_max is set to the maximum epoch count of each phase."""

    # ---- DataLoader ----
    num_workers: int = 2
    """DataLoader workers. Set to 0 in Colab if multiprocessing causes issues."""


@dataclass
class ExperimentConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    """Five random seeds (provisional). Each seed produces one paired observation
    (one ANN+hybrid run AND one direct-SG run). Post-pilot decision may change count."""

    accuracy_margin: float = 0.02
    """Accuracy threshold margin in absolute percentage points (2 pp).
    threshold_s = ANN_test_acc_s - accuracy_margin.
    The SNN qualifies at T if SNN_test_acc_s_T >= threshold_s."""

    threshold_set: str = "test"
    """Which accuracy determines the threshold. 'test' = test set (as per proposal).
    Validation is used only for model selection during training."""

    output_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    protocol_version: str = "v1.0"
    """Increment when any locked decision changes. Old runs must be re-run or
    marked as belonging to a different protocol version."""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    snn: SNNConfig = field(default_factory=SNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def get_config() -> Config:
    """Return the default experiment configuration. Modify fields as needed."""
    return Config()
