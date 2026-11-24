"""
Central configuration for the SNN-vs-ANN experiment.

All hyperparameters live here. Do not hard-code values anywhere else.
Import with:  from src.config import get_config

Protocol version: v2.2-draft  (2026-11-24)   — supersedes v2.1-draft, v2.0-draft, v1.0.
Decision log with rationale: docs/DECISIONS.md.

Unchanged from v1.0:
  - Neuron model:    LIF, tau=2, soft reset (v_reset=None), ATan surrogate, theta=1
  - Threshold basis: test-based — threshold_s = ANN_test_acc_s - 0.02 (D-C: kept)
  - Architecture:    Conv(2→32)→AvgPool→Conv(32→64)→AvgPool→FC(4096→256)→FC(256→10)
  - No BatchNorm

Changed (see docs/DECISIONS.md):
  - D-0  Input: timestamp-binned event sequences [T, 2, 34, 34] for the SNN
         (v1.0 repeated one static frame). Window [0, 315 000) µs.
  - D-G  Grid T_eval = {3, 6, 12, 24, 48}, T_train = n_time_bins = 24, so that
         every bin edge set contains the 105/210-ms saccade boundaries.
  - D-A  Evaluation at T: the same 315-ms window is re-binned into T bins and
         counts are scaled by T / T_train (per-step input rate preserved).
  - D-B  ANN input = MEAN over the T_train bins; conversion = layer-wise weight
         normalisation + LIF bias offset; hybrid arm gated by a conversion-
         fidelity check on validation data before fine-tuning.
  - D-F  Direct-SG arm: data-driven per-layer weight-gain initialisation so
         that no LIF layer is silent at initialisation.
  - D-I  (after the pilot) Main experiment = protocol P2: both SNNs trained
         separately at every T in {1, 2, 3, 6, 12, 24}; one ANN per seed,
         reused at every T; fine-tuning max 30 epochs; seeds {0, 1, 2}.
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
    """Reserved for future preprocessed-tensor caching. NOT used by the v2.0
    pipeline (events are binned on the fly). Any future cache MUST be keyed by
    preprocessing_signature(cfg) so that v1.0 single-frame caches can never be
    reused. Delete ./cache produced by v1.0 code."""

    train_val_split: float = 0.8
    """Fraction of the official 60k training set used for training.
    Remainder (0.2 → 12,000 samples) becomes the validation set."""

    seed_split: int = 42
    """Fixed RNG seed for the train/val split.
    This is DIFFERENT from the experiment seeds so that the split is
    identical across all experimental seeds."""


@dataclass
class PreprocessConfig:
    """Event-to-tensor preprocessing (see src/data/transforms.py)."""

    preprocessing_version: str = "event_bins_v2.1"
    """Bump whenever any field below changes meaning. Part of the signature."""

    sensor_width: int = 34
    sensor_height: int = 34
    n_polarities: int = 2
    """N-MNIST ATIS sensor: 34x34 pixels, polarity p in {0, 1}. Channel index == p.
    Naming (decision D-P): the dataset's 40-bit event format defines bit 23 as
    polarity with 0 = OFF and 1 = ON; Tonic reads exactly that bit
    ((byte[2] & 128) >> 7). So channel 0 = OFF, channel 1 = ON, per the
    dataset authors' definition. The smoke test's leading-edge diagnostic
    (p=0 leads the assumed saccade motion) is recorded but not used for naming:
    it depends on an unverified assumption about display contrast / motion
    direction. Both models see the same channel order, so results do not
    depend on the naming."""

    timestamp_unit: str = "us"
    """Tonic decodes N-MNIST timestamps as integer microseconds (23-bit
    counter plus overflow). Verified on real samples by the smoke test."""

    n_time_bins: int = 24
    """Number of chronologically ordered temporal bins used for TRAINING
    (= SNNConfig.t_train). 315 ms / 24 = 13.125 ms per bin; the saccade
    boundaries at 105 and 210 ms are exactly the starts of bins 8 and 16
    (D-G). Evaluation at other T re-bins the same window (D-A)."""

    binning_strategy: str = "fixed_window"
    """'fixed_window': every recording uses the same physical window, so bin k
    is the same saccade phase in every sample. 'per_recording': window =
    [min t, max t] of each recording (bin width varies per sample)."""

    window_start_us: int = 0
    window_duration_us: int = 315_000
    """Window [0, 315 000) µs = three 105-ms saccades (Orchard et al. 2015;
    Tonic's saccade constants). Measured on the full TRAIN split: 27 of
    250 316 228 events (in 24 of 60 000 recordings) fall at t >= 315 ms
    (test split: 155 of 42 036 067). For every T in {3, 6, 12, 24, 48} the
    integer bin edges include 105 000 and 210 000 µs exactly (tested)."""

    out_of_window_policy: str = "drop_and_report"
    """Events outside the window are dropped AND counted in BinningInfo
    (never silently). Use 'error' to make any such event fatal."""

    aggregation: str = "count"
    """Per-bin value = number of events at (bin, polarity, y, x)."""

    normalization: str = "none"
    """'none': raw counts. 'global_scale': divide every bin of every sample by
    the single constant normalization_scale (preserves relative intensity
    across bins and samples). Per-bin or per-sample normalisation is
    deliberately NOT offered (issue D)."""

    normalization_scale: Optional[float] = None

    ann_representation: str = "mean_over_bins"
    """ANN input = X_SNN.mean(dim=0) of the SAME T_train-bin recording, i.e. the
    average per-step input the SNN receives. Same information as the sum
    (constant factor 1/T_train) but on the SNN's per-step scale, which is what
    rate-based conversion requires (D-B)."""

    @property
    def sensor_size(self) -> tuple[int, int, int]:
        """(width, height, polarities) — Tonic convention."""
        return (self.sensor_width, self.sensor_height, self.n_polarities)


@dataclass
class NetworkConfig:
    in_channels: int = 2
    """Input channels: one per event polarity, channel index == p (0, 1)."""

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

    t_train: int = 24
    """Number of SNN simulation time-steps during TRAINING (training horizon).
    Each simulation step consumes one temporal event bin, so t_train must
    equal preprocess.n_time_bins (checked by validate_config). 24 keeps the
    position of the old 16 in the grid (3 smaller values, 1 larger; D-G)."""

    t_eval_grid: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    """Time-step grid (D-G, revised by D-I). Under P2 (main experiment) both
    SNNs are trained AND evaluated at every T in this grid. The pilot showed
    both arms qualify at T = 3 under P2, so T = 1, 2 were added to keep the
    primary outcome (smallest qualifying T) discriminative; T = 48 was dropped
    (most expensive, irrelevant for a minimum). T = 3, 6, 12, 24 put bin edges
    on the 105/210-ms saccade boundaries; T = 1 has no interior edge; T = 2
    splits the middle saccade (limitation, reported)."""

    t_eval_interpretation: str = "rebin_full_window"
    """D-A: at inference T, the SAME [0, 315 ms) window is re-binned into T
    equal bins, so every T sees all events of the recording; only temporal
    resolution (and number of steps) changes."""

    t_eval_input_scaling: str = "match_training_rate"
    """D-A consequence: a bin at T holds ~T_train/T times as many events as a
    training bin. 'match_training_rate' multiplies counts by T / T_train so
    the expected per-step input equals the training regime (value = events
    per 13.125-ms training-bin duration). 'none' feeds raw counts."""

    init_calibration: str = "firing_rate"
    """D-F: 'firing_rate' rescales each Conv/Linear layer of a freshly
    initialised SNN (direct-SG arm only) so its LIF layer reaches
    init_target_rate on training data. 'none' = PyTorch default init (measured:
    LIF2/LIF3 silent)."""

    init_target_rate: float = 0.05
    """Target mean firing rate (spikes/neuron/step) per LIF layer at init.
    Selected by a validation sweep over {0.01, 0.02, 0.05, 0.10, 0.20}
    (mean val acc 71.8/72.4/74.1/73.2/69.6 %; rule: best mean, ties within
    1 pp -> lower rate). See results/conversion_init_check/report_F_sweep.json."""

    init_calib_batches: int = 4


@dataclass
class TrainConfig:
    # ---- ANN training ----
    ann_epochs: int = 30
    ann_lr: float = 1e-3
    ann_weight_decay: float = 1e-4
    ann_batch_size: int = 128

    # ---- Hybrid fine-tuning (ANN→SNN then SG fine-tune) ----
    snn_ft_epochs: int = 30
    """D-I: was 10. Same rule as the other phases (max 30, early stopping
    with patience 7); the pilot's T=24 fine-tuning was still improving at 10."""
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

    ann_reference_bins: int = 24
    """D-I: the single ANN of each seed is trained on the mean over this many
    bins (= total events / 24). Converting it for a SNN at T multiplies its
    conv1 weights by T / ann_reference_bins, because the mean over T bins is
    total / T; the ANN function, accuracy and threshold are then identical."""

    # ---- Threshold calibration (Method B only) ----
    calib_batches: int = 10
    """Number of training batches to use for per-layer activation statistics
    during ANN→SNN threshold calibration.  At batch_size=128 this covers
    128*10=1,280 samples, which is sufficient for 99th-percentile estimation."""

    calib_percentile: float = 99.0
    """Percentile lambda_l of post-ReLU ANN activations per layer used for
    layer-wise weight normalisation (Rueckauer et al. 2017)."""

    conversion_input_map: str = "data_fitted"
    """LIF input map used by conversion (src/conversion.py): 'data_fitted' =
    LIF-curve fit refined per layer on real training recordings (D-B);
    'fitted' = curve fit only (constant-input assumption); 'offset' =
    X = theta*(1+u); 'none' = plain layer-wise normalisation (ablations)."""

    conversion_refine_batches: int = 4
    """Training batches used by the per-layer 'data_fitted' refinement."""

    # ---- Conversion-fidelity gate (Method B, D-B) ----
    conversion_gate: str = "staged"
    """Decision D-B ("recheck, then relax"), evaluated on the VALIDATION split
    for the converted SNN before fine-tuning:
    'staged' (default): the strict criteria below are always checked and
        recorded; if they fail, fine-tuning proceeds only if the sanity
        criterion passes (gate_sanity_fraction); otherwise ConversionGateError.
    'enforce': strict criteria must pass.
    'report': never blocks; outcome recorded (tests / smoke runs)."""

    gate_sanity_fraction: float = 0.5
    """Sanity criterion: acc_SNN >= chance + 0.5 * (acc_ANN - chance), i.e. the
    converted SNN keeps at least half of the ANN's above-chance accuracy
    (ANN 91.3 % -> 50.7 %). Implementation choice; to be confirmed."""

    gate_max_acc_drop: float = 0.02
    """Gate criterion 1: val_acc(converted SNN @ T_train) >= val_acc(ANN) - 0.02
    (same tolerance as the study's accuracy margin)."""

    gate_min_agreement: float = 0.90
    """Gate criterion 2: fraction of validation samples on which converted SNN
    and ANN predict the same class."""

    gate_batches: Optional[int] = None
    """Validation batches used by the gate (None = full validation split)."""

    # ---- LR scheduler ----
    lr_scheduler: str = "cosine"
    """Learning rate schedule. Currently only 'cosine' (CosineAnnealingLR)
    is supported. T_max is set to the maximum epoch count of each phase."""

    # ---- DataLoader ----
    num_workers: int = 2
    """DataLoader workers. Set to 0 in Colab if multiprocessing causes issues."""


@dataclass
class ExperimentConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    """D-I: three seeds (was five, provisional). Each seed gives one paired
    observation per T. Chosen from the pilot's wall time (~10 h per seed for
    P2 on a Colab T4) and the report deadline; a limitation to report."""

    accuracy_margin: float = 0.02
    """Accuracy threshold margin in absolute percentage points (2 pp).
    threshold_s = ANN_test_acc_s - accuracy_margin.
    The SNN qualifies at T if SNN_test_acc_s_T >= threshold_s."""

    threshold_set: str = "test"
    """Which accuracy determines the threshold. 'test' = test set, as in the
    submitted proposal (D-C: reviewed and kept). Validation is used for model
    selection and for the conversion gate only."""

    output_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    protocol_version: str = "v2.2-draft"
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


def validate_config(cfg: Config) -> None:
    """Raise ValueError on internally inconsistent settings."""
    pp = cfg.preprocess
    if cfg.snn.t_train != pp.n_time_bins:
        raise ValueError(
            f"snn.t_train ({cfg.snn.t_train}) must equal preprocess.n_time_bins "
            f"({pp.n_time_bins}): one simulation step consumes one event bin."
        )
    if cfg.network.in_channels != pp.n_polarities:
        raise ValueError("network.in_channels must equal preprocess.n_polarities")
    if pp.normalization not in ("none", "global_scale"):
        raise ValueError(f"unsupported normalization '{pp.normalization}'")
    if pp.normalization == "global_scale" and not (pp.normalization_scale and pp.normalization_scale > 0):
        raise ValueError("global_scale normalization needs normalization_scale > 0")
    if cfg.snn.t_eval_interpretation not in ("rebin_full_window",):
        raise ValueError(f"unsupported t_eval_interpretation '{cfg.snn.t_eval_interpretation}'")
    if cfg.snn.t_eval_input_scaling not in ("match_training_rate", "none"):
        raise ValueError(f"unsupported t_eval_input_scaling '{cfg.snn.t_eval_input_scaling}'")
    if pp.ann_representation not in ("mean_over_bins", "sum_over_bins"):
        raise ValueError(f"unsupported ann_representation '{pp.ann_representation}'")
    if pp.aggregation != "count":
        raise ValueError(f"unsupported aggregation '{pp.aggregation}'")
    if pp.timestamp_unit != "us":
        raise ValueError("only microsecond timestamps are supported (N-MNIST)")


def preprocessing_signature(cfg: Config) -> str:
    """Short stable hash of every preprocessing field. Use it to key caches,
    checkpoints and reports so v1.0 single-frame artefacts are never reused."""
    import hashlib
    import json
    from dataclasses import asdict
    payload = json.dumps(asdict(cfg.preprocess), sort_keys=True)
    return f"{cfg.preprocess.preprocessing_version}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def get_config() -> Config:
    """Return the default experiment configuration. Modify fields as needed."""
    cfg = Config()
    validate_config(cfg)
    return cfg
