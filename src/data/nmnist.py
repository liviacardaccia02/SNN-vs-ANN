"""
N-MNIST event dataset that preserves the original (x, y, t, p) events.

Source: tonic.datasets.NMNIST (decodes the original Orchard et al. 2015 .bin
files; timestamps in integer microseconds, polarity p in {0, 1}).

Representations returned by NMNISTEventDataset (see src/data/transforms.py):

    representation="events": (structured np.ndarray [N] with x, y, t, p, label)
    representation="snn":    (float32 Tensor [T, 2, 34, 34], label)
    representation="ann":    (float32 Tensor [2, 34, 34],    label)
                              == the "snn" tensor reduced over T (mean by
                              default, decision D-B)

Evaluation at inference T (decision D-A): the same 315-ms window is re-binned
into T bins and counts are multiplied by value_scale = T / T_train, so the
expected per-step input matches training (see make_eval_loaders_by_T).

Binning is computed on the fly from the original file on every access;
nothing is cached on disk, so no stale v1.0 single-frame cache can be read.

Batched by a DataLoader, "snn" samples are [B, T, C, H, W]; the training code
converts them to SpikingJelly's time-major [T, B, C, H, W] (see
src/trainer.py: to_time_major).
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import Config
from src.data.transforms import (
    BinningInfo,
    BinningSpec,
    EventValidationError,
    events_to_temporal_frames,
    temporal_to_aggregate,
)

SENSOR_SIZE: tuple[int, int, int] = (34, 34, 2)
N_CLASSES: int = 10
EXPECTED_TRAIN_SIZE: int = 60_000
EXPECTED_TEST_SIZE: int = 10_000

Representation = Literal["events", "snn", "ann"]


class EventFileError(RuntimeError):
    """A recording could not be read or failed validation. Never skipped silently."""


def binning_spec_from_config(cfg: Config, n_bins: Optional[int] = None) -> BinningSpec:
    """Binning spec from config; n_bins overrides n_time_bins (re-binning, D-A)."""
    pp = cfg.preprocess
    return BinningSpec(
        n_bins=pp.n_time_bins if n_bins is None else n_bins,
        sensor_width=pp.sensor_width,
        sensor_height=pp.sensor_height,
        n_polarities=pp.n_polarities,
        strategy=pp.binning_strategy,          # type: ignore[arg-type]
        window_start_us=pp.window_start_us,
        window_duration_us=pp.window_duration_us,
        out_of_window_policy=pp.out_of_window_policy,  # type: ignore[arg-type]
    )


def _split_dir(root: str, train: bool) -> str:
    return os.path.join(root, "NMNIST", "Train" if train else "Test")


def _count_bin_files(path: str) -> int:
    return sum(f.endswith(".bin") for _, _, files in os.walk(path) for f in files)


class NMNISTEventDataset(Dataset):
    """
    Original N-MNIST recordings with deterministic timestamp-based binning.

    Args:
        root:           Directory passed to Tonic as save_to (contains NMNIST/).
        train:          True -> official 60k training split, False -> 10k test split.
        spec:           Temporal binning specification.
        representation: "events", "snn" or "ann" (see module docstring).
        normalization:  "none" (raw counts) or "global_scale".
        normalization_scale: constant divisor for "global_scale".
        download:       If False and files are missing, raise FileNotFoundError
                        instead of triggering a ~1 GB download.
        value_scale:    Constant multiplier applied to every bin after
                        normalisation (T / T_train when re-binning, D-A).
        ann_reduction:  "mean" (default, D-B) or "sum" for representation="ann".
    """

    def __init__(
        self,
        root: str,
        train: bool,
        spec: BinningSpec,
        representation: Representation = "snn",
        normalization: str = "none",
        normalization_scale: Optional[float] = None,
        download: bool = True,
        value_scale: float = 1.0,
        ann_reduction: str = "mean",
        prefix_bins: Optional[int] = None,
    ) -> None:
        import tonic  # local import: keeps synthetic unit tests independent of Tonic

        if representation not in ("events", "snn", "ann"):
            raise ValueError(f"unknown representation '{representation}'")
        if normalization not in ("none", "global_scale"):
            raise ValueError(f"unknown normalization '{normalization}'")
        if normalization == "global_scale" and not (normalization_scale and normalization_scale > 0):
            raise ValueError("global_scale requires normalization_scale > 0")
        if not value_scale > 0:
            raise ValueError("value_scale must be > 0")
        if ann_reduction not in ("mean", "sum"):
            raise ValueError(f"unknown ann_reduction '{ann_reduction}'")

        expected = EXPECTED_TRAIN_SIZE if train else EXPECTED_TEST_SIZE
        split_dir = _split_dir(root, train)
        if not download and _count_bin_files(split_dir) < expected:
            raise FileNotFoundError(
                f"N-MNIST {'train' if train else 'test'} split not found or incomplete at "
                f"{split_dir} (expected {expected} .bin files). Re-run with download=True "
                "or see scripts/fix_nmnist_testset.py."
            )

        self._tonic = tonic.datasets.NMNIST(save_to=root, train=train)
        if len(self._tonic) != expected:
            raise EventFileError(
                f"Tonic found {len(self._tonic)} recordings in {split_dir}, expected {expected}"
            )
        if tuple(self._tonic.sensor_size) != (spec.sensor_width, spec.sensor_height, spec.n_polarities):
            raise ValueError(
                f"Tonic sensor_size {self._tonic.sensor_size} != spec "
                f"{(spec.sensor_width, spec.sensor_height, spec.n_polarities)}"
            )

        self.root = root
        self.train = train
        self.spec = spec
        self.representation: Representation = representation
        self.normalization = normalization
        self.normalization_scale = normalization_scale
        self.value_scale = float(value_scale)
        self.ann_reduction = ann_reduction
        if prefix_bins is not None and not 1 <= prefix_bins <= spec.n_bins:
            raise ValueError(f"prefix_bins must be in [1, {spec.n_bins}]")
        self.prefix_bins = prefix_bins

    # ------------------------------------------------------------------
    @property
    def targets(self) -> list[int]:
        """Labels from the directory structure — no files are read."""
        return list(self._tonic.targets)

    def file_path(self, index: int) -> str:
        return self._tonic.data[index]

    def __len__(self) -> int:
        return len(self._tonic)

    # ------------------------------------------------------------------
    def get_events(self, index: int) -> tuple[np.ndarray, int]:
        """Original, unmodified event array and label."""
        try:
            events, label = self._tonic[index]
        except Exception as exc:  # corrupt / unreadable file
            raise EventFileError(f"failed to read {self.file_path(index)}: {exc}") from exc
        return events, int(label)

    def get_binned(self, index: int) -> tuple[np.ndarray, BinningInfo, int]:
        """[T, C, H, W] float32 frames (after normalisation), BinningInfo, label."""
        events, label = self.get_events(index)
        try:
            frames, info = events_to_temporal_frames(events, self.spec)
        except EventValidationError as exc:
            raise EventFileError(f"invalid events in {self.file_path(index)}: {exc}") from exc
        if self.normalization == "global_scale":
            frames = frames / np.float32(self.normalization_scale)
        if self.value_scale != 1.0:
            frames = frames * np.float32(self.value_scale)
        if self.prefix_bins is not None:          # sensitivity variant "prefix"
            frames = frames[: self.prefix_bins]
        return frames, info, label

    def __getitem__(self, index: int):
        if self.representation == "events":
            return self.get_events(index)
        frames, _, label = self.get_binned(index)
        if self.representation == "ann":
            return torch.from_numpy(temporal_to_aggregate(frames, self.ann_reduction)), label
        return torch.from_numpy(frames), label


# ---------------------------------------------------------------------------
# Splits and loaders
# ---------------------------------------------------------------------------

def make_train_val_split(
    dataset: NMNISTEventDataset,
    val_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """
    Deterministic class-stratified split of the official training set.

    The split depends only on (labels, val_fraction, seed); it is identical for
    every representation, so ANN and SNN models see exactly the same recordings.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))


def _ann_reduction(cfg: Config) -> str:
    return {"mean_over_bins": "mean", "sum_over_bins": "sum"}[cfg.preprocess.ann_representation]


def eval_value_scale(cfg: Config, T: int) -> float:
    """Per-step input scaling for evaluation at T (decision D-A)."""
    if cfg.snn.t_eval_input_scaling == "match_training_rate":
        return T / cfg.snn.t_train
    return 1.0


def _make_ds(cfg: Config, train: bool, representation: Representation, download: bool,
             n_bins: Optional[int] = None, value_scale: float = 1.0,
             prefix_bins: Optional[int] = None) -> NMNISTEventDataset:
    pp = cfg.preprocess
    return NMNISTEventDataset(
        cfg.data.data_root, train, binning_spec_from_config(cfg, n_bins), representation,
        pp.normalization, pp.normalization_scale, download,
        value_scale=value_scale, ann_reduction=_ann_reduction(cfg), prefix_bins=prefix_bins,
    )


EVAL_VARIANTS = ("rebin_scaled", "rebin_raw", "prefix")
"""How a T-step input is formed from a recording at evaluation time (D-A):

rebin_scaled  PRIMARY. Whole 315-ms window split into T bins (the SpikingJelly
              'frames_number=T, split_by=time' convention), counts x T/T_train
              so the per-step input matches what the model was trained on.
rebin_raw     Same bins, raw counts (sensitivity: is the scaling what matters?).
prefix        First T bins of the T_train-bin sequence, raw counts: same
              temporal resolution as training, but only the first
              T*13.125 ms of the recording are seen (sensitivity: early
              decision). Defined only for T <= T_train.
"""


def _loader_kwargs(cfg: Config, batch_size: int, seed: int) -> dict:
    from src.utils.reproducibility import get_worker_init_fn
    return dict(batch_size=batch_size, num_workers=cfg.train.num_workers,
                worker_init_fn=get_worker_init_fn(seed))


def make_dataloaders(
    cfg: Config,
    experiment_seed: int,
    representation: Representation,
    batch_size: Optional[int] = None,
    download: bool = True,
) -> dict[str, DataLoader]:
    """
    {'train', 'val', 'test'} loaders for one representation ("snn" or "ann"),
    binned at T_train = cfg.preprocess.n_time_bins.

    The train/val split uses cfg.data.seed_split (fixed across experiment seeds);
    shuffling uses experiment_seed.
    """
    if representation not in ("snn", "ann"):
        raise ValueError("make_dataloaders supports representation 'snn' or 'ann'")
    full_train = _make_ds(cfg, True, representation, download)
    test = _make_ds(cfg, False, representation, download)
    train_sub, val_sub = make_train_val_split(
        full_train, 1.0 - cfg.data.train_val_split, cfg.data.seed_split
    )
    if batch_size is None:
        batch_size = cfg.train.ann_batch_size if representation == "ann" else cfg.train.snn_sg_batch_size
    gen = torch.Generator().manual_seed(experiment_seed)
    common = _loader_kwargs(cfg, batch_size, experiment_seed)
    return {
        "train": DataLoader(train_sub, shuffle=True, generator=gen, **common),
        "val": DataLoader(val_sub, shuffle=False, **common),
        "test": DataLoader(test, shuffle=False, **common),
    }


def make_eval_loaders_by_T(
    cfg: Config,
    split: str = "test",
    t_grid: Optional[list[int]] = None,
    batch_size: Optional[int] = None,
    indices: Optional[list[int]] = None,
    download: bool = False,
    variant: str = "rebin_scaled",
) -> dict[int, DataLoader]:
    """
    {T: DataLoader} of SNN sequences for every T in the evaluation grid
    (decision D-A): each loader re-bins the SAME [0, 315 ms) window of the
    SAME recordings into T bins, scaled by eval_value_scale(cfg, T).

    split: "test" (official test set) or "val" (validation part of the train
    split, as produced by make_train_val_split). `indices` optionally restricts
    to a subset of that split (positions within the split).
    variant: see EVAL_VARIANTS. For "prefix", T > T_train is skipped.
    """
    if split not in ("test", "val"):
        raise ValueError("split must be 'test' or 'val'")
    if variant not in EVAL_VARIANTS:
        raise ValueError(f"variant must be one of {EVAL_VARIANTS}")
    t_grid = list(cfg.snn.t_eval_grid if t_grid is None else t_grid)
    if variant == "prefix":
        t_grid = [T for T in t_grid if T <= cfg.snn.t_train]
    batch_size = batch_size or cfg.train.snn_sg_batch_size
    out: dict[int, DataLoader] = {}
    for T in sorted(t_grid):
        if variant == "rebin_scaled":
            ds = _make_ds(cfg, split == "val", "snn", download, n_bins=T,
                          value_scale=eval_value_scale(cfg, T))
        elif variant == "rebin_raw":
            ds = _make_ds(cfg, split == "val", "snn", download, n_bins=T, value_scale=1.0)
        else:
            ds = _make_ds(cfg, split == "val", "snn", download, prefix_bins=T)
        if split == "val":
            _, ds = make_train_val_split(ds, 1.0 - cfg.data.train_val_split, cfg.data.seed_split)
        if indices is not None:
            ds = Subset(ds, list(indices))
        out[T] = DataLoader(ds, shuffle=False, **_loader_kwargs(cfg, batch_size, 0))
    return out
