"""
Evaluation utilities for ANN and SNN models.

Provides:
  - EvalAtT          — per-T evaluation result dataclass
  - EvalResult       — full evaluation result for one seed/method
  - evaluate_ann_from_checkpoint()
  - evaluate_snn_grid_from_checkpoint()
  - compute_eval_result()

The test set is ONLY accessed from this module (and the pilot script).
All training code uses 'train' and 'val' splits only.

Threshold rule (locked, protocol v1.0):
  threshold_s = ANN_test_acc_s - accuracy_margin
  where accuracy_margin = cfg.experiment.accuracy_margin = 0.02

  The SNN qualifies at time-step count T if:
    snn_test_acc_s_T >= threshold_s

  The primary outcome is min{T ∈ t_eval_grid : qualifies}.
  If no T qualifies, min_qualifying_T = None (method fails).

Note on threshold calibration for SNN-FT checkpoints:
  The SNN-FT checkpoint stores 'calibrated_thresholds' in the extra payload.
  This module restores those thresholds after loading the state dict so that
  the evaluated model uses exactly the thresholds set during conversion —
  not the default v_threshold=1.0.

Note on evaluation at T ≠ T_train (decision D-A):
  For each T in t_eval_grid the SAME 315-ms window of the SAME test recordings
  is re-binned into T bins and scaled by T / T_train (per-step input rate kept
  equal to training). Pass {T: loader} from
  src/data/nmnist.py::make_eval_loaders_by_T. A single loader can only be
  evaluated at its own sequence length (evaluate() raises otherwise).

Note on SNN-FT checkpoints (decision D-B): conversion is encoded in the
weights (thresholds stay 1.0); 'calibrated_thresholds' are still restored for
backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.models.ann import ANN
from src.models.snn import SNN
from src.trainer import evaluate, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalAtT:
    """Evaluation metrics for the SNN at a specific inference time-step count T."""
    T: int
    test_loss: float
    test_acc: float
    qualifies: bool   # test_acc >= threshold_s
    margin: float     # test_acc - threshold_s  (positive → qualifies)


@dataclass
class EvalResult:
    """
    Complete evaluation result for one (seed, method) pair.

    Fields:
        seed:              Experiment random seed.
        method:            'snn_sg' or 'snn_ft'.
        ann_test_acc:      ANN test accuracy for this seed (used to set threshold).
        threshold_s:       Qualifying threshold = ann_test_acc - accuracy_margin.
        accuracy_margin:   The configured margin (0.02 pp).
        snn_evals:         List of EvalAtT, one per T in t_eval_grid (sorted by T).
        min_qualifying_T:  Smallest T at which snn_test_acc >= threshold_s.
                           None if no T qualifies.
    """
    seed: int
    method: str
    ann_test_acc: float
    threshold_s: float
    accuracy_margin: float
    snn_evals: list   # list[EvalAtT]
    min_qualifying_T: Optional[int]

    def to_dict(self) -> dict:
        """Serialisable dict for JSON output."""
        return {
            'seed':             self.seed,
            'method':           self.method,
            'ann_test_acc':     round(self.ann_test_acc, 6),
            'threshold_s':      round(self.threshold_s, 6),
            'accuracy_margin':  self.accuracy_margin,
            'snn_evals': [
                {
                    'T':         e.T,
                    'test_loss': round(e.test_loss, 6),
                    'test_acc':  round(e.test_acc, 6),
                    'qualifies': e.qualifies,
                    'margin':    round(e.margin, 6),
                }
                for e in self.snn_evals
            ],
            'min_qualifying_T': self.min_qualifying_T,
        }


# ---------------------------------------------------------------------------
# ANN evaluation
# ---------------------------------------------------------------------------

def evaluate_ann_from_checkpoint(
    checkpoint_path: Path,
    cfg: Config,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Load a trained ANN checkpoint and evaluate it on the test set.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by train_ann().
        cfg:             Experiment config (for architecture dimensions).
        test_loaders_by_T: {T: DataLoader} of re-binned test sequences
                         (make_eval_loaders_by_T(cfg, "test")).
        device:          Evaluation device.

    Returns:
        (test_loss, test_acc) — loss averaged over test set, accuracy in [0, 1].
    """
    model = ANN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
    ).to(device)
    load_model_from_checkpoint(checkpoint_path, model)
    criterion = nn.CrossEntropyLoss()
    return evaluate(model, test_loader, criterion, device, is_snn=False)


# ---------------------------------------------------------------------------
# SNN evaluation at multiple T
# ---------------------------------------------------------------------------

def evaluate_snn_grid_from_checkpoint(
    checkpoint_path: Path,
    cfg: Config,
    test_loaders_by_T: dict,
    device: torch.device,
    t_eval_grid: Optional[list] = None,
) -> dict:
    """
    Load a trained SNN checkpoint and evaluate at each T in the eval grid.

    For SNN-FT checkpoints: restores calibrated thresholds from the
    checkpoint payload (stored as 'calibrated_thresholds' by train_snn_hybrid).
    For SNN-SG checkpoints: uses default thresholds (v_threshold=1.0 for all
    layers, since SG training does not use conversion calibration).

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by train_snn_sg()
                         or train_snn_hybrid().
        cfg:             Experiment config.
        test_loader:     Test set DataLoader (representation="snn"). Its
                         sequence length must equal every T evaluated; with
                         a single 16-bin loader only T=16 is valid (issue A).
        device:          Evaluation device.
        t_eval_grid:     List of T values to evaluate at.
                         Defaults to cfg.snn.t_eval_grid = [2, 4, 8, 16, 32].

    Returns:
        dict: {T: (test_loss, test_acc)} for each T in t_eval_grid.
    """
    if t_eval_grid is None:
        t_eval_grid = cfg.snn.t_eval_grid

    # Build model with default thresholds
    model = SNN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
        tau=cfg.snn.tau,
        v_threshold=cfg.snn.v_threshold,
    ).to(device)

    # Load weights + optionally restore calibrated thresholds
    ckpt = load_model_from_checkpoint(checkpoint_path, model)
    if 'calibrated_thresholds' in ckpt:
        for lif_name, thresh in ckpt['calibrated_thresholds'].items():
            model.set_threshold(lif_name, thresh)

    criterion = nn.CrossEntropyLoss()
    results: dict[int, tuple[float, float]] = {}

    missing = [T for T in t_eval_grid if T not in test_loaders_by_T]
    if missing:
        raise KeyError(f"no re-binned loader for T={missing}")
    for T in sorted(t_eval_grid):
        loss, acc = evaluate(
            model, test_loaders_by_T[T], criterion, device, is_snn=True, T=T,
        )
        results[T] = (loss, acc)

    return results


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def compute_eval_result(
    seed: int,
    method: str,
    ann_test_acc: float,
    snn_accs_by_T: dict,
    cfg: Config,
) -> EvalResult:
    """
    Assemble an EvalResult from raw accuracy values.

    Args:
        seed:           Experiment seed.
        method:         'snn_sg' or 'snn_ft'.
        ann_test_acc:   ANN test accuracy for this seed.
        snn_accs_by_T:  {T: (loss, acc)} from evaluate_snn_grid_from_checkpoint().
        cfg:            Experiment config (for accuracy_margin).

    Returns:
        EvalResult with threshold, per-T qualification flags, and min qualifying T.
    """
    threshold_s = ann_test_acc - cfg.experiment.accuracy_margin
    snn_evals: list[EvalAtT] = []
    min_qualifying_T: Optional[int] = None

    for T in sorted(snn_accs_by_T):
        loss, acc = snn_accs_by_T[T]
        qualifies = acc >= threshold_s
        margin    = acc - threshold_s
        snn_evals.append(EvalAtT(
            T=T, test_loss=loss, test_acc=acc,
            qualifies=qualifies, margin=margin,
        ))
        if qualifies and min_qualifying_T is None:
            min_qualifying_T = T

    return EvalResult(
        seed=seed,
        method=method,
        ann_test_acc=ann_test_acc,
        threshold_s=threshold_s,
        accuracy_margin=cfg.experiment.accuracy_margin,
        snn_evals=snn_evals,
        min_qualifying_T=min_qualifying_T,
    )


# ---------------------------------------------------------------------------
# Pilot check helpers
# ---------------------------------------------------------------------------

def pilot_checks(
    ann_test_acc: float,
    sg_result,    # TrainingResult
    ft_result,    # HybridTrainingResult
    sg_eval: EvalResult,
    ft_eval: EvalResult,
    t_train: int,
) -> list[tuple[str, bool, str]]:
    """
    Run pilot sanity checks. Returns list of (label, passed, detail) tuples.

    Checks:
      1. Both SNNs learn: val_acc > 0.2 at end of training (well above chance 0.1)
      2. ANN achieves reasonable accuracy (> 0.8 on test)
      3. At least one SNN qualifies at some T
      4. Accuracy varies with T (T_eval grid is discriminative)
      5. SNN at T=T_train is close to ANN (< 10 pp = 0.10 accuracy degradation)
    """
    checks = []

    # 1. Both SNNs learn
    sg_learns = sg_result.best_val_acc > 0.2
    checks.append((
        "SNN-SG val_acc > 0.2 (learning check)",
        sg_learns,
        f"best_val_acc = {sg_result.best_val_acc:.4f}"
    ))
    ft_learns = ft_result.ft_result.best_val_acc > 0.2
    checks.append((
        "SNN-FT val_acc > 0.2 (learning check)",
        ft_learns,
        f"best_val_acc = {ft_result.ft_result.best_val_acc:.4f}"
    ))

    # 2. ANN accuracy
    ann_ok = ann_test_acc > 0.80
    checks.append((
        "ANN test_acc > 0.80",
        ann_ok,
        f"ann_test_acc = {ann_test_acc:.4f}"
    ))

    # 3. At least one SNN qualifies
    checks.append((
        "SNN-SG qualifies at some T",
        sg_eval.min_qualifying_T is not None,
        f"min_T = {sg_eval.min_qualifying_T}"
    ))
    checks.append((
        "SNN-FT qualifies at some T",
        ft_eval.min_qualifying_T is not None,
        f"min_T = {ft_eval.min_qualifying_T}"
    ))

    # 4. T_eval grid is discriminative (max - min acc > 0.01 across T)
    sg_accs = [e.test_acc for e in sg_eval.snn_evals]
    ft_accs = [e.test_acc for e in ft_eval.snn_evals]
    sg_range = max(sg_accs) - min(sg_accs)
    ft_range = max(ft_accs) - min(ft_accs)
    checks.append((
        "T_eval grid is discriminative (acc range > 0.01 for SNN-SG)",
        sg_range > 0.01,
        f"acc range = {sg_range:.4f} across T ∈ {[e.T for e in sg_eval.snn_evals]}"
    ))
    checks.append((
        "T_eval grid is discriminative (acc range > 0.01 for SNN-FT)",
        ft_range > 0.01,
        f"acc range = {ft_range:.4f}"
    ))

    # 5. Degradation at T=T_train relative to ANN
    sg_at_T_train = next((e.test_acc for e in sg_eval.snn_evals if e.T == t_train), None)
    ft_at_T_train = next((e.test_acc for e in ft_eval.snn_evals if e.T == t_train), None)
    if sg_at_T_train is not None:
        deg = ann_test_acc - sg_at_T_train
        checks.append((
            f"SNN-SG degradation at T={t_train} < 10 pp",
            deg < 0.10,
            f"ANN={ann_test_acc:.4f}, SNN-SG@T{t_train}={sg_at_T_train:.4f}, "
            f"degradation={deg:.4f}"
        ))
    if ft_at_T_train is not None:
        deg = ann_test_acc - ft_at_T_train
        checks.append((
            f"SNN-FT degradation at T={t_train} < 10 pp",
            deg < 0.10,
            f"ANN={ann_test_acc:.4f}, SNN-FT@T{t_train}={ft_at_T_train:.4f}, "
            f"degradation={deg:.4f}"
        ))

    return checks
