"""
Shared training utilities for ANN and SNN training loops.

Provides:
  - EpochLog          — per-epoch metrics dataclass
  - TrainingResult    — full training run result dataclass
  - HybridTrainingResult — combined ANN + SNN-FT result (Method B)
  - EarlyStopping     — patience-based early stopping
  - get_hardware_info — device / version metadata
  - evaluate()        — single-pass evaluation loop (ANN and SNN)
  - train_one_epoch() — single-epoch training loop (ANN and SNN)

Design notes:
  - All functions are model-agnostic; is_snn=True switches the SNN-specific
    path (functional.reset_net + time-step expansion of input).
  - SNN input expansion: x_seq = x.unsqueeze(0).expand(T, …) uses a
    memory-efficient view (no copy). .contiguous() is called before forwarding
    to avoid potential view errors inside SpikingJelly's reshape operations.
  - Gradient clipping is applied only to SNN training (surrogate gradients
    can explode; ATan is bounded but downstream gradients are not).
  - No in-place shuffle of data is performed; determinism relies on DataLoader
    being constructed with a fixed generator seed.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EpochLog:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    n_updates: int     # number of optimizer steps (= n_batches)
    elapsed_s: float   # wall-clock time for this epoch (seconds)


@dataclass
class TrainingResult:
    """Complete result of one training phase (ANN, SNN-SG, or SNN-FT)."""
    method: str                    # 'ann' | 'snn_sg' | 'snn_ft'
    seed: int
    best_epoch: int                # epoch at which best val_acc was achieved
    best_val_acc: float
    epoch_logs: list               # list[EpochLog]
    total_epochs: int              # actual epochs run (may be < max due to ES)
    total_updates: int             # total optimizer steps across all epochs
    total_wall_s: float            # total wall time (seconds) for this phase
    early_stopped: bool
    hardware: dict
    checkpoint_path: Optional[Path]  # path to saved best-model checkpoint


@dataclass
class HybridTrainingResult:
    """
    Combined result for Method B: ANN pre-training + SNN fine-tuning.

    IMPORTANT — computational cost accounting:
      The total cost of Method B is ann_result.total_wall_s + ft_result.total_wall_s.
      Epoch counts and update counts are tracked separately per phase because
      the two phases use different models, T-values (ANN has no T), and LRs.
      These are NOT directly comparable to Method A's counts.
    """
    ann_result: TrainingResult          # ANN pre-training phase
    ft_result: TrainingResult           # SNN fine-tuning phase
    calibrated_thresholds: dict         # {lif_name: threshold_value}
    total_wall_s: float                 # ann + ft wall time (excludes calib time)
    seed: int


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Patience-based early stopping on validation accuracy (higher = better).

    Usage:
        es = EarlyStopping(patience=7)
        for epoch in ...:
            if es.step(val_acc, epoch):
                break  # stop training
        best_epoch = es.best_epoch
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-5):
        """
        Args:
            patience:  Number of epochs without improvement before stopping.
            min_delta: Minimum improvement to count as a genuine improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc: float = -float('inf')
        self.best_epoch: int = 0
        self.counter: int = 0

    def step(self, val_acc: float, epoch: int) -> bool:
        """
        Update state and return True if training should stop.

        Args:
            val_acc: Validation accuracy for the current epoch.
            epoch:   Current epoch number (1-indexed).

        Returns:
            True → stop training.  False → continue.
        """
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    @property
    def improved(self) -> bool:
        """True if the most recent step() call registered an improvement."""
        return self.counter == 0


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def get_hardware_info() -> dict:
    """Return a dict of hardware and software version metadata.

    Logged once per training run for reproducibility reporting.
    Keys include: platform, python, torch, cuda_available, cuda_device,
    cuda_version.
    """
    import importlib.metadata
    info: dict = {
        'platform': platform.platform(),
        'python':   platform.python_version(),
        'torch':    torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    try:
        info['spikingjelly'] = importlib.metadata.version('spikingjelly')
    except Exception:
        info['spikingjelly'] = 'unknown'

    if torch.cuda.is_available():
        info['cuda_device']  = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = str(torch.backends.cudnn.version())
    else:
        info['cuda_device']  = None
        info['cuda_version'] = None
        info['cudnn_version'] = None
    return info


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_snn: bool = False,
    T: int = 16,
) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader.

    Args:
        model:     Model to evaluate (ANN or SNN).
        loader:    DataLoader (validation or test set).
        criterion: Loss function (CrossEntropyLoss).
        device:    Evaluation device.
        is_snn:    If True, applies reset + time-step expansion before forwarding.
        T:         Number of simulation time-steps (SNN only).

    Returns:
        (avg_loss, accuracy) — loss averaged over all samples, accuracy in [0, 1].
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if is_snn:
        from spikingjelly.activation_based import functional

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_snn:
                functional.reset_net(model)
                x_seq = x.unsqueeze(0).expand(T, -1, -1, -1, -1).contiguous()
                logits = model(x_seq)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_snn: bool = False,
    T: int = 16,
    max_grad_norm: Optional[float] = None,
) -> tuple[float, float, int]:
    """
    Train for one epoch.

    Args:
        model:         Model to train.
        loader:        Training DataLoader.
        optimizer:     Optimizer (Adam).
        criterion:     Loss function (CrossEntropyLoss).
        device:        Training device.
        is_snn:        If True, applies reset + time-step expansion.
        T:             Simulation time-steps (SNN only; ignored for ANN).
        max_grad_norm: If not None, clip gradient L2-norm to this value.
                       Recommended for SNN training (e.g., 1.0).

    Returns:
        (avg_loss, accuracy, n_updates) — loss, accuracy in [0,1], update count.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_updates = 0

    if is_snn:
        from spikingjelly.activation_based import functional

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if is_snn:
            functional.reset_net(model)
            x_seq = x.unsqueeze(0).expand(T, -1, -1, -1, -1).contiguous()
            logits = model(x_seq)
        else:
            logits = model(x)

        loss = criterion(logits, y)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
        n_updates += 1

    return total_loss / total, correct / total, n_updates


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    val_loss: float,
    seed: int,
    extra: Optional[dict] = None,
) -> None:
    """Save a model checkpoint to disk."""
    payload = {
        'epoch':               epoch,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc':             val_acc,
        'val_loss':            val_loss,
        'seed':                seed,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_from_checkpoint(
    path: Path,
    model: nn.Module,
) -> dict:
    """
    Load model weights from a checkpoint file.

    Args:
        path:  Path to .pt checkpoint.
        model: Model whose weights will be updated in-place.

    Returns:
        The full checkpoint dict (for inspection of metadata).
    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt
