"""
ANN training loop (Method B pre-training phase / ANN baseline).

Entry point:
    result = train_ann(cfg, seed, dataloaders, device, checkpoint_dir)

Returns a TrainingResult with the best model checkpoint, per-epoch logs,
and hardware/timing metadata. The test set is NEVER accessed here.

Early stopping:
  Monitors validation accuracy. Stops if no improvement for
  cfg.train.early_stop_patience consecutive epochs. Best model is saved
  when validation accuracy improves.

Checkpoint format (PyTorch .pt file):
  {
    'epoch':                int,
    'model_state_dict':     dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'val_acc':              float,
    'val_loss':             float,
    'seed':                 int,
    'method':               'ann',
  }

Methodological flags (do NOT change silently):
  1. CosineAnnealingLR T_max is set to ann_epochs (max budget), so if early
     stopping triggers at epoch k < ann_epochs, the LR at epoch k is lower
     than it would be in a full run ending at k. This is a minor inconsistency
     but is standard practice; the pilot should check if early stopping and
     cosine decay interact unfavourably.
  2. No BatchNorm → no eval/train mode effect on BN statistics.
     Only Dropout (absent here) and the eval/train distinction for gradient
     computation matters.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from src.config import Config
from src.models.ann import ANN
from src.trainer import (
    EarlyStopping, EpochLog, TrainingResult,
    get_hardware_info, train_one_epoch, evaluate, save_checkpoint,
)
from src.utils.reproducibility import set_all_seeds


def train_ann(
    cfg: Config,
    seed: int,
    dataloaders: dict,
    device: torch.device,
    checkpoint_dir: Path,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train an ANN on N-MNIST with Adam + CosineAnnealingLR + early stopping.

    Args:
        cfg:            Full experiment config (all hyperparameters are read
                        from cfg.train; see src/config.py).
        seed:           Experiment seed. Controls model init and DataLoader
                        worker seeds (NOT the train/val split seed).
        dataloaders:    Dict with keys 'train', 'val', 'test'.
                        Only 'train' and 'val' are used here.
                        'test' is NEVER touched in this function.
        device:         PyTorch device for training.
        checkpoint_dir: Directory where the best model checkpoint is saved.
                        Created if it does not exist.
        verbose:        If True, print per-epoch progress to stdout.

    Returns:
        TrainingResult with best_epoch, best_val_acc, epoch_logs,
        total_updates, total_wall_s, hardware, and checkpoint_path.
    """
    set_all_seeds(seed)
    hardware = get_hardware_info()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']

    # ---- Model ----
    model = ANN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
    ).to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.ann_lr,
        weight_decay=cfg.train.ann_weight_decay,
    )

    # ---- LR scheduler ----
    # T_max = ann_epochs (full budget) so that even if early stopping triggers
    # before ann_epochs, the LR follows the same schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.ann_epochs,
        eta_min=cfg.train.ann_lr * 1e-2,   # floor at 1% of initial LR
    )

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=cfg.train.early_stop_patience)

    epoch_logs: list[EpochLog] = []
    total_updates = 0
    sample_timesteps = 0          # D-E: ANN = 1 time-step per sample
    best_val_acc = -1.0
    best_epoch = 0
    ckpt_path = checkpoint_dir / f"ann_seed{seed:02d}_best.pt"

    wall_start = time.time()

    for epoch in range(1, cfg.train.ann_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, n_updates = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            is_snn=False,
            max_grad_norm=None,   # no clipping for ANN
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, is_snn=False,
        )
        scheduler.step()

        epoch_elapsed = time.time() - epoch_start
        total_updates += n_updates
        sample_timesteps += len(train_loader.dataset) * 1

        log = EpochLog(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss,     val_acc=val_acc,
            n_updates=n_updates,
            elapsed_s=epoch_elapsed,
        )
        epoch_logs.append(log)

        # ---- Best model checkpoint ----
        is_best = val_acc > best_val_acc   # strict improvement in validation accuracy
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                ckpt_path, model, optimizer, epoch, val_acc, val_loss, seed,
                extra={'scheduler_state_dict': scheduler.state_dict(),
                       'method': 'ann'},
            )

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[ANN seed={seed:02d}] "
                f"Epoch {epoch:3d}/{cfg.train.ann_epochs} | "
                f"lr={current_lr:.2e} | "
                f"train  loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val    loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={epoch_elapsed:.1f}s"
                + (" ← new best val acc (checkpoint saved)" if is_best else "")
            )

        if early_stopping.step(val_acc, epoch):
            if verbose:
                print(
                    f"[ANN seed={seed:02d}] Early stopping at epoch {epoch}. "
                    f"Best epoch {best_epoch} val_acc={best_val_acc:.4f}"
                )
            break

    total_wall_s = time.time() - wall_start

    return TrainingResult(
        method='ann',
        seed=seed,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        epoch_logs=epoch_logs,
        total_epochs=len(epoch_logs),
        total_updates=total_updates,
        train_sample_timesteps=sample_timesteps,
        total_wall_s=total_wall_s,
        early_stopped=early_stopping.counter >= early_stopping.patience,
        hardware=hardware,
        checkpoint_path=ckpt_path,
    )
