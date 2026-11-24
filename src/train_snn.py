"""
SNN training loops — Method A (direct surrogate-gradient) and Method B (hybrid).

Entry points:
    result = train_snn_sg(cfg, seed, dataloaders, device, checkpoint_dir)
    result = train_snn_hybrid(cfg, seed, snn_dataloaders, ann_dataloaders,
                              device, checkpoint_dir)

Method A — Direct SG training:
  The SNN is trained from random initialisation using surrogate-gradient
  backpropagation on temporal event sequences [T_train, B, 2, 34, 34]
  (one simulation step per event bin; protocol v2.0). Loss is
  CrossEntropyLoss on the time-averaged output logits.

Method B — Hybrid (ANN pre-train + convert + SNN fine-tune):
  1. Train ANN for up to ann_epochs with early stopping on the aggregate
     representation (sum over bins) of the SAME recordings and split.
  2. Copy Conv/Linear weights from best ANN checkpoint to SNN.
  3. Calibrate per-layer LIF thresholds using 99th-percentile activation
     method on ANN (mean-over-bins) training data, then layer-wise weight
     normalisation + LIF bias offset (src/conversion.py, decision D-B).
  3b. Conversion-fidelity gate on the validation split (staged, D-B): the
     strict criteria (accuracy within 2 pp, agreement >= 90 %) are checked and
     recorded; if they fail, fine-tuning proceeds only if the converted SNN
     keeps at least half of the ANN's above-chance accuracy.
  4. Fine-tune the converted SNN with surrogate gradients for up to
     snn_ft_epochs with early stopping.
  Total cost = ANN training cost + SNN fine-tuning cost (logged separately
  AND as a combined total — see HybridTrainingResult).

Methodological flags:
  1. T_train / T_eval separation: all SNN training (both methods) uses
     T=T_train=24. Evaluation at T_inf ∈ {3,6,12,24,48} is performed only
     in the final evaluation script. Validation loss/accuracy during training
     is computed at T=T_train for consistency.
  2. Input encoding (v2.0): the event bins of each recording are presented
     in chronological order, one bin per time step. The v1.0 direct coding
     (one accumulated frame repeated T times) has been removed. Evaluation
     at other T re-bins the same 315-ms window into T bins (decision D-A,
     src/data/nmnist.py::make_eval_loaders_by_T).
  3. Method B total cost: clock time includes ANN training + threshold
     calibration + SNN fine-tuning. Gradient updates are tracked separately
     per phase because ANN and SNN have the same parameter count but
     different computational graphs (T steps of surrogate gradient vs 1).
  4. Gradient clipping (max_grad_norm): applied per optimizer step to the
     full parameter vector L2-norm. This prevents rare exploding-gradient
     events from surrogate backpropagation through many time steps.
  5. Fine-tuning LR: snn_ft_lr is deliberately lower than snn_sg_lr to
     avoid catastrophically overwriting the calibrated threshold signal
     during the first few fine-tuning epochs.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from src.config import Config
from src.models.ann import ANN
from src.models.snn import SNN
from src.conversion import ConversionGateError, convert, conversion_fidelity, gate_decision
from src.init_calibration import calibrate_initial_firing, calibration_batches
from src.train_ann import train_ann
from src.trainer import (
    EarlyStopping, EpochLog, TrainingResult, HybridTrainingResult,
    get_hardware_info, train_one_epoch, evaluate, save_checkpoint,
    load_model_from_checkpoint,
)
from src.utils.reproducibility import set_all_seeds


# ---------------------------------------------------------------------------
# Method A — Direct surrogate-gradient training
# ---------------------------------------------------------------------------

def train_snn_sg(
    cfg: Config,
    seed: int,
    dataloaders: dict,
    device: torch.device,
    checkpoint_dir: Path,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train an SNN from scratch using surrogate-gradient backpropagation.

    Batches of event sequences [B, T_train, 2, 34, 34] are transposed to
    [T_train, B, 2, 34, 34]; bin k is simulated at step k.
    Loss = CrossEntropyLoss(time_averaged_logits, labels).

    Args:
        cfg:            Experiment config.
        seed:           Experiment seed (controls weight init and DataLoader).
        dataloaders:    SNN loaders (representation="snn") 'train', 'val',
                        'test' — test is never used.
        device:         Training device.
        checkpoint_dir: Directory for best-model checkpoint.
        verbose:        If True, print epoch-level progress.

    Returns:
        TrainingResult (method='snn_sg').
    """
    set_all_seeds(seed)
    hardware = get_hardware_info()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    T = cfg.snn.t_train

    # ---- Model ----
    model = SNN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
        tau=cfg.snn.tau,
        v_threshold=cfg.snn.v_threshold,
    ).to(device)

    # ---- Data-driven initialisation (decision D-F) ----
    init_report: dict = {}
    if cfg.snn.init_calibration == "firing_rate":
        seqs = calibration_batches(train_loader, cfg.snn.init_calib_batches, device)
        cost: dict = {}
        init_report = calibrate_initial_firing(model, seqs, cfg.snn.init_target_rate, counter=cost)
        if verbose:
            print(f"[SNN-SG seed={seed:02d}] init calibration: " + ", ".join(
                f"{k}: g={v['gain']:.3g} rate {v['rate_before']:.4f}->{v['rate_after']:.4f}"
                for k, v in init_report.items()))
    elif cfg.snn.init_calibration != "none":
        raise ValueError(f"unknown init_calibration '{cfg.snn.init_calibration}'")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.snn_sg_lr,
        weight_decay=cfg.train.snn_sg_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.snn_sg_epochs,
        eta_min=cfg.train.snn_sg_lr * 1e-2,
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=cfg.train.early_stop_patience)

    epoch_logs: list[EpochLog] = []
    total_updates = 0
    sample_timesteps = 0          # D-E: T_train time-steps per training sample
    best_val_acc = -1.0
    best_epoch = 0
    ckpt_path = checkpoint_dir / f"snn_sg_seed{seed:02d}_best.pt"

    wall_start = time.time()

    for epoch in range(1, cfg.train.snn_sg_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, n_updates = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            is_snn=True, T=T,
            max_grad_norm=cfg.train.max_grad_norm,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            is_snn=True, T=T,
        )
        scheduler.step()

        epoch_elapsed = time.time() - epoch_start
        total_updates += n_updates
        sample_timesteps += len(train_loader.dataset) * T

        log = EpochLog(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss,     val_acc=val_acc,
            n_updates=n_updates,
            elapsed_s=epoch_elapsed,
        )
        epoch_logs.append(log)

        is_best = val_acc > best_val_acc   # strict improvement in validation accuracy
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                ckpt_path, model, optimizer, epoch, val_acc, val_loss, seed,
                extra={'scheduler_state_dict': scheduler.state_dict(),
                       'method': 'snn_sg',
                       't_train': T,
                       'init_calibration': init_report},
            )

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[SNN-SG seed={seed:02d}] "
                f"Epoch {epoch:3d}/{cfg.train.snn_sg_epochs} | "
                f"T={T} | lr={current_lr:.2e} | "
                f"train  loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val    loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={epoch_elapsed:.1f}s"
                + (" ← new best val acc (checkpoint saved)" if is_best else "")
            )

        if early_stopping.step(val_acc, epoch):
            if verbose:
                print(
                    f"[SNN-SG seed={seed:02d}] Early stopping at epoch {epoch}. "
                    f"Best epoch {best_epoch} val_acc={best_val_acc:.4f}"
                )
            break

    total_wall_s = time.time() - wall_start

    return TrainingResult(
        method='snn_sg',
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
        extra={'init_calibration': init_report,
               'init_forward_sample_timesteps': (cost.get('forward_sample_timesteps', 0)
                                                 if cfg.snn.init_calibration == "firing_rate" else 0)},
    )


# ---------------------------------------------------------------------------
# Method B — Hybrid: ANN pre-train → convert → SNN fine-tune
# ---------------------------------------------------------------------------

def train_snn_hybrid(
    cfg: Config,
    seed: int,
    snn_dataloaders: dict,
    ann_dataloaders: dict,
    device: torch.device,
    checkpoint_dir: Path,
    verbose: bool = True,
) -> HybridTrainingResult:
    """
    Method B: ANN pre-training + ANN→SNN conversion + SNN fine-tuning.

    Phase 1 — ANN training:
      Calls train_ann() to produce a trained ANN checkpoint.
      Cost: ann_result.total_wall_s, ann_result.total_updates.

    Phase 2 — Conversion:
      Loads best ANN checkpoint, copies weights to SNN, calibrates LIF
      thresholds. Calibration uses calib_batches batches of training data.
      Wall time for calibration is NOT included in total_wall_s (it is
      negligible: a single forward pass over ~1,280 samples).

    Phase 3 — SNN fine-tuning:
      Fine-tunes the converted SNN with surrogate gradients for up to
      snn_ft_epochs with early stopping.
      Cost: ft_result.total_wall_s, ft_result.total_updates.

    Total cost (Method B) = ann_result.total_wall_s + ft_result.total_wall_s.
    This is logged in HybridTrainingResult.total_wall_s.

    Args:
        cfg:            Experiment config.
        seed:           Experiment seed.
        snn_dataloaders: SNN loaders (representation="snn"), used for
                         fine-tuning; 'test' never used.
        ann_dataloaders: ANN loaders (representation="ann") over the SAME
                         split, used for ANN pre-training and calibration.
        device:         Training device.
        checkpoint_dir: Directory for best-model checkpoints.
        verbose:        If True, print epoch-level progress.

    Returns:
        HybridTrainingResult containing both phases and calibrated thresholds.
    """
    checkpoint_dir = Path(checkpoint_dir)
    T = cfg.snn.t_train
    train_loader = snn_dataloaders['train']
    val_loader   = snn_dataloaders['val']

    # =========================================================
    # Phase 1 — ANN training
    # =========================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Hybrid seed={seed:02d}] Phase 1: ANN pre-training")
        print(f"{'='*60}")

    ann_result = train_ann(
        cfg, seed, ann_dataloaders, device, checkpoint_dir, verbose=verbose,
    )

    # =========================================================
    # Phase 2 — ANN→SNN conversion
    # =========================================================
    if verbose:
        print(f"\n[Hybrid seed={seed:02d}] Phase 2: ANN->SNN conversion "
              f"(percentile={cfg.train.calib_percentile}, "
              f"n_batches={cfg.train.calib_batches})")

    # Load the best ANN checkpoint
    ann_model = ANN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
    ).to(device)
    load_model_from_checkpoint(ann_result.checkpoint_path, ann_model)
    ann_model.eval()

    # Build SNN and convert
    snn_model = SNN(
        in_channels=cfg.network.in_channels,
        conv1_out=cfg.network.conv1_out,
        conv2_out=cfg.network.conv2_out,
        fc1_out=cfg.network.fc1_out,
        n_classes=cfg.network.n_classes,
        tau=cfg.snn.tau,
        v_threshold=cfg.snn.v_threshold,
    ).to(device)

    # Lambdas from ANN activations on TRAINING data (order irrelevant).
    snn_model, conversion_info = convert(
        ann_model, snn_model, ann_dataloaders['train'],
        percentile=cfg.train.calib_percentile,
        device=device,
        n_batches=cfg.train.calib_batches,
        v_threshold=cfg.snn.v_threshold,
        input_map=cfg.train.conversion_input_map,
        T=cfg.snn.t_train,
        snn_calibration_loader=snn_dataloaders['train'],
        refine_batches=cfg.train.conversion_refine_batches,
    )
    calibrated_thresholds = conversion_info['thresholds']
    if verbose:
        print(f"[Hybrid seed={seed:02d}] lambdas: "
              + ", ".join(f"{k}={v:.4f}" for k, v in conversion_info['lambdas'].items()))

    # ---- Conversion-fidelity gate on VALIDATION data (decision D-B) ----
    fidelity = conversion_fidelity(
        ann_model, snn_model, ann_dataloaders['val'], snn_dataloaders['val'],
        conversion_info['lambdas'], device=device, n_batches=cfg.train.gate_batches,
        max_acc_drop=cfg.train.gate_max_acc_drop, min_agreement=cfg.train.gate_min_agreement,
        sanity_fraction=cfg.train.gate_sanity_fraction, n_classes=cfg.network.n_classes,
    )
    proceed = gate_decision(fidelity, cfg.train.conversion_gate)
    if verbose:
        print(f"[Hybrid seed={seed:02d}] conversion gate ({cfg.train.conversion_gate}): "
              f"{fidelity.gate_outcome.upper()} | strict {'PASS' if fidelity.passed else 'FAIL'}, "
              f"sanity {'PASS' if fidelity.sanity_passed else 'FAIL'} (>= {fidelity.sanity_threshold:.4f}) | "
              f"ANN val={fidelity.ann_acc:.4f} SNN(T={fidelity.T}) val={fidelity.snn_acc:.4f} "
              f"agreement={fidelity.agreement:.4f}"
              + (f" | {'; '.join(fidelity.reasons)}" if fidelity.reasons else ""))
    if not proceed:
        raise ConversionGateError(
            "Converted SNN fails the conversion gate on validation data "
            f"(mode '{cfg.train.conversion_gate}'; {'; '.join(fidelity.reasons)}; "
            f"sanity threshold {fidelity.sanity_threshold:.4f}). Hybrid fine-tuning refused.", fidelity)

    # =========================================================
    # Phase 3 — SNN fine-tuning
    # =========================================================
    if verbose:
        print(f"\n[Hybrid seed={seed:02d}] Phase 3: SNN fine-tuning "
              f"(max {cfg.train.snn_ft_epochs} epochs, T={T})")

    # Re-seed for fine-tuning phase (same seed, deterministic from this point)
    set_all_seeds(seed)

    optimizer = torch.optim.Adam(
        snn_model.parameters(),
        lr=cfg.train.snn_ft_lr,
        weight_decay=cfg.train.snn_ft_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.snn_ft_epochs,
        eta_min=cfg.train.snn_ft_lr * 1e-2,
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=cfg.train.early_stop_patience)

    ft_epoch_logs: list[EpochLog] = []
    ft_total_updates = 0
    ft_sample_timesteps = 0
    ft_best_val_acc = -1.0
    ft_best_epoch = 0
    ft_ckpt_path = checkpoint_dir / f"snn_ft_seed{seed:02d}_best.pt"
    hardware = get_hardware_info()

    ft_wall_start = time.time()

    for epoch in range(1, cfg.train.snn_ft_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, n_updates = train_one_epoch(
            snn_model, train_loader, optimizer, criterion, device,
            is_snn=True, T=T,
            max_grad_norm=cfg.train.max_grad_norm,
        )
        val_loss, val_acc = evaluate(
            snn_model, val_loader, criterion, device,
            is_snn=True, T=T,
        )
        scheduler.step()

        epoch_elapsed = time.time() - epoch_start
        ft_total_updates += n_updates
        ft_sample_timesteps += len(train_loader.dataset) * T

        log = EpochLog(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss,     val_acc=val_acc,
            n_updates=n_updates,
            elapsed_s=epoch_elapsed,
        )
        ft_epoch_logs.append(log)

        is_best = val_acc > ft_best_val_acc   # strict improvement in validation accuracy
        if is_best:
            ft_best_val_acc = val_acc
            ft_best_epoch = epoch
            save_checkpoint(
                ft_ckpt_path, snn_model, optimizer, epoch,
                val_acc, val_loss, seed,
                extra={
                    'scheduler_state_dict':  scheduler.state_dict(),
                    'method':               'snn_ft',
                    't_train':              T,
                    'calibrated_thresholds': calibrated_thresholds,
                    'conversion_info':       conversion_info,
                    'conversion_fidelity':   fidelity.to_dict(),
                    'ann_checkpoint':        str(ann_result.checkpoint_path),
                },
            )

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[SNN-FT seed={seed:02d}] "
                f"Epoch {epoch:3d}/{cfg.train.snn_ft_epochs} | "
                f"T={T} | lr={current_lr:.2e} | "
                f"train  loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val    loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={epoch_elapsed:.1f}s"
                + (" ← new best val acc (checkpoint saved)" if is_best else "")
            )

        if early_stopping.step(val_acc, epoch):
            if verbose:
                print(
                    f"[SNN-FT seed={seed:02d}] Early stopping at epoch {epoch}. "
                    f"Best epoch {ft_best_epoch} val_acc={ft_best_val_acc:.4f}"
                )
            break

    ft_wall_s = time.time() - ft_wall_start

    ft_result = TrainingResult(
        method='snn_ft',
        seed=seed,
        best_epoch=ft_best_epoch,
        best_val_acc=ft_best_val_acc,
        epoch_logs=ft_epoch_logs,
        total_epochs=len(ft_epoch_logs),
        total_updates=ft_total_updates,
        train_sample_timesteps=ft_sample_timesteps,
        total_wall_s=ft_wall_s,
        early_stopped=early_stopping.counter >= early_stopping.patience,
        hardware=hardware,
        checkpoint_path=ft_ckpt_path,
    )

    return HybridTrainingResult(
        ann_result=ann_result,
        ft_result=ft_result,
        calibrated_thresholds=calibrated_thresholds,
        total_wall_s=ann_result.total_wall_s + ft_wall_s,
        seed=seed,
        conversion_info=conversion_info,
        conversion_fidelity=fidelity.to_dict(),
    )
