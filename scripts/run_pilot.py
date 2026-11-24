#!/usr/bin/env python3
"""
Pilot study runner — one seed, both methods, full evaluation grid.

Usage (local CPU, reduced epochs for smoke test):
    python scripts/run_pilot.py --seed 0 --ann-epochs 2 --sg-epochs 2 --ft-epochs 2

Usage (Colab GPU, full pilot):
    python scripts/run_pilot.py --seed 0

What this script does:
  1. Downloads / loads N-MNIST via the project DataLoader.
  2. Runs Method B: ANN pre-training → conversion → SNN fine-tuning.
  3. Evaluates the ANN on the test set → ANN test accuracy → threshold_s.
  4. Runs Method A: SNN direct surrogate-gradient training from scratch.
  5. Evaluates both SNNs at each T ∈ {3, 6, 12, 24, 48} on the test set,
     re-binning the same 315-ms window into T bins (decision D-A).
  6. Prints a structured summary and runs pilot sanity checks.
  7. Saves a machine-readable JSON result file.

Output files (in --output-dir, default ./results/pilot_seed{N}):
  summary.txt   — human-readable console output
  results.json  — machine-readable structured results
  checkpoints/  — model checkpoints for both methods

Pilot checks (printed after evaluation):
  [✓] Both SNNs learn (val_acc > 0.2)
  [✓] ANN baseline is reasonable (> 0.80)
  [✓] At least one T qualifies for each method
  [✓] T_eval grid is discriminative (accuracy varies across T)
  [✓] Degradation at T=T_train is < 10 pp (0.10 absolute accuracy)

FLAGS FOR SCIENTIFIC TUTOR:
  - All design decisions are read from src/config.py (protocol v1.0).
  - Epoch counts and accuracy_margin can be overridden via CLI args.
  - The test set is ONLY accessed in the evaluation phase (after all
    training and validation-based model selection is complete).
  - wall_time_method_b = ANN pre-training + SNN fine-tuning (not comparable
    directly to wall_time_method_a, which uses no ANN pre-training).
  - T_train=24 during training; T_eval grid = {3,6,12,24,48} at inference.
  - Method B uses the staged conversion gate (decision D-B): the strict
    fidelity criteria are checked and recorded on validation data; if they
    fail, fine-tuning continues only if the converted SNN keeps at least half
    of the ANN's above-chance accuracy; otherwise ConversionGateError.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Fix Windows cp1252 console encoding issues
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Allow running from repo root or from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.config import get_config
from src.data.nmnist import make_dataloaders, make_eval_loaders_by_T
from src.trainer import get_hardware_info
from src.train_snn import train_snn_sg, train_snn_hybrid
from src.evaluate import (
    evaluate_ann_from_checkpoint,
    evaluate_snn_grid_from_checkpoint,
    compute_eval_result,
    pilot_checks,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SNN-vs-ANN pilot study — single seed end-to-end run"
    )
    p.add_argument('--seed',        type=int,   default=0,
                   help='Experiment seed (default: 0)')
    p.add_argument('--ann-epochs',  type=int,   default=None,
                   help='Override cfg.train.ann_epochs for this run')
    p.add_argument('--sg-epochs',   type=int,   default=None,
                   help='Override cfg.train.snn_sg_epochs for this run')
    p.add_argument('--ft-epochs',   type=int,   default=None,
                   help='Override cfg.train.snn_ft_epochs for this run')
    p.add_argument('--output-dir',  type=str,   default=None,
                   help='Output directory (default: results/pilot_seed{N})')
    p.add_argument('--data-root',   type=str,   default=None,
                   help='Override cfg.data.data_root (N-MNIST download location)')
    p.add_argument('--no-gpu',      action='store_true',
                   help='Force CPU even if GPU is available')
    p.add_argument('--quiet',       action='store_true',
                   help='Suppress per-epoch training output')
    p.add_argument('--smoke-test',  action='store_true',
                   help='Run on a tiny subset of data (256 samples) for quick pipeline verification')
    p.add_argument('--t-train',     type=int,   default=None,
                   help='Train both SNNs at this T (sets n_time_bins = t_train). Protocol P2 '
                        '("train at each T") = one run per T in the grid. Default: config (24).')
    p.add_argument('--batch-size',  type=int,   default=None,
                   help='Override all training batch sizes (memory mitigation, research plan Risk 2; '
                        'record any use of it)')
    p.add_argument('--gate-mode',   choices=['staged', 'enforce', 'report'], default=None,
                   help="Override cfg.train.conversion_gate (default 'staged', decision D-B). "
                        "'report' never blocks; use it only for diagnosis and record it.")
    p.add_argument('--no-sensitivity', action='store_true',
                   help='Skip the rebin_raw / prefix evaluation variants (primary rebin_scaled only)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

SEP = '=' * 64

def header(title: str):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)


def section(title: str):
    print(f'\n--- {title} ---')


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f}s'
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f'{m}m {s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m {s:02d}s'


def print_eval_table(sg_eval, ft_eval, threshold_s: float):
    """Print a T vs accuracy table for both methods."""
    T_vals = [e.T for e in sg_eval.snn_evals]
    header_row = 'Method    ' + ''.join(f'   T={T:2d}' for T in T_vals)
    print(header_row)
    print('-' * len(header_row))

    def row(name, evals):
        cells = []
        for e in evals:
            mark = '*' if e.qualifies else ' '
            cells.append(f'{e.test_acc:.4f}{mark}')
        return f'{name:<10}' + ''.join(f'  {c:7s}' for c in cells)

    print(row('SNN-SG', sg_eval.snn_evals))
    print(row('SNN-FT', ft_eval.snn_evals))
    print(f'\nThreshold = {threshold_s:.4f}   (* = qualifies)\n')
    print(f'Min qualifying T:  SNN-SG = {sg_eval.min_qualifying_T}   '
          f'SNN-FT = {ft_eval.min_qualifying_T}')


def print_checks(checks):
    all_passed = True
    for label, passed, detail in checks:
        icon = '[PASS]' if passed else '[FAIL]'
        print(f'  {icon} {label}')
        if detail:
            print(f'       {detail}')
        if not passed:
            all_passed = False
    if all_passed:
        print('\n  All pilot checks passed.')
    else:
        print('\n  [WARN] One or more checks failed -- review before scaling to full experiment.')
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    pilot_start = time.time()

    # ---- Config ----
    cfg = get_config()
    if args.ann_epochs  is not None: cfg.train.ann_epochs    = args.ann_epochs
    if args.sg_epochs   is not None: cfg.train.snn_sg_epochs = args.sg_epochs
    if args.ft_epochs   is not None: cfg.train.snn_ft_epochs = args.ft_epochs
    if args.data_root   is not None: cfg.data.data_root      = args.data_root
    if args.gate_mode   is not None:
        cfg.train.conversion_gate = args.gate_mode
    if args.batch_size  is not None:
        cfg.train.ann_batch_size = cfg.train.snn_sg_batch_size = cfg.train.snn_ft_batch_size = args.batch_size
    if args.t_train     is not None:
        cfg.snn.t_train = args.t_train
        cfg.preprocess.n_time_bins = args.t_train
    from src.config import validate_config
    validate_config(cfg)

    seed = args.seed
    verbose = not args.quiet


    # ---- Device ----
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # ---- Output directory ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg.experiment.output_dir) / f'pilot_seed{seed:02d}_Ttrain{cfg.snn.t_train}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    hw = get_hardware_info()

    header(f'SNN-vs-ANN PILOT STUDY  |  seed={seed}  |  device={device}')
    print(f'  Protocol:      {cfg.experiment.protocol_version}')
    print(f'  Ann epochs:    {cfg.train.ann_epochs}  '
          f'(early stop patience={cfg.train.early_stop_patience})')
    print(f'  SNN-SG epochs: {cfg.train.snn_sg_epochs}')
    print(f'  SNN-FT epochs: {cfg.train.snn_ft_epochs}')
    print(f'  T_train:       {cfg.snn.t_train}')
    print(f'  T_eval_grid:   {cfg.snn.t_eval_grid}')
    print(f'  Threshold:     ANN_test_acc - {cfg.experiment.accuracy_margin:.2f}')
    print(f'  Device:        {hw["cuda_device"] if hw["cuda_available"] else "CPU"}')
    print(f'  PyTorch:       {hw["torch"]}')
    print(f'  Output dir:    {out_dir}')

    # ---- Data ----
    section('Loading N-MNIST')
    # Both representations derive from the same recordings and the same split.
    dataloaders = make_dataloaders(cfg, experiment_seed=seed, representation='snn')
    ann_dataloaders = make_dataloaders(cfg, experiment_seed=seed, representation='ann')
    test_subset = None
    if args.smoke_test:
        import numpy as np
        from torch.utils.data import Subset, DataLoader
        print('  [SMOKE TEST MODE: 256 random samples per split for quick CPU run]')
        cfg.train.calib_batches = 1
        # Smoke mode trains for ~1 epoch on 256 samples; the ANN is far from
        # converged, so the gate is recorded but not enforced here.
        cfg.train.conversion_gate = 'report'

        def _shrink(loaders: dict) -> dict:
            out = {}
            for split, dl in loaders.items():
                # Random (seeded) subset: N-MNIST files are ordered by class, so
                # range(256) would contain a single class (v1.0 bug).
                rng = np.random.default_rng(1234)
                idx = rng.choice(len(dl.dataset), min(256, len(dl.dataset)), replace=False)
                if split == 'test':
                    nonlocal_test[0] = sorted(idx.tolist())
                out[split] = DataLoader(
                    Subset(dl.dataset, sorted(idx.tolist())),
                    batch_size=min(64, cfg.train.ann_batch_size),
                    shuffle=(split == 'train'),
                    generator=torch.Generator().manual_seed(seed),
                )
            return out
        nonlocal_test = [None]
        dataloaders = _shrink(dataloaders)
        ann_dataloaders = _shrink(ann_dataloaders)
        test_subset = nonlocal_test[0]
    n_train = len(dataloaders['train'].dataset)
    n_val   = len(dataloaders['val'].dataset)
    n_test  = len(dataloaders['test'].dataset)
    print(f'  train={n_train:,}  val={n_val:,}  test={n_test:,}')

    # =========================================================
    # Phase 1 -- Method B: ANN + Hybrid SNN
    # =========================================================
    header(f'METHOD B -- Hybrid (seed={seed})')
    print(f'  ANN pre-train ({cfg.train.ann_epochs} epochs max) -> '
          f'convert -> SNN fine-tune ({cfg.train.snn_ft_epochs} epochs max)')

    def _reset_peak() -> None:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    def _peak_mb():
        return round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == 'cuda' else None

    _reset_peak()
    hybrid_result = train_snn_hybrid(
        cfg, seed, dataloaders, ann_dataloaders, device, ckpt_dir, verbose=verbose,
    )
    peak_mb_b = _peak_mb()
    ann_res = hybrid_result.ann_result
    ft_res  = hybrid_result.ft_result

    section('Method B summary')
    print(f'  ANN    -- best_val_acc={ann_res.best_val_acc:.4f} '
          f'at epoch {ann_res.best_epoch}/{ann_res.total_epochs} | '
          f'wall={fmt_time(ann_res.total_wall_s)} | '
          f'updates={ann_res.total_updates:,}')
    print(f'  SNN-FT -- best_val_acc={ft_res.best_val_acc:.4f} '
          f'at epoch {ft_res.best_epoch}/{ft_res.total_epochs} | '
          f'wall={fmt_time(ft_res.total_wall_s)} | '
          f'updates={ft_res.total_updates:,}')
    print(f'  Calibrated thresholds: '
          + ', '.join(f'{k}={v:.4f}'
                      for k, v in hybrid_result.calibrated_thresholds.items()))
    print(f'  Method B total wall: {fmt_time(hybrid_result.total_wall_s)}')
    print(f'  Method B training budget (D-E): {hybrid_result.train_sample_timesteps:,} sample-time-steps '
          f'(ANN {ann_res.train_sample_timesteps:,} + FT {ft_res.train_sample_timesteps:,}); '
          f'conversion forward-only: {hybrid_result.conversion_info.get("refine_forward_sample_timesteps", 0):,}')
    fid = hybrid_result.conversion_fidelity
    print(f'  Conversion gate ({cfg.train.conversion_gate}): {fid.get("gate_outcome", "?")} | '
          f'strict {"PASS" if fid.get("passed") else "FAIL"}, '
          f'sanity {"PASS" if fid.get("sanity_passed") else "FAIL"} (>= {fid.get("sanity_threshold", float("nan")):.4f}) | '
          f'SNN {fid.get("snn_acc", float("nan")):.4f} vs ANN {fid.get("ann_acc", float("nan")):.4f}, '
          f'agreement {fid.get("agreement", float("nan")):.4f}')
    print(f'  Peak GPU memory (Method B): {peak_mb_b} MB')

    # =========================================================
    # Phase 2 -- Evaluate ANN on test set -> threshold_s
    # =========================================================
    section('ANN test evaluation -> threshold_s')
    ann_test_loss, ann_test_acc = evaluate_ann_from_checkpoint(
        ann_res.checkpoint_path, cfg, ann_dataloaders['test'], device,
    )
    threshold_s = ann_test_acc - cfg.experiment.accuracy_margin
    print(f'  ANN test_acc = {ann_test_acc:.4f}   test_loss = {ann_test_loss:.4f}')
    print(f'  Threshold_s  = {threshold_s:.4f}  '
          f'(= {ann_test_acc:.4f} - {cfg.experiment.accuracy_margin:.2f})')

    # =========================================================
    # Phase 3 -- Method A: Direct SG training
    # =========================================================
    header(f'METHOD A -- Direct SG (seed={seed})')
    print(f'  SG training from scratch ({cfg.train.snn_sg_epochs} epochs max, '
          f'T_train={cfg.snn.t_train})')

    _reset_peak()
    sg_result = train_snn_sg(
        cfg, seed, dataloaders, device, ckpt_dir, verbose=verbose,
    )
    peak_mb_a = _peak_mb()
    section('Method A summary')
    print(f'  SNN-SG -- best_val_acc={sg_result.best_val_acc:.4f} '
          f'at epoch {sg_result.best_epoch}/{sg_result.total_epochs} | '
          f'wall={fmt_time(sg_result.total_wall_s)} | '
          f'updates={sg_result.total_updates:,}')
    print(f'  Method A training budget (D-E): {sg_result.train_sample_timesteps:,} sample-time-steps; '
          f'init calibration forward-only: {sg_result.extra.get("init_forward_sample_timesteps", 0):,}')
    print(f'  Peak GPU memory (Method A): {peak_mb_a} MB')

    # =========================================================
    # Phase 4 — Evaluate both SNNs on test grid
    # =========================================================
    header('EVALUATION ON TEST SET')
    print(f'  T_eval grid: {cfg.snn.t_eval_grid}')
    print(f'  Threshold:   {threshold_s:.4f}')

    section('Evaluating SNN-SG at each T')
    # Decision D-A: one re-binned loader per T over the same test recordings.
    test_loaders_by_T = make_eval_loaders_by_T(
        cfg, 'test', indices=test_subset,
        batch_size=dataloaders['test'].batch_size,
    )
    sg_ops_by_T: dict = {}
    sg_accs_by_T = evaluate_snn_grid_from_checkpoint(
        sg_result.checkpoint_path, cfg, test_loaders_by_T, device, ops_out=sg_ops_by_T,
    )
    sg_eval = compute_eval_result(seed, 'snn_sg', ann_test_acc, sg_accs_by_T, cfg)

    section('Evaluating SNN-FT at each T')
    ft_ops_by_T: dict = {}
    ft_accs_by_T = evaluate_snn_grid_from_checkpoint(
        ft_res.checkpoint_path, cfg, test_loaders_by_T, device, ops_out=ft_ops_by_T,
    )
    ft_eval = compute_eval_result(seed, 'snn_ft', ann_test_acc, ft_accs_by_T, cfg)

    section('Results table')
    print_eval_table(sg_eval, ft_eval, threshold_s)

    # Decision D-H: hardware-independent operation counts (src/opcount.py),
    # measured on the test set in the same pass as the accuracies above.
    section('Operation counts per test sample (D-H; hardware-independent)')
    ann_macs_total = next(iter(sg_ops_by_T.values()))['ann_macs']
    print(f'  ANN reference: {ann_macs_total:,} MACs  '
          f'(est. {next(iter(sg_ops_by_T.values()))["ann_energy_pj_est"] / 1e3:.1f} nJ)')
    print(f'  {"Method":7s} {"T":>3s} | {"SynOps (AC)":>12s} {"input MACs":>11s} | '
          f'{"rate l1/l2/l3":>17s} | {"est. nJ":>7s} {"vs ANN":>6s} | {"dense-sim MACs":>14s}')
    for name, ops_by_T in (('SNN-SG', sg_ops_by_T), ('SNN-FT', ft_ops_by_T)):
        for T, o in sorted(ops_by_T.items()):
            r = o['firing_rate_per_layer']
            print(f'  {name:7s} {T:3d} | {o["synops_ac"]:12,.0f} {o["input_macs_event"]:11,.0f} | '
                  f'{r["lif1"]:.3f}/{r["lif2"]:.3f}/{r["lif3"]:.3f} | '
                  f'{o["energy_pj_est"] / 1e3:7.1f} {o["energy_ratio_snn_over_ann"]:6.2f} | '
                  f'{o["dense_macs_equivalent"]:14,}')
    print('  SynOps: spike x fan-out (additions). Input MACs: non-zero event counts x fan-out.')
    print('  Energy: Horowitz 45 nm, MAC 4.6 pJ / AC 0.9 pJ (estimate; memory access ignored).')

    # Sensitivity of the accuracy-vs-T curve to HOW the T-step input is formed
    # (decision D-A). Primary outcome = rebin_scaled above; these are secondary.
    sensitivity: dict = {}
    if not args.no_sensitivity:
        for variant in ('rebin_raw', 'prefix'):
            section(f'Sensitivity: evaluation variant "{variant}"')
            loaders_v = make_eval_loaders_by_T(
                cfg, 'test', indices=test_subset,
                batch_size=dataloaders['test'].batch_size, variant=variant,
            )
            res_v = {}
            for name, ckpt in (('snn_sg', sg_result.checkpoint_path), ('snn_ft', ft_res.checkpoint_path)):
                accs = evaluate_snn_grid_from_checkpoint(ckpt, cfg, loaders_v, device,
                                                         t_eval_grid=sorted(loaders_v))
                res_v[name] = compute_eval_result(seed, name, ann_test_acc, accs, cfg).to_dict()
                print(f'  {name}: ' + ', '.join(f'T={T}: {acc:.4f}' for T, (_, acc) in sorted(accs.items()))
                      + f'  | min qualifying T = {res_v[name]["min_qualifying_T"]}')
            sensitivity[variant] = res_v

    # =========================================================
    # Pilot checks
    # =========================================================
    header('PILOT SANITY CHECKS')
    checks = pilot_checks(
        ann_test_acc, sg_result, hybrid_result, sg_eval, ft_eval,
        cfg.snn.t_train,
    )
    all_passed = print_checks(checks)

    # =========================================================
    # Wall-time summary and extrapolation
    # =========================================================
    header('TIMING SUMMARY')
    n_seeds = len(cfg.experiment.seeds)
    print(f'  Method B (ANN+FT): {fmt_time(hybrid_result.total_wall_s)} for 1 seed')
    print(f'  Method A (SG):     {fmt_time(sg_result.total_wall_s)} for 1 seed')
    total_1 = hybrid_result.total_wall_s + sg_result.total_wall_s
    print(f'  Both methods, 1 seed: {fmt_time(total_1)}')
    print(f'  Estimated {n_seeds} seeds:  {fmt_time(total_1 * n_seeds)} '
          f'(assumes no parallelism)')

    pilot_total = time.time() - pilot_start
    print(f'  Pilot total wall:  {fmt_time(pilot_total)}')

    # =========================================================
    # Save JSON
    # =========================================================
    results_dict = {
        'protocol_version': cfg.experiment.protocol_version,
        'seed':             seed,
        'hardware':         hw,
        'config': {
            'ann_epochs':    cfg.train.ann_epochs,
            'sg_epochs':     cfg.train.snn_sg_epochs,
            'ft_epochs':     cfg.train.snn_ft_epochs,
            't_train':       cfg.snn.t_train,
            't_eval_grid':   cfg.snn.t_eval_grid,
            'accuracy_margin': cfg.experiment.accuracy_margin,
        },
        'ann_baseline': {
            'test_acc':   round(ann_test_acc, 6),
            'test_loss':  round(ann_test_loss, 6),
            'threshold_s': round(threshold_s, 6),
            'best_epoch': ann_res.best_epoch,
            'best_val_acc': round(ann_res.best_val_acc, 6),
            'total_epochs': ann_res.total_epochs,
            'total_updates': ann_res.total_updates,
            'total_wall_s': round(ann_res.total_wall_s, 3),
            'early_stopped': ann_res.early_stopped,
        },
        'method_a_sg': {
            'training': {
                'best_epoch':    sg_result.best_epoch,
                'best_val_acc':  round(sg_result.best_val_acc, 6),
                'total_epochs':  sg_result.total_epochs,
                'total_updates': sg_result.total_updates,
                'total_wall_s':  round(sg_result.total_wall_s, 3),
                'early_stopped': sg_result.early_stopped,
                'train_sample_timesteps': sg_result.train_sample_timesteps,
                'init_calibration': sg_result.extra.get('init_calibration'),
                'init_forward_sample_timesteps': sg_result.extra.get('init_forward_sample_timesteps'),
                'peak_gpu_mb': peak_mb_a,
            },
            'evaluation': sg_eval.to_dict(),
            'op_counts_by_T': {str(T): o for T, o in sorted(sg_ops_by_T.items())},
        },
        'method_b_hybrid': {
            'training': {
                'ann_phase': {
                    'best_epoch':    ann_res.best_epoch,
                    'best_val_acc':  round(ann_res.best_val_acc, 6),
                    'total_epochs':  ann_res.total_epochs,
                    'total_updates': ann_res.total_updates,
                    'total_wall_s':  round(ann_res.total_wall_s, 3),
                    'early_stopped': ann_res.early_stopped,
                },
                'ft_phase': {
                    'best_epoch':    ft_res.best_epoch,
                    'best_val_acc':  round(ft_res.best_val_acc, 6),
                    'total_epochs':  ft_res.total_epochs,
                    'total_updates': ft_res.total_updates,
                    'total_wall_s':  round(ft_res.total_wall_s, 3),
                    'early_stopped': ft_res.early_stopped,
                },
                'calibrated_thresholds': {
                    k: round(v, 6)
                    for k, v in hybrid_result.calibrated_thresholds.items()
                },
                'total_wall_s': round(hybrid_result.total_wall_s, 3),
                'train_sample_timesteps': hybrid_result.train_sample_timesteps,
                'ann_sample_timesteps': ann_res.train_sample_timesteps,
                'ft_sample_timesteps': ft_res.train_sample_timesteps,
                'conversion_info': hybrid_result.conversion_info,
                'conversion_fidelity': hybrid_result.conversion_fidelity,
                'peak_gpu_mb': peak_mb_b,
            },
            'evaluation': ft_eval.to_dict(),
            'op_counts_by_T': {str(T): o for T, o in sorted(ft_ops_by_T.items())},
        },
        'pilot_checks': [
            {'label': label, 'passed': passed, 'detail': detail}
            for label, passed, detail in checks
        ],
        'protocol': {
            't_train': cfg.snn.t_train,
            't_eval_grid': cfg.snn.t_eval_grid,
            'batch_sizes': {'ann': cfg.train.ann_batch_size, 'snn_sg': cfg.train.snn_sg_batch_size,
                            'snn_ft': cfg.train.snn_ft_batch_size},
            'primary_eval_variant': 'rebin_scaled',
            'conversion_gate': cfg.train.conversion_gate,
        },
        'evaluation_sensitivity': sensitivity,
        'timing': {
            'pilot_total_wall_s': round(pilot_total, 3),
            'estimate_5seeds_wall_s': round(total_1 * n_seeds, 3),
        },
    }

    json_path = out_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f'\n  Results saved to: {json_path}')

    # =========================================================
    # Final verdict
    # =========================================================
    header('PILOT VERDICT')
    if all_passed:
        print('  [PASS] Pilot passed all checks.')
        print('  Recommendation: proceed to full experiment (5 seeds).')
        print('  Review timing estimate above and ensure GPU budget is adequate.')
    else:
        failed = [l for l, p, _ in checks if not p]
        print(f'  [FAIL] {len(failed)} check(s) failed:')
        for l in failed:
            print(f'    - {l}')
        print('  Action required: consult scientific tutor before scaling.')
        print('  Do NOT silently adjust protocol to pass checks.')

    print(f'\n{SEP}\n')
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
