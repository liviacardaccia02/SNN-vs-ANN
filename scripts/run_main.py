#!/usr/bin/env python3
"""
Main experiment runner — protocol P2 (decision D-I).

For every seed:
  1. Train ONE ANN on the mean over cfg.train.ann_reference_bins (24) bins and
     evaluate it on the test set -> threshold_s = ANN test acc - 0.02 (fixed
     for this seed, all T).
  2. For every T in cfg.snn.t_eval_grid ({1, 2, 3, 6, 12, 24}):
       - Method B: convert the seed's ANN (conv1 weights x T/24) -> gate ->
         fine-tune the SNN at T (max 30 epochs, early stopping);
       - Method A: train the SNN with surrogate gradients at T from scratch;
       - evaluate both SNNs on the test set at T (the T they were trained at),
         with hardware-independent operation counts (D-H).
  3. Per seed: smallest qualifying T per method (primary outcome).
Across seeds: mean / sd of accuracy and op counts per T, the per-seed minimum T
and the paired difference.

Resumable: each (seed, T) writes seed{S}/T{T}/result.json and is skipped when
that file exists; the seed's ANN is reused from seed{S}/ann.json. On Colab,
re-running the same command after a disconnect continues where it stopped.
Independent seeds can run in parallel sessions (--seeds 0 / --seeds 1 / ...)
into the same --output-dir; run with --summarize-only at the end.

Usage:
    python scripts/run_main.py --data-root /content/data --output-dir /content/drive/MyDrive/snn_results/main
    python scripts/run_main.py --smoke-test --no-gpu --data-root ./data --output-dir /tmp/main_smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.config import get_config, validate_config
from src.conversion import ConversionGateError
from src.data.nmnist import make_dataloaders
from src.evaluate import evaluate_ann_from_checkpoint, evaluate_snn_grid_from_checkpoint
from src.train_ann import train_ann
from src.train_snn import train_snn_hybrid, train_snn_sg
from src.trainer import TrainingResult, get_hardware_info

SEP = '=' * 72


def parse_args():
    p = argparse.ArgumentParser(description='SNN-vs-ANN main experiment (protocol P2, D-I)')
    p.add_argument('--seeds', type=int, nargs='+', default=None, help='default: cfg.experiment.seeds')
    p.add_argument('--t-grid', type=int, nargs='+', default=None, help='default: cfg.snn.t_eval_grid')
    p.add_argument('--data-root', type=str, default=None)
    p.add_argument('--output-dir', type=str, default='./results/main')
    p.add_argument('--ann-epochs', type=int, default=None)
    p.add_argument('--sg-epochs', type=int, default=None)
    p.add_argument('--ft-epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--smoke-test', action='store_true',
                   help='256 samples per split, gate in report mode (CPU check of the pipeline)')
    p.add_argument('--summarize-only', action='store_true',
                   help='only aggregate existing result files')
    return p.parse_args()


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m {s:02d}s' if h else f'{m}m {s:02d}s'


def _json_dump(obj, path: Path) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)                 # atomic: a disconnect never leaves half a file


def _shrink(loaders: dict, seed: int, n: int = 256, batch_size: int = 64) -> dict:
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    out = {}
    for split, dl in loaders.items():
        rng = np.random.default_rng(1234)    # same subset for every T and representation
        idx = sorted(rng.choice(len(dl.dataset), min(n, len(dl.dataset)), replace=False).tolist())
        out[split] = DataLoader(Subset(dl.dataset, idx), batch_size=batch_size,
                                shuffle=(split == 'train'),
                                generator=torch.Generator().manual_seed(seed))
    return out


def cfg_at_T(cfg, T: int):
    c = copy.deepcopy(cfg)
    c.snn.t_train = T
    c.preprocess.n_time_bins = T
    validate_config(c)
    return c


def _peak_reset(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device):
    return round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == 'cuda' else None


def _training_summary(r: TrainingResult) -> dict:
    return {
        'best_epoch': r.best_epoch, 'best_val_acc': round(r.best_val_acc, 6),
        'total_epochs': r.total_epochs, 'total_updates': r.total_updates,
        'total_wall_s': round(r.total_wall_s, 3), 'early_stopped': r.early_stopped,
        'train_sample_timesteps': r.train_sample_timesteps,
        'val_acc_per_epoch': [round(e.val_acc, 6) for e in r.epoch_logs],
    }


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def run_ann(cfg, seed, out: Path, device, smoke, verbose) -> tuple[TrainingResult, dict]:
    """Train (or reload) the seed's single ANN and its test accuracy / threshold."""
    path = out / 'ann.json'
    if path.exists():
        a = json.loads(path.read_text())
        print(f'  ANN for seed {seed} reused from {path} (test acc {a["test_acc"]:.4f})')
        res = TrainingResult(
            method='ann', seed=seed, best_epoch=a['training']['best_epoch'],
            best_val_acc=a['training']['best_val_acc'], epoch_logs=[],
            total_epochs=a['training']['total_epochs'], total_updates=a['training']['total_updates'],
            total_wall_s=a['training']['total_wall_s'], early_stopped=a['training']['early_stopped'],
            hardware={}, checkpoint_path=Path(a['checkpoint_path']),
            train_sample_timesteps=a['training']['train_sample_timesteps'],
        )
        return res, a

    cfg_ann = cfg_at_T(cfg, cfg.train.ann_reference_bins)
    loaders = make_dataloaders(cfg_ann, experiment_seed=seed, representation='ann')
    if smoke:
        loaders = _shrink(loaders, seed)
    _peak_reset(device)
    res = train_ann(cfg_ann, seed, loaders, device, out / 'ann_ckpt', verbose=verbose)
    test_loss, test_acc = evaluate_ann_from_checkpoint(res.checkpoint_path, cfg_ann, loaders['test'], device)
    a = {
        'seed': seed, 'input_bins': cfg.train.ann_reference_bins,
        'test_acc': test_acc, 'test_loss': test_loss,
        'threshold_s': test_acc - cfg.experiment.accuracy_margin,
        'checkpoint_path': str(res.checkpoint_path),
        'training': _training_summary(res), 'peak_gpu_mb': _peak_mb(device),
    }
    _json_dump(a, path)
    print(f'  ANN seed {seed}: test acc {test_acc:.4f} -> threshold_s {a["threshold_s"]:.4f}')
    return res, a


def run_T(cfg, seed, T, ann_res, ann_info, out: Path, device, smoke, verbose) -> dict:
    cfg_T = cfg_at_T(cfg, T)
    snn_loaders = make_dataloaders(cfg_T, experiment_seed=seed, representation='snn')
    ann_loaders = make_dataloaders(cfg_T, experiment_seed=seed, representation='ann')
    if smoke:
        snn_loaders, ann_loaders = _shrink(snn_loaders, seed), _shrink(ann_loaders, seed)
    ckpt = out / 'ckpt'
    ckpt.mkdir(parents=True, exist_ok=True)
    thr = ann_info['threshold_s']
    test = {T: snn_loaders['test']}        # trained at T, evaluated at T: no rescaling (T/T_train = 1)
    result = {'seed': seed, 'T': T, 'threshold_s': thr, 'ann_test_acc': ann_info['test_acc'],
              'protocol_version': cfg.experiment.protocol_version}

    # ---- Method B: hybrid ----
    print(f'\n--- seed {seed} | T={T} | Method B (convert seed ANN, fine-tune) ---')
    _peak_reset(device)
    t0 = time.time()
    try:
        hyb = train_snn_hybrid(cfg_T, seed, snn_loaders, ann_loaders, device, ckpt, verbose=verbose,
                               pretrained_ann=ann_res, ann_input_bins=cfg.train.ann_reference_bins)
        ops: dict = {}
        (loss, acc), = evaluate_snn_grid_from_checkpoint(
            hyb.ft_result.checkpoint_path, cfg_T, test, device, t_eval_grid=[T], ops_out=ops).values()
        result['hybrid'] = {
            'status': 'ok', 'test_acc': acc, 'test_loss': loss, 'qualifies': acc >= thr,
            'conversion_fidelity': hyb.conversion_fidelity,
            'conversion_info': hyb.conversion_info,
            'ft_training': _training_summary(hyb.ft_result),
            'train_sample_timesteps_total': hyb.train_sample_timesteps,
            'train_sample_timesteps_ann': ann_res.train_sample_timesteps,
            'train_sample_timesteps_ft': hyb.ft_result.train_sample_timesteps,
            'op_counts': ops[T], 'peak_gpu_mb': _peak_mb(device),
            'wall_s_ft': round(time.time() - t0, 3),
        }
    except ConversionGateError as e:
        print(f'  conversion gate FAILED at T={T}: {e}')
        result['hybrid'] = {'status': 'gate_fail', 'test_acc': None, 'qualifies': False,
                            'conversion_fidelity': e.report.to_dict()}

    # ---- Method A: direct SG ----
    print(f'\n--- seed {seed} | T={T} | Method A (direct surrogate gradient) ---')
    _peak_reset(device)
    sg = train_snn_sg(cfg_T, seed, snn_loaders, device, ckpt, verbose=verbose)
    ops = {}
    (loss, acc), = evaluate_snn_grid_from_checkpoint(
        sg.checkpoint_path, cfg_T, test, device, t_eval_grid=[T], ops_out=ops).values()
    result['sg'] = {
        'status': 'ok', 'test_acc': acc, 'test_loss': loss, 'qualifies': acc >= thr,
        'training': _training_summary(sg),
        'init_calibration': sg.extra.get('init_calibration'),
        'train_sample_timesteps_total': sg.train_sample_timesteps,
        'op_counts': ops[T], 'peak_gpu_mb': _peak_mb(device),
    }
    h = result['hybrid']
    print(f'\n  seed {seed} T={T}: threshold {thr:.4f} | '
          f'SG {acc:.4f}{" *" if acc >= thr else ""} | '
          f'hybrid {h["test_acc"]:.4f}{" *" if h["qualifies"] else ""}' if h['test_acc'] is not None
          else f'\n  seed {seed} T={T}: SG {acc:.4f} | hybrid gate_fail')
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarize(out_root: Path, grid: list[int]) -> dict:
    per_seed: dict = {}
    for sd in sorted(out_root.glob('seed*')):
        seed = int(sd.name[4:])
        rows = {}
        for T in grid:
            f = sd / f'T{T:02d}' / 'result.json'
            if f.exists():
                rows[T] = json.loads(f.read_text())
        if rows:
            per_seed[seed] = rows
    if not per_seed:
        print('  no results yet')
        return {}

    def min_T(rows, arm):
        ok = [T for T in sorted(rows) if rows[T][arm]['qualifies']]
        return ok[0] if ok else None

    summary = {'grid': grid, 'seeds': sorted(per_seed), 'per_seed': {}, 'per_T': {}}
    for seed, rows in per_seed.items():
        complete = all(T in rows for T in grid)
        summary['per_seed'][seed] = {
            'complete': complete,
            'min_T_sg': min_T(rows, 'sg'), 'min_T_hybrid': min_T(rows, 'hybrid'),
            'threshold_s': next(iter(rows.values()))['threshold_s'],
        }
    for T in grid:
        rs = [per_seed[s][T] for s in per_seed if T in per_seed[s]]
        if not rs:
            continue
        entry = {'n_seeds': len(rs)}
        for arm in ('sg', 'hybrid'):
            accs = [r[arm]['test_acc'] for r in rs]
            ok = [r[arm] for r in rs if r[arm]['status'] == 'ok']
            entry[arm] = {
                'acc_mean_sd': _mean_sd(accs),
                'n_qualify': sum(r[arm]['qualifies'] for r in rs),
                'synops_mean_sd': _mean_sd([o['op_counts']['synops_ac'] for o in ok]),
                'energy_ratio_mean_sd': _mean_sd([o['op_counts']['energy_ratio_snn_over_ann'] for o in ok]),
                'train_sample_timesteps_mean': _mean_sd([o['train_sample_timesteps_total'] for o in ok])[0],
                'n_gate_fail': sum(r[arm]['status'] == 'gate_fail' for r in rs),
            }
        entry['paired_acc_diff_hybrid_minus_sg'] = _mean_sd(
            [r['hybrid']['test_acc'] - r['sg']['test_acc'] for r in rs if r['hybrid']['test_acc'] is not None])
        summary['per_T'][T] = entry

    print(f'\n{SEP}\n  SUMMARY ({len(per_seed)} seed(s); mean ± sd over seeds)\n{SEP}')
    print(f'  {"T":>3} | {"SG acc":>15} q | {"hybrid acc":>15} q | {"SG SynOps":>11} {"hyb SynOps":>11} | '
          f'{"E/E_ANN SG":>10} {"hyb":>5}')
    for T, e in summary['per_T'].items():
        def ms(v, pct=True):
            m, s = v
            return '—' if m is None else (f'{100*m:6.2f}±{100*s:4.2f}%' if pct else f'{m:,.0f}')
        n = e['n_seeds']
        print(f'  {T:3d} | {ms(e["sg"]["acc_mean_sd"]):>15} {e["sg"]["n_qualify"]}/{n} | '
              f'{ms(e["hybrid"]["acc_mean_sd"]):>15} {e["hybrid"]["n_qualify"]}/{n} | '
              f'{ms(e["sg"]["synops_mean_sd"], False):>11} {ms(e["hybrid"]["synops_mean_sd"], False):>11} | '
              f'{(e["sg"]["energy_ratio_mean_sd"][0] or 0):10.2f} {(e["hybrid"]["energy_ratio_mean_sd"][0] or 0):5.2f}')
    print('  q = seeds qualifying (acc >= that seed\'s ANN test acc - 2 pp)')
    print('\n  Primary outcome, smallest qualifying T per seed:')
    for seed, s in summary['per_seed'].items():
        print(f'    seed {seed}: SG = {s["min_T_sg"]}   hybrid = {s["min_T_hybrid"]}'
              f'{"" if s["complete"] else "   (incomplete: some T missing)"}')
    _json_dump(summary, out_root / 'summary.json')
    print(f'\n  summary saved: {out_root / "summary.json"}')
    return summary


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = get_config()
    if args.data_root:   cfg.data.data_root = args.data_root
    if args.ann_epochs:  cfg.train.ann_epochs = args.ann_epochs
    if args.sg_epochs:   cfg.train.snn_sg_epochs = args.sg_epochs
    if args.ft_epochs:   cfg.train.snn_ft_epochs = args.ft_epochs
    if args.batch_size:
        cfg.train.ann_batch_size = cfg.train.snn_sg_batch_size = cfg.train.snn_ft_batch_size = args.batch_size
    if args.smoke_test:
        cfg.train.conversion_gate = 'report'
        cfg.train.calib_batches = 1
        cfg.train.conversion_refine_batches = 1
    seeds = args.seeds or cfg.experiment.seeds
    grid = sorted(args.t_grid or cfg.snn.t_eval_grid)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize(out_root, grid)
        return

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    hw = get_hardware_info()
    print(f'{SEP}\n  SNN-vs-ANN MAIN EXPERIMENT (P2) | protocol {cfg.experiment.protocol_version}\n{SEP}')
    print(f'  seeds {seeds} | T grid {grid} | ANN on mean over {cfg.train.ann_reference_bins} bins')
    print(f'  max epochs: ANN {cfg.train.ann_epochs}, SG {cfg.train.snn_sg_epochs}, '
          f'FT {cfg.train.snn_ft_epochs} (patience {cfg.train.early_stop_patience})')
    print(f'  device {hw["cuda_device"] if device.type == "cuda" else "CPU"} | output {out_root}')
    _json_dump({'hardware': hw, 'seeds': seeds, 'grid': grid, 'smoke_test': args.smoke_test,
                'protocol_version': cfg.experiment.protocol_version}, out_root / f'run_info_{int(time.time())}.json')

    start = time.time()
    for seed in seeds:
        sdir = out_root / f'seed{seed:02d}'
        sdir.mkdir(exist_ok=True)
        print(f'\n{SEP}\n  SEED {seed}\n{SEP}')
        ann_res, ann_info = run_ann(cfg, seed, sdir, device, args.smoke_test, not args.quiet)
        for T in grid:
            tdir = sdir / f'T{T:02d}'
            if (tdir / 'result.json').exists():
                print(f'  seed {seed} T={T}: done earlier, skipped')
                continue
            tdir.mkdir(exist_ok=True)
            t0 = time.time()
            res = run_T(cfg, seed, T, ann_res, ann_info, tdir, device, args.smoke_test, not args.quiet)
            res['wall_s'] = round(time.time() - t0, 3)
            _json_dump(res, tdir / 'result.json')
            print(f'  seed {seed} T={T} finished in {fmt_time(res["wall_s"])} '
                  f'(elapsed {fmt_time(time.time() - start)})')
    summarize(out_root, grid)


if __name__ == '__main__':
    main()
