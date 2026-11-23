#!/usr/bin/env python3
"""
End-to-end smoke test of the event-based N-MNIST pipeline (protocol v2.0).

This is NOT a training run and produces no accuracy result. It verifies, on
a small subset of ORIGINAL N-MNIST recordings, that:

   1-5  original events load with valid x, y, t (µs), p
   6-8  timestamp-based binning gives [24, 2, 34, 34] and conserves events
  9-10  the ANN aggregate equals the SNN sequence averaged over time (D-B)
   10b  re-binning for every T in the eval grid conserves events and puts the
        105/210-ms saccade boundaries on bin edges (D-A, D-G)
    11  a temporal visualisation is written to the results directory
 12-17  an SNN forward pass, CE loss, backward pass and one optimizer update
        succeed on genuine temporal input with finite loss/gradients, and
        membrane state is reset between batches
 18-19  environment metadata and a machine-readable JSON report are saved

Usage:
    python scripts/test_event_pipeline.py                     # CPU / CUDA auto
    python scripts/test_event_pipeline.py --n-samples 64 --batch-size 16 --device cuda

Exit status: 0 if every critical check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.config import get_config, preprocessing_signature  # noqa: E402
from src.data.nmnist import (NMNISTEventDataset, binning_spec_from_config,  # noqa: E402
                             eval_value_scale)
from src.init_calibration import calibrate_initial_firing  # noqa: E402
from src.data.transforms import temporal_to_aggregate, validate_events  # noqa: E402
from src.models.ann import ANN  # noqa: E402
from src.models.snn import SNN  # noqa: E402
from src.trainer import get_hardware_info, to_time_major  # noqa: E402
from src.utils.reproducibility import make_run_id, set_all_seeds  # noqa: E402


def display_path(p: Path) -> str:
    """Path relative to the repo if it is inside it, else absolute (e.g. Google Drive)."""
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(p).resolve())


# ---------------------------------------------------------------------------
# Check bookkeeping
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "", critical: bool = True) -> bool:
        self.items.append({"name": name, "passed": bool(ok), "critical": critical, "detail": detail})
        tag = "PASS" if ok else ("FAIL" if critical else "WARN")
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    @property
    def critical_failures(self) -> list[dict]:
        return [c for c in self.items if c["critical"] and not c["passed"]]


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Polarity diagnostic
# ---------------------------------------------------------------------------

# Apparent image motion on the sensor during each 105-ms saccade, as implied by
# the motion compensation in tonic.datasets.nmnist.stabilize (dx, dy per saccade).
SACCADE_MOTION = [(3.5, 7.0), (3.5, -7.0), (-7.0, 0.0)]
SACCADE_US = 105_000


def polarity_leading_edge_score(events: np.ndarray) -> list[float]:
    """
    For each saccade, mean projection onto the motion direction of p=1 events
    minus that of p=0 events, computed within 5-ms slices (so the common drift
    cancels). A consistent sign across all three saccades shows the two
    channels carry distinct, motion-dependent (leading vs trailing edge)
    information. Mapping the sign to ON/OFF additionally requires knowing the
    display contrast (bright-on-dark -> leading edge is ON), which is not
    verified here.
    """
    scores = []
    for s, (dx, dy) in enumerate(SACCADE_MOTION):
        d = np.array([dx, dy]) / np.hypot(dx, dy)
        diffs = []
        for t0 in range(s * SACCADE_US, (s + 1) * SACCADE_US, 5_000):
            sl = events[(events["t"] >= t0) & (events["t"] < t0 + 5_000)]
            on, off = sl[sl["p"] == 1], sl[sl["p"] == 0]
            if on.size > 10 and off.size > 10:
                proj_on = (on["x"] * d[0] + on["y"] * d[1]).mean()
                proj_off = (off["x"] * d[0] + off["y"] * d[1]).mean()
                diffs.append(proj_on - proj_off)
        scores.append(float(np.mean(diffs)) if diffs else float("nan"))
    return scores


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_temporal_figure(frames: np.ndarray, info, label: int, file_name: str,
                         n_events_total: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    T = frames.shape[0]
    vmax = max(1.0, float(frames.max()))            # ONE global scale for all bins
    fig = plt.figure(figsize=(1.0 * T + 2, 6.0))
    gs = GridSpec(4, T, height_ratios=[1, 1, 0.1, 1.2], hspace=0.45, wspace=0.08, figure=fig)
    rows = [(0, "p=0 (OFF)", "Reds"), (1, "p=1 (ON)", "Greens")]
    for r, (p, name, cmap) in enumerate(rows):
        for k in range(T):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(frames[k, p], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            n_kp = int(frames[k, p].sum())
            if n_kp == 0:
                for sp in ax.spines.values():
                    sp.set_edgecolor("black"); sp.set_linestyle("--"); sp.set_linewidth(1.2)
                ax.text(16.5, 16.5, "EMPTY", ha="center", va="center", fontsize=7)
            if r == 0:
                ax.set_title(f"bin {k}\n{info.bin_start_us[k]/1000:.1f}–{(info.bin_end_us[k]+1)/1000:.1f} ms",
                             fontsize=7)
            ax.set_xlabel(f"n={n_kp}", fontsize=7, labelpad=2)
            if k == 0:
                ax.set_ylabel(name, fontsize=9)

    ax = fig.add_subplot(gs[3, :])
    c0 = frames[:, 0].sum(axis=(1, 2)); c1 = frames[:, 1].sum(axis=(1, 2))
    idx = np.arange(T)
    ax.bar(idx - 0.2, c0, width=0.4, color="tab:red", label="p=0 (OFF)")
    ax.bar(idx + 0.2, c1, width=0.4, color="tab:green", label="p=1 (ON)")
    for b in (SACCADE_US, 2 * SACCADE_US):            # saccade boundaries in bin units
        pos = (b - info.t_start_us) / (info.t_end_us - info.t_start_us + 1) * T - 0.5
        ax.axvline(pos, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{k}\n{info.bin_start_us[k]/1000:.0f}" for k in idx], fontsize=7)
    ax.set_xlabel("bin index / bin start (ms); dotted = 105-ms saccade boundaries", fontsize=8)
    ax.set_ylabel("events per bin", fontsize=8)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"N-MNIST {file_name}  label={label}  — {T} temporal bins (ON/OFF per dataset bit-23 definition), window "
        f"[{info.t_start_us/1000:.0f}, {(info.t_end_us+1)/1000:.0f}) ms, colour scale shared by all bins (vmax={vmax:.0f})\n"
        f"original events={n_events_total}, binned={info.n_events_binned}, "
        f"dropped (t ≥ window end)={info.n_events_outside_window}, empty bins={info.empty_bins}",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=40, help="recordings to inspect (stratified)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "event_pipeline_smoke"))
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_config()
    if args.data_root:
        cfg.data.data_root = args.data_root
    set_all_seeds(args.seed, force_deterministic=True)
    run_id = make_run_id("event_pipeline_smoke", args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    checks = Checks()
    report: dict = {"run_id": run_id, "protocol_version": cfg.experiment.protocol_version,
                    "preprocessing_signature": preprocessing_signature(cfg),
                    "preprocess_config": asdict(cfg.preprocess), "args": vars(args)}
    t_start = time.time()
    print(f"run_id={run_id}  device={device}  signature={report['preprocessing_signature']}")

    # ------------------------------------------------------------------ 1
    section("1. Load original N-MNIST recordings")
    spec = binning_spec_from_config(cfg)
    train = args.split == "train"
    ds_events = NMNISTEventDataset(cfg.data.data_root, train, spec, "events", download=not args.no_download)
    ds_snn = NMNISTEventDataset(cfg.data.data_root, train, spec, "snn", download=False)
    ds_ann = NMNISTEventDataset(cfg.data.data_root, train, spec, "ann", download=False)
    labels = np.asarray(ds_events.targets)
    rng = np.random.default_rng(args.seed)
    per_class = max(1, args.n_samples // 10)
    subset = sorted(int(i) for c in range(10)
                    for i in rng.choice(np.flatnonzero(labels == c), per_class, replace=False))
    print(f"  split={args.split} size={len(ds_events)}; inspecting {len(subset)} recordings "
          f"({per_class}/class, seed={args.seed})")
    report["subset_indices"] = subset

    # ------------------------------------------------------------------ 2-5, 6-10
    section("2-10. Event statistics, binning, conservation, ANN aggregate")
    per_sample = []
    all_ok = {"coords": True, "shape": True, "conserve": True, "ann": True, "finite": True,
              "monotonic": True, "temporal": True}
    dtype_str = None
    pol_scores = []
    for i in subset:
        ev, y = ds_events[i]
        dtype_str = str(ev.dtype)
        try:
            validate_events(ev, spec.sensor_width, spec.sensor_height, spec.n_polarities)
        except Exception:
            all_ok["coords"] = False
        frames, info, y2 = ds_snn.get_binned(i)
        x_ann, y3 = ds_ann[i]
        n_p = np.bincount(ev["p"], minlength=2)
        all_ok["shape"] &= frames.shape == spec.output_shape
        all_ok["conserve"] &= (info.n_events_binned + info.n_events_outside_window == ev.size
                               and int(frames.sum()) == info.n_events_binned
                               and info.n_events_outside_window == int((ev["t"] > info.t_end_us).sum()))
        all_ok["ann"] &= bool(np.array_equal(x_ann.numpy(), temporal_to_aggregate(frames, "mean"))) and y == y2 == y3
        all_ok["finite"] &= bool(np.isfinite(frames).all())
        all_ok["monotonic"] &= bool(np.all(np.diff(ev["t"]) >= 0))
        all_ok["temporal"] &= bool(np.std(info.events_per_bin) > 0)
        pol_scores.append(polarity_leading_edge_score(ev))
        per_sample.append({
            "index": i, "file": os.path.relpath(ds_events.file_path(i), cfg.data.data_root),
            "label": y, "n_events": int(ev.size),
            "t_min_us": int(ev["t"].min()), "t_max_us": int(ev["t"].max()),
            "n_p0": int(n_p[0]), "n_p1": int(n_p[1]),
            "n_binned": info.n_events_binned, "n_dropped_outside_window": info.n_events_outside_window,
            "events_per_bin": info.events_per_bin, "empty_bins": info.empty_bins,
        })
    n_ev = np.array([s["n_events"] for s in per_sample])
    tmins = np.array([s["t_min_us"] for s in per_sample]); tmaxs = np.array([s["t_max_us"] for s in per_sample])
    p0 = sum(s["n_p0"] for s in per_sample); p1 = sum(s["n_p1"] for s in per_sample)
    dropped = sum(s["n_dropped_outside_window"] for s in per_sample)
    print(f"  event dtype: {dtype_str}")
    print(f"  events/recording: min={n_ev.min()} median={int(np.median(n_ev))} max={n_ev.max()}")
    print(f"  timestamps (µs): t_min in [{tmins.min()}, {tmins.max()}], t_max in [{tmaxs.min()}, {tmaxs.max()}]"
          f"  -> recording length ≈ {np.median(tmaxs)/1e3:.1f} ms")
    print(f"  polarity: p=0 {p0} ({p0/(p0+p1):.1%}), p=1 {p1} ({p1/(p0+p1):.1%})")
    print(f"  window [{spec.window_start_us}, {spec.window_start_us + spec.window_duration_us}) µs, "
          f"{spec.n_bins} bins; events dropped outside window: {dropped} of {int(n_ev.sum())}")
    checks.add("events have integer x,y,t,p fields", dtype_str is not None and all(
        f in dtype_str for f in ("'x'", "'y'", "'t'", "'p'")), dtype_str or "")
    checks.add("coordinates/polarity/timestamps valid", all_ok["coords"])
    checks.add("timestamps monotonic within each recording", all_ok["monotonic"])
    checks.add("timestamps consistent with µs units (recording ≈ 0.3 s)",
               250_000 < float(np.median(tmaxs)) < 400_000, f"median t_max={int(np.median(tmaxs))} µs")
    checks.add(f"SNN representation shape {spec.output_shape}", all_ok["shape"])
    checks.add("event-count conservation (binned + explicitly dropped == original)", all_ok["conserve"],
               f"dropped={dropped}")
    checks.add("ANN representation == SNN sequence averaged over bins (same recording)", all_ok["ann"])
    checks.add("no NaN/Inf in binned frames", all_ok["finite"])
    checks.add("events per bin vary over time in every recording", all_ok["temporal"])
    ps = np.nanmean(np.array(pol_scores), axis=0)
    leader = "p=1" if np.all(ps > 0) else ("p=0" if np.all(ps < 0) else "inconsistent")
    checks.add("polarity channels show consistent leading/trailing-edge structure",
               leader != "inconsistent",
               f"leading polarity={leader}; score (p1-p0) per saccade = {np.round(ps, 3).tolist()} px",
               critical=False)
    sample0_frames, sample0_info, _ = ds_snn.get_binned(subset[0])
    x_example = torch.from_numpy(sample0_frames)
    print(f"  example SNN tensor shape: {tuple(x_example.shape)}; ANN tensor shape: {tuple(ds_ann[subset[0]][0].shape)}")
    report["event_stats"] = {
        "dtype": dtype_str, "timestamp_unit": cfg.preprocess.timestamp_unit,
        "events_per_recording": {"min": int(n_ev.min()), "median": float(np.median(n_ev)), "max": int(n_ev.max())},
        "t_min_us_range": [int(tmins.min()), int(tmins.max())],
        "t_max_us_range": [int(tmaxs.min()), int(tmaxs.max())],
        "polarity_counts": {"p0": int(p0), "p1": int(p1)},
        "events_dropped_outside_window": int(dropped), "events_total": int(n_ev.sum()),
        "polarity_leading_edge_score_per_saccade_px": ps.tolist(),
        "snn_shape": list(x_example.shape), "ann_shape": list(ds_ann[subset[0]][0].shape),
        "bin_edges_us": {"start": sample0_info.bin_start_us, "end_inclusive": sample0_info.bin_end_us},
    }
    report["per_sample"] = per_sample

    # ------------------------------------------------------------------ 10b
    section("10b. Re-binning for the evaluation grid (D-A, D-G)")
    rebin_ok, align_ok = True, True
    rebin_rows = {}
    for T in cfg.snn.t_eval_grid:
        spec_T = binning_spec_from_config(cfg, n_bins=T)
        ds_T = NMNISTEventDataset(cfg.data.data_root, train, spec_T, "snn", download=False,
                                  value_scale=eval_value_scale(cfg, T))
        means = []
        for i in subset[:10]:
            fr, inf, _ = ds_T.get_binned(i)
            rebin_ok &= fr.shape[0] == T and abs(float(fr.sum()) / ds_T.value_scale - inf.n_events_binned) < 1e-3
            align_ok &= 105_000 in inf.bin_start_us and 210_000 in inf.bin_start_us
            means.append(float(fr.mean()))
        rebin_rows[T] = {"value_scale": ds_T.value_scale, "mean_input_per_step": float(np.mean(means))}
        print(f"  T={T:2d}: bin width {315000 / T / 1000:6.3f} ms, scale {ds_T.value_scale:.3f}, "
              f"mean input/step {np.mean(means):.4f}")
    ref = rebin_rows[cfg.snn.t_train]["mean_input_per_step"]
    rate_ok = all(abs(r["mean_input_per_step"] - ref) <= 1e-5 for r in rebin_rows.values())
    checks.add("re-binned sequences have T bins and conserve events for every grid T", rebin_ok)
    checks.add("saccade boundaries (105, 210 ms) are bin edges for every grid T", align_ok)
    checks.add("per-step mean input equal across T (match_training_rate scaling)", rate_ok)
    report["rebinning"] = rebin_rows

    # ------------------------------------------------------------------ 11
    section("11. Temporal visualisation")
    viz_idx = subset[0]
    ev0, y0 = ds_events[viz_idx]
    viz_path = out_dir / "temporal_bins_sample.png"
    save_temporal_figure(sample0_frames, sample0_info, y0, os.path.relpath(ds_events.file_path(viz_idx), cfg.data.data_root),
                         int(ev0.size), viz_path)
    checks.add("temporal visualisation saved", viz_path.exists(), display_path(viz_path))
    report["visualization"] = display_path(viz_path)

    # ------------------------------------------------------------------ 12-17
    section("12-17. SNN forward / loss / backward / optimizer step on temporal input")
    from torch.utils.data import DataLoader, Subset
    from spikingjelly.activation_based import functional
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Subset(ds_snn, subset), batch_size=args.batch_size, shuffle=True, generator=g)
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    x_seq = to_time_major(xb, cfg.snn.t_train)
    print(f"  batch from DataLoader: {tuple(xb.shape)} -> time-major {tuple(x_seq.shape)}")
    checks.add("SNN batch is [T, B, 2, 34, 34] with T = n_time_bins",
               tuple(x_seq.shape) == (cfg.preprocess.n_time_bins, xb.shape[0], 2, 34, 34))
    checks.add("time-major transpose preserves bin order", torch.equal(x_seq[3, 0], xb[0, 3]))

    torch.manual_seed(args.seed)
    snn = SNN(cfg.network.in_channels, cfg.network.conv1_out, cfg.network.conv2_out,
              cfg.network.fc1_out, cfg.network.n_classes, tau=cfg.snn.tau,
              v_threshold=cfg.snn.v_threshold).to(device).train()
    lifs = {"lif1": snn.features[1], "lif2": snn.features[4], "lif3": snn.classifier[2]}
    # Decision D-F: data-driven firing-rate initialisation on training batches.
    calib_seqs = [to_time_major(b[0]).to(device) for b, _ in zip(
        DataLoader(Subset(ds_snn, subset), batch_size=args.batch_size, shuffle=False), range(2))]
    init_report = calibrate_initial_firing(snn, calib_seqs, cfg.snn.init_target_rate)
    print("  init calibration: " + ", ".join(
        f"{k}: gain={v['gain']:.3g}, rate {v['rate_before']:.4f} -> {v['rate_after']:.4f}"
        for k, v in init_report.items()))
    report["init_calibration"] = init_report
    snn.train()
    rates: dict[str, float] = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: rates.__setitem__(n, float(out.detach().mean())))
             for n, m in lifs.items()]
    opt = torch.optim.Adam(snn.parameters(), lr=cfg.train.snn_sg_lr)
    before = [p.detach().clone() for p in snn.parameters()]

    ok_step = True
    try:
        functional.reset_net(snn)
        logits = snn(x_seq)
        loss = nn.functional.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward()
        grads = {n: p.grad for n, p in snn.named_parameters()}
        grad_norms = {n: (float(gr.norm()) if gr is not None else None) for n, gr in grads.items()}
        nn.utils.clip_grad_norm_(snn.parameters(), cfg.train.max_grad_norm)
        opt.step()
    except Exception as exc:
        ok_step = False
        traceback.print_exc()
        checks.add("SNN forward/backward/step executed", False, repr(exc))
    for h in hooks:
        h.remove()
    if ok_step:
        changed = {n: not torch.equal(b, p) for b, (n, p) in zip(before, snn.named_parameters())}
        print(f"  logits {tuple(logits.shape)}, loss={loss.item():.4f}")
        print(f"  LIF mean firing rate (spikes/neuron/step) at init: "
              + ", ".join(f"{k}={v:.4f}" for k, v in rates.items()))
        print("  grad L2 norms: " + ", ".join(f"{k}={v:.2e}" for k, v in grad_norms.items()))
        checks.add("SNN forward pass output [B, 10]", tuple(logits.shape) == (xb.shape[0], 10))
        checks.add("logits finite", bool(torch.isfinite(logits).all()))
        checks.add("cross-entropy loss finite", bool(torch.isfinite(loss)), f"{loss.item():.4f}")
        checks.add("all gradients exist and are finite",
                   all(gr is not None and bool(torch.isfinite(gr).all()) for gr in grads.values()))
        checks.add("non-zero gradient reaches first conv layer", grad_norms["features.0.weight"] > 0,
                   f"{grad_norms['features.0.weight']:.2e}")
        checks.add("optimizer update changed parameters", any(changed.values()),
                   f"{sum(changed.values())}/{len(changed)} tensors changed")
        silent = [k for k, v in rates.items() if v == 0.0]
        checks.add("no LIF layer silent after D-F initialisation (on this batch)", not silent,
                   f"rates: { {k: round(v, 4) for k, v in rates.items()} }")
        checks.add("all weights receive non-zero gradient", all(
            (v or 0) > 0 for k, v in grad_norms.items()),
            ", ".join(f"{k}={v:.1e}" for k, v in grad_norms.items() if k.endswith('weight')))
        report["snn_step"] = {"logits_shape": list(logits.shape), "loss": loss.item(),
                              "lif_firing_rate_at_init": rates, "grad_l2_norms": grad_norms,
                              "params_changed": changed}

        # State management
        snn.eval()
        with torch.no_grad():
            functional.reset_net(snn)
            o1 = snn(x_seq)
            v_after = {k: (float(m.v.abs().max()) if isinstance(m.v, torch.Tensor) else float(m.v))
                       for k, m in lifs.items()}
            functional.reset_net(snn)
            v_reset = {k: (float(m.v.abs().max()) if isinstance(m.v, torch.Tensor) else float(m.v))
                       for k, m in lifs.items()}
            o2 = snn(x_seq)
        checks.add("membrane potential non-zero after a sequence (state carried across bins)",
                   any(v > 0 for v in v_after.values()), str({k: round(v, 4) for k, v in v_after.items()}))
        checks.add("functional.reset_net zeroes all membrane potentials", all(v == 0 for v in v_reset.values()))
        checks.add("identical output after reset (no state leakage between batches)",
                   torch.allclose(o1, o2, atol=1e-6))

    # ANN functional check on the aggregate of the same batch
    ann = ANN(cfg.network.in_channels, cfg.network.conv1_out, cfg.network.conv2_out,
              cfg.network.fc1_out, cfg.network.n_classes).to(device)
    with torch.no_grad():
        ann_logits = ann(xb.mean(dim=1))
    checks.add("ANN forward on aggregate of the same batch -> [B, 10]",
               tuple(ann_logits.shape) == (xb.shape[0], 10) and bool(torch.isfinite(ann_logits).all()))

    # ------------------------------------------------------------------ 18-19
    section("18. Environment")
    hw = get_hardware_info()
    for pkg in ("numpy", "tonic", "matplotlib"):
        try:
            hw[pkg] = importlib.metadata.version(pkg)
        except Exception:
            hw[pkg] = "unknown"
    hw["device_used"] = str(device)
    hw["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    for k, v in hw.items():
        print(f"  {k}: {v}")
    report["environment"] = hw

    section("19. Report")
    report["checks"] = checks.items
    report["critical_failures"] = len(checks.critical_failures)
    report["wall_s"] = round(time.time() - t_start, 2)
    report["status"] = "PASS" if not checks.critical_failures else "FAIL"
    rpath = out_dir / "smoke_report.json"
    rpath.write_text(json.dumps(report, indent=2, default=str))
    print(f"  report: {display_path(rpath)}")
    print(f"\nSTATUS: {report['status']}  ({len(checks.items)} checks, "
          f"{len(checks.critical_failures)} critical failures, "
          f"{sum(1 for c in checks.items if not c['passed'] and not c['critical'])} warnings)")
    return 0 if not checks.critical_failures else 1


if __name__ == "__main__":
    sys.exit(main())
