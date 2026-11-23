#!/usr/bin/env python3
"""
Report figures, generated only from data produced in this repository:
original N-MNIST recordings, simulated LIF neurons, and the JSON reports
written by scripts/test_event_pipeline.py and scripts/check_conversion_and_init.py.

Outputs (PDF for LaTeX/Overleaf + PNG preview) in report/figures/:
  fig1_event_rate_and_bins     mean event rate over time, saccades, old vs new bin edges (D-0, D-G)
  fig2_temporal_bins_example   one recording as 24 ordered ON/OFF bins (D-0)
  fig3_lif_transfer            LIF rate vs target for IF-style, offset and fitted maps (D-B)
  fig4_conversion_gate         converted-SNN validation accuracy per conversion variant (D-B)
  fig5_init_calibration        firing rates at init and early learning, default vs calibrated (D-F)
  fig6_training_budget         configured maximum training budget per arm, sample-time-steps (D-E)

Usage:
    python scripts/make_report_figures.py [--n-recordings 2000]
Missing JSON inputs are reported and the corresponding figure is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data.nmnist import _make_ds  # noqa: E402
from src.data.transforms import bin_boundaries  # noqa: E402

# Validated categorical palette (light surface); aqua/yellow need direct labels.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OFF_C, ON_C = BLUE, ORANGE            # channel identity used in every figure

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.dpi": 200,
    "pdf.fonttype": 42,
})
OUT = PROJECT_ROOT / "report" / "figures"
SACCADES_MS = (105, 210)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote report/figures/{name}.pdf/.png")


def load_json(rel: str):
    p = PROJECT_ROOT / rel
    if not p.exists():
        print(f"  [skip] {rel} not found")
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Layout rules (all figures): no in-figure titles (the LaTeX caption carries
# the description); legends in their own strip ABOVE the axes, never on data;
# reference-line labels outside the plotting area; panel tags (a), (b).

def legend_above(ax, ncol: int, **kw) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=ncol, frameon=False,
              borderaxespad=0.0, handlelength=1.8, columnspacing=1.6, fontsize=8, **kw)


def panel_tag(ax, tag: str) -> None:
    ax.text(-0.02, 1.02, tag, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)


def fig1_event_rate(cfg, n_rec: int, seed: int) -> dict:
    """Mean events per ms over a stratified sample of TRAINING recordings."""
    ds = _make_ds(cfg, True, "events", download=False)
    labels = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), n_rec // 10, replace=False) for c in range(10)])
    W = cfg.preprocess.window_duration_us
    edges = np.arange(0, 336_001, 1_000)                         # 1-ms histogram to 336 ms
    h = np.zeros((2, len(edges) - 1))
    for i in idx:
        ev, _ = ds.get_events(int(i))
        for p in (0, 1):
            h[p] += np.histogram(ev["t"][ev["p"] == p], edges)[0]
    h /= len(idx)                                                # mean events per ms per recording
    t_ms = edges[:-1] / 1000 + 0.5

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.plot(t_ms, h[0], color=OFF_C, label="OFF events (p = 0)")
    ax.plot(t_ms, h[1], color=ON_C, label="ON events (p = 1)")
    peak = h.max()
    top = peak * 1.18                                            # head-room for saccade labels
    for s in SACCADES_MS:
        ax.axvline(s, color=INK2, lw=0.9, ls="--", label="saccade boundary" if s == SACCADES_MS[0] else None)
    ax.axvspan(W / 1000, 336, color=GRID, alpha=0.7, lw=0, label="dropped (t ≥ 315 ms)")
    for k, s in enumerate((0, 105, 210)):
        ax.text(s + 52.5, peak * 1.10, f"saccade {k + 1}", ha="center", va="center", fontsize=7.5, color=INK2)
    # bin edges: old 16-bin grid vs new 24-bin grid, drawn in a strip below zero
    for row, (n, col, lab) in enumerate(((16, INK2, "old 16 bins"), (24, BLUE, "new 24 bins"))):
        st, _ = bin_boundaries(n, 0, W - 1)
        y = -peak * (0.11 + 0.11 * row)
        ax.vlines(np.array(st + [W]) / 1000, y - peak * 0.035, y + peak * 0.035, color=col, lw=0.8)
        ax.text(-4, y, lab, ha="right", va="center", fontsize=7, color=col)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-peak * 0.29, top)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlim(0, 336)
    ax.set_xlabel("time within recording (ms)")
    ax.set_ylabel("mean events per ms")
    legend_above(ax, ncol=4)
    save(fig, "fig1_event_rate_and_bins")
    return {"n_recordings": int(len(idx)), "peak_events_per_ms": float(peak)}


def fig2_example(cfg, index: int) -> dict:
    ds = _make_ds(cfg, True, "snn", download=False)
    frames, info, label = ds.get_binned(index)
    T = frames.shape[0]
    vmax = float(frames.max())
    cols = T // 2
    fig, axes = plt.subplots(4, cols, figsize=(6.3, 2.75),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.05, "height_ratios": [1, 1, 1, 1]})
    for half in (0, 1):
        for r, (p, name, cmap) in enumerate(((1, "ON", "Oranges"), (0, "OFF", "Blues"))):
            for j in range(cols):
                k = half * cols + j
                ax = axes[half * 2 + r, j]
                ax.imshow(frames[k, p], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(GRID); sp.set_linewidth(0.5)
                if r == 0:
                    ax.set_title(f"{info.bin_start_us[k] / 1000:.0f} ms", fontsize=6, pad=1.5, color=INK2)
                if j == 0:
                    ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
    save(fig, "fig2_temporal_bins_example")
    return {"index": index, "label": int(label), "vmax_count": vmax}


def fig3_lif_transfer(cfg) -> dict:
    from spikingjelly.activation_based import neuron
    from src.conversion import fit_lif_input_map
    T = cfg.snn.t_train
    u = torch.linspace(0, 1, 201)

    def rate(X):
        lif = neuron.LIFNode(tau=cfg.snn.tau, v_threshold=1.0, v_reset=None, step_mode="m")
        with torch.no_grad():
            return lif(X.expand(T, -1).contiguous()).mean(0).numpy()

    c, k, rmse = fit_lif_input_map(cfg.snn.tau, T, 1.0)
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.plot(u, u, color=INK2, lw=1.0, ls=":", label="target (= u)")
    ax.plot(u, rate(u.clone()), color=YELLOW, label="IF-style  X = u")
    ax.plot(u, rate(1 + u), color=AQUA, label="offset  X = 1 + u")
    ax.plot(u, rate(c + k * u), color=BLUE, label=f"fitted  X = {c:.2f} + {k:.2f}u")
    ax.set_xlabel("normalised ANN activation u")
    ax.set_ylabel(f"LIF firing rate (spikes / step, T = {T})")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    legend_above(ax, ncol=2)
    save(fig, "fig3_lif_transfer")
    return {"c": c, "k": k, "rmse": rmse}


def fig4_conversion(rep_b, sanity_fraction: float = 0.5, n_classes: int = 10) -> None:
    v = rep_b["B"]["variants"]
    order = [("none", "plain normalisation"), ("offset", "+θ bias offset"),
             ("fitted", "fitted LIF map"), ("data_fitted", "fitted + per-layer data")]
    rows = [(lab, v[key]["fidelity"]["snn_acc"], v[key]["fidelity"]["agreement"], BLUE)
            for key, lab in order if key in v]
    ann = v["data_fitted"]["fidelity"]["ann_acc"]
    drop = v["data_fitted"]["fidelity"]["max_acc_drop"]
    ft = v["data_fitted"].get("fine_tune")
    if ft:
        rows.append(("+ 1 fine-tuning epoch", ft["val_acc_per_epoch"][-1], None, ORANGE))
    # Bars whose label would sit on a reference line get it outside instead.
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, acc, agr, col) in zip(y, rows):
        ax.barh(yi, acc * 100, height=0.6, color=col)
        if acc * 100 >= 30:                       # label inside the bar end
            ax.text(acc * 100 - 1.2, yi, f"{acc * 100:.1f} %", va="center", ha="right",
                    fontsize=7.5, color="white", fontweight="bold")
        else:
            ax.text(acc * 100 + 1.2, yi, f"{acc * 100:.1f} %", va="center", ha="left", fontsize=7.5, color=INK)

    ax.axvline(ann * 100, color=INK, lw=1.0, label=f"ANN accuracy ({ann * 100:.1f} %)")
    ax.axvline((ann - drop) * 100, color=INK2, lw=1.0, ls="--",
               label=f"strict gate ({(ann - drop) * 100:.1f} %)")
    chance = 1.0 / n_classes
    sanity = chance + sanity_fraction * (ann - chance)
    ax.axvline(sanity * 100, color=INK2, lw=1.0, ls=":", label=f"relaxed gate ({sanity * 100:.1f} %)")
    legend_above(ax, ncol=3)
    ax.set_yticks(y, [r[0] + (f"\nagreement {r[2] * 100:.1f} %" if r[2] is not None else "") for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("validation accuracy (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_conversion_gate")


def fig5_init(rep_f) -> None:
    F = rep_f["F"]
    cal = F["calibration"]
    layers = ["lif1", "lif2", "lif3"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.35})
    x = np.arange(3)
    before = [cal[l]["rate_before"] * 100 for l in layers]
    after = [cal[l]["rate_after"] * 100 for l in layers]
    a.bar(x - 0.2, before, 0.38, color=BLUE, label="default init")
    a.bar(x + 0.2, after, 0.38, color=ORANGE, label="calibrated init")
    for xi, vb, va in zip(x, before, after):
        a.text(xi - 0.2, vb + 0.12, f"{vb:.1f}", ha="center", va="bottom", fontsize=7)
        a.text(xi + 0.2, va + 0.12, f"{va:.1f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(x, ["LIF 1", "LIF 2", "LIF 3"])
    a.set_ylim(0, max(after) * 1.25)
    a.set_ylabel("firing rate at init (%)")
    a.grid(axis="x", visible=False)
    legend_above(a, ncol=2)
    panel_tag(a, "(a)")
    accs = [F["default_init"]["val_acc_per_epoch"][-1] * 100, F["firing_rate_init"]["val_acc_per_epoch"][-1] * 100]
    b.bar([0, 1], accs, 0.55, color=[BLUE, ORANGE])
    for xi, v in enumerate(accs):
        b.text(xi, v + 1.5, f"{v:.1f} %", ha="center", va="bottom", fontsize=7.5)
    b.axhline(10, color=INK2, lw=0.9, ls=":", label="chance (10 %)")
    legend_above(b, ncol=1)
    nu = F["firing_rate_init"]["n_updates"]
    b.set_xticks([0, 1], ["default", "calibrated"])
    b.set_ylim(0, 100)
    b.set_ylabel(f"val. accuracy after {nu} updates (%)")
    b.grid(axis="x", visible=False)
    panel_tag(b, "(b)")
    save(fig, "fig5_init_calibration")


def fig6_budget(cfg) -> None:
    n = int(60_000 * cfg.data.train_val_split)
    T = cfg.snn.t_train
    ann = cfg.train.ann_epochs * n * 1
    ft = cfg.train.snn_ft_epochs * n * T
    sg = cfg.train.snn_sg_epochs * n * T
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    ax.barh(1, sg / 1e6, 0.5, color=BLUE, label=f"SNN training ({T} steps / sample)")
    ax.barh(0, ft / 1e6, 0.5, left=ann / 1e6, color=BLUE)
    ax.barh(0, ann / 1e6, 0.5, color=AQUA, label="ANN training (1 step / sample)")
    ax.text(sg / 1e6 + 0.6, 1, f"{sg / 1e6:.1f} M  (SG, {cfg.train.snn_sg_epochs} epochs)", va="center", fontsize=7.5)
    ax.text((ann + ft) / 1e6 + 0.6, 0,
            f"{(ann + ft) / 1e6:.1f} M  (ANN {cfg.train.ann_epochs} ep: {ann / 1e6:.1f} M + "
            f"fine-tune {cfg.train.snn_ft_epochs} ep: {ft / 1e6:.1f} M)", va="center", fontsize=7.5)
    ax.set_yticks([0, 1], ["Hybrid (B)", "Direct SG (A)"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("maximum training budget (million sample-time-steps)")
    ax.set_xlim(0, sg / 1e6 * 1.85)
    ax.grid(axis="y", visible=False)
    legend_above(ax, ncol=2)
    save(fig, "fig6_training_budget")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recordings", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--example-index", type=int, default=15927)
    args = ap.parse_args()
    cfg = get_config()
    meta = {}
    print("fig1 …"); meta["fig1"] = fig1_event_rate(cfg, args.n_recordings, args.seed)
    print("fig2 …"); meta["fig2"] = fig2_example(cfg, args.example_index)
    rep_b = load_json("results/conversion_init_check/report_B.json")
    print("fig3 …"); meta["fig3"] = fig3_lif_transfer(cfg)
    if rep_b:
        print("fig4 …"); fig4_conversion(rep_b, cfg.train.gate_sanity_fraction, cfg.network.n_classes)
    rep_f = load_json("results/conversion_init_check/report_F.json")
    if rep_f:
        print("fig5 …"); fig5_init(rep_f)
    print("fig6 …"); fig6_budget(cfg)
    (OUT / "figures_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
