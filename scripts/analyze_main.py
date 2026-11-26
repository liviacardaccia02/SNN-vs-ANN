#!/usr/bin/env python3
"""
Analysis of the main experiment (protocol P2, decision D-I).

Reads results/main/seed*/ann.json and seed*/T*/result.json (written by
scripts/run_main.py) and produces

  results/main_analysis/analysis.json   every number used in the write-up
  results/main_analysis/tables.md       the same numbers as markdown tables
  report/figures/main_*.{pdf,png}       figures

Sections of analysis.json
  verification      completeness and internal-consistency checks
  ann               per-seed ANN baseline and threshold_s
  per_T             accuracy (per seed, mean, sd), paired hybrid - SG difference
                    with a t-based 95 % CI (n = 3 seeds, df = 2)
  primary           pre-registered outcome: smallest qualifying T per seed/arm
  posthoc_threshold POST-HOC: smallest qualifying T under stricter margins
  conversion        gate / fidelity per T (converted SNN before fine-tuning)
  cost              training budget (sample-time-steps), wall-clock, SynOps,
                    firing rates, energy estimate

Usage:
    python scripts/analyze_main.py [--results results/main]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
GRID = [1, 2, 3, 6, 12, 24]
SEEDS = [0, 1, 2]
N_TRAIN, N_TEST = 48_000, 10_000

# Same palette / style as scripts/make_report_figures.py
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
C_SG, C_HYB, C_ANN = BLUE, ORANGE, INK2
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def msd(xs):
    xs = [x for x in xs if x is not None]
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def ci95(xs):
    """Mean and t-based 95 % CI half-width (df = n - 1)."""
    m, s = msd(xs)
    h = stats.t.ppf(0.975, len(xs) - 1) * s / math.sqrt(len(xs))
    return m, h


def load(res: Path):
    ann = {s: json.loads((res / f"seed{s:02d}" / "ann.json").read_text()) for s in SEEDS}
    R = {s: {T: json.loads((res / f"seed{s:02d}" / f"T{T:02d}" / "result.json").read_text())
             for T in GRID} for s in SEEDS}
    return ann, R


# ---------------------------------------------------------------------------

def verify(ann, R):
    checks = []

    def chk(name, ok):
        checks.append({"check": name, "ok": bool(ok)})

    chk("18 result files present (3 seeds x 6 T)", sum(len(R[s]) for s in SEEDS) == 18)
    chk("same protocol version in every run",
        len({R[s][T]["protocol_version"] for s in SEEDS for T in GRID}) == 1)
    chk("threshold_s identical across T within each seed (one ANN per seed)",
        all(len({R[s][T]["threshold_s"] for T in GRID}) == 1 for s in SEEDS))
    chk("threshold_s = ANN test acc - 0.02",
        all(abs(ann[s]["threshold_s"] - (ann[s]["test_acc"] - 0.02)) < 1e-9 for s in SEEDS))
    chk("hybrid reused the seed's ANN with conv1 scale T/24",
        all(R[s][T]["hybrid"]["conversion_info"]["ann_reused"]
            and abs(R[s][T]["hybrid"]["conversion_info"]["ann_input_scale"] - T / 24) < 1e-9
            for s in SEEDS for T in GRID))
    chk("no conversion-gate failure (all hybrid runs status ok)",
        all(R[s][T]["hybrid"]["status"] == "ok" for s in SEEDS for T in GRID))
    chk("every SNN evaluated on the full test set (10 000) at its training T",
        all(R[s][T][a]["op_counts"]["n_samples"] == N_TEST and R[s][T][a]["op_counts"]["T"] == T
            for s in SEEDS for T in GRID for a in ("sg", "hybrid")))
    chk("sample-time-step budgets = epochs x 48 000 x T (SG, FT) and ANN + FT (hybrid)",
        all(R[s][T]["sg"]["train_sample_timesteps_total"] == R[s][T]["sg"]["training"]["total_epochs"] * N_TRAIN * T
            and R[s][T]["hybrid"]["train_sample_timesteps_ft"] == R[s][T]["hybrid"]["ft_training"]["total_epochs"] * N_TRAIN * T
            and R[s][T]["hybrid"]["train_sample_timesteps_total"]
            == R[s][T]["hybrid"]["train_sample_timesteps_ann"] + R[s][T]["hybrid"]["train_sample_timesteps_ft"]
            for s in SEEDS for T in GRID))
    chk("logged best_val_acc equals max of per-epoch validation curve",
        all(max(R[s][T]["sg"]["training"]["val_acc_per_epoch"]) == R[s][T]["sg"]["training"]["best_val_acc"]
            and max(R[s][T]["hybrid"]["ft_training"]["val_acc_per_epoch"]) == R[s][T]["hybrid"]["ft_training"]["best_val_acc"]
            for s in SEEDS for T in GRID))
    chk("ANN trained once per seed on 24-bin mean (1 440 000 sample-time-steps = 30 epochs)",
        all(ann[s]["input_bins"] == 24 and ann[s]["training"]["train_sample_timesteps"] == 30 * N_TRAIN
            for s in SEEDS))

    def stop_info(tr):
        return {"epochs": tr["total_epochs"], "best_epoch": tr["best_epoch"],
                "early_stopped": tr["early_stopped"]}

    epochs = {f"seed{s}_T{T}": {"sg": stop_info(R[s][T]["sg"]["training"]),
                                "ft": stop_info(R[s][T]["hybrid"]["ft_training"])}
              for s in SEEDS for T in GRID}
    sg_cap = sum(not v["sg"]["early_stopped"] for v in epochs.values())
    ft_cap = sum(not v["ft"]["early_stopped"] for v in epochs.values())
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks), "epochs": epochs,
            "n_runs_reaching_epoch_cap": {"sg": sg_cap, "ft": ft_cap, "of": 18}}


def accuracy(ann, R):
    out = {}
    for T in GRID:
        sg = [R[s][T]["sg"]["test_acc"] for s in SEEDS]
        hy = [R[s][T]["hybrid"]["test_acc"] for s in SEEDS]
        d = [h - g for h, g in zip(hy, sg)]
        dm, dh = ci95(d)
        t_res = stats.ttest_rel(hy, sg)
        out[T] = {
            "sg": {"per_seed": sg, "mean_sd": msd(sg),
                   "val_best_per_seed": [R[s][T]["sg"]["training"]["best_val_acc"] for s in SEEDS]},
            "hybrid": {"per_seed": hy, "mean_sd": msd(hy),
                       "val_best_per_seed": [R[s][T]["hybrid"]["ft_training"]["best_val_acc"] for s in SEEDS]},
            "diff_hybrid_minus_sg": {"per_seed": d, "mean": dm, "ci95_halfwidth": dh,
                                     "paired_t": float(t_res.statistic), "p_two_sided": float(t_res.pvalue)},
            "sg_minus_ann": [R[s][T]["sg"]["test_acc"] - ann[s]["test_acc"] for s in SEEDS],
            "hybrid_minus_ann": [R[s][T]["hybrid"]["test_acc"] - ann[s]["test_acc"] for s in SEEDS],
        }
    # binomial standard error of one test-set accuracy (n = 10 000), for scale
    p = st.mean(out[24]["hybrid"]["per_seed"])
    se = math.sqrt(p * (1 - p) / N_TEST)
    return out, {"binomial_se_at_98pct": se}


def min_T(accs_by_T, thr):
    ok = [T for T in GRID if accs_by_T[T] >= thr]
    return ok[0] if ok else None


def primary_and_posthoc(ann, R):
    acc = {a: {s: {T: R[s][T][a]["test_acc"] for T in GRID} for s in SEEDS} for a in ("sg", "hybrid")}
    primary = {s: {a: min_T(acc[a][s], ann[s]["threshold_s"]) for a in acc} for s in SEEDS}

    # POST-HOC 1: stricter margins around the ANN (margin < 0 = must beat the ANN)
    margins = [0.02, 0.01, 0.005, 0.0, -0.005, -0.01, -0.015]
    ann_margin = {f"{m:+.3f}": {s: {a: min_T(acc[a][s], ann[s]["test_acc"] - m) for a in acc} for s in SEEDS}
                  for m in margins}
    # POST-HOC 2: iso-accuracy relative to each arm's own T = 24 accuracy
    own = {}
    for m in (0.02, 0.01, 0.005):
        own[f"{m:.3f}"] = {s: {a: min_T(acc[a][s], acc[a][s][24] - m) for a in acc} for s in SEEDS}
    # POST-HOC 3: common absolute targets (same bar for both arms)
    absolute = {f"{t:.3f}": {s: {a: min_T(acc[a][s], t) for a in acc} for s in SEEDS}
                for t in (0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99)}
    return primary, {"ann_margin": ann_margin, "own_T24_margin": own, "absolute_target": absolute}


def conversion(R):
    out = {}
    for T in GRID:
        cf = [R[s][T]["hybrid"]["conversion_fidelity"] for s in SEEDS]
        ci = [R[s][T]["hybrid"]["conversion_info"] for s in SEEDS]
        out[T] = {
            "gate_outcomes": [c["gate_outcome"] for c in cf],
            "strict_passed": [c["passed"] for c in cf],
            "ann_val_acc": [c["ann_acc"] for c in cf],
            "converted_val_acc": [c["snn_acc"] for c in cf],
            "agreement": [c["agreement"] for c in cf],
            "sanity_threshold": [c["sanity_threshold"] for c in cf],
            "rate_corr": {l: [c["layer_rate_correlation"][l] for c in cf] for l in ("lif1", "lif2", "lif3")},
            "mean_rate": {l: [c["layer_mean_rate"][l] for c in cf] for l in ("lif1", "lif2", "lif3")},
            "target_rate": {l: [c["layer_mean_target_rate"][l] for c in cf] for l in ("lif1", "lif2", "lif3")},
            "per_layer_map": [c["per_layer_map"] for c in ci],
            "ft_best_val": [R[s][T]["hybrid"]["ft_training"]["best_val_acc"] for s in SEEDS],
            "ft_first_epoch_val": [R[s][T]["hybrid"]["ft_training"]["val_acc_per_epoch"][0] for s in SEEDS],
        }
    return out


def cost(ann, R):
    out = {}
    for T in GRID:
        e = {}
        for a in ("sg", "hybrid"):
            oc = [R[s][T][a]["op_counts"] for s in SEEDS]
            e[a] = {
                "train_sample_timesteps": [R[s][T][a]["train_sample_timesteps_total"] for s in SEEDS],
                "synops_ac": msd([o["synops_ac"] for o in oc]),
                "synops_ac_per_layer": {l: msd([o["synops_ac_per_layer"][l] for o in oc]) for l in ("lif1", "lif2", "lif3")},
                "firing_rate": {l: msd([o["firing_rate_per_layer"][l] for o in oc]) for l in ("lif1", "lif2", "lif3")},
                "spikes_total": msd([sum(o["spikes_per_layer"].values()) for o in oc]),
                "input_macs_event": msd([o["input_macs_event"] for o in oc]),
                "energy_ratio_vs_ann": msd([o["energy_ratio_snn_over_ann"] for o in oc]),
                "energy_uj": msd([o["energy_pj_est"] / 1e6 for o in oc]),
                "peak_gpu_mb": msd([R[s][T][a]["peak_gpu_mb"] for s in SEEDS]),
            }
        e["sg"]["wall_h"] = msd([R[s][T]["sg"]["training"]["total_wall_s"] / 3600 for s in SEEDS])
        e["hybrid"]["wall_h_ft"] = msd([R[s][T]["hybrid"]["ft_training"]["total_wall_s"] / 3600 for s in SEEDS])
        e["hybrid"]["train_sample_timesteps_ft"] = [R[s][T]["hybrid"]["train_sample_timesteps_ft"] for s in SEEDS]
        e["ann_macs"] = R[0][T]["sg"]["op_counts"]["ann_macs"]
        e["ann_energy_uj"] = R[0][T]["sg"]["op_counts"]["ann_energy_pj_est"] / 1e6
        e["input_nonzero_fraction"] = R[0][T]["sg"]["op_counts"]["input_nonzero_fraction"]
        out[T] = e
    out["ann_wall_h"] = msd([ann[s]["training"]["total_wall_s"] / 3600 for s in SEEDS])
    out["total_gpu_h_per_seed"] = {s: sum(R[s][T]["wall_s"] for T in GRID) / 3600
                                   + ann[s]["training"]["total_wall_s"] / 3600 for s in SEEDS}
    return out


# ---------------------------------------------------------------------------
# Figures

def _xT(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(GRID)
    ax.set_xticklabels([str(t) for t in GRID])
    ax.set_xlabel("time-steps T (train = test)")


def fig_accuracy(ann, R, A):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for a, c, lab in (("sg", C_SG, "Method A: direct SG"), ("hybrid", C_HYB, "Method B: hybrid")):
        m = np.array([A[T][a]["mean_sd"][0] for T in GRID]) * 100
        sd = np.array([A[T][a]["mean_sd"][1] for T in GRID]) * 100
        for s in SEEDS:
            ax.plot(GRID, [R[s][T][a]["test_acc"] * 100 for T in GRID], color=c, lw=0.6, alpha=0.35)
        ax.errorbar(GRID, m, yerr=sd, color=c, marker="o", ms=3.5, capsize=2, label=lab)
    am = st.mean(ann[s]["test_acc"] for s in SEEDS) * 100
    tm = st.mean(ann[s]["threshold_s"] for s in SEEDS) * 100
    ax.axhline(am, color=C_ANN, ls="--", lw=1)
    ax.axhline(tm, color=C_ANN, ls=":", lw=1)
    ax.text(24, am + 0.08, "ANN (24-bin mean)", ha="right", va="bottom", fontsize=7, color=C_ANN)
    ax.text(1, tm + 0.08, "threshold = ANN − 2 pp", ha="left", va="bottom", fontsize=7, color=C_ANN)
    _xT(ax)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(94.8, 99.6)
    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.38))
    ax.set_title("(a) accuracy vs T, mean ± sd over 3 seeds (thin: single seeds)")

    d = np.array([A[T]["diff_hybrid_minus_sg"]["mean"] for T in GRID]) * 100
    h = np.array([A[T]["diff_hybrid_minus_sg"]["ci95_halfwidth"] for T in GRID]) * 100
    bx.axhline(0, color=INK2, lw=0.8)
    bx.errorbar(GRID, d, yerr=h, color=INK, marker="o", ms=3.5, capsize=2)
    for s in SEEDS:
        bx.scatter(GRID, [(R[s][T]["hybrid"]["test_acc"] - R[s][T]["sg"]["test_acc"]) * 100 for T in GRID],
                   s=8, color=C_HYB, alpha=0.6, zorder=3)
    _xT(bx)
    bx.set_ylabel("hybrid − SG (pp)")
    bx.set_title("(b) paired difference, 95 % CI (t, df = 2)")
    fig.tight_layout()
    save(fig, "main_fig1_accuracy_vs_T")


def fig_thresholds(P):
    """POST-HOC: min qualifying T vs required margin relative to the ANN."""
    margins = list(P["ann_margin"].keys())
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ylab = GRID + [48]                  # 48 = 'never qualifies' on the plotted axis
    for a, c, off, lab in (("sg", C_SG, -0.06, "direct SG"), ("hybrid", C_HYB, 0.06, "hybrid")):
        xs, ys = [], []
        for i, m in enumerate(margins):
            for s in SEEDS:
                v = P["ann_margin"][m][s][a]
                xs.append(i + off + (s - 1) * 0.04)
                ys.append(v if v is not None else 48)
        ax.scatter(xs, ys, color=c, s=14, label=lab, zorder=3)
    ax.set_yscale("log", base=2)
    ax.set_yticks(ylab)
    ax.set_yticklabels([str(t) for t in GRID] + ["never"])
    ax.set_xticks(range(len(margins)))
    ax.set_xticklabels([f"{(-float(m)*100)+0.0:+.1f}" for m in margins])
    ax.set_xlabel("required accuracy relative to ANN test acc (pp)   [pre-registered: −2.0]")
    ax.set_ylabel("smallest qualifying T")
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Post-hoc: minimum T under stricter thresholds (dots = seeds)")
    fig.tight_layout()
    save(fig, "main_fig2_threshold_sensitivity")


def fig_conversion(C, A):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    conv = np.array([np.mean(C[T]["converted_val_acc"]) for T in GRID]) * 100
    first = np.array([np.mean(C[T]["ft_first_epoch_val"]) for T in GRID]) * 100
    best = np.array([np.mean(C[T]["ft_best_val"]) for T in GRID]) * 100
    annv = np.mean([C[GRID[0]]["ann_val_acc"]]) * 100
    sgv = np.array([np.mean(A[T]["sg"]["val_best_per_seed"]) for T in GRID]) * 100
    ax.plot(GRID, conv, marker="o", ms=3.5, color=YELLOW, label="converted, before FT")
    ax.plot(GRID, first, marker="o", ms=3.5, color=AQUA, label="after 1 FT epoch")
    ax.plot(GRID, best, marker="o", ms=3.5, color=C_HYB, label="after FT (best)")
    ax.plot(GRID, sgv, marker="o", ms=3.5, color=C_SG, label="direct SG (best)")
    ax.axhline(annv, color=C_ANN, ls="--", lw=1)
    ax.text(24, annv - 2.6, "ANN val", fontsize=7, color=C_ANN, ha="right")
    _xT(ax)
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("Conversion fidelity and recovery by fine-tuning (mean of 3 seeds)")
    fig.tight_layout()
    save(fig, "main_fig3_conversion_recovery")


def fig_curves(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, T in zip(axs, (1, 3, 24)):
        for a, key, c in (("sg", "training", C_SG), ("hybrid", "ft_training", C_HYB)):
            for s in SEEDS:
                v = np.array(R[s][T][a][key]["val_acc_per_epoch"]) * 100
                ax.plot(range(1, len(v) + 1), v, color=c, lw=0.9, alpha=0.8,
                        label=("direct SG" if a == "sg" else "hybrid FT") if s == 0 else None)
        ax.set_title(f"T = {T}")
        ax.set_xlabel("epoch")
        ax.set_ylim(93, 99.5)
    axs[0].set_ylabel("validation accuracy (%)")
    axs[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Validation curves, 3 seeds each (max 30 epochs, patience 7)", fontsize=9)
    fig.tight_layout()
    save(fig, "main_fig4_validation_curves")


def _log_ticks(ax, ticks):
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


def fig_cost(R, Cst, A, ann):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for a, c, lab in (("sg", C_SG, "direct SG"), ("hybrid", C_HYB, "hybrid")):
        for s in SEEDS:
            xs = [R[s][T][a]["train_sample_timesteps_total"] / 1e6 for T in GRID]
            ys = [R[s][T][a]["test_acc"] * 100 for T in GRID]
            ax.plot(xs, ys, color=c, lw=0.7, alpha=0.5, marker="o", ms=2.5,
                    label=lab if s == 0 else None)
        for T in GRID:
            ax.annotate(f"{T}", (np.mean([R[s][T][a]["train_sample_timesteps_total"] / 1e6 for s in SEEDS]),
                                 A[T][a]["mean_sd"][0] * 100),
                        textcoords="offset points", xytext=(3, -8 if a == "hybrid" else 3), fontsize=6, color=c)
    _log_ticks(ax, [1.5, 3, 6, 12, 24, 48])
    ax.set_xlabel("training budget (M sample-time-steps, as used)")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) accuracy vs training cost (labels = T)")

    for a, c, lab in (("sg", C_SG, "direct SG"), ("hybrid", C_HYB, "hybrid")):
        xs = [Cst[T][a]["synops_ac"][0] / 1e6 for T in GRID]
        ys = [A[T][a]["mean_sd"][0] * 100 for T in GRID]
        bx.plot(xs, ys, color=c, marker="o", ms=3.5, label=lab)
        for T, x, y in zip(GRID, xs, ys):
            bx.annotate(f"{T}", (x, y), textcoords="offset points", xytext=(3, -8 if a == "hybrid" else 3),
                        fontsize=6, color=c)
    bx.axvline(Cst[24]["ann_macs"] / 1e6, color=C_ANN, ls="--", lw=1)
    bx.text(Cst[24]["ann_macs"] / 1e6 * 1.06, 96.4, "ANN MACs\n(6.6 M)", fontsize=7, color=C_ANN)
    _log_ticks(bx, [2, 4, 8, 16, 32, 64])
    bx.set_xlabel("SynOps (AC) per test sample, millions")
    bx.set_ylabel("test accuracy (%)")
    bx.set_title("(b) accuracy vs inference operations")
    fig.tight_layout()
    save(fig, "main_fig5_cost_tradeoffs")


def fig_rates(Cst):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharex=True)
    for ax, l in zip(axs, ("lif1", "lif2", "lif3")):
        for a, c, lab in (("sg", C_SG, "direct SG"), ("hybrid", C_HYB, "hybrid")):
            m = [Cst[T][a]["firing_rate"][l][0] for T in GRID]
            s = [Cst[T][a]["firing_rate"][l][1] for T in GRID]
            ax.errorbar(GRID, m, yerr=s, color=c, marker="o", ms=3, capsize=2, label=lab)
        _xT(ax)
        ax.set_title({"lif1": "LIF1 (after conv1)", "lif2": "LIF2 (after conv2)", "lif3": "LIF3 (after fc1)"}[l])
    axs[0].set_ylabel("firing rate (spikes / neuron / step)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "main_fig6_firing_rates")


# ---------------------------------------------------------------------------

def tables_md(ann, A, P, Pp, C, Cst, V, se):
    L = []
    pct = lambda x: f"{100*x:.2f}"
    L.append("## Verification\n")
    for c in V["checks"]:
        L.append(f"- [{'x' if c['ok'] else ' '}] {c['check']}")
    L.append(f"\nRuns reaching the 30-epoch cap without early stopping: SG {V['n_runs_reaching_epoch_cap']['sg']}/18, "
             f"FT {V['n_runs_reaching_epoch_cap']['ft']}/18\n")
    L.append("| seed | T | SG epochs / best | SG early-stopped | FT epochs / best | FT early-stopped |\n|---|---|---|---|---|---|")
    for k, v in V["epochs"].items():
        s, T = k.split("_")
        L.append(f"| {s[4:]} | {T[1:]} | {v['sg']['epochs']} / {v['sg']['best_epoch']} | {v['sg']['early_stopped']} | "
                 f"{v['ft']['epochs']} / {v['ft']['best_epoch']} | {v['ft']['early_stopped']} |")
    L.append("\n## ANN baseline\n\n| seed | val (best) | test | threshold_s | best epoch |\n|---|---|---|---|---|")
    for s in SEEDS:
        a = ann[s]
        L.append(f"| {s} | {pct(a['training']['best_val_acc'])} | {pct(a['test_acc'])} | {pct(a['threshold_s'])} | "
                 f"{a['training']['best_epoch']}/30 |")
    L.append(f"\nBinomial SE of one test accuracy near 98 % (n = 10 000): {100*se:.2f} pp\n")
    L.append("## Test accuracy (%)\n\n| T | SG s0 / s1 / s2 | SG mean ± sd | hybrid s0 / s1 / s2 | hybrid mean ± sd | "
             "hybrid − SG (pp), 95 % CI | paired t, p |\n|---|---|---|---|---|---|---|")
    for T in GRID:
        e = A[T]
        d = e["diff_hybrid_minus_sg"]
        L.append(f"| {T} | {' / '.join(pct(x) for x in e['sg']['per_seed'])} | {pct(e['sg']['mean_sd'][0])} ± {pct(e['sg']['mean_sd'][1])} | "
                 f"{' / '.join(pct(x) for x in e['hybrid']['per_seed'])} | {pct(e['hybrid']['mean_sd'][0])} ± {pct(e['hybrid']['mean_sd'][1])} | "
                 f"{100*d['mean']:+.2f} ± {100*d['ci95_halfwidth']:.2f} | t = {d['paired_t']:.1f}, p = {d['p_two_sided']:.4f} |")
    L.append("\n## Primary outcome: smallest qualifying T (threshold = ANN test − 2 pp)\n\n| seed | SG | hybrid |\n|---|---|---|")
    for s in SEEDS:
        L.append(f"| {s} | {P[s]['sg']} | {P[s]['hybrid']} |")
    L.append("\n## POST-HOC: smallest qualifying T, threshold = ANN test acc − margin (seeds 0/1/2)\n\n"
             "| required vs ANN | SG | hybrid |\n|---|---|---|")
    for m, v in Pp["ann_margin"].items():
        f = lambda a: " / ".join(str(v[s][a]) if v[s][a] else "never" for s in SEEDS)
        L.append(f"| {(-float(m)*100)+0.0:+.1f} pp | {f('sg')} | {f('hybrid')} |")
    L.append("\n## POST-HOC: common absolute targets (seeds 0/1/2)\n\n| target | SG | hybrid |\n|---|---|---|")
    for t, v in Pp["absolute_target"].items():
        f = lambda a: " / ".join(str(v[s][a]) if v[s][a] else "never" for s in SEEDS)
        L.append(f"| {100*float(t):.1f} % | {f('sg')} | {f('hybrid')} |")
    L.append("\n## POST-HOC: within-arm iso-accuracy (own T = 24 accuracy − margin; seeds 0/1/2)\n\n| margin | SG | hybrid |\n|---|---|---|")
    for m, v in Pp["own_T24_margin"].items():
        f = lambda a: " / ".join(str(v[s][a]) if v[s][a] else "never" for s in SEEDS)
        L.append(f"| {100*float(m):.1f} pp | {f('sg')} | {f('hybrid')} |")
    L.append("\n## Conversion (validation, before fine-tuning; mean of 3 seeds)\n\n"
             "| T | gate outcome | ANN val | converted val | agreement | rate corr. lif1/2/3 | after 1 FT epoch | after FT (best) |\n"
             "|---|---|---|---|---|---|---|---|")
    for T in GRID:
        c = C[T]
        L.append(f"| {T} | {', '.join(sorted(set(c['gate_outcomes'])))} (strict pass {sum(c['strict_passed'])}/3) | "
                 f"{pct(np.mean(c['ann_val_acc']))} | {pct(np.mean(c['converted_val_acc']))} | {pct(np.mean(c['agreement']))} | "
                 f"{' / '.join(f'{np.mean(c['rate_corr'][l]):.2f}' for l in ('lif1','lif2','lif3'))} | "
                 f"{pct(np.mean(c['ft_first_epoch_val']))} | {pct(np.mean(c['ft_best_val']))} |")
    L.append("\n## Training cost (as used; M sample-time-steps, mean of 3 seeds) and wall-clock (T4)\n\n"
             "| T | SG budget | hybrid budget (ANN + FT) | SG wall (h) | hybrid FT wall (h) |\n|---|---|---|---|---|")
    for T in GRID:
        e = Cst[T]
        L.append(f"| {T} | {np.mean(e['sg']['train_sample_timesteps'])/1e6:.2f} | "
                 f"{np.mean(e['hybrid']['train_sample_timesteps'])/1e6:.2f} (1.44 + {np.mean(e['hybrid']['train_sample_timesteps_ft'])/1e6:.2f}) | "
                 f"{e['sg']['wall_h'][0]:.2f} | {e['hybrid']['wall_h_ft'][0]:.2f} |")
    L.append(f"\nANN training wall-clock: {Cst['ann_wall_h'][0]:.2f} h per seed. Total GPU time per seed: "
             + ", ".join(f"seed {s} {v:.1f} h" for s, v in Cst["total_gpu_h_per_seed"].items()))
    L.append("\n## Inference operations per test sample (mean of 3 seeds)\n\n"
             "| T | SG SynOps (M) | hybrid SynOps (M) | SG E/E_ANN | hybrid E/E_ANN | SG rate lif1/2/3 | hybrid rate lif1/2/3 |\n"
             "|---|---|---|---|---|---|---|")
    for T in GRID:
        e = Cst[T]
        r = lambda a: " / ".join(f"{e[a]['firing_rate'][l][0]:.3f}" for l in ("lif1", "lif2", "lif3"))
        L.append(f"| {T} | {e['sg']['synops_ac'][0]/1e6:.2f} | {e['hybrid']['synops_ac'][0]/1e6:.2f} | "
                 f"{e['sg']['energy_ratio_vs_ann'][0]:.2f} | {e['hybrid']['energy_ratio_vs_ann'][0]:.2f} | {r('sg')} | {r('hybrid')} |")
    L.append(f"\nANN reference: {Cst[24]['ann_macs']:,} MACs per sample.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "main"))
    args = ap.parse_args()
    res = Path(args.results)
    ann, R = load(res)
    V = verify(ann, R)
    A, extra = accuracy(ann, R)
    P, Pp = primary_and_posthoc(ann, R)
    C = conversion(R)
    Cst = cost(ann, R)

    fig_accuracy(ann, R, A)
    fig_thresholds(Pp)
    fig_conversion(C, A)
    fig_curves(R)
    fig_cost(R, Cst, A, ann)
    fig_rates(Cst)

    out = ROOT / "results" / "main_analysis"
    out.mkdir(parents=True, exist_ok=True)
    analysis = {"verification": V, "ann": ann, "per_T": A, "extra": extra, "primary": P,
                "posthoc_threshold": Pp, "conversion": C, "cost": Cst}
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    (out / "tables.md").write_text(tables_md(ann, A, P, Pp, C, Cst, V, extra["binomial_se_at_98pct"]))
    print(f"verification all_ok = {V['all_ok']}")
    print(f"wrote {out/'analysis.json'}, {out/'tables.md'} and report/figures/main_fig*.png/pdf")


if __name__ == "__main__":
    main()
