"""
ANN-to-SNN conversion for LIF neurons on temporal event input (decision D-B),
plus the conversion-fidelity gate that must pass before fine-tuning.

Pipeline:
  1. copy_weights(ann, snn)                 — copy Conv/Linear parameters
  2. activation_percentiles(ann, loader)    — lambda_l = p-th percentile of the
                                              post-ReLU ANN activation of layer l
  3. normalize_for_lif(snn, lambdas, theta) — layer-wise weight normalisation
                                              + LIF bias offset (below)
  4. conversion_fidelity(...)               — compare converted SNN with the
                                              ANN on VALIDATION data (gate)

Why the v1.0 method was replaced
--------------------------------
v1.0 set theta_l = lambda_l per layer and copied weights unchanged. Spikes have
amplitude 1, so layer l+1 then received its input scaled by 1/lambda_l relative
to the ANN, and the LIF leak was ignored. It also calibrated on whole-recording
counts while the SNN receives per-bin counts.

Rate model of the LIF used here
-------------------------------
SpikingJelly LIFNode (decay_input=True, soft reset, tau=2):
    H[t] = V[t-1] + (X[t] - V[t-1]) / tau,  spike if H >= theta, V = H - theta*spike
For constant input X the steady state without spiking is V = X, so the neuron
is silent for X <= theta and fires every step for X >= 2*theta (tau=2);
measured: X = 1.2, 1.5, 1.8, 2.1 -> rate 0.33, 0.50, 0.67, 1.0 (theta = 1).
This is a thresholded, clipped and CONCAVE rate curve (near threshold the
inter-spike interval grows like log2(1/(X - theta))) — NOT the IF rate X/theta
that classic conversion assumes. Hence an affine input map X = theta*(c + k*u)
is fitted so that rate(X) ≈ u for u in [0, 1] (fit_lif_input_map; for tau=2,
theta=1, T=24: c=0.83, k=1.36, rate RMSE 0.061 vs 0.118 for the naive
X = theta*(1 + u)):

    target:      rate_l ≈ u_l = clip(a_l / lambda_l, 0, 1),  a_l = ReLU(z_l)
    weights:     W'_l = W_l * k * lambda_{l-1} / lambda_l      (lambda_0 = 1)
    biases:      b'_l = b_l * k / lambda_l + c * theta
    so that      mean X_l = theta * (c + k * z_l / lambda_l)   (theta = 1)
    output FC2:  W'_4 = W_4 * lambda_3,  b'_4 = b_4            (logits ≈ ANN logits)

input_map="data_fitted" (default, D-B) starts from the fitted map and then
refines (c_l, k_l) PER LAYER on real TRAINING recordings, sequentially from
layer 1 to 3, minimising the mean squared error between the SNN's actual
per-neuron firing rate (mean over T) and the ANN target u_l. This corrects for
the fact that event input is not constant in time: bins alternate between
saccade bursts and near-empty turn-arounds, and with a concave rate curve
the rate of a fluctuating input differs from the rate of its mean (Jensen).
Data-based balancing on SNN activity follows Sengupta et al. (2019).

input_map="offset" uses (c, k) = (1, 1); input_map="none" uses (0, 1), i.e.
plain Rueckauer normalisation, which ignores the leak (ablations).
    thresholds:  theta = cfg.snn.v_threshold (1.0) in every LIF layer, i.e. the
                 converted network has exactly the SG-arm hyper-parameters.

This rests on (a) the ANN input being the MEAN per-step SNN input
(ann_representation="mean_over_bins") and (b) a rate approximation that is only
approximate for time-varying, saccade-modulated input over T_train=24 steps.
Whether it is good enough is NOT assumed: conversion_fidelity() measures it, and
the hybrid arm refuses to fine-tune if the gate fails (cfg.train.conversion_gate).

References: Rueckauer et al. (2017) Front. Neurosci.; Rathi et al. (2020).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, neuron

from src.models.ann import ANN
from src.models.snn import SNN

# (container, ann_idx, snn_idx, activation key, lif name)
_LAYER_MAP: list[tuple[str, int, int, str, str]] = [
    ('features',    0, 0, 'conv1', 'lif1'),
    ('features',    3, 3, 'conv2', 'lif2'),
    ('classifier',  1, 1, 'fc1',   'lif3'),
]
_OUTPUT_LAYER: tuple[str, int, int] = ('classifier', 3, 3)


class ConversionGateError(RuntimeError):
    """Raised when the converted SNN does not reproduce the ANN closely enough."""

    def __init__(self, message: str, report: "FidelityReport") -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# 1. Weight copy
# ---------------------------------------------------------------------------

def copy_weights(ann: ANN, snn: SNN) -> None:
    """Copy Conv/Linear weights and biases from ANN to SNN in place (exact copy)."""
    ann.eval()
    snn.eval()
    pairs = [(c, a, s) for c, a, s, _, _ in _LAYER_MAP] + [_OUTPUT_LAYER]
    with torch.no_grad():
        for container, ai, si in pairs:
            a_layer = getattr(ann, container)[ai]
            s_layer = getattr(snn, container)[si]
            _check_shapes(a_layer, s_layer, f"{container}[{ai}]")
            s_layer.weight.copy_(a_layer.weight)
            if a_layer.bias is not None and s_layer.bias is not None:
                s_layer.bias.copy_(a_layer.bias)


# ---------------------------------------------------------------------------
# 2. Activation statistics
# ---------------------------------------------------------------------------

def exact_percentile(v: torch.Tensor, percentile: float) -> float:
    """
    Exact percentile of a 1-D tensor of any size, with linear interpolation
    (identical to torch.quantile's default).

    torch.quantile refuses inputs above 2**24 elements. Full-batch calibration
    on N-MNIST produces ~47 M conv1 activations (128 x 10 batches x 32 x 34 x 34),
    so NumPy is used instead. An earlier float32 linspace subsampling crashed
    there: float32 cannot represent indices above 2**24 exactly.
    """
    import numpy as np
    if v.numel() == 0:
        raise ValueError("empty tensor")
    a = v.detach().cpu().double().numpy() if v.dtype != torch.float64 else v.detach().cpu().numpy()
    return float(np.percentile(a, percentile, method="linear"))


def activation_percentiles(
    ann: ANN,
    loader: DataLoader,
    percentile: float = 99.0,
    device: torch.device = torch.device('cpu'),
    n_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    lambda_l = `percentile`-th percentile of post-ReLU ANN activations
    (clamp(z, 0)) of conv1, conv2, fc1 over the given ANN-input loader.

    Returns {'conv1': ..., 'conv2': ..., 'fc1': ...}. A non-positive value
    (dead layer) raises instead of being replaced by an arbitrary default.
    """
    if not (0.0 < percentile <= 100.0):
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    ann = ann.to(device).eval()
    acc: dict[str, list[torch.Tensor]] = {'conv1': [], 'conv2': [], 'fc1': []}
    with torch.no_grad():
        for b, (x, _) in enumerate(loader):
            if n_batches is not None and b >= n_batches:
                break
            if x.dim() != 4:
                raise ValueError("activation_percentiles expects ANN inputs [B, C, H, W]")
            acts = ann.get_activations(x.to(device))
            for k in acc:
                acc[k].append(acts[k].clamp(min=0).flatten().cpu())
    if not acc['conv1']:
        raise RuntimeError("calibration loader provided zero samples")
    out: dict[str, float] = {}
    for k, vals in acc.items():
        lam = exact_percentile(torch.cat(vals), percentile)
        if not lam > 0.0:
            raise RuntimeError(f"layer '{k}' has no positive activations (lambda={lam}); dead layer")
        out[k] = lam
    return out


# ---------------------------------------------------------------------------
# 2b. LIF input map
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def fit_lif_input_map(tau: float, T: int, v_threshold: float = 1.0) -> tuple[float, float, float]:
    """
    Least-squares fit of (c, k) such that the mean firing rate over T steps of
    a SpikingJelly LIFNode (tau, soft reset, decay_input=True, threshold theta)
    driven by the constant input X = theta*(c + k*u) approximates u, for u on a
    uniform grid in [0, 1]. Data-free and deterministic (grid search).

    Returns (c, k, rmse).
    """
    u = torch.linspace(0.0, 1.0, 101)
    best = (float("inf"), 1.0, 1.0)
    cs = torch.arange(0.50, 1.2001, 0.01)
    ks = torch.arange(0.50, 3.0001, 0.02)
    with torch.no_grad():
        for c in cs:
            X = (c + ks[:, None] * u[None, :]) * v_threshold            # [K, U]
            lif = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=None, step_mode="m")
            r = lif(X.unsqueeze(0).expand(T, -1, -1).contiguous()).mean(0)
            err = ((r - u) ** 2).mean(1)
            i = int(err.argmin())
            if float(err[i]) < best[0]:
                best = (float(err[i]), round(float(c), 2), round(float(ks[i]), 2))
    return best[1], best[2], best[0] ** 0.5


# ---------------------------------------------------------------------------
# 3. Normalisation for LIF
# ---------------------------------------------------------------------------

def normalize_for_lif(
    snn: SNN,
    lambdas: dict[str, float],
    v_threshold: float = 1.0,
    c: float = 1.0,
    k: float = 1.0,
) -> dict[str, float]:
    """
    In-place layer-wise weight normalisation of an SNN whose weights were
    copied from the ANN, with affine LIF input map (c, k) (module docstring).
    Sets every LIF threshold to v_threshold. Returns the thresholds dict.
    """
    prev = 1.0
    with torch.no_grad():
        for container, _, si, key, lif in _LAYER_MAP:
            layer = getattr(snn, container)[si]
            lam = float(lambdas[key])
            layer.weight.mul_(v_threshold * k * prev / lam)
            if layer.bias is not None:
                layer.bias.mul_(v_threshold * k / lam).add_(v_threshold * c)
            snn.set_threshold(lif, v_threshold)
            prev = lam
        container, _, si = _OUTPUT_LAYER
        getattr(snn, container)[si].weight.mul_(prev)
    return snn.get_thresholds()


def _layer_rates_and_targets(ann, snn, xa_list, xs_list, lambdas, upto: int):
    """SNN rates (mean over T) and ANN targets for LIF layer index `upto` (0..2)."""
    from src.trainer import to_time_major
    lif_mod = [snn.features[1], snn.features[4], snn.classifier[2]][upto]
    key = _LAYER_MAP[upto][3]
    out = {}
    h = lif_mod.register_forward_hook(lambda m, i, o: out.__setitem__('r', o.mean(0)))
    rs, us = [], []
    try:
        with torch.no_grad():
            for xa, xs in zip(xa_list, xs_list):
                functional.reset_net(snn)
                snn(to_time_major(xs))
                rs.append(out['r'].flatten())
                us.append((ann.get_activations(xa)[key].clamp(min=0) / lambdas[key]).clamp(max=1).flatten())
    finally:
        h.remove(); functional.reset_net(snn)
    return torch.cat(rs), torch.cat(us)


def refine_input_map_on_data(
    ann: ANN, snn: SNN, base_weights: dict, lambdas: dict, xa_list, xs_list,
    c0: float, k0: float, v_threshold: float = 1.0, rounds: int = 3,
    counter: Optional[dict] = None,
) -> dict[str, tuple[float, float, float]]:
    """
    Sequential per-layer coordinate search for (c_l, k_l). `base_weights` holds
    the plain copied ANN weights/biases per layer index. Modifies snn in place.
    Returns {lif_name: (c_l, k_l, rate_mse)}.
    """
    cs = [round(0.40 + 0.05 * i, 2) for i in range(15)]          # 0.40 .. 1.10
    ks = [round(0.6 + 0.2 * i, 2) for i in range(18)]            # 0.6 .. 4.0
    result = {}
    prev = 1.0
    for li, (container, _, si, key, lif) in enumerate(_LAYER_MAP):
        layer = getattr(snn, container)[si]
        w0, b0 = base_weights[li]
        lam = float(lambdas[key])

        def apply(c, k):
            with torch.no_grad():
                layer.weight.copy_(w0 * (v_threshold * k * prev / lam))
                layer.bias.copy_(b0 * (v_threshold * k / lam) + v_threshold * c)

        def mse(c, k):
            apply(c, k)
            if counter is not None:   # forward-only cost (D-E): SNN T steps + ANN 1 step
                counter['forward_sample_timesteps'] = counter.get('forward_sample_timesteps', 0) + sum(
                    int(xs.shape[0]) * (int(xs.shape[1]) + 1) for xs in xs_list)
            r, u = _layer_rates_and_targets(ann, snn, xa_list, xs_list, lambdas, li)
            return float(((r - u) ** 2).mean())

        c, k = c0, k0
        best = mse(c, k)
        for _ in range(rounds):
            for cand in cs:
                e = mse(cand, k)
                if e < best:
                    best, c = e, cand
            for cand in ks:
                e = mse(c, cand)
                if e < best:
                    best, k = e, cand
        apply(c, k)
        result[lif] = (c, k, best)
        prev = lam
    return result


def convert(
    ann: ANN,
    snn: SNN,
    calibration_loader: DataLoader,
    percentile: float = 99.0,
    device: torch.device = torch.device('cpu'),
    n_batches: Optional[int] = None,
    v_threshold: float = 1.0,
    input_map: str = "data_fitted",
    T: int = 24,
    snn_calibration_loader: Optional[DataLoader] = None,
    refine_batches: int = 4,
) -> tuple[SNN, dict]:
    """
    Full conversion: copy weights, measure lambdas on ANN-input training data,
    normalise for LIF with the chosen input map
    ("data_fitted" | "fitted" | "offset" | "none").
    T is the simulation length the map is fitted for (T_train).
    "data_fitted" needs snn_calibration_loader: SNN sequences of the SAME
    training recordings as calibration_loader (both are read unshuffled from
    their .dataset, and labels are checked).
    Returns (snn, info) with keys 'method', 'thresholds', 'lambdas',
    'percentile', 'input_map', 'c', 'k', 'map_rmse', 'per_layer_map'.
    """
    if input_map in ("fitted", "data_fitted"):
        c, k, rmse = fit_lif_input_map(float(snn.tau), int(T), float(v_threshold))
    elif input_map == "offset":
        c, k, rmse = 1.0, 1.0, float("nan")
    elif input_map == "none":
        c, k, rmse = 0.0, 1.0, float("nan")
    else:
        raise ValueError(f"unknown input_map '{input_map}'")
    cost: dict = {}
    copy_weights(ann, snn)
    base = {li: (getattr(snn, cont)[si].weight.detach().clone(), getattr(snn, cont)[si].bias.detach().clone())
            for li, (cont, _, si, _, _) in enumerate(_LAYER_MAP)}
    lambdas = activation_percentiles(ann, calibration_loader, percentile, device, n_batches)
    thresholds = normalize_for_lif(snn, lambdas, v_threshold, c, k)
    per_layer = {lif: (c, k, float("nan")) for *_, lif in _LAYER_MAP}
    if input_map == "data_fitted":
        if snn_calibration_loader is None:
            raise ValueError("input_map='data_fitted' requires snn_calibration_loader")
        bs = calibration_loader.batch_size or 32
        dla = DataLoader(calibration_loader.dataset, batch_size=bs, shuffle=False)
        dls = DataLoader(snn_calibration_loader.dataset, batch_size=bs, shuffle=False)
        xa_list, xs_list = [], []
        for b, ((xa, ya), (xs, ys)) in enumerate(zip(dla, dls)):
            if b >= refine_batches:
                break
            if not torch.equal(ya, ys):
                raise ValueError("ANN and SNN calibration datasets are not aligned")
            xa_list.append(xa.to(device)); xs_list.append(xs.to(device))
        ann.to(device).eval(); snn.to(device).eval()
        per_layer = refine_input_map_on_data(ann, snn, base, lambdas, xa_list, xs_list,
                                             c, k, v_threshold, counter=cost)
    info = {
        'method': f'layerwise_norm_lif_{input_map}',
        'thresholds': thresholds,
        'lambdas': lambdas,
        'percentile': percentile,
        'input_map': input_map,
        'c': c, 'k': k, 'map_rmse': rmse,
        'per_layer_map': {k_: {'c': v[0], 'k': v[1], 'rate_mse': v[2]} for k_, v in per_layer.items()},
        # Forward-only cost of the data refinement in sample-time-steps (D-E);
        # the lambda statistics add n_batches * batch_size ANN forward passes.
        'refine_forward_sample_timesteps': int(cost.get('forward_sample_timesteps', 0)),
    }
    return snn, info


# ---------------------------------------------------------------------------
# 4. Fidelity gate
# ---------------------------------------------------------------------------

@dataclass
class FidelityReport:
    n_samples: int
    T: int
    ann_acc: float
    snn_acc: float
    agreement: float
    layer_rate_correlation: dict          # Pearson r(ANN clip(a/lambda,0,1), SNN rate)
    layer_mean_rate: dict                 # SNN mean firing rate per LIF layer
    layer_mean_target_rate: dict          # ANN clip(a/lambda,0,1) mean per layer
    max_acc_drop: float
    min_agreement: float
    passed: bool                          # STRICT criteria (recheck), always reported
    reasons: list = field(default_factory=list)
    chance: float = 0.1
    sanity_fraction: float = 0.5
    sanity_threshold: float = float('nan')
    sanity_passed: bool = False           # relaxed criterion (D-B staged gate)
    gate_outcome: str = ""                # set by gate_decision(): strict_pass | sanity_pass | fail | not_enforced

    def to_dict(self) -> dict:
        return asdict(self)


def gate_decision(report: FidelityReport, mode: str) -> bool:
    """
    Decide whether hybrid fine-tuning may proceed (decision D-B, staged gate).

    mode="enforce": proceed only if the strict criteria pass.
    mode="staged" (default): the strict criteria are checked and recorded
        first (the recheck); if they fail, proceed only if the sanity criterion
        passes: converted SNN keeps at least `sanity_fraction` of the ANN's
        above-chance accuracy, acc_SNN >= chance + f * (acc_ANN - chance).
    mode="report": always proceed; outcome is recorded.

    Sets report.gate_outcome and returns True if fine-tuning may proceed.
    """
    if mode not in ("enforce", "staged", "report"):
        raise ValueError(f"unknown conversion_gate mode '{mode}'")
    if report.passed:
        report.gate_outcome = "strict_pass"
    elif report.sanity_passed and mode in ("staged", "report"):
        report.gate_outcome = "sanity_pass"
    elif mode == "report":
        report.gate_outcome = "not_enforced"
    else:
        report.gate_outcome = "fail"
    return report.gate_outcome != "fail"


def conversion_fidelity(
    ann: ANN,
    snn: SNN,
    ann_loader: DataLoader,
    snn_loader: DataLoader,
    lambdas: dict[str, float],
    device: torch.device = torch.device('cpu'),
    n_batches: Optional[int] = None,
    max_acc_drop: float = 0.02,
    min_agreement: float = 0.90,
    sanity_fraction: float = 0.5,
    n_classes: int = 10,
) -> FidelityReport:
    """
    Compare the converted (not yet fine-tuned) SNN with the ANN on the SAME
    samples. ann_loader and snn_loader must iterate the same recordings in the
    same order (e.g. the two unshuffled 'val' loaders of one split); labels are
    checked batch by batch.

    Strict criteria (report.passed):
        snn_acc >= ann_acc - max_acc_drop   and   agreement >= min_agreement.
    Sanity criterion (report.sanity_passed), used by the staged gate:
        snn_acc >= chance + sanity_fraction * (ann_acc - chance), chance = 1/n_classes.
    Per-layer rate correlations are diagnostics (not gate criteria).
    """
    from src.trainer import to_time_major

    ann = ann.to(device).eval()
    snn = snn.to(device).eval()
    lifs = {'lif1': snn.features[1], 'lif2': snn.features[4], 'lif3': snn.classifier[2]}
    keymap = {'lif1': 'conv1', 'lif2': 'conv2', 'lif3': 'fc1'}
    rates: dict[str, torch.Tensor] = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, n=n: rates.__setitem__(n, o.mean(0)))
             for n, m in lifs.items()]
    # Streaming Pearson statistics (float64 running sums). Storing every
    # per-neuron value would need ~3.5 GB for conv1 on the full validation set.
    stats = {k: torch.zeros(6, dtype=torch.float64) for k in lifs}   # n, Sx, Sy, Sxx, Syy, Sxy
    n = correct_a = correct_s = agree = 0
    T = None
    try:
        with torch.no_grad():
            for b, ((xa, ya), (xs_, ys_)) in enumerate(zip(ann_loader, snn_loader)):
                if n_batches is not None and b >= n_batches:
                    break
                if not torch.equal(ya, ys_):
                    raise ValueError("ann_loader and snn_loader are not aligned (labels differ)")
                xa, xs_, y = xa.to(device), xs_.to(device), ya.to(device)
                pa = ann(xa).argmax(1)
                acts = ann.get_activations(xa)
                functional.reset_net(snn)
                seq = to_time_major(xs_)
                T = seq.shape[0]
                ps = snn(seq).argmax(1)
                correct_a += int((pa == y).sum()); correct_s += int((ps == y).sum())
                agree += int((pa == ps).sum()); n += y.numel()
                for lif, key in keymap.items():
                    tgt = (acts[key].clamp(min=0) / lambdas[key]).clamp(max=1.0)
                    a = tgt.flatten().double(); r = rates[lif].flatten().double()
                    stats[lif] += torch.stack([
                        torch.tensor(float(a.numel()), dtype=torch.float64, device=a.device),
                        a.sum(), r.sum(), (a * a).sum(), (r * r).sum(), (a * r).sum(),
                    ]).cpu()
    finally:
        for h in hooks:
            h.remove()
        functional.reset_net(snn)
    if n == 0:
        raise RuntimeError("fidelity check received zero samples")

    corr, mean_rate, mean_tgt = {}, {}, {}
    for lif in lifs:
        m, sx, sy, sxx, syy, sxy = (float(v) for v in stats[lif])
        mx, my = sx / m, sy / m
        vx, vy = sxx / m - mx * mx, syy / m - my * my
        cov = sxy / m - mx * my
        mean_rate[lif] = my; mean_tgt[lif] = mx
        corr[lif] = cov / (vx * vy) ** 0.5 if vx > 1e-12 and vy > 1e-12 else float('nan')

    ann_acc, snn_acc, agreement = correct_a / n, correct_s / n, agree / n
    reasons = []
    if snn_acc < ann_acc - max_acc_drop:
        reasons.append(f"accuracy drop {ann_acc - snn_acc:.4f} > {max_acc_drop}")
    if agreement < min_agreement:
        reasons.append(f"prediction agreement {agreement:.4f} < {min_agreement}")
    chance = 1.0 / n_classes
    sanity_threshold = chance + sanity_fraction * (ann_acc - chance)
    return FidelityReport(
        n_samples=n, T=int(T), ann_acc=ann_acc, snn_acc=snn_acc, agreement=agreement,
        layer_rate_correlation=corr, layer_mean_rate=mean_rate,
        layer_mean_target_rate=mean_tgt, max_acc_drop=max_acc_drop,
        min_agreement=min_agreement, passed=not reasons, reasons=reasons,
        chance=chance, sanity_fraction=sanity_fraction, sanity_threshold=sanity_threshold,
        sanity_passed=bool(snn_acc >= sanity_threshold and snn_acc > chance),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_shapes(ann_layer: nn.Module, snn_layer: nn.Module, label: str) -> None:
    if not hasattr(ann_layer, 'weight') or not hasattr(snn_layer, 'weight'):
        raise ValueError(f"Layer '{label}': missing 'weight' "
                         f"(ANN {type(ann_layer).__name__}, SNN {type(snn_layer).__name__})")
    if ann_layer.weight.shape != snn_layer.weight.shape:
        raise ValueError(f"Layer '{label}': weight shape mismatch "
                         f"ANN {tuple(ann_layer.weight.shape)} != SNN {tuple(snn_layer.weight.shape)}")
