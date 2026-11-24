"""
Unit tests for ANN-to-SNN conversion and the fidelity gate (src/conversion.py,
decision D-B). Synthetic data only.

Verified:
  - copy_weights copies exactly and independently
  - activation_percentiles: positive, finite, percentile=100 -> max, bad input raises
  - normalize_for_lif implements W'_l = W_l*lambda_{l-1}/lambda_l,
    b'_l = b_l/lambda_l + theta, W'_out = W_out*lambda_3, thresholds = theta
  - the LIF rate model the method relies on (shifted, clipped ReLU)
  - converted SNN: shape, finiteness, determinism
  - end-to-end: on constant-in-time input the converted SNN reproduces the
    ANN's predictions (gate passes); a deliberately broken conversion fails it
  - gate rejects misaligned loaders
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import functional, neuron

from src.config import get_config
from src.conversion import (
    activation_percentiles,
    conversion_fidelity,
    convert,
    copy_weights,
    normalize_for_lif,
)
from src.models.ann import ANN
from src.models.snn import SNN

T = 24


@pytest.fixture(scope="module")
def cfg():
    return get_config()


def make_ann(seed: int = 0) -> ANN:
    torch.manual_seed(seed)
    return ANN().eval()


def make_snn() -> SNN:
    return SNN(tau=2.0, v_threshold=1.0).eval()


def ann_loader(x: torch.Tensor, y: torch.Tensor, bs: int = 8) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(32, 2, 34, 34, generator=g) * 2.0
    y = torch.randint(0, 10, (32,), generator=g)
    return x, y


# ---------------------------------------------------------------------------
# copy_weights
# ---------------------------------------------------------------------------

class TestCopyWeights:

    @pytest.mark.parametrize("c,i", [("features", 0), ("features", 3), ("classifier", 1), ("classifier", 3)])
    def test_weights_and_bias_match(self, c, i):
        ann, snn = make_ann(), make_snn()
        copy_weights(ann, snn)
        assert torch.equal(getattr(ann, c)[i].weight, getattr(snn, c)[i].weight)
        assert torch.equal(getattr(ann, c)[i].bias, getattr(snn, c)[i].bias)

    def test_copy_is_independent(self):
        ann, snn = make_ann(), make_snn()
        copy_weights(ann, snn)
        with torch.no_grad():
            ann.features[0].weight.add_(1.0)
        assert not torch.equal(ann.features[0].weight, snn.features[0].weight)


# ---------------------------------------------------------------------------
# activation_percentiles
# ---------------------------------------------------------------------------

class TestPercentiles:

    def test_keys_positive_finite(self, data):
        lam = activation_percentiles(make_ann(), ann_loader(*data))
        assert set(lam) == {"conv1", "conv2", "fc1"}
        assert all(v > 0 and torch.isfinite(torch.tensor(v)) for v in lam.values())

    def test_percentile_100_is_max(self, data):
        ann = make_ann()
        lam = activation_percentiles(ann, ann_loader(*data), percentile=100.0)
        with torch.no_grad():
            z = ann.get_activations(data[0])["conv1"].clamp(min=0)
        assert lam["conv1"] == pytest.approx(float(z.max()))

    def test_bad_percentile_raises(self, data):
        with pytest.raises(ValueError):
            activation_percentiles(make_ann(), ann_loader(*data), percentile=0.0)

    def test_rejects_sequence_input(self):
        dl = ann_loader(torch.rand(4, T, 2, 34, 34), torch.zeros(4, dtype=torch.long))
        with pytest.raises(ValueError):
            activation_percentiles(make_ann(), dl)


# ---------------------------------------------------------------------------
# normalize_for_lif
# ---------------------------------------------------------------------------

def test_normalisation_formulas():
    ann, snn = make_ann(), make_snn()
    copy_weights(ann, snn)
    lam = {"conv1": 2.0, "conv2": 4.0, "fc1": 0.5}
    th = normalize_for_lif(snn, lam, v_threshold=1.0, c=0.8, k=1.5)
    assert th == {"lif1": 1.0, "lif2": 1.0, "lif3": 1.0}
    pairs = [(("features", 0), 1.0, 2.0), (("features", 3), 2.0, 4.0), (("classifier", 1), 4.0, 0.5)]
    for (c, i), prev, cur in pairs:
        a, s = getattr(ann, c)[i], getattr(snn, c)[i]
        assert torch.allclose(s.weight, a.weight * 1.5 * prev / cur)
        assert torch.allclose(s.bias, a.bias * 1.5 / cur + 0.8)
    assert torch.allclose(snn.classifier[3].weight, ann.classifier[3].weight * 0.5)
    assert torch.equal(snn.classifier[3].bias, ann.classifier[3].bias)


def test_plain_normalisation_variant():
    ann, snn = make_ann(), make_snn()
    copy_weights(ann, snn)
    normalize_for_lif(snn, {"conv1": 2.0, "conv2": 2.0, "fc1": 2.0}, c=0.0, k=1.0)
    assert torch.allclose(snn.features[0].bias, ann.features[0].bias / 2.0)


def test_fitted_input_map_beats_naive_offset():
    from src.conversion import fit_lif_input_map
    c, k, rmse = fit_lif_input_map(2.0, 24, 1.0)
    assert 0.5 <= c <= 1.2 and k > 0
    u = torch.linspace(0, 1, 101)
    lif = neuron.LIFNode(tau=2.0, v_reset=None, step_mode="m")
    naive = float(((lif((1 + u).expand(24, -1).contiguous()).mean(0) - u) ** 2).mean()) ** 0.5
    assert rmse < naive


def test_lif_rate_model_is_shifted_clipped_relu():
    """The conversion relies on: LIF(tau=2, decay_input, theta=1) is silent for
    X <= 1, fires every step for X >= 2, and is monotone in between."""
    X = torch.tensor([0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])
    lif = neuron.LIFNode(tau=2.0, v_reset=None, step_mode="m")
    r = lif(X.expand(200, -1).contiguous()).mean(0)
    assert r[0] == 0 and r[1] < 0.05     # X == theta approaches threshold only asymptotically
    assert r[5] == pytest.approx(1.0) and r[6] == pytest.approx(1.0)
    assert torch.all(r[1:] >= r[:-1])
    assert 0.0 < r[3] < 1.0


# ---------------------------------------------------------------------------
# convert + gate
# ---------------------------------------------------------------------------

def _trained_ann(x, y, steps=150) -> ANN:
    """ANN fitted to a synthetic, linearly separable task (deterministic)."""
    ann = make_ann(1).train()
    opt = torch.optim.Adam(ann.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(ann(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return ann.eval()


@pytest.fixture(scope="module")
def separable():
    """10 class prototypes + noise; ANN input = mean per-step input."""
    g = torch.Generator().manual_seed(3)
    protos = (torch.rand(10, 2, 34, 34, generator=g) < 0.08).float() * 1.5
    y = torch.arange(10).repeat(6)
    x = protos[y] + 0.05 * torch.rand(len(y), 2, 34, 34, generator=g)
    return x, y


def _seq_loader(x, y, bs=10):
    """Constant-in-time sequence whose mean over T equals x (ideal rate input)."""
    return DataLoader(TensorDataset(x.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous(), y),
                      batch_size=bs, shuffle=False)


def test_converted_snn_shape_finite_deterministic(data):
    ann = make_ann()
    snn, info = convert(ann, make_snn(), ann_loader(*data), input_map="fitted")
    assert set(info) >= {"thresholds", "lambdas", "method", "input_map", "c", "k"}
    x = torch.rand(T, 4, 2, 34, 34)
    functional.reset_net(snn); o1 = snn(x)
    functional.reset_net(snn); o2 = snn(x)
    assert o1.shape == (4, 10) and torch.isfinite(o1).all() and torch.equal(o1, o2)


def test_gate_passes_when_conversion_is_faithful(separable):
    x, y = separable
    ann = _trained_ann(x, y)
    assert (ann(x).argmax(1) == y).float().mean() == 1.0
    snn, info = convert(ann, make_snn(), ann_loader(x, y, 10), percentile=99.0, input_map="fitted")
    rep = conversion_fidelity(ann, snn, ann_loader(x, y, 10), _seq_loader(x, y), info["lambdas"],
                              max_acc_drop=0.02, min_agreement=0.9)
    assert rep.T == T and rep.n_samples == len(y)
    assert rep.passed, rep.to_dict()
    assert all(c > 0.5 for c in rep.layer_rate_correlation.values()), rep.layer_rate_correlation


def test_gate_fails_for_broken_conversion(separable):
    x, y = separable
    ann = _trained_ann(x, y)
    snn, info = convert(ann, make_snn(), ann_loader(x, y, 10), input_map="fitted")
    with torch.no_grad():
        snn.classifier[3].weight.mul_(-1.0)        # sabotage the output layer
    rep = conversion_fidelity(ann, snn, ann_loader(x, y, 10), _seq_loader(x, y), info["lambdas"])
    assert not rep.passed and rep.reasons


def test_gate_rejects_misaligned_loaders(separable):
    x, y = separable
    ann = _trained_ann(x, y, steps=5)
    snn, info = convert(ann, make_snn(), ann_loader(x, y, 10), input_map="fitted")
    perm = torch.roll(y, 1)
    with pytest.raises(ValueError, match="aligned"):
        conversion_fidelity(ann, snn, ann_loader(x, y, 10), _seq_loader(x, perm), info["lambdas"])


def test_data_fitted_requires_aligned_snn_loader(separable):
    x, y = separable
    ann = _trained_ann(x, y, steps=5)
    with pytest.raises(ValueError, match="snn_calibration_loader"):
        convert(ann, make_snn(), ann_loader(x, y, 10), input_map="data_fitted")
    with pytest.raises(ValueError, match="aligned"):
        convert(ann, make_snn(), ann_loader(x, y, 10), input_map="data_fitted",
                snn_calibration_loader=_seq_loader(x, torch.roll(y, 1)))


def test_data_fitted_on_bursty_input_does_not_degrade_fit(separable):
    """On time-varying (bursty) input, per-layer data refinement must give a
    rate MSE no worse than the constant-input curve fit, and the gate report
    must be produced."""
    x, y = separable
    ann = _trained_ann(x, y)
    g = torch.Generator().manual_seed(5)
    profile = torch.rand(T, generator=g) * 2.0                 # mean ≈ 1, bursty
    profile = profile / profile.mean()
    seq = x.unsqueeze(1) * profile.view(1, T, 1, 1, 1)
    sl = DataLoader(TensorDataset(seq, y), batch_size=10, shuffle=False)
    snn_d, info_d = convert(ann, make_snn(), ann_loader(x, y, 10), input_map="data_fitted",
                            snn_calibration_loader=sl, refine_batches=2)
    snn_f, _ = convert(ann, make_snn(), ann_loader(x, y, 10), input_map="fitted")
    from src.conversion import _layer_rates_and_targets
    # Layer 1 sees identical input in both variants, so its fit is directly
    # comparable on the refinement batches (samples 0..19).
    xa, xs = [x[:20]], [seq[:20]]
    r_d, u = _layer_rates_and_targets(ann, snn_d, xa, xs, info_d["lambdas"], 0)
    r_f, _ = _layer_rates_and_targets(ann, snn_f, xa, xs, info_d["lambdas"], 0)
    assert ((r_d - u) ** 2).mean() <= ((r_f - u) ** 2).mean() + 1e-6
    assert set(info_d["per_layer_map"]) == {"lif1", "lif2", "lif3"}
    rep = conversion_fidelity(ann, snn_d, ann_loader(x, y, 10), sl, info_d["lambdas"])
    assert 0.0 <= rep.agreement <= 1.0


# ---------------------------------------------------------------------------
# Staged gate (decision D-B: recheck strictly, then relax to a sanity check)
# ---------------------------------------------------------------------------

def _report(snn_acc, ann_acc=0.913, agreement=0.77, frac=0.5):
    from src.conversion import FidelityReport
    reasons = []
    if snn_acc < ann_acc - 0.02:
        reasons.append("drop")
    if agreement < 0.9:
        reasons.append("agreement")
    thr = 0.1 + frac * (ann_acc - 0.1)
    return FidelityReport(n_samples=100, T=24, ann_acc=ann_acc, snn_acc=snn_acc, agreement=agreement,
                          layer_rate_correlation={}, layer_mean_rate={}, layer_mean_target_rate={},
                          max_acc_drop=0.02, min_agreement=0.9, passed=not reasons, reasons=reasons,
                          sanity_fraction=frac, sanity_threshold=thr,
                          sanity_passed=snn_acc >= thr and snn_acc > 0.1)


def test_sanity_threshold_formula(separable):
    x, y = separable
    ann = _trained_ann(x, y)
    snn, info = convert(ann, make_snn(), ann_loader(x, y, 10), input_map="fitted")
    rep = conversion_fidelity(ann, snn, ann_loader(x, y, 10), _seq_loader(x, y), info["lambdas"],
                              sanity_fraction=0.5, n_classes=10)
    assert rep.sanity_threshold == pytest.approx(0.1 + 0.5 * (rep.ann_acc - 0.1))


@pytest.mark.parametrize("snn_acc,mode,proceed,outcome", [
    (0.905, "staged", True, "strict_pass"),
    (0.737, "staged", True, "sanity_pass"),     # real-data case: strict fails, sanity passes
    (0.231, "staged", False, "fail"),           # offset-only conversion: refused
    (0.737, "enforce", False, "fail"),
    (0.905, "enforce", True, "strict_pass"),
    (0.231, "report", True, "not_enforced"),
])
def test_gate_decision_modes(snn_acc, mode, proceed, outcome):
    from src.conversion import gate_decision
    agreement = 0.95 if snn_acc > 0.9 else 0.77
    rep = _report(snn_acc, agreement=agreement)
    assert gate_decision(rep, mode) is proceed
    assert rep.gate_outcome == outcome


def test_gate_decision_rejects_unknown_mode():
    from src.conversion import gate_decision
    with pytest.raises(ValueError):
        gate_decision(_report(0.8), "lenient")


# ---------------------------------------------------------------------------
# Regression: full-scale sizes (Colab pilot crash, 2026-11-23)
# ---------------------------------------------------------------------------

def test_exact_percentile_matches_torch_quantile_small():
    from src.conversion import exact_percentile
    g = torch.Generator().manual_seed(0)
    v = torch.rand(10_000, generator=g) ** 3
    for p in (50.0, 99.0, 99.9, 100.0):
        assert exact_percentile(v, p) == pytest.approx(float(torch.quantile(v.double(), p / 100)), rel=1e-9)


def test_exact_percentile_above_torch_quantile_limit():
    """47 349 760 = size that crashed the pilot (> 2**24, torch.quantile's limit)."""
    from src.conversion import exact_percentile
    n = 2**24 + 1001
    v = torch.arange(n, dtype=torch.float32)
    assert exact_percentile(v, 100.0) == float(n - 1)
    assert exact_percentile(v, 50.0) == pytest.approx((n - 1) / 2, rel=1e-6)


def test_streaming_pearson_formula_matches_corrcoef():
    """The running-sum Pearson formula used by conversion_fidelity, on fixed
    numbers (no spiking, so no threshold sensitivity)."""
    g = torch.Generator().manual_seed(0)
    a = torch.rand(50_000, generator=g, dtype=torch.float64)
    r = 0.7 * a + 0.3 * torch.rand(50_000, generator=g, dtype=torch.float64)
    sums = torch.zeros(6, dtype=torch.float64)
    for ca, cr in zip(a.split(777), r.split(777)):            # uneven chunks, as in batching
        sums += torch.stack([torch.tensor(float(ca.numel()), dtype=torch.float64),
                             ca.sum(), cr.sum(), (ca * ca).sum(), (cr * cr).sum(), (ca * cr).sum()])
    m, sx, sy, sxx, syy, sxy = (float(v) for v in sums)
    mx, my = sx / m, sy / m
    corr = (sxy / m - mx * my) / ((sxx / m - mx * mx) * (syy / m - my * my)) ** 0.5
    assert corr == pytest.approx(float(torch.corrcoef(torch.stack([a, r]))[0, 1]), abs=1e-10)


def test_streaming_correlation_equals_batch_correlation(separable):
    """End to end: the gate's streamed correlation equals a direct computation
    over the SAME batches. (Different batch sizes change floating-point
    rounding, which can flip near-threshold spikes; the first version of this
    test used one 60-sample batch and failed on Colab for that reason.)"""
    x, y = separable
    ann = _trained_ann(x, y)
    snn, info = convert(ann, make_snn(), ann_loader(x, y, 10), input_map="fitted")
    rep = conversion_fidelity(ann, snn, ann_loader(x, y, 10), _seq_loader(x, y), info["lambdas"])
    from src.conversion import _layer_rates_and_targets
    xa = list(x.split(10))
    xs = [c.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous() for c in x.split(10)]
    r, u = _layer_rates_and_targets(ann, snn, xa, xs, info["lambdas"], 0)
    direct = float(torch.corrcoef(torch.stack([u.double(), r.double()]))[0, 1])
    assert rep.layer_rate_correlation["lif1"] == pytest.approx(direct, abs=1e-6)
    assert rep.layer_mean_rate["lif1"] == pytest.approx(float(r.mean()), abs=1e-6)
