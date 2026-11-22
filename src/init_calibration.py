"""
Data-driven initialisation of the direct-SG SNN (decision D-F).

Problem (measured on real N-MNIST, protocol v2.0 smoke test): with PyTorch's
default Conv/Linear initialisation, theta=1, tau=2 and raw event counts, the
untrained SNN fires at lif1 ≈ 0.005, lif2 = 0, lif3 = 0 spikes/neuron/step.
FC1/FC2 then receive zero gradient and learning depends entirely on the
surrogate gradient of sub-threshold membranes.

Fix: layer-sequential gain calibration (in the spirit of LSUV, Mishkin &
Matas 2016, and data-driven SNN initialisation, e.g. Rossbroich et al. 2022).
For LIF layer l = 1, 2, 3 in order, multiply the weights and bias of the
preceding Conv/Linear layer by a scalar g_l chosen by bisection so that the
mean firing rate of LIF l on a fixed set of TRAINING batches equals
`target_rate`. Neuron model, threshold, tau, architecture and the random
weight pattern are unchanged; only per-layer scale changes. The procedure is
deterministic given the seed (fixed batches, fixed bisection).

Applied to the direct-SG arm only. The hybrid arm's initial weights come from
conversion (src/conversion.py) and are not modified.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from spikingjelly.activation_based import functional

from src.models.snn import SNN

# Flattened module order of SNN: features[0..5] + classifier[0..3]
#   0 conv1, 1 lif1, 2 pool1, 3 conv2, 4 lif2, 5 pool2, 6 flatten, 7 fc1, 8 lif3, 9 fc2
_CALIB_PAIRS = [(0, 1, 'lif1'), (3, 4, 'lif2'), (7, 8, 'lif3')]


def _modules(snn: SNN) -> list[torch.nn.Module]:
    return list(snn.features) + list(snn.classifier)


def _rate_at(snn: SNN, seqs: list[torch.Tensor], lif_pos: int,
             counter: Optional[dict] = None) -> float:
    mods = _modules(snn)
    if counter is not None:       # forward-only cost in sample-time-steps (D-E)
        counter['forward_sample_timesteps'] = counter.get('forward_sample_timesteps', 0) + sum(
            int(x.shape[0]) * int(x.shape[1]) for x in seqs)
    total, count = 0.0, 0
    with torch.no_grad():
        for x in seqs:
            functional.reset_net(snn)
            h = x
            for m in mods[: lif_pos + 1]:
                h = m(h)
            total += float(h.sum()); count += h.numel()
    functional.reset_net(snn)
    return total / count


def calibrate_initial_firing(
    snn: SNN,
    seqs: list[torch.Tensor],
    target_rate: float = 0.05,
    tol: float = 0.1,
    max_iter: int = 40,
    g_bounds: tuple[float, float] = (1e-3, 1e4),
    counter: Optional[dict] = None,
) -> dict[str, dict]:
    """
    Rescale snn in place so each LIF layer fires at ≈ target_rate.

    Args:
        snn:   freshly initialised SNN (eval mode is set temporarily).
        seqs:  list of time-major input batches [T, B, C, H, W] from TRAINING data.
        target_rate: desired mean spikes/neuron/step.
        tol:   relative tolerance on the achieved rate.
        counter: optional dict; 'forward_sample_timesteps' accumulates the
               forward-only cost of the procedure (decision D-E).

    Returns:
        {lif_name: {'gain': g, 'rate_before': r0, 'rate_after': r1}}

    Raises:
        RuntimeError if a layer cannot reach the target within g_bounds.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must be in (0, 1)")
    was_training = snn.training
    snn.eval()
    mods = _modules(snn)
    report: dict[str, dict] = {}
    for w_pos, lif_pos, name in _CALIB_PAIRS:
        layer = mods[w_pos]
        w0 = layer.weight.detach().clone()
        b0 = None if layer.bias is None else layer.bias.detach().clone()

        def set_gain(g: float) -> None:
            with torch.no_grad():
                layer.weight.copy_(w0 * g)
                if b0 is not None:
                    layer.bias.copy_(b0 * g)

        r_before = _rate_at(snn, seqs, lif_pos, counter)
        lo, hi = math.log(g_bounds[0]), math.log(g_bounds[1])
        set_gain(math.exp(hi))
        if _rate_at(snn, seqs, lif_pos, counter) < target_rate:
            set_gain(1.0)
            raise RuntimeError(f"{name}: target rate {target_rate} unreachable (gain <= {g_bounds[1]})")
        g = 1.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            g = math.exp(mid)
            set_gain(g)
            r = _rate_at(snn, seqs, lif_pos, counter)
            if abs(r - target_rate) <= tol * target_rate:
                break
            if r < target_rate:
                lo = mid
            else:
                hi = mid
        report[name] = {'gain': g, 'rate_before': r_before, 'rate_after': _rate_at(snn, seqs, lif_pos, counter)}
    snn.train(was_training)
    return report


def calibration_batches(loader, n_batches: int, device: torch.device) -> list[torch.Tensor]:
    """
    First n_batches of an UNSHUFFLED view of loader.dataset, time-major, on device.
    Uses a separate DataLoader so the training loader's shuffle RNG is untouched.
    """
    from torch.utils.data import DataLoader
    from src.trainer import to_time_major
    dl = DataLoader(loader.dataset, batch_size=loader.batch_size or 32, shuffle=False)
    out = []
    for i, (x, _) in enumerate(dl):
        if i >= n_batches:
            break
        out.append(to_time_major(x).to(device))
    return out
