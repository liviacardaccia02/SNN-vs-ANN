"""
Hardware-independent operation counts for the ANN and the SNN (decision D-H).

Why this exists
---------------
Wall-clock time depends on the machine, and a GPU simulates an SNN densely:
every neuron is updated at every step whether it spiked or not. So the GPU
time of an SNN measures the simulator, not the network. The field-standard,
hardware-independent alternative is to count operations analytically from
the network structure and the activity actually observed on the data:

* ANN: one multiply-accumulate (MAC) per synapse per inference.
* SNN: a binary spike reaching a synapse needs only an addition
  (accumulate, AC) of the weight; a neuron that does not spike costs no
  synaptic operation. SynOps = sum over spikes of the neuron's fan-out.

Counting rules (all per test sample, averaged over the test set)
---------------------------------------------------------------
* ``synops_ac``: every spike of LIF layer l is multiplied by its fan-out,
  i.e. the number of synapses it drives in the next Conv/Linear layer.
  Average pooling between a LIF layer and the next weight layer is linear,
  so it is folded into those weights (each pooled spike still reaches the
  same synapses); spikes in rows/columns that the pooling drops have fan-out 0.
* ``input_macs_event``: the first layer receives event COUNTS (not binary
  spikes), so its operations are MACs. Event-driven counting: one MAC per
  non-zero input element per synapse it drives, summed over the T steps.
* ``input_macs_dense``: the same layer computed densely at every step
  (T x conv1 MACs), for reference.
* ``neuron_updates``: membrane-state updates, T x number of LIF neurons.
  Reported separately because they are not synaptic operations.
* ``dense_macs_equivalent``: T x ANN MACs, what a clock-driven dense
  simulation (our GPU runs) executes. Reported to show the gap between
  "what the network needs" and "what the simulator did".
* Biases, pooling additions and the output averaging are not counted
  (they are small and identical in kind for both methods).

Energy estimate (derived, optional)
-----------------------------------
``energy_pj_est`` converts the counts with the 45-nm CMOS figures of
Horowitz (ISSCC 2014) for 32-bit floating point: MAC = 3.7 pJ (mult) +
0.9 pJ (add) = 4.6 pJ, AC = 0.9 pJ. This is the convention used across
the SNN literature; it is an estimate under stated assumptions, not a
measurement, and ignores memory access.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron

E_MAC_PJ = 4.6
E_AC_PJ = 0.9

_WEIGHT_LAYERS = (nn.Conv2d, nn.Linear)


def _leaves(model: nn.Module) -> list[nn.Module]:
    """Leaf modules in registration order (= forward order for our Sequential
    models). A spiking neuron counts as a leaf even though SpikingJelly gives it
    a surrogate-function child module."""
    out: list[nn.Module] = []

    def walk(m: nn.Module) -> None:
        children = list(m.children())
        if isinstance(m, neuron.BaseNode) or not children:
            out.append(m)
            return
        for c in children:
            walk(c)

    walk(model)
    return out


def _structural(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply m with every weight set to 1 and no bias, pooling as a SUM, so
    that d(sum out)/d(x_i) is the number of synapses input i drives."""
    if isinstance(m, nn.Conv2d):
        return F.conv2d(x, torch.ones(m.weight.shape, dtype=x.dtype, device=x.device), None,
                        m.stride, m.padding, m.dilation, m.groups)
    if isinstance(m, nn.Linear):
        return F.linear(x, torch.ones(m.weight.shape, dtype=x.dtype, device=x.device))
    if isinstance(m, nn.AvgPool2d):
        k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        return F.avg_pool2d(x, m.kernel_size, m.stride, m.padding, m.ceil_mode,
                            m.count_include_pad) * (k[0] * k[1])
    if isinstance(m, nn.Flatten):
        return x.flatten(1)
    if isinstance(m, (nn.ReLU, nn.Identity, nn.Dropout)):
        return x
    raise TypeError(f"op counting does not support {type(m).__name__}")


def _segments(model: nn.Module) -> list[tuple[str, list[nn.Module]]]:
    """Split the leaf sequence at the LIF layers.

    Returns [(source, modules)], where source is 'input' or 'lif1', 'lif2', ...
    and modules are the layers between that source and the next LIF/output.
    """
    segs: list[tuple[str, list[nn.Module]]] = [("input", [])]
    k = 0
    for m in _leaves(model):
        if isinstance(m, neuron.BaseNode):
            k += 1
            segs.append((f"lif{k}", []))
        else:
            segs[-1][1].append(m)
    for name, mods in segs:
        n_w = sum(isinstance(m, _WEIGHT_LAYERS) for m in mods)
        if n_w != 1:
            raise ValueError(f"segment after '{name}' has {n_w} weight layers; expected 1")
    return segs


def fanout_maps(model: nn.Module, in_shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """{source: tensor of the source's shape (no batch/time dims)} giving the
    number of synapses each element drives in the next weight layer."""
    maps: dict[str, torch.Tensor] = {}
    shape = tuple(in_shape)
    for name, mods in _segments(model):
        x = torch.ones((1,) + shape, dtype=torch.float64, requires_grad=True)
        y = x
        for m in mods:
            y = _structural(m, y)
        y.sum().backward()
        maps[name] = x.grad[0].detach().float().round()
        # shape of the next LIF layer's output = output of this segment's weight layer
        shape = tuple(y.shape[1:])
    return maps


def ann_macs(model: nn.Module, in_shape: tuple[int, ...]) -> dict[str, int]:
    """MACs per sample of every Conv/Linear layer of a (non-spiking) model.

    Counted as real synapses: products with zero padding are excluded (the
    SNN fan-out count excludes them by construction, so both arms use the
    same rule). Tools such as ptflops include padding and report e.g.
    665 856 instead of 640 000 MACs for conv1."""
    out: dict[str, int] = {}
    hooks = []

    def make_hook(name):
        def hook(m, inp, o):
            out[name] = _synapse_count(m, tuple(inp[0].shape[1:]))
        return hook

    for name, m in model.named_modules():
        if isinstance(m, _WEIGHT_LAYERS):
            hooks.append(m.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        p = next(model.parameters())
        with torch.no_grad():
            model(torch.zeros((1,) + tuple(in_shape), device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return out


class SNNOpCounter:
    """Accumulates spike-driven operation counts while an SNN is evaluated.

    Usage:
        counter = SNNOpCounter(snn, in_shape=(2, 34, 34), ann_macs_total=...)
        with counter:
            evaluate(snn, loader, ...)       # any loop feeding [T, B, C, H, W]
        counts = counter.result()
    """

    def __init__(self, model: nn.Module, in_shape: tuple[int, ...] = (2, 34, 34),
                 ann_macs_total: Optional[int] = None) -> None:
        self.model = model
        self.in_shape = tuple(in_shape)
        self.ann_macs_total = ann_macs_total
        self._fan_cpu = fanout_maps(model, self.in_shape)
        self._lif = [m for m in _leaves(model) if isinstance(m, neuron.BaseNode)]
        first_w = next(m for m in _segments(model)[0][1] if isinstance(m, _WEIGHT_LAYERS))
        self._input_layer_macs = self._dense_macs(first_w)
        self._hooks: list = []
        self._fan: dict[str, torch.Tensor] = {}
        self.reset()

    def _dense_macs(self, layer: nn.Module) -> int:
        return _synapse_count(layer, self.in_shape)

    def reset(self) -> None:
        self.n_samples = 0
        self.T: Optional[int] = None
        self.input_macs_event = 0.0
        self.input_nonzero = 0.0
        self.spikes = {f"lif{i + 1}": 0.0 for i in range(len(self._lif))}
        self.synops = {f"lif{i + 1}": 0.0 for i in range(len(self._lif))}
        self.neurons = {f"lif{i + 1}": 0 for i in range(len(self._lif))}

    # -- hooks ---------------------------------------------------------------

    def _fan_on(self, name: str, device) -> torch.Tensor:
        f = self._fan.get(name)
        if f is None or f.device != torch.device(device):
            f = self._fan_cpu[name].to(device)
            self._fan[name] = f
        return f

    def _pre_hook(self, m, args):
        x = args[0]                                   # [T, B, C, H, W]
        T, B = int(x.shape[0]), int(x.shape[1])
        if self.T is None:
            self.T = T
        elif self.T != T:
            raise ValueError(f"SNNOpCounter saw T={T} after T={self.T}; use one counter per T")
        self.n_samples += B
        nz = (x != 0)
        self.input_nonzero += float(nz.sum(dtype=torch.float64))
        self.input_macs_event += float((nz * self._fan_on("input", x.device)).sum(dtype=torch.float64))

    def _make_lif_hook(self, name: str):
        def hook(m, inp, out):                        # out: [T, B, ...] binary spikes
            s = out.detach()
            self.spikes[name] += float(s.sum(dtype=torch.float64))
            self.synops[name] += float((s * self._fan_on(name, s.device)).sum(dtype=torch.float64))
            self.neurons[name] = int(s[0, 0].numel())
        return hook

    def __enter__(self) -> "SNNOpCounter":
        self._hooks.append(self.model.register_forward_pre_hook(self._pre_hook))
        for i, m in enumerate(self._lif):
            self._hooks.append(m.register_forward_hook(self._make_lif_hook(f"lif{i + 1}")))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- result --------------------------------------------------------------

    def result(self) -> dict:
        n = max(self.n_samples, 1)
        T = self.T or 0
        synops = sum(self.synops.values()) / n
        in_ev = self.input_macs_event / n
        out = {
            "T": T,
            "n_samples": self.n_samples,
            "synops_ac": synops,
            "synops_ac_per_layer": {k: v / n for k, v in self.synops.items()},
            "input_macs_event": in_ev,
            "input_macs_dense": T * self._input_layer_macs,
            "input_nonzero_fraction": (self.input_nonzero / n) / (T * _numel(self.in_shape)) if T else 0.0,
            "spikes_per_layer": {k: v / n for k, v in self.spikes.items()},
            "firing_rate_per_layer": {
                k: (self.spikes[k] / n) / (T * self.neurons[k]) if T and self.neurons[k] else 0.0
                for k in self.spikes
            },
            "neuron_updates": T * sum(self.neurons.values()),
            "energy_pj_est": E_MAC_PJ * in_ev + E_AC_PJ * synops,
        }
        if self.ann_macs_total is not None:
            ann_e = E_MAC_PJ * self.ann_macs_total
            out["dense_macs_equivalent"] = T * self.ann_macs_total
            out["ann_macs"] = self.ann_macs_total
            out["ann_energy_pj_est"] = ann_e
            out["energy_ratio_snn_over_ann"] = out["energy_pj_est"] / ann_e if ann_e else None
        return out


def _synapse_count(layer: nn.Module, in_shape: tuple[int, ...]) -> int:
    """Number of (input, weight) products of one Conv/Linear layer per sample,
    excluding products with zero padding."""
    with torch.no_grad():
        y = _structural(layer, torch.ones((1,) + tuple(in_shape), dtype=torch.float64))
    return int(round(float(y.sum())))


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n
