"""
Tests for the hardware-independent operation counts (src/opcount.py, decision D-H).

Run from the project root:
    python -m pytest tests/test_opcount.py -v

Checks:
  - ANN MACs per layer equal the hand-computed values
  - fan-out of individual neurons (interior, corner, pool-cropped, FC)
  - sum of fan-outs equals the layer's synapse count (x pool area when pooled)
  - adjoint (fan-out) SynOps equal a forward brute-force count on random spikes
  - the counter attached to a real SNN forward reproduces spike counts,
    neuron counts and the all-events input identity
  - the counter works through trainer.evaluate and refuses mixed T
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import functional

from src.models.ann import ANN
from src.models.snn import SNN
from src.opcount import E_AC_PJ, E_MAC_PJ, SNNOpCounter, ann_macs, fanout_maps
from src.trainer import evaluate

IN = (2, 34, 34)
# Real synapses (padding excluded): per spatial dim a 3x3/pad-1 conv over n
# positions has 3n - 2 valid (input, kernel-tap) pairs.
CONV1 = 32 * 2 * (3 * 34 - 2) ** 2    # 640 000   (665 856 if padding were counted)
CONV2 = 64 * 32 * (3 * 17 - 2) ** 2   # 4 917 248 (5 326 848 with padding)
FC1 = 4096 * 256                      # 1 048 576
FC2 = 256 * 10                        # 2 560


def test_ann_macs_match_hand_computation():
    m = ann_macs(ANN(), IN)
    assert list(m.values()) == [CONV1, CONV2, FC1, FC2]
    assert sum(m.values()) == 6_608_384


def test_fanout_of_individual_neurons():
    f = fanout_maps(SNN(), IN)
    assert f["input"][0, 10, 10] == 32 * 9          # interior pixel -> 9 positions x 32 maps
    assert f["input"][0, 0, 0] == 32 * 4            # corner with padding 1
    assert f["lif1"][5, 10, 10] == 64 * 9           # pools to (5,5): interior of 17x17
    assert f["lif1"][5, 0, 0] == 64 * 4             # pools to the corner
    assert f["lif2"][3, 0, 0] == 256                # pooled position kept -> all of FC1
    assert f["lif2"][3, 16, 5] == 0                 # row 16 dropped by floor(17/2)
    assert torch.all(f["lif3"] == 10)


def test_fanout_sums_equal_synapse_counts():
    f = fanout_maps(SNN(), IN)
    assert f["input"].sum() == CONV1
    assert f["lif1"].sum() == 4 * CONV2   # 4 spikes share each pooled input (weight / 4 each)
    assert f["lif2"].sum() == 4 * FC1
    assert f["lif3"].sum() == FC2


def test_adjoint_synops_equal_forward_bruteforce():
    """Fan-out counting (gradient) vs counting forward through the layers."""
    g = torch.Generator().manual_seed(0)
    f = fanout_maps(SNN(), IN)
    s1 = (torch.rand(3, 32, 34, 34, generator=g) < 0.1).double()
    s2 = (torch.rand(3, 64, 17, 17, generator=g) < 0.1).double()
    ones_c2 = torch.ones(64, 32, 3, 3, dtype=torch.float64)
    brute1 = F.conv2d(F.avg_pool2d(s1, 2) * 4, ones_c2, padding=1).sum()
    brute2 = (F.avg_pool2d(s2, 2) * 4).flatten(1).sum() * 256
    assert float((s1 * f["lif1"].double()).sum()) == pytest.approx(float(brute1))
    assert float((s2 * f["lif2"].double()).sum()) == pytest.approx(float(brute2))


def _spike_counts(snn, x_seq):
    lifs = [snn.features[1], snn.features[4], snn.classifier[2]]
    got = {}
    hooks = [m.register_forward_hook(lambda m, i, o, k=k: got.__setitem__(k, o.detach().clone()))
             for k, m in zip(("lif1", "lif2", "lif3"), lifs)]
    functional.reset_net(snn)
    with torch.no_grad():
        snn(x_seq)
    for h in hooks:
        h.remove()
    return got


def test_counter_matches_direct_spike_counts():
    torch.manual_seed(0)
    snn = SNN().eval()
    with torch.no_grad():                      # make every layer fire
        for m in (snn.features[0], snn.features[3], snn.classifier[1]):
            m.weight.mul_(6.0)
    T, B = 4, 3
    x = (torch.rand(T, B, *IN) < 0.05).float() * 3.0
    spikes = _spike_counts(snn, x)
    f = fanout_maps(snn, IN)

    functional.reset_net(snn)
    with SNNOpCounter(snn, IN, ann_macs_total=6_608_384) as c, torch.no_grad():
        snn(x)
    r = c.result()

    assert r["T"] == T and r["n_samples"] == B
    for k in ("lif1", "lif2", "lif3"):
        assert r["spikes_per_layer"][k] == pytest.approx(float(spikes[k].sum()) / B)
        assert r["synops_ac_per_layer"][k] == pytest.approx(float((spikes[k] * f[k]).sum()) / B)
        assert 0.0 <= r["firing_rate_per_layer"][k] <= 1.0
    assert sum(spikes[k].sum() for k in spikes) > 0
    assert r["synops_ac"] == pytest.approx(sum(r["synops_ac_per_layer"].values()))
    assert r["input_macs_event"] == pytest.approx(float(((x != 0) * f["input"]).sum()) / B)
    assert r["input_macs_dense"] == T * CONV1
    assert r["neuron_updates"] == T * (32 * 34 * 34 + 64 * 17 * 17 + 256)
    assert r["dense_macs_equivalent"] == T * 6_608_384
    assert r["energy_pj_est"] == pytest.approx(E_MAC_PJ * r["input_macs_event"] + E_AC_PJ * r["synops_ac"])
    assert r["energy_ratio_snn_over_ann"] == pytest.approx(r["energy_pj_est"] / (E_MAC_PJ * 6_608_384))


def test_all_nonzero_input_gives_dense_input_macs():
    snn = SNN().eval()
    T = 2
    functional.reset_net(snn)
    with SNNOpCounter(snn, IN) as c, torch.no_grad():
        snn(torch.ones(T, 1, *IN))
    r = c.result()
    assert r["input_macs_event"] == T * CONV1 == r["input_macs_dense"]
    assert r["input_nonzero_fraction"] == pytest.approx(1.0)


def test_counter_through_trainer_evaluate_and_hooks_removed():
    snn = SNN()
    T = 3
    data = TensorDataset((torch.rand(5, T, *IN) < 0.05).float(), torch.randint(0, 10, (5,)))
    loader = DataLoader(data, batch_size=2)
    with SNNOpCounter(snn, IN) as c:
        evaluate(snn, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), is_snn=True, T=T)
    assert c.result()["n_samples"] == 5
    assert not snn._forward_pre_hooks and not snn.features[1]._forward_hooks


def test_counter_rejects_mixed_T():
    snn = SNN().eval()
    with SNNOpCounter(snn, IN), torch.no_grad():
        functional.reset_net(snn)
        snn(torch.zeros(2, 1, *IN))
        functional.reset_net(snn)
        with pytest.raises(ValueError, match="one counter per T"):
            snn(torch.zeros(3, 1, *IN))
