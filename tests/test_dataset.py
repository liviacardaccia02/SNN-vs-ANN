"""
Real-data integration tests for NMNISTEventDataset (protocol v2.0).

These tests read original N-MNIST recordings from cfg.data.data_root and are
SKIPPED (not failed) when the split is not on disk, so ordinary unit tests
never trigger a dataset download. Fetch the data first with:

    python -c "import tonic; tonic.datasets.NMNIST('./data', train=False)"

What is verified on real recordings:
  - original (x, y, t, p) fields are returned unmodified (no static aggregate)
  - coordinates, polarity and timestamp ranges / units
  - SNN representation [24, 2, 34, 34] with distinct temporal bins
  - ANN representation [2, 34, 34] == SNN representation averaged over T
  - re-binned evaluation loaders for every T in the grid (decision D-A)
  - event-count conservation (binned + explicitly dropped == original)
  - determinism, stratified/deterministic train-val split
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.config import get_config
from src.data.nmnist import (
    EXPECTED_TEST_SIZE,
    EXPECTED_TRAIN_SIZE,
    N_CLASSES,
    SENSOR_SIZE,
    NMNISTEventDataset,
    binning_spec_from_config,
    make_train_val_split,
)
from src.data.transforms import temporal_to_aggregate

CFG = get_config()


def _split_available(train: bool) -> bool:
    d = os.path.join(CFG.data.data_root, "NMNIST", "Train" if train else "Test")
    if not os.path.isdir(d):
        return False
    n = sum(f.endswith(".bin") for _, _, fs in os.walk(d) for f in fs)
    return n >= (EXPECTED_TRAIN_SIZE if train else EXPECTED_TEST_SIZE)


needs_test_split = pytest.mark.skipif(not _split_available(False), reason="N-MNIST test split not on disk")
needs_train_split = pytest.mark.skipif(not _split_available(True), reason="N-MNIST train split not on disk")

SAMPLE_IDX = [0, 999, 1000, 4321, 9999]     # spans several classes (files are class-ordered)


@pytest.fixture(scope="module")
def spec():
    return binning_spec_from_config(CFG)


def _ds(spec, representation, train=False):
    return NMNISTEventDataset(CFG.data.data_root, train, spec, representation, download=False)


# ---------------------------------------------------------------------------
# Constants (no data needed)
# ---------------------------------------------------------------------------

def test_constants_match_config():
    assert SENSOR_SIZE == CFG.preprocess.sensor_size == (34, 34, 2)
    assert N_CLASSES == CFG.network.n_classes == 10
    assert CFG.snn.t_train == CFG.preprocess.n_time_bins


def test_missing_data_raises_without_download(tmp_path, spec):
    with pytest.raises(FileNotFoundError):
        NMNISTEventDataset(str(tmp_path), False, spec, "snn", download=False)


# ---------------------------------------------------------------------------
# Original events
# ---------------------------------------------------------------------------

@needs_test_split
class TestOriginalEvents:

    @pytest.fixture(scope="class")
    def ds(self, spec):
        return _ds(spec, "events")

    def test_length(self, ds):
        assert len(ds) == EXPECTED_TEST_SIZE

    def test_fields_and_integer_dtype(self, ds):
        ev, label = ds[0]
        assert ev.dtype.names == ("x", "y", "t", "p")
        for f in ev.dtype.names:
            assert np.issubdtype(ev.dtype[f], np.integer)
        assert isinstance(label, int) and 0 <= label < 10

    def test_is_event_stream_not_aggregate(self, ds):
        ev, _ = ds[0]
        assert ev.ndim == 1 and ev.size > 500, "expected thousands of individual events"

    @pytest.mark.parametrize("i", SAMPLE_IDX)
    def test_bounds_polarity_and_timestamps(self, ds, i):
        ev, _ = ds[i]
        assert ev["x"].min() >= 0 and ev["x"].max() < 34
        assert ev["y"].min() >= 0 and ev["y"].max() < 34
        assert set(np.unique(ev["p"]).tolist()) <= {0, 1}
        t = ev["t"]
        assert t.min() >= 0
        assert np.all(np.diff(t) >= 0), "Tonic returns events in timestamp order"
        # Microseconds: three ~105 ms saccades -> recording spans ~0.3 s.
        assert 250_000 < t.max() < 400_000, f"t_max={t.max()} inconsistent with µs units"

    def test_both_polarities_present(self, ds):
        ev, _ = ds[0]
        assert (ev["p"] == 0).any() and (ev["p"] == 1).any()

    def test_labels_cover_all_classes(self, ds):
        assert sorted(set(ds.targets)) == list(range(10))


# ---------------------------------------------------------------------------
# Temporal and aggregate representations from the same recording
# ---------------------------------------------------------------------------

@needs_test_split
class TestRepresentations:

    @pytest.fixture(scope="class")
    def ds_events(self, spec):
        return _ds(spec, "events")

    @pytest.fixture(scope="class")
    def ds_snn(self, spec):
        return _ds(spec, "snn")

    @pytest.fixture(scope="class")
    def ds_ann(self, spec):
        return _ds(spec, "ann")

    @pytest.mark.parametrize("i", SAMPLE_IDX)
    def test_snn_shape_finite_and_conservation(self, ds_snn, ds_events, i):
        x, label = ds_snn[i]
        ev, label_ev = ds_events[i]
        assert label == label_ev
        assert x.shape == (CFG.preprocess.n_time_bins, 2, 34, 34) and x.dtype == torch.float32
        assert torch.isfinite(x).all() and (x >= 0).all()
        frames, info, _ = ds_snn.get_binned(i)
        assert info.n_events_in == ev.size
        assert info.n_events_binned + info.n_events_outside_window == ev.size
        assert int(x.sum()) == info.n_events_binned
        # Out-of-window events are exactly those at t >= window end.
        assert info.n_events_outside_window == int((ev["t"] >= CFG.preprocess.window_duration_us).sum())

    @pytest.mark.parametrize("i", SAMPLE_IDX)
    def test_bins_follow_timestamps(self, ds_snn, ds_events, i):
        """Independent per-bin recount from the raw events using bin edges."""
        ev, _ = ds_events[i]
        frames, info, _ = ds_snn.get_binned(i)
        for k, (s, e) in enumerate(zip(info.bin_start_us, info.bin_end_us)):
            sel = ev[(ev["t"] >= s) & (ev["t"] <= e)]
            for p in (0, 1):
                assert frames[k, p].sum() == (sel["p"] == p).sum()

    @pytest.mark.parametrize("i", SAMPLE_IDX)
    def test_ann_equals_mean_over_bins(self, ds_snn, ds_ann, i):
        x_snn, y1 = ds_snn[i]
        x_ann, y2 = ds_ann[i]
        assert y1 == y2
        assert x_ann.shape == (2, 34, 34)
        assert torch.allclose(x_ann, x_snn.mean(dim=0))
        assert np.array_equal(x_ann.numpy(), temporal_to_aggregate(x_snn.numpy(), "mean"))

    def test_bins_are_temporally_distinct(self, ds_snn):
        x, _ = ds_snn[0]
        per_bin = x.sum(dim=(1, 2, 3))
        assert (per_bin > 0).sum() >= 12, "most bins of a real recording carry events"
        assert per_bin.std() > 0, "event counts must vary over time"
        n = x.shape[0] - 1
        distinct = sum(not torch.equal(x[k], x[k + 1]) for k in range(n))
        assert distinct == n, "consecutive bins must not be copies of one frame"

    def test_deterministic(self, ds_snn):
        a, _ = ds_snn[4321]
        b, _ = ds_snn[4321]
        assert torch.equal(a, b)

    def test_raw_counts_not_normalised(self, ds_snn):
        x, _ = ds_snn[0]
        assert torch.equal(x, x.round()), "v2.0 default representation is raw integer counts"


# ---------------------------------------------------------------------------
# Re-binned evaluation loaders (decision D-A)
# ---------------------------------------------------------------------------

@needs_test_split
class TestRebinnedEvaluation:

    @pytest.fixture(scope="class")
    def loaders(self):
        from src.data.nmnist import make_eval_loaders_by_T
        return make_eval_loaders_by_T(CFG, "test", batch_size=4, indices=[0, 4321, 9999])

    def test_one_loader_per_grid_T_with_matching_length(self, loaders):
        assert sorted(loaders) == sorted(CFG.snn.t_eval_grid)
        for T, dl in loaders.items():
            x, _ = next(iter(dl))
            assert x.shape[1:] == (T, 2, 34, 34)

    def test_same_recordings_same_labels_all_T(self, loaders):
        labels = {T: torch.cat([y for _, y in dl]).tolist() for T, dl in loaders.items()}
        assert len({tuple(v) for v in labels.values()}) == 1

    def test_total_events_and_per_step_rate_preserved(self, loaders):
        t_train = CFG.snn.t_train
        ref_total, ref_mean = None, None
        for T, dl in sorted(loaders.items()):
            x = torch.cat([b for b, _ in dl])
            total = x.sum(dim=(1, 2, 3, 4)) * (t_train / T)   # undo value_scale
            mean_step = x.mean(dim=(1, 2, 3, 4))
            if ref_total is None:
                ref_total, ref_mean = total, mean_step
            assert torch.allclose(total, ref_total, rtol=1e-5)
            assert torch.allclose(mean_step, ref_mean, rtol=1e-5)

    def test_coarse_bins_are_sums_of_fine_bins(self, loaders):
        """With an aligned grid, T=12 bin k == (T=24 bins 2k, 2k+1) summed
        (after undoing the scale): re-binning loses no events and no order."""
        x12 = next(iter(loaders[12]))[0] * (CFG.snn.t_train / 12)
        x24 = next(iter(loaders[24]))[0]
        assert torch.allclose(x12, x24.reshape(x24.shape[0], 12, 2, 2, 34, 34).sum(2), atol=1e-4)


@needs_test_split
class TestEvaluationVariants:
    """Sensitivity variants of how a T-step input is formed (decision D-A)."""

    IDX = [0, 4321]

    def _first(self, variant):
        from src.data.nmnist import make_eval_loaders_by_T
        L = make_eval_loaders_by_T(CFG, "test", batch_size=2, indices=self.IDX, variant=variant)
        return {T: next(iter(dl))[0] for T, dl in L.items()}

    def test_rebin_raw_is_rebin_scaled_without_scaling(self):
        scaled, raw = self._first("rebin_scaled"), self._first("rebin_raw")
        for T in raw:
            assert torch.allclose(scaled[T], raw[T] * (T / CFG.snn.t_train), rtol=1e-5)

    def test_prefix_is_first_bins_of_training_sequence(self):
        pre = self._first("prefix")
        assert max(pre) <= CFG.snn.t_train and 48 not in pre
        full = pre[CFG.snn.t_train]
        for T, x in pre.items():
            assert x.shape[1] == T
            assert torch.equal(x, full[:, :T])

    def test_unknown_variant_rejected(self):
        from src.data.nmnist import make_eval_loaders_by_T
        with pytest.raises(ValueError):
            make_eval_loaders_by_T(CFG, "test", variant="random")


# ---------------------------------------------------------------------------
# Train/val split (labels only — no files read)
# ---------------------------------------------------------------------------

@needs_train_split
class TestSplit:

    @pytest.fixture(scope="class")
    def ds(self, spec):
        return _ds(spec, "snn", train=True)

    def test_split_sizes_disjoint_deterministic_stratified(self, ds):
        val_frac = 1.0 - CFG.data.train_val_split
        tr, va = make_train_val_split(ds, val_frac, CFG.data.seed_split)
        tr2, va2 = make_train_val_split(ds, val_frac, CFG.data.seed_split)
        assert len(tr) + len(va) == EXPECTED_TRAIN_SIZE
        assert abs(len(va) - EXPECTED_TRAIN_SIZE * val_frac) <= 10
        assert not set(tr.indices) & set(va.indices)
        assert tr.indices == tr2.indices and va.indices == va2.indices
        labels = np.asarray(ds.targets)
        assert np.bincount(labels[va.indices], minlength=10).min() > 0
