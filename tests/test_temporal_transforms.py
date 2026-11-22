"""
Deterministic unit tests for timestamp-based temporal binning
(src/data/transforms.py). Synthetic events only — no dataset download.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch

from src.data.transforms import (
    BinningSpec,
    EventValidationError,
    bin_boundaries,
    bin_indices,
    events_to_temporal_frames,
    temporal_to_aggregate,
    validate_events,
)

DTYPE = np.dtype([("x", np.int64), ("y", np.int64), ("t", np.int64), ("p", np.int64)])
WINDOW = 315_000
N_BINS = 16


def make_events(rows) -> np.ndarray:
    """rows: iterable of (x, y, t, p)."""
    return np.array([tuple(r) for r in rows], dtype=DTYPE)


def fixed_spec(**kw) -> BinningSpec:
    base = dict(n_bins=N_BINS, strategy="fixed_window", window_start_us=0,
                window_duration_us=WINDOW, out_of_window_policy="error")
    base.update(kw)
    return BinningSpec(**base)


def reference_bin(t: int, n_bins: int, t_start: int, t_end: int) -> int:
    """Exact rational reference: bin k <=> k <= (t - t_start) * n / D < k + 1."""
    d = t_end - t_start + 1
    return int(Fraction(t - t_start) * n_bins // d)


# ---------------------------------------------------------------------------
# Shapes and basic properties
# ---------------------------------------------------------------------------

def test_output_shape_and_dtype():
    ev = make_events([(0, 0, 0, 0), (33, 33, WINDOW - 1, 1)])
    frames, info = events_to_temporal_frames(ev, fixed_spec())
    assert frames.shape == (16, 2, 34, 34)
    assert frames.dtype == np.float32
    assert len(info.bin_start_us) == len(info.bin_end_us) == 16


def test_first_and_last_valid_timestamp_land_in_first_and_last_bin():
    ev = make_events([(1, 2, 0, 0), (3, 4, WINDOW - 1, 1)])
    frames, info = events_to_temporal_frames(ev, fixed_spec())
    assert frames[0, 0, 2, 1] == 1
    assert frames[15, 1, 4, 3] == 1
    assert info.n_events_binned == 2 and info.n_events_outside_window == 0


def test_bin_boundaries_are_contiguous_equal_and_cover_window():
    starts, ends = bin_boundaries(N_BINS, 0, WINDOW - 1)
    assert starts[0] == 0 and ends[-1] == WINDOW - 1
    for k in range(N_BINS - 1):
        assert ends[k] + 1 == starts[k + 1]
    widths = [e - s + 1 for s, e in zip(starts, ends)]
    assert sum(widths) == WINDOW
    assert max(widths) - min(widths) <= 1          # 19687 / 19688 µs


def test_events_on_every_bin_edge_go_to_the_bin_that_starts_there():
    starts, ends = bin_boundaries(N_BINS, 0, WINDOW - 1)
    t = np.array(starts + ends, dtype=np.int64)
    k = bin_indices(t, N_BINS, 0, WINDOW - 1)
    assert k[:N_BINS].tolist() == list(range(N_BINS))     # starts
    assert k[N_BINS:].tolist() == list(range(N_BINS))     # inclusive ends


def test_bin_indices_match_exact_rational_reference():
    rng = np.random.default_rng(0)
    for t_start, t_end, n in [(0, WINDOW - 1, 16), (17, 300_017, 16), (0, 99, 7), (5, 5 + 15, 16)]:
        t = rng.integers(t_start, t_end + 1, size=2000)
        t = np.concatenate([t, [t_start, t_end]])
        k = bin_indices(t, n, t_start, t_end)
        ref = [reference_bin(int(v), n, t_start, t_end) for v in t]
        assert k.tolist() == ref
        assert k.min() >= 0 and k.max() <= n - 1


def test_bin_indices_against_boundaries_consistency():
    t = np.arange(0, WINDOW, dtype=np.int64)
    k = bin_indices(t, N_BINS, 0, WINDOW - 1)
    starts, ends = bin_boundaries(N_BINS, 0, WINDOW - 1)
    for b in range(N_BINS):
        assert (k[starts[b]:ends[b] + 1] == b).all()


# ---------------------------------------------------------------------------
# Temporal ordering, polarity separation, repeated events, empty bins
# ---------------------------------------------------------------------------

def test_events_in_different_bins_preserve_chronological_order():
    # One event per bin at pixel (b, b) — bin index must follow time order.
    starts, _ = bin_boundaries(N_BINS, 0, WINDOW - 1)
    ev = make_events([(b, b, starts[b] + 5, b % 2) for b in range(N_BINS)])
    frames, info = events_to_temporal_frames(ev, fixed_spec())
    for b in range(N_BINS):
        assert frames[b, b % 2, b, b] == 1
        assert frames[b].sum() == 1
    assert info.events_per_bin == [1] * N_BINS


def test_unsorted_input_gives_same_result_as_sorted():
    rng = np.random.default_rng(1)
    ev = make_events(zip(rng.integers(0, 34, 500), rng.integers(0, 34, 500),
                         rng.integers(0, WINDOW, 500), rng.integers(0, 2, 500)))
    shuffled = ev[rng.permutation(ev.size)]
    a, _ = events_to_temporal_frames(np.sort(ev, order="t"), fixed_spec())
    b, _ = events_to_temporal_frames(shuffled, fixed_spec())
    assert np.array_equal(a, b), "binning must depend on timestamps, not array index"


def test_on_off_channels_are_independent():
    ev = make_events([(10, 10, 100, 0), (10, 10, 100, 1), (10, 10, 101, 1)])
    frames, info = events_to_temporal_frames(ev, fixed_spec())
    assert frames[0, 0, 10, 10] == 1
    assert frames[0, 1, 10, 10] == 2
    assert info.events_per_polarity == [1, 2]


def test_repeated_events_at_same_pixel_accumulate_counts():
    ev = make_events([(5, 6, 1000, 1)] * 7)
    frames, _ = events_to_temporal_frames(ev, fixed_spec())
    assert frames[0, 1, 6, 5] == 7
    assert frames.sum() == 7


def test_xy_orientation_row_is_y_column_is_x():
    ev = make_events([(30, 2, 0, 0)])
    frames, _ = events_to_temporal_frames(ev, fixed_spec())
    assert frames[0, 0, 2, 30] == 1 and frames[0, 0, 30, 2] == 0


def test_empty_bins_are_zero_and_reported():
    starts, _ = bin_boundaries(N_BINS, 0, WINDOW - 1)
    ev = make_events([(0, 0, starts[3], 0), (0, 0, starts[9], 1)])
    frames, info = events_to_temporal_frames(ev, fixed_spec())
    assert info.empty_bins == [b for b in range(N_BINS) if b not in (3, 9)]
    for b in info.empty_bins:
        assert frames[b].sum() == 0


def test_empty_recording_returns_all_zero_frames():
    frames, info = events_to_temporal_frames(np.zeros(0, dtype=DTYPE), fixed_spec())
    assert frames.shape == (16, 2, 34, 34) and frames.sum() == 0
    assert info.n_events_in == 0 and info.empty_bins == list(range(16))


def test_event_count_conservation_random():
    rng = np.random.default_rng(2)
    n = 5000
    ev = make_events(zip(rng.integers(0, 34, n), rng.integers(0, 34, n),
                         rng.integers(0, WINDOW, n), rng.integers(0, 2, n)))
    frames, info = events_to_temporal_frames(ev, fixed_spec())
    assert frames.sum() == n == info.n_events_binned == sum(info.events_per_bin)
    assert np.isfinite(frames).all()


def test_deterministic():
    rng = np.random.default_rng(3)
    ev = make_events(zip(rng.integers(0, 34, 800), rng.integers(0, 34, 800),
                         rng.integers(0, WINDOW, 800), rng.integers(0, 2, 800)))
    a, ia = events_to_temporal_frames(ev, fixed_spec())
    b, ib = events_to_temporal_frames(ev.copy(), fixed_spec())
    assert np.array_equal(a, b) and ia.to_dict() == ib.to_dict()


# ---------------------------------------------------------------------------
# Out-of-window and invalid events
# ---------------------------------------------------------------------------

def test_out_of_window_raises_by_default():
    ev = make_events([(0, 0, 10, 0), (0, 0, WINDOW, 1)])          # t == WINDOW is outside
    with pytest.raises(EventValidationError):
        events_to_temporal_frames(ev, fixed_spec())


def test_out_of_window_drop_and_report_counts_dropped_events():
    ev = make_events([(0, 0, 10, 0), (0, 0, WINDOW, 1), (0, 0, WINDOW + 50_000, 1)])
    frames, info = events_to_temporal_frames(ev, fixed_spec(out_of_window_policy="drop_and_report"))
    assert info.n_events_in == 3
    assert info.n_events_outside_window == 2
    assert info.n_events_binned == 1 == frames.sum()


@pytest.mark.parametrize("bad", [(-1, 0, 0, 0), (34, 0, 0, 0), (0, -1, 0, 0),
                                 (0, 34, 0, 0), (0, 0, 0, 2), (0, 0, 0, -1), (0, 0, -5, 0)])
def test_invalid_events_raise(bad):
    ev = make_events([(0, 0, 0, 0), bad])
    with pytest.raises(EventValidationError):
        events_to_temporal_frames(ev, fixed_spec())


def test_float_timestamps_rejected():
    ev = np.array([(0, 0, 0.5, 0)], dtype=[("x", int), ("y", int), ("t", float), ("p", int)])
    with pytest.raises(EventValidationError):
        validate_events(ev)


def test_missing_field_rejected():
    ev = np.array([(0, 0, 0)], dtype=[("x", int), ("y", int), ("t", int)])
    with pytest.raises(EventValidationError):
        validate_events(ev)


def test_invalid_spec_rejected():
    with pytest.raises(ValueError):
        BinningSpec(n_bins=0, window_duration_us=WINDOW)
    with pytest.raises(ValueError):
        BinningSpec(n_bins=16, strategy="fixed_window", window_duration_us=None)


# ---------------------------------------------------------------------------
# Per-recording strategy
# ---------------------------------------------------------------------------

def test_per_recording_window_uses_min_and_max_timestamp():
    ev = make_events([(0, 0, 1000, 0), (1, 1, 5000, 1), (2, 2, 17_000, 0)])
    spec = BinningSpec(n_bins=4, strategy="per_recording")
    frames, info = events_to_temporal_frames(ev, spec)
    assert (info.t_start_us, info.t_end_us) == (1000, 17_000)
    assert frames[0, 0, 0, 0] == 1 and frames[3, 0, 2, 2] == 1
    assert frames.sum() == 3


def test_per_recording_single_timestamp_goes_to_bin_zero():
    ev = make_events([(0, 0, 42, 0), (1, 0, 42, 1)])
    frames, info = events_to_temporal_frames(ev, BinningSpec(n_bins=16, strategy="per_recording"))
    assert frames[0].sum() == 2 and frames.sum() == 2


# ---------------------------------------------------------------------------
# ANN aggregate vs SNN sequence
# ---------------------------------------------------------------------------

def test_ann_representation_is_sum_over_bins_and_snn_keeps_bins_distinct():
    starts, _ = bin_boundaries(N_BINS, 0, WINDOW - 1)
    ev = make_events([(4, 4, starts[1], 0), (4, 4, starts[12], 0), (7, 8, starts[12] + 3, 1)])
    frames, _ = events_to_temporal_frames(ev, fixed_spec())
    agg = temporal_to_aggregate(frames)
    assert agg.shape == (2, 34, 34)
    assert np.array_equal(agg, frames.sum(axis=0))
    assert agg[0, 4, 4] == 2 and agg[1, 8, 7] == 1
    # The SNN sequence distinguishes when events happened; the aggregate cannot.
    assert frames[1, 0, 4, 4] == 1 and frames[12, 0, 4, 4] == 1
    assert not np.array_equal(frames[1], frames[12])


def test_aggregate_rejects_non_sequence():
    with pytest.raises(ValueError):
        temporal_to_aggregate(np.zeros((2, 34, 34), dtype=np.float32))


def test_to_time_major_rejects_static_frames_and_t_mismatch():
    from src.trainer import to_time_major
    with pytest.raises(ValueError):
        to_time_major(torch.zeros(4, 2, 34, 34))
    with pytest.raises(ValueError):
        to_time_major(torch.zeros(4, 16, 2, 34, 34), T=8)
    x = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16, 1, 1, 1).expand(4, 16, 2, 34, 34)
    y = to_time_major(x, T=16)
    assert y.shape == (16, 4, 2, 34, 34)
    assert torch.equal(y[5, 2], x[2, 5])


# ---------------------------------------------------------------------------
# Decision D-G: saccade alignment; D-B: mean aggregate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [3, 6, 12, 24, 48])
def test_saccade_boundaries_are_bin_starts_for_new_grid(T):
    starts, _ = bin_boundaries(T, 0, WINDOW - 1)
    assert 105_000 in starts and 210_000 in starts


@pytest.mark.parametrize("T", [2, 4, 8, 16, 32])
def test_old_grid_was_not_saccade_aligned(T):
    starts, _ = bin_boundaries(T, 0, WINDOW - 1)
    assert not (105_000 in starts and 210_000 in starts)


def test_mean_aggregate():
    ev = make_events([(4, 4, 0, 0), (4, 4, WINDOW - 1, 0)])
    frames, _ = events_to_temporal_frames(ev, fixed_spec())
    agg = temporal_to_aggregate(frames, "mean")
    assert agg[0, 4, 4] == pytest.approx(2 / N_BINS)
    with pytest.raises(ValueError):
        temporal_to_aggregate(frames, "max")
