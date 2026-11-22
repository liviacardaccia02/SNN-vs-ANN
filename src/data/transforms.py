"""
Deterministic, timestamp-based temporal binning of event streams.

Three representations are distinguished throughout this project:

1. Original event stream (source of truth)
       E = {e_i = (x_i, y_i, t_i, p_i)}_{i=1..N}
   as decoded from the N-MNIST .bin file by Tonic (structured NumPy array
   with fields x, y, t, p; t in integer microseconds).

2. Binned temporal representation (SNN input)
       X_SNN[k, c, y, x] = #{ i : bin(t_i) = k, p_i = c, y_i = y, x_i = x }
   shape [T, C, H, W] = [n_bins, 2, 34, 34]. Bins are contiguous, ordered
   chronologically, and of (near-)equal physical duration.

3. Aggregate representation (ANN input)
       X_ANN[c, y, x] = (1/T) sum_k X_SNN[k, c, y, x]     (mean over bins; D-B)
   shape [C, H, W]. Temporal ordering is intentionally discarded here only.

Bin-assignment rule (integer arithmetic, no floating-point rounding):

    A window is the inclusive integer tick range [t_start, t_end] with
    D = t_end - t_start + 1 ticks (1 tick = 1 µs for N-MNIST).
    bin(t) = floor((t - t_start) * n_bins / D)

    Bin k therefore contains exactly the ticks
        [t_start + ceil(k*D/n_bins),  t_start + ceil((k+1)*D/n_bins) - 1]
    i.e. half-open on the right in continuous terms. Bin durations differ by
    at most 1 tick. The first tick t_start falls in bin 0 and the last tick
    t_end falls in bin n_bins-1, so no clipping is required and no event at
    the window boundaries can produce an out-of-range index.

Two window strategies are supported:

  - "fixed_window":   [t_start, t_end] = [window_start_us,
                       window_start_us + window_duration_us - 1], identical
                       for every recording. Bin k corresponds to the same
                       physical time (same saccade phase) in every sample.
  - "per_recording":  [t_start, t_end] = [min(t), max(t)] of each recording.
                       Bin duration varies slightly between recordings.

Events outside a fixed window are never dropped silently: the default policy
raises; the alternative "drop_and_report" drops them and reports the count.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np

EVENT_FIELDS: tuple[str, ...] = ("x", "y", "t", "p")

BinningStrategy = Literal["fixed_window", "per_recording"]
OutOfWindowPolicy = Literal["error", "drop_and_report"]


class EventValidationError(ValueError):
    """Raised when an event array violates the expected N-MNIST event format."""


# ---------------------------------------------------------------------------
# Specification and bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinningSpec:
    """Complete, hashable description of the temporal binning procedure."""

    n_bins: int
    sensor_width: int = 34
    sensor_height: int = 34
    n_polarities: int = 2
    strategy: BinningStrategy = "fixed_window"
    window_start_us: int = 0
    window_duration_us: Optional[int] = None
    out_of_window_policy: OutOfWindowPolicy = "error"

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        if self.strategy not in ("fixed_window", "per_recording"):
            raise ValueError(f"Unknown binning strategy '{self.strategy}'")
        if self.strategy == "fixed_window":
            if self.window_duration_us is None or self.window_duration_us < self.n_bins:
                raise ValueError(
                    "fixed_window requires window_duration_us >= n_bins "
                    f"(got {self.window_duration_us})"
                )
        if self.out_of_window_policy not in ("error", "drop_and_report"):
            raise ValueError(f"Unknown out_of_window_policy '{self.out_of_window_policy}'")

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """[T, C, H, W]."""
        return (self.n_bins, self.n_polarities, self.sensor_height, self.sensor_width)


@dataclass
class BinningInfo:
    """Traceability record produced for every binned recording."""

    t_start_us: int
    t_end_us: int                      # inclusive
    bin_start_us: list[int]            # inclusive start tick of each bin
    bin_end_us: list[int]              # inclusive end tick of each bin
    n_events_in: int
    n_events_binned: int
    n_events_outside_window: int
    events_per_bin: list[int]
    events_per_polarity: list[int]
    empty_bins: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_events(
    events: np.ndarray,
    sensor_width: int = 34,
    sensor_height: int = 34,
    n_polarities: int = 2,
) -> None:
    """
    Check that `events` is a structured array with integer x, y, t, p fields,
    in-bounds coordinates, valid polarities and non-negative timestamps.

    Raises EventValidationError on the first violated condition; never
    modifies or filters the input.
    """
    if not isinstance(events, np.ndarray) or events.dtype.names is None:
        raise EventValidationError("events must be a NumPy structured array")
    missing = [f for f in EVENT_FIELDS if f not in events.dtype.names]
    if missing:
        raise EventValidationError(f"events missing fields {missing}")
    for f in EVENT_FIELDS:
        if not np.issubdtype(events.dtype[f], np.integer):
            raise EventValidationError(
                f"field '{f}' must have an integer dtype, got {events.dtype[f]}"
            )
    if events.size == 0:
        return

    x, y, t, p = (events[f] for f in EVENT_FIELDS)
    bad_x = (x < 0) | (x >= sensor_width)
    bad_y = (y < 0) | (y >= sensor_height)
    bad_p = (p < 0) | (p >= n_polarities)
    bad_t = t < 0
    for name, mask in (("x", bad_x), ("y", bad_y), ("p", bad_p), ("t", bad_t)):
        if mask.any():
            i = int(np.flatnonzero(mask)[0])
            raise EventValidationError(
                f"{int(mask.sum())} event(s) with invalid '{name}' "
                f"(first at index {i}: x={x[i]}, y={y[i]}, t={t[i]}, p={p[i]}); "
                f"sensor={sensor_width}x{sensor_height}, polarities={n_polarities}"
            )


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_indices(t: np.ndarray, n_bins: int, t_start: int, t_end: int) -> np.ndarray:
    """
    Map integer timestamps in the inclusive window [t_start, t_end] to bin
    indices 0..n_bins-1 using floor((t - t_start) * n_bins / D), D = t_end-t_start+1.

    Caller must ensure all t lie inside the window.
    """
    duration = int(t_end) - int(t_start) + 1
    if duration < 1:
        raise ValueError(f"empty window [{t_start}, {t_end}]")
    rel = t.astype(np.int64) - np.int64(t_start)
    idx = (rel * np.int64(n_bins)) // np.int64(duration)
    return idx


def bin_boundaries(n_bins: int, t_start: int, t_end: int) -> tuple[list[int], list[int]]:
    """Inclusive [start, end] tick of every bin, consistent with bin_indices()."""
    duration = int(t_end) - int(t_start) + 1
    starts = [int(t_start) + -(-k * duration // n_bins) for k in range(n_bins)]   # ceil
    ends = [s - 1 for s in starts[1:]] + [int(t_end)]
    return starts, ends


def events_to_temporal_frames(
    events: np.ndarray,
    spec: BinningSpec,
) -> tuple[np.ndarray, BinningInfo]:
    """
    Convert an event stream into chronologically ordered count frames.

    Args:
        events: Structured array with integer fields x, y, t, p.
        spec:   Binning specification.

    Returns:
        frames: float32 array [n_bins, n_polarities, H, W] of raw event counts.
                Channel index == polarity value p.
        info:   BinningInfo with window, bin edges and event bookkeeping.
    """
    validate_events(events, spec.sensor_width, spec.sensor_height, spec.n_polarities)
    frames = np.zeros(spec.output_shape, dtype=np.float32)
    n_in = int(events.size)

    # ---- Window ----
    if spec.strategy == "fixed_window":
        t_start = int(spec.window_start_us)
        t_end = t_start + int(spec.window_duration_us) - 1
    elif n_in == 0:
        t_start, t_end = 0, spec.n_bins - 1     # degenerate: nothing to bin
    else:
        t_start = int(events["t"].min())
        t_end = int(events["t"].max())
        # A recording whose events share a single timestamp: widen the window
        # so D >= n_bins; all events then land in bin 0 (documented behaviour).
        t_end = max(t_end, t_start + spec.n_bins - 1)

    # ---- Out-of-window handling (fixed window only) ----
    if n_in:
        t = events["t"]
        inside = (t >= t_start) & (t <= t_end)
        n_out = int((~inside).sum())
        if n_out and spec.out_of_window_policy == "error":
            raise EventValidationError(
                f"{n_out} event(s) outside window [{t_start}, {t_end}] µs "
                f"(t range [{int(t.min())}, {int(t.max())}]). Increase "
                "window_duration_us or use out_of_window_policy='drop_and_report'."
            )
        ev = events[inside] if n_out else events
    else:
        n_out = 0
        ev = events

    # ---- Accumulate counts ----
    if ev.size:
        k = bin_indices(ev["t"], spec.n_bins, t_start, t_end)
        np.add.at(
            frames,
            (k, ev["p"].astype(np.int64), ev["y"].astype(np.int64), ev["x"].astype(np.int64)),
            1.0,
        )

    starts, ends = bin_boundaries(spec.n_bins, t_start, t_end)
    per_bin = frames.sum(axis=(1, 2, 3)).astype(np.int64).tolist()
    info = BinningInfo(
        t_start_us=t_start,
        t_end_us=t_end,
        bin_start_us=starts,
        bin_end_us=ends,
        n_events_in=n_in,
        n_events_binned=int(ev.size),
        n_events_outside_window=n_out,
        events_per_bin=per_bin,
        events_per_polarity=frames.sum(axis=(0, 2, 3)).astype(np.int64).tolist(),
        empty_bins=[i for i, c in enumerate(per_bin) if c == 0],
    )
    if sum(per_bin) != info.n_events_binned:     # conservation invariant
        raise AssertionError(
            f"event-count conservation violated: {sum(per_bin)} != {info.n_events_binned}"
        )
    return frames, info


def temporal_to_aggregate(frames: np.ndarray, reduction: str = "sum") -> np.ndarray:
    """ANN representation: reduce the [T, C, H, W] sequence over T -> [C, H, W].

    reduction="sum":  total event counts of the recording.
    reduction="mean": average per-bin value = the mean per-step input the SNN
                      receives (protocol default, decision D-B).
    """
    if frames.ndim != 4:
        raise ValueError(f"expected [T, C, H, W], got shape {frames.shape}")
    if reduction == "sum":
        return frames.sum(axis=0)
    if reduction == "mean":
        return frames.mean(axis=0, dtype=np.float32)
    raise ValueError(f"unknown reduction '{reduction}'")
