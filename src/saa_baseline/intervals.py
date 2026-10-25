"""Half-open intervals and non-bridging observation support.

An event is a positive RUN, not a connected support component or physical event.
Runs end at negatives or acquisition gaps > max_gap. Each run's support may
contain several disjoint pieces. Cadence jitter never creates new runs.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def utc_times(values):
    t = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    if t.hasnans or t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError("Observation timestamps must be unique, finite and increasing.")
    return t


def merge_intervals(intervals):
    result = []
    for s, e in sorted(intervals):
        if pd.isna(s) or pd.isna(e) or e <= s:
            raise ValueError("Intervals must have finite start < end.")
        if result and s <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], e))
        else:
            result.append((s, e))
    return result


def rasterize(times, intervals):
    t = pd.DatetimeIndex(times)
    out = np.zeros(len(t), dtype=bool)
    for s, e in merge_intervals(intervals):
        out |= (t >= s) & (t < e)
    return out


def interval_seconds(interval):
    # Integer nanoseconds retain sub-microsecond jitter rather than rounding it.
    return (interval[1].value - interval[0].value) / 1e9


@dataclass(frozen=True)
class SupportEvent:
    pieces: tuple

    def __post_init__(self):
        merged = tuple(merge_intervals(self.pieces))
        if not merged:
            raise ValueError("An event needs nonempty positive-duration support.")
        object.__setattr__(self, "pieces", merged)

    @property
    def start(self):
        return self.pieces[0][0]

    @property
    def end(self):
        return self.pieces[-1][1]

    @property
    def seconds(self):
        return sum(interval_seconds(p) for p in self.pieces)

    @property
    def envelope_seconds(self):
        return interval_seconds((self.start, self.end))


def observation_support(times, cadence, analysis_end=None):
    t = utc_times(times)
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("cadence must be finite and positive")
    cap = pd.Timedelta(seconds=float(cadence))
    if analysis_end is not None and (pd.isna(analysis_end) or analysis_end.tzinfo is None):
        raise ValueError("analysis_end must be finite and timezone-aware")
    if analysis_end is not None and len(t) and analysis_end <= t[-1]:
        raise ValueError("analysis_end must be strictly after the last sample")
    pieces = []
    for i, s in enumerate(t):
        e = min(s + cap, t[i + 1]) if i + 1 < len(t) else s + cap
        if analysis_end is not None:
            e = min(e, analysis_end)
        pieces.append((s, e))
    return pieces


def positive_runs(times, positive, cadence, max_gap=30., analysis_end=None):
    t = utc_times(times)
    y = np.asarray(positive)
    if y.ndim != 1 or len(y) != len(t) or not np.isin(y, [0, 1]).all():
        raise ValueError("positive must be a binary vector aligned with times")
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive")
    support = observation_support(t, cadence, analysis_end)
    result, current = [], []
    for i, is_positive in enumerate(y):
        gap = i > 0 and (t[i].value - t[i - 1].value) / 1e9 > max_gap
        if current and (gap or not is_positive):
            result.append(SupportEvent(tuple(current)))
            current = []
        if is_positive:
            current.append(support[i])
    if current:
        result.append(SupportEvent(tuple(current)))
    return result


def legacy_envelope_runs(times, positive, cadence, max_gap=30.):
    """Exact 21-Oct algorithm; only for legacy sensitivity comparison."""
    times = list(times)
    positive = np.asarray(positive, dtype=bool)
    result, begin, last = [], None, None
    for i, t in enumerate(times):
        if begin is not None and (t - last).total_seconds() > max_gap:
            result.append((begin, last + pd.Timedelta(seconds=cadence)))
            begin = None
        if positive[i]:
            if begin is None:
                begin = t
            last = t
        elif begin is not None:
            result.append((begin, min(t, last + pd.Timedelta(seconds=cadence))))
            begin = None
    if begin is not None:
        result.append((begin, last + pd.Timedelta(seconds=cadence)))
    return [SupportEvent((x,)) for x in result]


def intersection_seconds(a, b):
    i = j = 0
    total_ns = 0
    while i < len(a.pieces) and j < len(b.pieces):
        x, y = a.pieces[i], b.pieces[j]
        total_ns += max(0, min(x[1].value, y[1].value) - max(x[0].value, y[0].value))
        if x[1] <= y[1]:
            i += 1
        else:
            j += 1
    return total_ns / 1e9


def event_match(reference, predicted, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1].")
    matrix = np.zeros((len(reference), len(predicted)))
    for i, a in enumerate(reference):
        for j, b in enumerate(predicted):
            overlap = intersection_seconds(a, b)
            matrix[i, j] = overlap / (a.seconds + b.seconds - overlap) if overlap else 0.
    matches = 0
    matched_ious = []
    if matrix.size:
        eligible = matrix >= threshold
        # Cardinality dominates the sum of all IoU terms, then maximize IoU.
        objective = eligible * (min(matrix.shape) + 1 + matrix)
        rows, cols = linear_sum_assignment(-objective)
        matched_ious = [float(matrix[i, j]) for i, j in zip(rows, cols) if eligible[i, j]]
        matches = len(matched_ious)
    return {"IoU_threshold": threshold, "matches": matches,
            "human_events": len(reference), "predicted_runs": len(predicted),
            "fraction_human_events_matched": matches / len(reference) if reference else None,
            "fraction_predicted_runs_matched": matches / len(predicted) if predicted else None,
            "mean_matched_IoU": float(np.mean(matched_ious)) if matched_ious else None}
