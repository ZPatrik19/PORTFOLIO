from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float


def _percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank-style percentile kept compatible with earlier benchmark reports."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def summarize_latencies(values_ms: list[float]) -> LatencyStats:
    if not values_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
    values = [float(value) for value in values_ms]
    median = statistics.median(values)
    return LatencyStats(
        mean_ms=statistics.fmean(values),
        median_ms=median,
        p50_ms=_percentile(values, 0.50),
        p90_ms=_percentile(values, 0.90),
        p95_ms=_percentile(values, 0.95),
        p99_ms=_percentile(values, 0.99),
        min_ms=min(values),
        max_ms=max(values),
        stddev_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
    )
