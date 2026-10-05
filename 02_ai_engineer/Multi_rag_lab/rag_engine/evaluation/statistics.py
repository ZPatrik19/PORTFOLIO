from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    confidence: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class PairedComparison:
    mean_a: float
    mean_b: float
    mean_delta_b_minus_a: float
    ci_low: float
    ci_high: float
    probability_b_better: float
    relative_change_percent: float | None
    paired_effect_size: float
    samples: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _array(values: Iterable[float]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Non-parametric bootstrap confidence interval for a mean.

    The implementation deliberately depends only on NumPy so it remains available in
    the default project environment and in CI.
    """

    data = _array(values)
    if data.size == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, 0)
    if data.size == 1 or bootstrap_samples <= 0:
        value = float(data.mean())
        return ConfidenceInterval(value, value, value, confidence, int(data.size))
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(bootstrap_samples, data.size))
    means = data[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return ConfidenceInterval(float(data.mean()), float(low), float(high), confidence, int(data.size))


def paired_bootstrap_comparison(
    values_a: Iterable[float],
    values_b: Iterable[float],
    *,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 4000,
    seed: int = 42,
) -> PairedComparison:
    """Compare two configurations measured on the same observations.

    Pairing is important for RAG experiments because the same questions are evaluated
    by both configurations. The reported probability is empirical bootstrap support,
    not a frequentist p-value.
    """

    a = _array(values_a)
    b = _array(values_b)
    if a.size != b.size:
        raise ValueError("Paired comparisons require the same number of observations")
    if a.size == 0:
        return PairedComparison(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0)

    delta = b - a
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    mean_delta = float(delta.mean())
    if a.size == 1 or bootstrap_samples <= 0:
        low = high = mean_delta
        probability = float(mean_delta > 0) if higher_is_better else float(mean_delta < 0)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, a.size, size=(bootstrap_samples, a.size))
        boot_delta = (b[indices] - a[indices]).mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        low, high = np.quantile(boot_delta, [alpha, 1.0 - alpha])
        probability = float(np.mean(boot_delta > 0.0) if higher_is_better else np.mean(boot_delta < 0.0))

    relative = (mean_delta / abs(mean_a) * 100.0) if abs(mean_a) > 1e-12 else None
    if a.size > 1:
        std = float(np.std(delta, ddof=1))
        effect = mean_delta / std if std > 1e-12 else 0.0
    else:
        effect = 0.0
    if not higher_is_better:
        effect *= -1.0

    return PairedComparison(
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta_b_minus_a=mean_delta,
        ci_low=float(low),
        ci_high=float(high),
        probability_b_better=probability,
        relative_change_percent=relative,
        paired_effect_size=effect,
        samples=int(a.size),
    )


def coefficient_of_variation(values: Iterable[float]) -> float:
    data = _array(values)
    if data.size < 2:
        return 0.0
    mean = float(data.mean())
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(data, ddof=1) / abs(mean))


def standard_error(values: Iterable[float]) -> float:
    data = _array(values)
    if data.size < 2:
        return 0.0
    return float(np.std(data, ddof=1) / sqrt(data.size))
