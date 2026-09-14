from __future__ import annotations

import numpy as np
import pandas as pd

from prompt_benchmark.constants import INVALID_LABEL, LABELS


def _macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 over the fixed benchmark labels without sklearn call overhead."""
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_values.append(float((2 * true_positive) / denominator) if denominator else 0.0)
    return float(np.mean(f1_values))


def bootstrap_macro_f1_ci(
    frame: pd.DataFrame,
    iterations: int = 1000,
    random_seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap confidence interval for Macro F1."""
    rng = np.random.default_rng(random_seed)
    y_true = frame["true_label"].astype(str).to_numpy()
    y_pred = frame["predicted_label"].fillna(INVALID_LABEL).astype(str).to_numpy()
    n = len(frame)
    if n == 0:
        return float("nan"), float("nan")

    scores = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample_indices = rng.integers(0, n, size=n)
        scores[iteration] = _macro_f1_fast(y_true[sample_indices], y_pred[sample_indices])

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1.0 - alpha))
