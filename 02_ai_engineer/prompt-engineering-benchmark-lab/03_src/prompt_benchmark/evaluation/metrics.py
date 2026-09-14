from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support

from prompt_benchmark.constants import INVALID_LABEL, LABELS


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + (z * z / total)
    center = (p + (z * z / (2.0 * total))) / denom
    margin = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute quality, reliability, token, latency and cost metrics."""
    if frame.empty:
        raise ValueError("Cannot calculate metrics for an empty benchmark frame.")

    y_true = frame["true_label"].astype(str).tolist()
    y_pred = frame["predicted_label"].fillna(INVALID_LABEL).astype(str).tolist()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0
    )

    valid_output = frame["valid_output"].astype(bool)
    valid_json_values = frame["valid_json"].dropna() if "valid_json" in frame else pd.Series(dtype=bool)
    correct = frame["correct"].astype(bool) if "correct" in frame else pd.Series(
        [a == b for a, b in zip(y_true, y_pred)], index=frame.index
    )
    errors = frame["error"].fillna("").astype(bool) if "error" in frame else pd.Series(False, index=frame.index)

    n = int(len(frame))
    correct_count = int(correct.sum())
    invalid_output_count = int((~valid_output).sum())
    total_latency = float(frame["latency_seconds"].sum())
    total_tokens = float(frame["total_tokens"].sum())
    total_cost = float(frame["estimated_cost_usd"].sum())
    accuracy_ci_low, accuracy_ci_high = wilson_interval(correct_count, n)

    return {
        # Volume / counts
        "requests": float(n),
        "correct_count": float(correct_count),
        "incorrect_count": float(n - correct_count),
        "invalid_output_count": float(invalid_output_count),
        "api_error_count": float(errors.sum()),
        # Quality
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci_low": float(accuracy_ci_low),
        "accuracy_ci_high": float(accuracy_ci_high),
        "balanced_accuracy": float(recall_macro),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        # Reliability / contract
        "invalid_output_rate": float((~valid_output).mean()),
        "output_contract_valid_rate": float(valid_output.mean()),
        "invalid_json_rate": float((~valid_json_values.astype(bool)).mean()) if len(valid_json_values) else float("nan"),
        "json_grammar_valid_rate": float(valid_json_values.astype(bool).mean()) if len(valid_json_values) else float("nan"),
        "error_rate": float(errors.mean()),
        # Tokens
        "mean_input_tokens": float(frame["input_tokens"].mean()),
        "mean_output_tokens": float(frame["output_tokens"].mean()),
        "mean_total_tokens": float(frame["total_tokens"].mean()),
        "total_input_tokens": float(frame["input_tokens"].sum()),
        "total_output_tokens": float(frame["output_tokens"].sum()),
        "total_tokens": total_tokens,
        "tokens_per_correct_prediction": float(total_tokens / correct_count) if correct_count else float("inf"),
        # Latency
        "mean_latency_seconds": float(frame["latency_seconds"].mean()),
        "median_latency_seconds": float(frame["latency_seconds"].median()),
        "p50_latency_seconds": float(frame["latency_seconds"].quantile(0.50)),
        "p95_latency_seconds": float(frame["latency_seconds"].quantile(0.95)),
        "p99_latency_seconds": float(frame["latency_seconds"].quantile(0.99)),
        "total_latency_seconds": total_latency,
        "sequential_throughput_rps": float(n / total_latency) if total_latency > 0 else float("inf"),
        # Cost
        "mean_cost_usd": float(frame["estimated_cost_usd"].mean()),
        "cost_per_1000_requests_usd": float(frame["estimated_cost_usd"].mean() * 1000),
        "benchmark_cost_usd": total_cost,
        "cost_per_correct_prediction_usd": float(total_cost / correct_count) if correct_count else float("inf"),
        # Execution complexity
        "mean_branch_count": float(frame["branch_count"].mean()) if "branch_count" in frame else 1.0,
    }


def per_class_report(frame: pd.DataFrame) -> pd.DataFrame:
    y_pred = frame["predicted_label"].fillna(INVALID_LABEL).astype(str)
    report = classification_report(
        frame["true_label"], y_pred, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).T.loc[list(LABELS), ["precision", "recall", "f1-score", "support"]]


def grouped_classification_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Calculate a compact metric set for each synthetic scenario/difficulty group."""
    if group_column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(group_column, dropna=False):
        if group.empty:
            continue
        metrics = classification_metrics(group)
        rows.append(
            {
                group_column: value,
                "requests": int(metrics["requests"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "invalid_output_rate": metrics["invalid_output_rate"],
                "mean_total_tokens": metrics["mean_total_tokens"],
                "p95_latency_seconds": metrics["p95_latency_seconds"],
            }
        )
    return pd.DataFrame(rows)


def confusion_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a confusion table that keeps invalid/unparseable predictions visible.

    Unlike ``sklearn.metrics.confusion_matrix``, this helper safely handles NaN
    predictions by mapping them to ``INVALID_LABEL``. Rows are the six valid
    ground-truth classes; columns include the six labels plus ``INVALID_LABEL``.
    """
    if frame.empty:
        raise ValueError("Cannot build a confusion table from an empty frame.")
    true_values = frame["true_label"].fillna(INVALID_LABEL).astype(str)
    pred_values = frame["predicted_label"].fillna(INVALID_LABEL).astype(str)
    table = pd.crosstab(true_values, pred_values, dropna=False)
    return table.reindex(index=list(LABELS), columns=[*LABELS, INVALID_LABEL], fill_value=0)
