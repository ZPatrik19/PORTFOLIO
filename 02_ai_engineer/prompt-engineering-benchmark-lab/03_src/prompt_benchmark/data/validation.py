"""DataFrame schema validation used by benchmark and UI entry points."""
from __future__ import annotations

import pandas as pd

from prompt_benchmark.constants import LABELS


class DataValidationError(ValueError):
    """Raised when an input dataset cannot be safely benchmarked."""


def validate_benchmark_frame(frame: pd.DataFrame) -> None:
    """Validate minimum benchmark schema and label integrity.

    The function is intentionally strict at the benchmark boundary so malformed
    uploads fail before expensive external API calls start.
    """
    if frame.empty:
        raise DataValidationError("Benchmark dataset is empty.")
    required = {"sample_id", "text", "true_label"}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"Benchmark dataset is missing required columns: {sorted(missing)}")
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).str.strip().eq("").any():
        raise DataValidationError("sample_id contains empty values.")
    if frame["sample_id"].astype(str).duplicated().any():
        raise DataValidationError("sample_id must be unique within one benchmark run.")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise DataValidationError("text contains empty values.")
    labels = set(frame["true_label"].astype(str).str.lower())
    unknown = labels - set(LABELS)
    if unknown:
        raise DataValidationError(f"Unknown true_label values: {sorted(unknown)}")
