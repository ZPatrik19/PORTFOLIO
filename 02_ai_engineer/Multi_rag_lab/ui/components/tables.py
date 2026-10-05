from __future__ import annotations

import json
from numbers import Number
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # unit-test/minimal environments
    st = None  # type: ignore[assignment]


def _stringify(value: Any) -> Any:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except TypeError, ValueError:
        pass
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)


def arrow_safe_frame(data: Any) -> pd.DataFrame:
    """Return a DataFrame whose columns have Arrow-compatible scalar types.

    Streamlit serializes DataFrames through PyArrow. Pandas ``object`` columns can
    legally contain mixed Python types (for example ``0`` and ``"nincs"``), but
    Arrow requires one coherent type per column. This function preserves numeric
    and boolean object columns when possible and converts genuinely mixed/display
    columns to Pandas' nullable string dtype.
    """
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    for column in frame.columns:
        series = frame[column]
        if series.dtype != object:
            continue

        values = [value for value in series.tolist() if value is not None and value is not pd.NA]
        cleaned: list[Any] = []
        for value in values:
            try:
                if pd.isna(value):
                    continue
            except TypeError, ValueError:
                pass
            cleaned.append(value)

        if not cleaned:
            frame[column] = series.astype("string")
            continue

        if all(isinstance(value, bool) for value in cleaned):
            frame[column] = series.astype("boolean")
            continue

        if all(isinstance(value, Number) and not isinstance(value, bool) for value in cleaned):
            numeric = pd.to_numeric(series, errors="coerce")
            integral = all(float(value).is_integer() for value in cleaned)
            frame[column] = numeric.astype("Int64" if integral else "Float64")
            continue

        # Any string + number mixture, Path/object value or nested structure is a
        # display column. Normalize it explicitly instead of letting PyArrow guess.
        frame[column] = pd.Series((_stringify(value) for value in series.tolist()), index=series.index, dtype="string")

    return frame


def safe_dataframe(data: Any, *args: Any, **kwargs: Any):
    if st is None:
        raise RuntimeError("Streamlit is required to render a dataframe.")
    return st.dataframe(arrow_safe_frame(data), *args, **kwargs)


def safe_data_editor(data: Any, *args: Any, **kwargs: Any):
    if st is None:
        raise RuntimeError("Streamlit is required to render a data editor.")
    return st.data_editor(arrow_safe_frame(data), *args, **kwargs)
