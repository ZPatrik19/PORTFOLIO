"""EN: Strict benchmark schema validation before any paid/external request can be sent.

HU: Szigorú benchmark-sémavalidációt ellenőriz még bármilyen fizetős/külső API-hívás előtt.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prompt_benchmark.data.validation import DataValidationError, validate_benchmark_frame


def _valid() -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ["x1"], "text": ["Cancel my plan"], "true_label": ["cancellation"]})


def test_valid_benchmark_schema_passes() -> None:
    """EN: Confirms a correctly shaped benchmark frame is accepted.

    HU: Megerősíti, hogy a helyes benchmark DataFrame sémát a validátor elfogadja.
    """
    validate_benchmark_frame(_valid())


def test_duplicate_ids_fail_before_api_calls() -> None:
    """EN: Rejects duplicate sample IDs before any provider request can consume quota or money.

    HU: Duplikált sample_id esetén még provider-hívás előtt leállítja a folyamatot, így nem fogy quota vagy pénz.
    """
    frame = pd.concat([_valid(), _valid()], ignore_index=True)
    with pytest.raises(DataValidationError, match="unique"):
        validate_benchmark_frame(frame)


def test_unknown_label_fails_before_api_calls() -> None:
    """EN: Rejects labels outside the supported intent vocabulary before inference starts.

    HU: Ismeretlen label esetén inference előtt hibát jelez.
    """
    frame = _valid()
    frame.loc[0, "true_label"] = "unknown"
    with pytest.raises(DataValidationError, match="Unknown"):
        validate_benchmark_frame(frame)
