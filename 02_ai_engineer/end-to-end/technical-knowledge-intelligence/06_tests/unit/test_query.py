"""Unit tests for deterministic query-intent classification."""
from __future__ import annotations

import pytest

from tkip.query_understanding import classify_intent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("Keress PyTorch kód példát", "CODE_SEARCH"),
        ("Tanítsd meg nekem a PCA-t", "LEARNING"),
        ("Compare three books on transformer attention", "COMPARISON"),
    ],
)
def test_intent_classifier_returns_expected_intent(question: str, expected_intent: str) -> None:
    assert classify_intent(question) == expected_intent
