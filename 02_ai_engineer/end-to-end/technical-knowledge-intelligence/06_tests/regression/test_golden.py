"""Golden regression tests for stable deterministic behaviors."""
from __future__ import annotations

import pytest

from tkip.query_understanding import classify_intent

pytestmark = pytest.mark.regression


GOLDEN_INTENT_CASES = [
    ("Compare three books on transformers", "COMPARISON"),
    ("Find PyTorch code example", "CODE_SEARCH"),
    ("Tanítsd meg a PCA-t", "LEARNING"),
]


@pytest.mark.parametrize(("question", "expected_intent"), GOLDEN_INTENT_CASES)
def test_intent_classification_does_not_regress(question: str, expected_intent: str) -> None:
    assert classify_intent(question) == expected_intent
