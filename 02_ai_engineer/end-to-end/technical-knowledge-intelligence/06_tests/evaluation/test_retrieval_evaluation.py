"""Deterministic evaluation tests for retrieval and tool metrics.

These tests do not ask whether an AI answer *sounds* good. They validate the
mathematical evaluation functions against hand-calculable examples.
"""

from __future__ import annotations

import pytest
from tkip.evaluation import _ndcg, retrieval_metrics, tool_call_metrics

pytestmark = pytest.mark.evaluation


def test_retrieval_metrics_match_hand_calculated_example() -> None:
    # Relevant chunk appears at rank 2.
    ranked_chunk_ids = [["irrelevant", "relevant", "other"]]
    expected_relevant_ids = [{"relevant"}]

    summary, raw = retrieval_metrics(ranked_chunk_ids, expected_relevant_ids, ks=(1, 3))

    assert len(raw) == 1
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["recall@1"] == pytest.approx(0.0)
    assert summary["recall@3"] == pytest.approx(1.0)
    assert summary["precision@3"] == pytest.approx(1 / 3)


def test_ndcg_returns_zero_when_no_relevant_item_is_retrieved() -> None:
    score = _ndcg(["a", "b", "c"], {"relevant"}, k=3)

    assert score == pytest.approx(0.0)


def test_ndcg_rewards_earlier_relevant_items() -> None:
    relevant = {"relevant"}

    first = _ndcg(["relevant", "a", "b"], relevant, k=3)
    third = _ndcg(["a", "b", "relevant"], relevant, k=3)

    assert first == pytest.approx(1.0)
    assert first > third > 0.0


def test_tool_metrics_distinguish_correct_and_unnecessary_calls() -> None:
    records = [
        {
            "expected_tool": "search_library",
            "selected_tool": "search_library",
            "execution_success": True,
            "arguments_valid": True,
            "tool_calls": 1,
        },
        {
            "expected_tool": None,
            "selected_tool": None,
            "execution_success": False,
            "arguments_valid": False,
            "tool_calls": 0,
        },
    ]

    metrics = tool_call_metrics(records)

    assert metrics["tool_selection_accuracy"] == pytest.approx(1.0)
    assert metrics["unnecessary_tool_call_rate"] == pytest.approx(0.0)
    assert metrics["tool_execution_success"] == pytest.approx(1.0)
    assert metrics["tool_argument_accuracy"] == pytest.approx(1.0)
