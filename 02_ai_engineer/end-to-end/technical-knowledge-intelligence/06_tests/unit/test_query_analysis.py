"""Unit tests for query analysis and pipeline quality scoring."""

from __future__ import annotations

import pytest
from test_support.factories import make_chunk, make_search_hit
from tkip.query_analysis import analyze_query, evaluate_pipeline

pytestmark = pytest.mark.unit


def test_code_search_analysis_detects_topic_and_recommended_tool() -> None:
    analysis = analyze_query("Keress PyTorch Dataset kód példát", "CODE_SEARCH", "hu")

    assert "pytorch" in analysis["topics"]
    assert analysis["needs_code"] is True
    assert "search_code_examples" in analysis["recommended_tools"]


def test_pipeline_evaluation_scores_successful_tool_and_valid_citation() -> None:
    hit = make_search_hit(
        make_chunk(text="PyTorch Dataset implementation example"),
        reranker_score=0.8,
    )

    evaluation = evaluate_pipeline(
        analysis={"topics": ["pytorch"], "recommended_tools": ["search_code_examples"]},
        selected_hits=[hit],
        tool_results=[{"tool": "search_code_examples", "status": "success"}],
        ranking=[{"reranker_score": 0.8}],
        citation_valid=True,
    )

    assert evaluation["tool_calling_score"] == 100
    assert evaluation["citation_score"] == 100
