"""Unit tests for benchmark catalog, nDCG, experiment tracking and Gemini errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from tkip.benchmark_catalog import BENCHMARK_CATALOG
from tkip.config import load_config
from tkip.evaluation import _ndcg
from tkip.experiment_tracking import ExperimentTracker
from tkip.gemini_service import GeminiGenerationError, GeminiService
from tkip.models import SourceCitation

pytestmark = pytest.mark.unit


EXPECTED_BENCHMARK_FAMILIES = {
    "Text Classification",
    "Text Clustering",
    "Topic Modeling",
    "Prompt Engineering",
    "Text Generation",
    "Structured Output",
    "Tool Calling",
    "Retrieval",
    "Reranker",
    "Full RAG",
    "Hallucination / Abstention",
    "Prompt Injection / Robustness",
    "Regression",
    "System / Production",
}


def test_catalog_contains_expected_ai_evaluation_families_and_metrics() -> None:
    assert EXPECTED_BENCHMARK_FAMILIES.issubset(BENCHMARK_CATALOG)
    assert "Macro F1" in BENCHMARK_CATALOG["Text Classification"]["metrics"]
    assert "nDCG@K" in BENCHMARK_CATALOG["Retrieval"]["metrics"]
    assert "Tool Selection Accuracy" in BENCHMARK_CATALOG["Tool Calling"]["metrics"]


def test_ndcg_is_one_when_relevant_result_is_ranked_first() -> None:
    score = _ndcg(["relevant", "other-1", "other-2"], {"relevant"}, k=3)

    assert score == pytest.approx(1.0)


def test_ndcg_penalizes_relevant_result_when_it_moves_down_the_ranking() -> None:
    early = _ndcg(["relevant", "other-1", "other-2"], {"relevant"}, k=3)
    late = _ndcg(["other-1", "other-2", "relevant"], {"relevant"}, k=3)

    assert early > late > 0.0


def test_experiment_tracker_persists_and_reads_jsonl_trace(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path)

    written = tracker.append({"benchmark_family": "Prompt Engineering", "status": "success"})
    rows = tracker.recent()

    assert written["experiment_id"].startswith("exp_")
    assert len(rows) == 1
    assert rows[0]["benchmark_family"] == "Prompt Engineering"


def test_configured_gemini_failure_is_raised_instead_of_hidden_by_fallback() -> None:
    class BrokenInteractions:
        def create(self, **kwargs):
            raise RuntimeError("interactions unavailable")

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("generateContent unavailable")

    class BrokenClient:
        interactions = BrokenInteractions()
        models = BrokenModels()

    service = GeminiService(load_config(), api_key=None)
    service.key = "test-key"
    service.client = BrokenClient()
    citation = SourceCitation(
        document_id="doc",
        document_title="Demo",
        page=1,
        chunk_id="chunk",
        quote_or_evidence="Grounded evidence.",
    )

    with pytest.raises(GeminiGenerationError):
        service.generate_structured(
            "Explain RAG",
            {"prompt": "RETRIEVED SOURCES: evidence", "selected_hits": []},
            [citation],
            "CONCEPTUAL",
            max_output_tokens=500,
        )


def test_gemini_quota_error_is_classified_without_exponential_retry(monkeypatch) -> None:
    from tkip.exceptions import QuotaExceededError

    calls = {"interactions": 0, "fallback": 0}

    class QuotaInteractions:
        def create(self, **kwargs):
            calls["interactions"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    class QuotaModels:
        def generate_content(self, **kwargs):
            calls["fallback"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    class QuotaClient:
        interactions = QuotaInteractions()
        models = QuotaModels()

    service = GeminiService(load_config(), api_key=None)
    service.key = "test-key"
    service.client = QuotaClient()

    with pytest.raises(QuotaExceededError):
        service.generate_structured(
            "Explain RAG",
            {"prompt": "RETRIEVED SOURCES: evidence", "selected_hits": []},
            [],
            "CONCEPTUAL",
            max_output_tokens=500,
        )

    assert calls["interactions"] == 1
    assert calls["fallback"] <= 1
