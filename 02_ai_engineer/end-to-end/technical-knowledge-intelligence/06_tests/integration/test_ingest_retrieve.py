"""Offline integration test for ingestion -> indexing -> retrieval -> orchestration."""

from __future__ import annotations

import pytest
from tkip.models import AskRequest
from tkip.orchestration import KnowledgePlatform

pytestmark = pytest.mark.integration


def test_demo_corpus_can_be_ingested_cached_retrieved_and_orchestrated(
    isolated_config: dict,
) -> None:
    # Arrange
    platform = KnowledgePlatform(isolated_config)

    # Act 1: cold ingestion/indexing.
    first_report = platform.ingest_and_index()

    # Assert 1: public demo corpus produced a searchable index.
    assert first_report["chunks"] > 0
    assert first_report["failures"] == []

    # Act 2: repeat ingestion to validate caches.
    second_report = platform.ingest_and_index()

    # Assert 2: unchanged corpus does not recompute embeddings.
    assert second_report["embeddings_computed"] == 0
    assert second_report["embedding_cache_hits"] == second_report["chunks"]

    # Act 3: query the local hybrid retriever.
    hits = platform.retriever.search("Docker networking bridge")

    # Assert 3: relevant Docker evidence appears near the top.
    assert hits
    assert any("docker" in hit.chunk.text.lower() for hit in hits[:5])

    # Act 4: run orchestration without a live Gemini key.
    answer = platform.ask(
        AskRequest(
            question="Explain Docker networking from the documentation.",
            debug=True,
            prompt_language_check=False,
            quality_review=False,
        )
    )

    # Assert 4: offline orchestration still exposes deterministic diagnostics.
    diagnostics = answer.diagnostics
    assert diagnostics.get("pipeline_steps")
    assert diagnostics.get("ranking")
    assert any(step.get("stage") == "Context engineering" for step in diagnostics["pipeline_steps"])

    required_ranking_fields = {
        "bm25_score",
        "dense_score",
        "hybrid_score",
        "reranker_score",
        "final_rank",
    }
    assert required_ranking_fields.issubset(diagnostics["ranking"][0])
