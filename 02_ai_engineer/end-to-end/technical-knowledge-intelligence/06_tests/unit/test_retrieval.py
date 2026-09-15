"""Unit tests for lexical retrieval and reciprocal-rank fusion."""
from __future__ import annotations

import pytest

from test_support.factories import make_chunk
from tkip.retrieval import BM25Index, reciprocal_rank_fusion

pytestmark = pytest.mark.unit


def test_bm25_prefers_chunk_with_matching_technical_terms() -> None:
    # Arrange
    docker_chunk = make_chunk(
        chunk_id="docker",
        text="Docker bridge networking exposes container ports through virtual networks.",
    )
    ml_chunk = make_chunk(
        chunk_id="ml",
        text="Random forest models aggregate decision trees for supervised learning.",
    )
    index = BM25Index([docker_chunk, ml_chunk])

    # Act
    results = index.search("docker networking", 2)

    # Assert
    assert results[0][0].chunk_id == "docker"


def test_bm25_uses_title_metadata_when_query_names_a_topic() -> None:
    # Arrange
    etl_chunk = make_chunk(
        chunk_id="etl-1",
        document_id="etl-doc",
        title="Understanding ETL",
        text="Extract, transform and load pipelines move and prepare data.",
    )
    generic_chunk = make_chunk(
        chunk_id="ml-1",
        document_id="ml-doc",
        title="Machine Learning",
        text="What are the main steps and features of a model training workflow?",
    )
    index = BM25Index([etl_chunk, generic_chunk])

    # Act
    results = index.search("What is ETL and what are its main steps?", 2)

    # Assert
    assert results[0][0].document_id == "etl-doc"


def test_rrf_keeps_candidates_from_both_ranked_lists() -> None:
    # Arrange
    lexical_results = [(make_chunk(chunk_id="lexical", text="exact term match"), 3.0)]
    dense_results = [(make_chunk(chunk_id="semantic", text="semantic match"), 0.9)]

    # Act
    fused = reciprocal_rank_fusion(lexical_results, dense_results)
    fused_ids = {hit.chunk.chunk_id for hit in fused}

    # Assert
    assert fused_ids == {"lexical", "semantic"}
