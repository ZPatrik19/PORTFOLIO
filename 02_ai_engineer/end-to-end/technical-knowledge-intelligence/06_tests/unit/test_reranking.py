from __future__ import annotations

from test_support.factories import make_chunk, make_search_hit
from tkip.reranking import lightweight_rerank, rerank


def test_lightweight_reranker_prefers_matching_title_and_body() -> None:
    # Arrange
    relevant = make_search_hit(
        make_chunk(
            chunk_id="rag",
            title="RAG Architecture",
            text="RAG retrieval context generation architecture",
        ),
        hybrid_score=0.02,
    )
    unrelated = make_search_hit(
        make_chunk(
            chunk_id="docker",
            title="Docker Networking",
            text="Containers use bridge networks.",
        ),
        hybrid_score=0.04,
    )

    # Act
    ranked = lightweight_rerank("RAG architecture", [unrelated, relevant], top_n=2)

    # Assert
    assert ranked[0].chunk.chunk_id == "rag"
    assert ranked[0].reranker_score > ranked[1].reranker_score


def test_disabled_reranker_preserves_candidate_order() -> None:
    # Arrange
    first = make_search_hit(make_chunk(chunk_id="first"), rank=1)
    second = make_search_hit(make_chunk(chunk_id="second"), rank=2)
    config = {"reranking": {"enabled": False, "top_n": 1}}

    # Act
    ranked = rerank("anything", [first, second], config)

    # Assert
    assert [hit.chunk.chunk_id for hit in ranked] == ["first"]
