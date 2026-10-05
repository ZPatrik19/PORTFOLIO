from __future__ import annotations

from rag_engine.evaluation.pipeline_matrix import build_matrix_plan


def test_reranked_rag_expands_by_reranker_and_device() -> None:
    plan = build_matrix_plan(
        embedding_modes=["multilingual"],
        embedding_devices=["cpu"],
        vector_devices=["cpu"],
        reranker_devices=["cpu", "cuda"],
        chunkings=["recursive"],
        retrieval_modes=["hybrid-rrf"],
        rerankers=["none", "lexical", "cross-encoder"],
        rag_strategies=["baseline", "reranked"],
        include_rag=True,
        include_components=False,
    )
    # baseline = 1; lexical is CPU-only (1 variant), cross-encoder = CPU + CUDA (2)
    assert plan.rag_configurations == 4


def test_lexical_and_none_do_not_duplicate_across_cuda_device() -> None:
    plan = build_matrix_plan(
        embedding_modes=["multilingual"],
        embedding_devices=["cpu"],
        vector_devices=["cpu"],
        reranker_devices=["cpu", "cuda"],
        chunkings=["recursive"],
        retrieval_modes=["hybrid-rrf"],
        rerankers=["none", "lexical"],
        rag_strategies=["baseline"],
        include_rag=False,
        include_components=False,
    )
    assert plan.retrieval_configurations == 2
