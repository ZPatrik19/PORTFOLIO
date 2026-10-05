from rag_engine.evaluation.generation import (
    answer_redundancy_proxy,
    citation_coverage,
    citation_source_coverage,
)
from rag_engine.evaluation.runner import evaluate_retrieval


def test_extended_retrieval_metrics_are_bounded_and_rank_aware():
    metrics = evaluate_retrieval(["x", "a", "b", "z"], {"a", "b"}, k=4)
    assert 0 <= metrics.f1_at_k <= 1
    assert 0 <= metrics.map_at_k <= 1
    assert metrics.first_relevant_rank == 2
    assert metrics.mrr == 0.5


def test_generation_citation_coverage_and_source_coverage():
    answer = "Első állítás. Második állítás [S1]. Harmadik állítás [S2]."
    assert citation_coverage(answer) == 2 / 3
    assert citation_source_coverage(answer, 4) == 0.5


def test_answer_redundancy_proxy_detects_repeated_sentences():
    assert answer_redundancy_proxy("A. A. B.") > 0
    assert answer_redundancy_proxy("A. B. C.") == 0
