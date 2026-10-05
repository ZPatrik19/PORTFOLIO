from rag_engine.evaluation.runner import evaluate_retrieval


def test_retrieval_metrics():
    result = evaluate_retrieval(["a", "b", "c"], {"b", "x"}, k=3)
    assert result.recall_at_k == 0.5
    assert result.hit_rate_at_k == 1.0
    assert result.mrr == 0.5
