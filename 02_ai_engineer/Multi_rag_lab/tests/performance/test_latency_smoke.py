import pytest

from rag_engine.evaluation.latency import summarize_latencies


@pytest.mark.performance
def test_latency_statistics_smoke():
    stats = summarize_latencies([1, 2, 3, 4, 100])
    assert stats.mean_ms > stats.median_ms
    assert stats.p95_ms == 100
