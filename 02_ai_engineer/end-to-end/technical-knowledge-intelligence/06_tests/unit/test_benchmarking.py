"""Unit tests for benchmark aggregation and chunk statistics."""
from __future__ import annotations

import pandas as pd
import pytest

from tkip.evaluation import chunk_statistics, summarize_retrieval_frame
from test_support.factories import make_chunk

pytestmark = pytest.mark.unit


def test_retrieval_summary_aggregates_quality_and_latency_metrics() -> None:
    raw = pd.DataFrame(
        [
            {
                "method": "BM25", "mrr": 1.0, "hit_rate": 1.0,
                "recall@1": 1.0, "precision@1": 1.0, "ndcg@1": 1.0,
                "recall@3": 1.0, "precision@3": 1 / 3, "ndcg@3": 1.0,
                "recall@5": 1.0, "precision@5": 0.2, "ndcg@5": 1.0,
                "recall@10": 1.0, "precision@10": 0.1, "ndcg@10": 1.0,
                "latency_ms": 10.0,
            },
            {
                "method": "BM25", "mrr": 0.5, "hit_rate": 1.0,
                "recall@1": 0.0, "precision@1": 0.0, "ndcg@1": 0.0,
                "recall@3": 1.0, "precision@3": 1 / 3, "ndcg@3": 0.63,
                "recall@5": 1.0, "precision@5": 0.2, "ndcg@5": 0.63,
                "recall@10": 1.0, "precision@10": 0.1, "ndcg@10": 0.63,
                "latency_ms": 30.0,
            },
        ]
    )

    summary = summarize_retrieval_frame(raw, group_cols=("method",))

    assert len(summary) == 1
    assert int(summary.iloc[0]["samples"]) == 2
    assert summary.iloc[0]["mrr"] == pytest.approx(0.75)
    assert summary.iloc[0]["recall@5"] == pytest.approx(1.0)
    assert summary.iloc[0]["p50_latency_ms"] == pytest.approx(10.0)
    assert summary.iloc[0]["p95_latency_ms"] == pytest.approx(10.0)


def test_chunk_statistics_report_size_distribution_and_code_ratio() -> None:
    chunks = [
        make_chunk(chunk_id="text", text="abc" * 100, chunk_type="paragraph"),
        make_chunk(chunk_id="code", text="print('x')" * 10, chunk_type="code"),
    ]

    stats = chunk_statistics(chunks)

    assert stats["chunk_count"] == 2
    assert stats["avg_chunk_chars"] > 0
    assert stats["unique_documents"] == 1
    assert stats["code_ratio"] == pytest.approx(0.5)
