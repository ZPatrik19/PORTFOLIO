from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.components.evaluation_view_models import (
    build_rag_display_frame,
    build_retrieval_display_frame,
    rag_summary,
    retrieval_summary,
)


def test_retrieval_view_model_uses_hungarian_latency_columns_without_keyerror() -> None:
    rows = [
        {
            "chunking": "recursive",
            "retriever": "hybrid-rrf",
            "reranker": "lexical",
            "questions": 20,
            "recall_at_k": 0.82,
            "precision_at_k": 0.61,
            "f1_at_k": 0.70,
            "hit_rate_at_k": 0.95,
            "mrr": 0.78,
            "map_at_k": 0.73,
            "ndcg_at_k": 0.80,
            "mean_first_relevant_rank": 1.4,
            "source_diversity_at_k": 0.75,
            "duplicate_ratio_at_k": 0.05,
            "mean_latency_ms": 18.5,
            "p50_latency_ms": 17.0,
            "p95_latency_ms": 27.2,
            "p99_latency_ms": 31.0,
            "queries_per_second": 54.0,
            "labeling_coverage": 1.0,
        },
        {
            "chunking": "sentence",
            "retriever": "dense",
            "reranker": "none",
            "recall_at_k": 0.72,
            "precision_at_k": 0.55,
            "f1_at_k": 0.62,
            "hit_rate_at_k": 0.90,
            "mrr": 0.69,
            "map_at_k": 0.65,
            "ndcg_at_k": 0.70,
            "source_diversity_at_k": 0.67,
            "mean_latency_ms": 9.3,
            "p50_latency_ms": 8.8,
            "p95_latency_ms": 13.5,
            "p99_latency_ms": 16.2,
            "queries_per_second": 107.0,
            "labeling_coverage": 1.0,
        },
    ]

    frame = build_retrieval_display_frame(rows)
    summary = retrieval_summary(frame)

    assert "Átlagos késleltetés ms" in frame.columns
    assert "Mean latency ms" not in frame.columns
    assert "P95 késleltetés ms" in frame.columns
    assert "Összesített pontszám" in frame.columns
    assert summary["fastest_latency_ms"] == 9.3
    assert summary["best_configuration"]


def test_retrieval_view_model_tolerates_older_partial_rows() -> None:
    frame = build_retrieval_display_frame(
        [
            {
                "chunking": "recursive",
                "retriever": "dense",
                "reranker": "none",
                "recall_at_k": 0.5,
                "mrr": 0.4,
                "ndcg_at_k": 0.45,
                "mean_latency_ms": 12.0,
            }
        ]
    )

    summary = retrieval_summary(frame)
    assert summary["fastest_latency_ms"] == 12.0
    assert frame.loc[0, "Címkézési lefedettség"] == 0.0


def test_rag_view_model_has_consistent_hungarian_names_and_summary() -> None:
    rows = [
        {
            "rag_strategy": "hybrid",
            "citation_accuracy": 1.0,
            "citation_coverage": 0.8,
            "citation_source_coverage": 0.7,
            "key_fact_coverage": 0.75,
            "context_utilization": 0.68,
            "mean_retrieval_latency_ms": 18.0,
            "mean_reranking_latency_ms": 4.0,
            "mean_generation_latency_ms": 2800.0,
            "mean_ttft_ms": 520.0,
            "mean_tokens_per_second": 18.2,
            "mean_total_latency_ms": 2822.0,
            "p50_total_latency_ms": 2700.0,
            "p95_total_latency_ms": 3400.0,
            "p99_total_latency_ms": 3900.0,
            "mean_context_tokens": 1320,
        },
        {
            "rag_strategy": "baseline",
            "citation_accuracy": 0.8,
            "citation_coverage": 0.6,
            "citation_source_coverage": 0.5,
            "key_fact_coverage": 0.60,
            "context_utilization": 0.55,
            "mean_ttft_ms": 410.0,
            "mean_total_latency_ms": 2200.0,
        },
    ]

    frame = build_rag_display_frame(rows)
    summary = rag_summary(frame)

    assert "P95 teljes idő ms" in frame.columns
    assert "Átlagos kontextustoken" in frame.columns
    assert "Összesített pontszám" in frame.columns
    assert summary["best_ttft_ms"] == 410.0
    assert summary["best_strategy"] == "hybrid"
