from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_engine.evaluation.diagnostics import Severity, diagnose_generation, diagnose_retrieval
from rag_engine.evaluation.load import benchmark_retrieval_load
from rag_engine.evaluation.retrieval import context_precision_at_k, r_precision, reciprocal_rank_at_k
from rag_engine.evaluation.robustness import evaluate_retriever_robustness, generate_query_variants
from rag_engine.evaluation.statistics import bootstrap_mean_ci, paired_bootstrap_comparison


@dataclass
class _Result:
    chunk_id: str


class _StableRetriever:
    def retrieve(self, query: str, top_k: int = 5, candidate_count: int | None = None):
        _ = query, candidate_count
        return [_Result(f"c{i}") for i in range(top_k)]


def test_bootstrap_ci_is_deterministic_and_contains_mean():
    ci = bootstrap_mean_ci([1, 2, 3, 4, 5], bootstrap_samples=500, seed=7)
    ci_again = bootstrap_mean_ci([1, 2, 3, 4, 5], bootstrap_samples=500, seed=7)
    assert ci == ci_again
    assert ci.low <= ci.mean <= ci.high


def test_paired_bootstrap_detects_lower_latency_as_better():
    result = paired_bootstrap_comparison([10, 11, 12, 13], [5, 6, 7, 8], higher_is_better=False, seed=7)
    assert result.mean_delta_b_minus_a < 0
    assert result.probability_b_better > 0.95
    assert result.ci_high < 0


def test_query_variants_and_stable_retriever_have_perfect_robustness():
    variants = generate_query_variants("Mikor szükséges orvosi kivizsgálás?")
    assert {item.name for item in variants} >= {"original", "lowercase", "accentless"}
    summary = evaluate_retriever_robustness(_StableRetriever(), variants[0].query, top_k=3)
    assert summary.mean_jaccard_at_k == pytest.approx(1.0)
    assert summary.first_result_retention_rate == pytest.approx(1.0)


def test_load_benchmark_reports_percentiles_ci_and_throughput():
    result = benchmark_retrieval_load(
        _StableRetriever(),
        ["q1", "q2"],
        concurrency=1,
        repeats=2,
        warmup_requests=1,
        top_k=3,
        bootstrap_samples=100,
    )
    assert result.requests == 4
    assert result.requests_per_second > 0
    assert result.p50_latency_ms <= result.p95_latency_ms <= result.p99_latency_ms
    assert result.mean_ci_low_ms <= result.mean_latency_ms <= result.mean_ci_high_ms


def test_failure_diagnostics_explain_retrieval_and_generation_failures():
    retrieval = diagnose_retrieval(
        retrieved_ids=["x", "x", "y"],
        relevant_ids={"r"},
        source_ids=["s1", "s1", "s1"],
        latency_ms=120,
        latency_budget_ms=50,
        top_k=3,
    )
    codes = {item.code for item in retrieval}
    assert "retrieval.no_hit" in codes
    assert "retrieval.duplicates" in codes
    assert any(item.severity == Severity.ERROR for item in retrieval)

    generation = diagnose_generation(
        citation_accuracy=0.5,
        citation_coverage=0.2,
        key_fact_coverage=0.4,
        context_utilization=0.1,
        generation_latency_ms=1000,
        latency_budget_ms=500,
    )
    assert {item.code for item in generation} >= {
        "generation.invalid_citation",
        "generation.citation_gap",
        "generation.key_fact_gap",
        "generation.low_context_use",
        "generation.latency_budget",
    }


def test_extended_ir_metrics_are_rank_sensitive():
    retrieved = ["x", "r1", "r2", "y"]
    relevant = {"r1", "r2"}
    assert reciprocal_rank_at_k(retrieved, relevant, 3) == pytest.approx(0.5)
    assert r_precision(retrieved, relevant) == pytest.approx(0.5)
    assert 0.0 < context_precision_at_k(retrieved, relevant, 3) < 1.0
