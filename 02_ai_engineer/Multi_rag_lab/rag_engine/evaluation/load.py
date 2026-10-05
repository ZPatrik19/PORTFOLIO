from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from rag_engine.evaluation.latency import summarize_latencies
from rag_engine.evaluation.statistics import bootstrap_mean_ci, coefficient_of_variation


@dataclass(frozen=True)
class LoadBenchmarkResult:
    concurrency: int
    requests: int
    warmup_requests: int
    total_ms: float
    requests_per_second: float
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    stddev_latency_ms: float
    latency_cv: float
    mean_ci_low_ms: float
    mean_ci_high_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _single_request(retriever: Any, query: str, top_k: int, candidate_count: int) -> float:
    started = time.perf_counter()
    try:
        retriever.retrieve(query, top_k=top_k, candidate_count=max(top_k, candidate_count))
    except TypeError:
        retriever.retrieve(query, top_k=top_k)
    return (time.perf_counter() - started) * 1000.0


def benchmark_retrieval_load(
    retriever: Any,
    queries: list[str],
    *,
    concurrency: int = 1,
    repeats: int = 3,
    warmup_requests: int = 2,
    top_k: int = 5,
    candidate_count: int = 20,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> LoadBenchmarkResult:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not queries:
        raise ValueError("queries must not be empty")

    for index in range(warmup_requests):
        _single_request(retriever, queries[index % len(queries)], top_k, candidate_count)

    workload = queries * repeats
    latencies: list[float] = []
    wall_started = time.perf_counter()
    if concurrency == 1:
        for query in workload:
            latencies.append(_single_request(retriever, query, top_k, candidate_count))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_single_request, retriever, query, top_k, candidate_count) for query in workload]
            for future in as_completed(futures):
                latencies.append(future.result())
    total_ms = (time.perf_counter() - wall_started) * 1000.0

    stats = summarize_latencies(latencies)
    ci = bootstrap_mean_ci(latencies, bootstrap_samples=bootstrap_samples, seed=seed)
    return LoadBenchmarkResult(
        concurrency=concurrency,
        requests=len(workload),
        warmup_requests=warmup_requests,
        total_ms=total_ms,
        requests_per_second=len(workload) / max(total_ms / 1000.0, 1e-9),
        mean_latency_ms=stats.mean_ms,
        p50_latency_ms=stats.p50_ms,
        p90_latency_ms=stats.p90_ms,
        p95_latency_ms=stats.p95_ms,
        p99_latency_ms=stats.p99_ms,
        stddev_latency_ms=stats.stddev_ms,
        latency_cv=coefficient_of_variation(latencies),
        mean_ci_low_ms=ci.low,
        mean_ci_high_ms=ci.high,
    )
