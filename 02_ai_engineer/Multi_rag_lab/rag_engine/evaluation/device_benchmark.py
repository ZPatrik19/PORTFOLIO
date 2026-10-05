from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Callable

import numpy as np
import psutil

from rag_engine.evaluation.latency import summarize_latencies
from rag_engine.indexing.embedding_base import EmbeddingProvider
from rag_engine.platform.hardware import get_hardware_profile
from rag_engine.platform.memory import gpu_memory_mb


@dataclass
class BenchmarkResult:
    component: str
    device: str
    total_ms: float
    throughput_per_sec: float
    mean_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    ram_mb: float | None = None
    peak_gpu_memory_mb: float | None = None
    workload_size: int = 0
    cold_start_ms: float | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    output_tokens: float | None = None
    context_tokens: float | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_embedding(
    provider_factory: Callable[[str], EmbeddingProvider], texts: list[str], device: str, *, warmup: bool = True
) -> BenchmarkResult:
    cold_start = time.perf_counter()
    provider = provider_factory(device)
    cold_ms = (time.perf_counter() - cold_start) * 1000
    if warmup and texts:
        provider.embed_documents(texts[: min(4, len(texts))])
    start = time.perf_counter()
    provider.embed_documents(texts)
    total_ms = (time.perf_counter() - start) * 1000
    actual_device = str(getattr(provider, "device", device))
    return BenchmarkResult(
        component="embedding",
        device=actual_device,
        total_ms=total_ms,
        throughput_per_sec=len(texts) / max(total_ms / 1000, 1e-9),
        mean_ms=total_ms / max(1, len(texts)),
        median_ms=total_ms / max(1, len(texts)),
        p95_ms=total_ms / max(1, len(texts)),
        ram_mb=psutil.Process().memory_info().rss / 1024**2,
        peak_gpu_memory_mb=gpu_memory_mb()["peak_mb"] if actual_device == "cuda" else None,
        workload_size=len(texts),
        cold_start_ms=cold_ms,
        timestamp=datetime.now(UTC).isoformat(),
    )


def benchmark_retrieval(retriever, queries: list[str], device: str = "cpu", top_k: int = 5) -> BenchmarkResult:
    latencies = []
    for query in queries:
        start = time.perf_counter()
        try:
            retriever.retrieve(query, top_k=top_k, candidate_count=max(top_k, 20))
        except TypeError:
            retriever.retrieve(query, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
    stats = summarize_latencies(latencies)
    total = sum(latencies)
    return BenchmarkResult(
        component="retrieval",
        device=device,
        total_ms=total,
        throughput_per_sec=len(queries) / max(total / 1000, 1e-9),
        mean_ms=stats.mean_ms,
        median_ms=stats.median_ms,
        p95_ms=stats.p95_ms,
        ram_mb=psutil.Process().memory_info().rss / 1024**2,
        peak_gpu_memory_mb=gpu_memory_mb()["peak_mb"] if device == "cuda" else None,
        workload_size=len(queries),
        timestamp=datetime.now(UTC).isoformat(),
    )


def benchmark_metadata(device: str) -> dict[str, object]:
    profile = get_hardware_profile(device)
    return {
        "hardware": profile.model_dump(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def benchmark_reranking(reranker, query: str, chunks, *, repeats: int = 5, top_k: int = 5) -> BenchmarkResult:
    if chunks:
        reranker.rerank(query, chunks, top_k=top_k)  # warm-up outside measured runs
    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        reranker.rerank(query, chunks, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
    stats = summarize_latencies(latencies)
    total = sum(latencies)
    device = str(getattr(reranker, "device", "cpu"))
    return BenchmarkResult(
        component="reranking",
        device=device,
        total_ms=total,
        throughput_per_sec=(len(chunks) * repeats) / max(total / 1000, 1e-9),
        mean_ms=stats.mean_ms,
        median_ms=stats.median_ms,
        p95_ms=stats.p95_ms,
        ram_mb=psutil.Process().memory_info().rss / 1024**2,
        peak_gpu_memory_mb=gpu_memory_mb()["peak_mb"] if device == "cuda" else None,
        workload_size=len(chunks) * repeats,
        timestamp=datetime.now(UTC).isoformat(),
    )


def benchmark_pipeline(pipeline, queries: list[str]) -> BenchmarkResult:
    latencies: list[float] = []
    ttfts: list[float] = []
    token_speeds: list[float] = []
    output_tokens: list[int] = []
    context_tokens: list[int] = []
    for query in queries:
        start = time.perf_counter()
        result = pipeline.answer(query)
        latencies.append((time.perf_counter() - start) * 1000)
        if getattr(result, "generation_ttft_ms", None) is not None:
            ttfts.append(float(result.generation_ttft_ms or 0.0))
        if getattr(result, "tokens_per_second", None):
            token_speeds.append(float(result.tokens_per_second or 0.0))
        output_tokens.append(int(getattr(result, "output_tokens", 0) or 0))
        context_tokens.append(int(getattr(result, "context_tokens", 0) or 0))
    stats = summarize_latencies(latencies)
    total = sum(latencies)
    device = str(getattr(pipeline, "execution_device", "cpu"))
    return BenchmarkResult(
        component="end-to-end-rag",
        device=device,
        total_ms=total,
        throughput_per_sec=len(queries) / max(total / 1000, 1e-9),
        mean_ms=stats.mean_ms,
        median_ms=stats.median_ms,
        p95_ms=stats.p95_ms,
        ram_mb=psutil.Process().memory_info().rss / 1024**2,
        peak_gpu_memory_mb=gpu_memory_mb()["peak_mb"] if device == "cuda" else None,
        workload_size=len(queries),
        ttft_ms=float(np.mean(ttfts)) if ttfts else None,
        tokens_per_second=float(np.mean(token_speeds)) if token_speeds else None,
        output_tokens=float(np.mean(output_tokens)) if output_tokens else None,
        context_tokens=float(np.mean(context_tokens)) if context_tokens else None,
        timestamp=datetime.now(UTC).isoformat(),
    )


def benchmark_llm(llm, prompts: list[str], *, device: str = "runtime") -> BenchmarkResult:
    """Benchmark the active LLM provider without changing external runtime state."""
    if not prompts:
        return BenchmarkResult(
            component="llm-generation",
            device=device,
            total_ms=0.0,
            throughput_per_sec=0.0,
            workload_size=0,
            timestamp=datetime.now(UTC).isoformat(),
        )
    latencies: list[float] = []
    ttfts: list[float] = []
    speeds: list[float] = []
    output_counts: list[int] = []
    for prompt in prompts:
        start = time.perf_counter()
        llm.generate(prompt)
        latencies.append((time.perf_counter() - start) * 1000)
        metrics = dict(getattr(llm, "last_metrics", {}) or {})
        if metrics.get("ttft_ms") is not None:
            ttfts.append(float(metrics.get("ttft_ms") or 0.0))
        if float(metrics.get("tokens_per_second", 0.0) or 0.0) > 0:
            speeds.append(float(metrics.get("tokens_per_second") or 0.0))
        output_counts.append(int(metrics.get("output_tokens", 0) or 0))
    stats = summarize_latencies(latencies)
    total = sum(latencies)
    output_total = sum(output_counts)
    throughput = output_total / max(total / 1000, 1e-9) if output_total else len(prompts) / max(total / 1000, 1e-9)
    return BenchmarkResult(
        component="llm-generation",
        device=device,
        total_ms=total,
        throughput_per_sec=throughput,
        mean_ms=stats.mean_ms,
        median_ms=stats.median_ms,
        p95_ms=stats.p95_ms,
        ram_mb=psutil.Process().memory_info().rss / 1024**2,
        workload_size=len(prompts),
        ttft_ms=float(np.mean(ttfts)) if ttfts else None,
        tokens_per_second=float(np.mean(speeds)) if speeds else None,
        output_tokens=float(np.mean(output_counts)) if output_counts else None,
        timestamp=datetime.now(UTC).isoformat(),
    )
