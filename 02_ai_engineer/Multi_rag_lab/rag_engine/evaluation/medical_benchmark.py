from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rag_engine.platform.config import load_settings
from rag_engine.evaluation.generation import (
    answer_redundancy_proxy,
    answer_token_estimate,
    citation_accuracy,
    citation_coverage,
    citation_density,
    citation_source_coverage,
    context_utilization,
)
from rag_engine.evaluation.medical_dataset import (
    MedicalEvaluationItem,
    key_fact_coverage,
    resolve_relevant_chunk_ids,
)
from rag_engine.evaluation.runner import evaluate_retrieval
from rag_engine.evaluation.statistics import bootstrap_mean_ci, coefficient_of_variation
from rag_engine.platform.profiles import load_llm_profiles
from rag_engine.models import ChunkingConfig
from rag_engine.retrieval.rerank_cross_encoder import CrossEncoderReranker
from rag_engine.retrieval.rerank_lexical import LexicalReranker
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.service import LabBundle, build_lab, create_rag_pipeline


@dataclass(frozen=True)
class RetrievalBenchmarkRow:
    chunking: str
    retriever: str
    reranker: str
    questions: int
    recall_at_k: float
    precision_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    mrr: float
    map_at_k: float
    ndcg_at_k: float
    mean_first_relevant_rank: float
    reciprocal_rank_at_k: float
    r_precision: float
    context_precision_at_k: float
    recall_ci_low: float
    recall_ci_high: float
    ndcg_ci_low: float
    ndcg_ci_high: float
    no_hit_rate: float
    late_hit_rate: float
    source_diversity_at_k: float
    duplicate_ratio_at_k: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    latency_cv: float
    mean_latency_ci_low_ms: float
    mean_latency_ci_high_ms: float
    queries_per_second: float
    mean_relevant_chunks: float
    labeling_coverage: float
    embedding_device: str
    vector_backend: str
    vector_device: str


@dataclass(frozen=True)
class RAGBenchmarkRow:
    rag_strategy: str
    questions: int
    citation_accuracy: float
    citation_coverage: float
    citation_source_coverage: float
    citation_density_per_100_words: float
    key_fact_coverage: float
    key_fact_ci_low: float
    key_fact_ci_high: float
    context_utilization: float
    answer_redundancy: float
    fallback_rate: float
    repair_rate: float
    mean_answer_tokens: float
    mean_retrieval_latency_ms: float
    mean_reranking_latency_ms: float
    mean_generation_latency_ms: float
    mean_ttft_ms: float
    mean_tokens_per_second: float
    mean_total_latency_ms: float
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    p99_total_latency_ms: float
    latency_cv: float
    mean_total_latency_ci_low_ms: float
    mean_total_latency_ci_high_ms: float
    mean_context_tokens: float
    llm_provider: str
    execution_device: str


def corpus_paths_from_manifest(manifest_path: Path, raw_dir: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for record in manifest.get("documents", []):
        if isinstance(record, dict) and record.get("exists", True):
            path = raw_dir / str(record.get("filename", ""))
            if path.exists():
                paths.append(path)
    return paths


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _source_key(result) -> str:
    return str(
        result.metadata.get("source_id") or result.metadata.get("title") or result.source or result.document_id
        if hasattr(result, "document_id")
        else ""
    )


def _result_diversity(results: list) -> float:
    if not results:
        return 0.0
    sources = [
        str(item.metadata.get("source_id") or item.metadata.get("title") or item.source or "") for item in results
    ]
    return len({source for source in sources if source}) / len(results)


def _duplicate_ratio(results: list) -> float:
    if not results:
        return 0.0
    ids = [item.chunk_id for item in results]
    return 1.0 - len(set(ids)) / len(ids)


def _retrieve(
    bundle,
    mode: str,
    query: str,
    top_k: int,
    candidate_count: int,
    reranker,
    *,
    rrf_k: int,
    dense_weight: float,
) -> list:
    requested_k = candidate_count if reranker else top_k
    if mode == "dense":
        results = bundle.dense.retrieve(query, requested_k)
    elif mode == "bm25":
        results = bundle.sparse.retrieve(query, requested_k)
    elif mode in {"hybrid", "hybrid-rrf"}:
        retriever = HybridRetriever(bundle.dense, bundle.sparse, fusion="rrf", rrf_k=rrf_k, dense_weight=dense_weight)
        results = retriever.retrieve(query, top_k=requested_k, candidate_count=candidate_count)
    elif mode == "hybrid-weighted":
        retriever = HybridRetriever(
            bundle.dense, bundle.sparse, fusion="weighted", rrf_k=rrf_k, dense_weight=dense_weight
        )
        results = retriever.retrieve(query, top_k=requested_k, candidate_count=candidate_count)
    else:
        raise ValueError(f"Ismeretlen retrieval mód: {mode}")
    if reranker is not None:
        return reranker.rerank(query, results, top_k=top_k)
    return results[:top_k]


def _create_reranker(kind: str, device: str):
    if kind == "none":
        return None
    if kind == "lexical":
        return LexicalReranker()
    if kind == "cross-encoder":
        settings = load_settings()
        return CrossEncoderReranker(settings.reranker_model, device=device)
    raise ValueError(f"Ismeretlen reranker: {kind}")


def run_retrieval_benchmark(
    *,
    paths: list[Path],
    items: list[MedicalEvaluationItem],
    chunking_strategies: Iterable[str],
    retrieval_modes: Iterable[str],
    rerankers: Iterable[str] = ("none",),
    chunk_size: int = 700,
    overlap: int = 100,
    semantic_threshold: float = 0.72,
    embedding_model: str | None = None,
    embedding_device: str = "auto",
    vector_device: str = "cpu",
    top_k: int = 5,
    candidate_count: int = 20,
    fusion: str = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    fallback_embedding: bool = False,
    reranker_device: str = "auto",
) -> list[RetrievalBenchmarkRow]:
    settings = load_settings()
    model = embedding_model or settings.multilingual_embedding_model
    rows: list[RetrievalBenchmarkRow] = []
    reranker_cache: dict[str, object | None] = {}

    for chunking_strategy in chunking_strategies:
        bundle = build_lab(
            paths,
            chunking=ChunkingConfig(
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                semantic_threshold=semantic_threshold,
            ),
            embedding_model=model,
            embedding_device=embedding_device,
            vector_device=vector_device,
            llm_provider="dummy",
            fallback_embedding=fallback_embedding,
            fusion=fusion,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
        )
        index_chunks = [chunk for chunk in bundle.ingestion.chunks if chunk.metadata.get("role") != "parent"]
        relevant_map = {item.id: resolve_relevant_chunk_ids(item, index_chunks) for item in items}
        coverage = sum(bool(relevant_map[item.id]) for item in items) / max(1, len(items))

        for retrieval_mode in retrieval_modes:
            for reranker_kind in rerankers:
                if reranker_kind not in reranker_cache:
                    reranker_cache[reranker_kind] = _create_reranker(reranker_kind, reranker_device)
                reranker = reranker_cache[reranker_kind]
                evals = []
                latencies: list[float] = []
                relevant_counts: list[int] = []
                diversities: list[float] = []
                duplicates: list[float] = []
                for item in items:
                    relevant = relevant_map[item.id]
                    relevant_counts.append(len(relevant))
                    started = time.perf_counter()
                    results = _retrieve(
                        bundle,
                        retrieval_mode,
                        item.query,
                        top_k,
                        candidate_count,
                        reranker,
                        rrf_k=rrf_k,
                        dense_weight=dense_weight,
                    )
                    latencies.append((time.perf_counter() - started) * 1000)
                    diversities.append(_result_diversity(results))
                    duplicates.append(_duplicate_ratio(results))
                    evals.append(evaluate_retrieval([result.chunk_id for result in results], relevant, top_k))

                n = max(1, len(evals))
                mean_latency = statistics.fmean(latencies) if latencies else 0.0
                recall_values = [e.recall_at_k for e in evals]
                ndcg_values = [e.ndcg_at_k for e in evals]
                recall_ci = bootstrap_mean_ci(recall_values, seed=42)
                ndcg_ci = bootstrap_mean_ci(ndcg_values, seed=43)
                latency_ci = bootstrap_mean_ci(latencies, seed=44)
                no_hit_rate = sum(1 for e in evals if e.hit_rate_at_k <= 0.0) / n
                late_hit_rate = sum(1 for e in evals if e.first_relevant_rank > min(3, top_k)) / n
                rows.append(
                    RetrievalBenchmarkRow(
                        chunking=chunking_strategy,
                        retriever=retrieval_mode,
                        reranker=reranker_kind,
                        questions=len(items),
                        recall_at_k=sum(e.recall_at_k for e in evals) / n,
                        precision_at_k=sum(e.precision_at_k for e in evals) / n,
                        f1_at_k=sum(e.f1_at_k for e in evals) / n,
                        hit_rate_at_k=sum(e.hit_rate_at_k for e in evals) / n,
                        mrr=sum(e.mrr for e in evals) / n,
                        map_at_k=sum(e.map_at_k for e in evals) / n,
                        ndcg_at_k=sum(e.ndcg_at_k for e in evals) / n,
                        mean_first_relevant_rank=sum(e.first_relevant_rank for e in evals) / n,
                        reciprocal_rank_at_k=sum(e.reciprocal_rank_at_k for e in evals) / n,
                        r_precision=sum(e.r_precision for e in evals) / n,
                        context_precision_at_k=sum(e.context_precision_at_k for e in evals) / n,
                        recall_ci_low=recall_ci.low,
                        recall_ci_high=recall_ci.high,
                        ndcg_ci_low=ndcg_ci.low,
                        ndcg_ci_high=ndcg_ci.high,
                        no_hit_rate=no_hit_rate,
                        late_hit_rate=late_hit_rate,
                        source_diversity_at_k=statistics.fmean(diversities) if diversities else 0.0,
                        duplicate_ratio_at_k=statistics.fmean(duplicates) if duplicates else 0.0,
                        mean_latency_ms=mean_latency,
                        p50_latency_ms=_quantile(latencies, 0.50),
                        p95_latency_ms=_quantile(latencies, 0.95),
                        p99_latency_ms=_quantile(latencies, 0.99),
                        latency_cv=coefficient_of_variation(latencies),
                        mean_latency_ci_low_ms=latency_ci.low,
                        mean_latency_ci_high_ms=latency_ci.high,
                        queries_per_second=1000.0 / mean_latency if mean_latency > 0 else 0.0,
                        mean_relevant_chunks=statistics.fmean(relevant_counts) if relevant_counts else 0.0,
                        labeling_coverage=coverage,
                        embedding_device=str(getattr(bundle.embedder, "device", embedding_device)),
                        vector_backend=str(getattr(bundle.vector_store, "backend_name", "unknown")),
                        vector_device=str(getattr(bundle.vector_store, "device", vector_device)),
                    )
                )
    return rows


def run_rag_benchmark(
    *,
    paths: list[Path],
    items: list[MedicalEvaluationItem],
    rag_strategies: Iterable[str],
    chunking_strategy: str = "recursive",
    chunk_size: int = 700,
    overlap: int = 100,
    semantic_threshold: float = 0.72,
    embedding_model: str | None = None,
    embedding_device: str = "auto",
    vector_device: str = "cpu",
    llm_provider: str = "dummy",
    ollama_model: str | None = None,
    ollama_profile: str = "balanced",
    top_k: int = 5,
    candidate_count: int = 20,
    max_context_tokens: int = 1800,
    fallback_embedding: bool = False,
    context_profile: str = "balanced",
    prompt_profile: str = "professional",
    reranker=None,
) -> list[RAGBenchmarkRow]:
    settings = load_settings()
    model = embedding_model or settings.multilingual_embedding_model
    llm_profiles = load_llm_profiles()
    profile = llm_profiles.get(ollama_profile, llm_profiles.get("balanced", {}))
    rows: list[RAGBenchmarkRow] = []
    # A single matrix block can evaluate several RAG strategies over the exact same
    # ingestion/embedding/vector resources. Reusing the bundle avoids repeatedly
    # parsing, chunking and embedding the whole corpus for every strategy.
    bundle_cache: dict[str, LabBundle] = {}

    for strategy in rag_strategies:
        effective_chunking = "parent-child" if strategy == "parent-document" else chunking_strategy
        bundle = bundle_cache.get(effective_chunking)
        if bundle is None:
            bundle = build_lab(
                paths,
                chunking=ChunkingConfig(
                    strategy=effective_chunking,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    semantic_threshold=semantic_threshold,
                ),
                embedding_model=model,
                embedding_device=embedding_device,
                vector_device=vector_device,
                llm_provider=llm_provider,
                ollama_base_url=settings.ollama_base_url,
                ollama_model=ollama_model or str(profile.get("alias", settings.ollama_model)),
                fallback_embedding=fallback_embedding,
                ollama_connect_timeout=settings.ollama_connect_timeout_seconds,
                ollama_read_timeout=settings.ollama_read_timeout_seconds,
                ollama_max_retries=settings.ollama_max_retries,
                ollama_num_predict=int(profile.get("num_predict", settings.ollama_num_predict)),
                ollama_temperature=float(profile.get("temperature", 0.15)),
                ollama_keep_alive=str(profile.get("keep_alive", "5m")),
            )
            bundle_cache[effective_chunking] = bundle
        pipeline = create_rag_pipeline(
            bundle,
            strategy,
            max_context_tokens=max_context_tokens,
            execution_device=str(getattr(bundle.embedder, "device", embedding_device)),
            reranker=reranker,
            top_k=top_k,
            candidate_count=candidate_count,
            context_profile=context_profile,
            prompt_profile=prompt_profile,
        )
        citations: list[float] = []
        citation_coverages: list[float] = []
        source_coverages: list[float] = []
        citation_densities: list[float] = []
        fact_coverages: list[float] = []
        context_scores: list[float] = []
        redundancies: list[float] = []
        generation_modes: list[str] = []
        answer_tokens: list[int] = []
        retrieval_latencies: list[float] = []
        rerank_latencies: list[float] = []
        generation_latencies: list[float] = []
        ttfts: list[float] = []
        token_speeds: list[float] = []
        latencies: list[float] = []
        context_tokens: list[int] = []

        for item in items:
            result = pipeline.answer(item.query)
            context_text = "\n".join(chunk.text for chunk in result.retrieved_chunks)
            source_count = len(result.retrieved_chunks)
            citations.append(citation_accuracy(result.answer, source_count))
            citation_coverages.append(citation_coverage(result.answer))
            source_coverages.append(citation_source_coverage(result.answer, source_count))
            citation_densities.append(citation_density(result.answer))
            fact_coverages.append(key_fact_coverage(result.answer, item.expected_key_facts))
            context_scores.append(context_utilization(result.answer, context_text))
            redundancies.append(answer_redundancy_proxy(result.answer))
            generation_modes.append(str(result.trace.get("generation_mode", "standard")))
            answer_tokens.append(answer_token_estimate(result.answer))
            retrieval_latencies.append(result.retrieval_latency_ms)
            rerank_latencies.append(float(result.reranking_latency_ms or 0.0))
            generation_latencies.append(result.generation_latency_ms)
            if result.generation_ttft_ms is not None:
                ttfts.append(float(result.generation_ttft_ms))
            if result.tokens_per_second is not None and result.tokens_per_second > 0:
                token_speeds.append(float(result.tokens_per_second))
            latencies.append(result.total_latency_ms)
            context_tokens.append(result.context_tokens)

        n = max(1, len(items))
        key_fact_ci = bootstrap_mean_ci(fact_coverages, seed=45)
        total_latency_ci = bootstrap_mean_ci(latencies, seed=46)
        fallback_rate = sum(mode == "source-synthesis-fallback" for mode in generation_modes) / n
        repair_rate = sum(mode == "grounded-repair" for mode in generation_modes) / n
        rows.append(
            RAGBenchmarkRow(
                rag_strategy=strategy,
                questions=len(items),
                citation_accuracy=sum(citations) / n,
                citation_coverage=sum(citation_coverages) / n,
                citation_source_coverage=sum(source_coverages) / n,
                citation_density_per_100_words=sum(citation_densities) / n,
                key_fact_coverage=sum(fact_coverages) / n,
                key_fact_ci_low=key_fact_ci.low,
                key_fact_ci_high=key_fact_ci.high,
                context_utilization=sum(context_scores) / n,
                answer_redundancy=sum(redundancies) / n,
                fallback_rate=fallback_rate,
                repair_rate=repair_rate,
                mean_answer_tokens=statistics.fmean(answer_tokens) if answer_tokens else 0.0,
                mean_retrieval_latency_ms=statistics.fmean(retrieval_latencies) if retrieval_latencies else 0.0,
                mean_reranking_latency_ms=statistics.fmean(rerank_latencies) if rerank_latencies else 0.0,
                mean_generation_latency_ms=statistics.fmean(generation_latencies) if generation_latencies else 0.0,
                mean_ttft_ms=statistics.fmean(ttfts) if ttfts else 0.0,
                mean_tokens_per_second=statistics.fmean(token_speeds) if token_speeds else 0.0,
                mean_total_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
                p50_total_latency_ms=_quantile(latencies, 0.50),
                p95_total_latency_ms=_quantile(latencies, 0.95),
                p99_total_latency_ms=_quantile(latencies, 0.99),
                latency_cv=coefficient_of_variation(latencies),
                mean_total_latency_ci_low_ms=total_latency_ci.low,
                mean_total_latency_ci_high_ms=total_latency_ci.high,
                mean_context_tokens=statistics.fmean(context_tokens) if context_tokens else 0.0,
                llm_provider=str(getattr(bundle.llm, "name", llm_provider)),
                execution_device=str(getattr(bundle.embedder, "device", embedding_device)),
            )
        )
    return rows


def rows_to_dicts(rows: Iterable[RetrievalBenchmarkRow | RAGBenchmarkRow]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]
