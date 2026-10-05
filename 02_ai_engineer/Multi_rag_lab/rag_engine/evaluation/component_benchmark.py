from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil

from rag_engine.ingestion.chunking.factory import create_chunker
from rag_engine.platform.config import load_settings
from rag_engine.ingestion.cleaning import clean_documents
from rag_engine.ingestion.parser import parse_file
from rag_engine.indexing.embedding_factory import create_embedding_provider
from rag_engine.models import ChunkingConfig
from rag_engine.models import Chunk, Document
from rag_engine.platform.memory import gpu_memory_mb
from rag_engine.indexing.vector_factory import create_vector_store


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def _gpu_peak_if_cuda(device: str) -> float | None:
    if device != "cuda":
        return None
    try:
        return float(gpu_memory_mb().get("peak_mb") or 0.0)
    except Exception:
        return None


def _parse_and_clean(paths: Iterable[Path], max_documents: int) -> tuple[list[Document], dict[str, object]]:
    started = time.perf_counter()
    raw_documents: list[Document] = []
    for path in paths:
        raw_documents.extend(parse_file(path))
        if len(raw_documents) >= max_documents:
            break
    raw_documents = raw_documents[:max_documents]
    documents, stats = clean_documents(raw_documents)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    total_chars = sum(len(document.text) for document in documents)
    row = {
        "component": "parsing-cleaning",
        "variant": "parse+clean",
        "documents": len(documents),
        "total_chars": total_chars,
        "total_ms": elapsed_ms,
        "mean_ms_per_document": elapsed_ms / max(1, len(documents)),
        "documents_per_second": len(documents) / max(elapsed_ms / 1000.0, 1e-9),
        "chars_per_second": total_chars / max(elapsed_ms / 1000.0, 1e-9),
        "ram_mb": _rss_mb(),
        "removed_empty": getattr(stats, "removed_empty", 0),
        "removed_duplicates": getattr(stats, "removed_duplicates", 0),
    }
    return documents, row


def _chunk_metrics(chunks: list[Chunk], elapsed_ms: float, documents: list[Document]) -> dict[str, float | int]:
    lengths = [len(chunk.text) for chunk in chunks]
    unique_ratio = len({chunk.text for chunk in chunks}) / max(1, len(chunks))
    total_chars = sum(lengths)
    return {
        "chunks": len(chunks),
        "chunks_per_document": len(chunks) / max(1, len(documents)),
        "mean_chunk_chars": statistics.fmean(lengths) if lengths else 0.0,
        "median_chunk_chars": statistics.median(lengths) if lengths else 0.0,
        "p95_chunk_chars": _quantile([float(x) for x in lengths], 0.95),
        "unique_chunk_ratio": unique_ratio,
        "total_ms": elapsed_ms,
        "mean_ms_per_document": elapsed_ms / max(1, len(documents)),
        "documents_per_second": len(documents) / max(elapsed_ms / 1000.0, 1e-9),
        "chars_per_second": total_chars / max(elapsed_ms / 1000.0, 1e-9),
    }


def run_component_benchmarks(
    *,
    paths: list[Path],
    chunkings: list[str],
    embedding_modes: list[str],
    embedding_devices: list[str],
    vector_devices: list[str],
    chunk_size: int,
    overlap: int,
    semantic_threshold: float,
    max_documents: int = 20,
    max_embedding_chunks: int = 256,
    vector_query_count: int = 20,
    top_k: int = 5,
) -> dict[str, list[dict[str, object]]]:
    """Micro-benchmark major RAG components on one stable corpus sample.

    This deliberately measures components separately from retrieval quality. It is useful
    for answering *where* latency comes from without mixing chunking, model loading,
    embedding and vector search into one end-to-end number.
    """
    component_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    documents, parse_row = _parse_and_clean(paths, max_documents)
    component_rows.append(parse_row)
    if not documents:
        return {"components": component_rows, "errors": [{"phase": "components", "error": "Nincs feldolgozható dokumentum."}]}

    settings = load_settings()
    spec_map = {
        "multilingual": (settings.multilingual_embedding_model, False),
        "english": (settings.embedding_model, False),
        "e5-small": (settings.e5_embedding_model, False),
        "hashing": ("__offline_hashing_fallback__", True),
    }
    specs = [
        (mode, *spec_map[mode])
        for mode in embedding_modes
        if mode in spec_map
    ]

    # A semantic chunker needs an embedder. Reuse one representative selected model;
    # all non-semantic chunkers are model-independent.
    semantic_embedder = None
    if "semantic" in chunkings and specs:
        try:
            semantic_embedder = create_embedding_provider(
                specs[0][1],
                device=embedding_devices[0] if embedding_devices else "cpu",
                fallback_to_hashing=specs[0][2],
            )
        except Exception as exc:
            errors.append({"phase": "chunking", "variant": "semantic", "error": str(exc)})

    chunk_sets: dict[str, list[Chunk]] = {}
    for strategy in chunkings:
        try:
            if strategy == "semantic" and semantic_embedder is None:
                continue
            config = ChunkingConfig(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                semantic_threshold=semantic_threshold,
            )
            chunker = create_chunker(config, embedder=semantic_embedder if strategy == "semantic" else None)
            started = time.perf_counter()
            chunks = chunker.chunk(documents)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            chunk_sets[strategy] = chunks
            row: dict[str, object] = {
                "component": "chunking",
                "variant": strategy,
                "chunking": strategy,
                "ram_mb": _rss_mb(),
                **_chunk_metrics(chunks, elapsed_ms, documents),
            }
            if strategy == "semantic" and semantic_embedder is not None:
                row["embedding_model"] = str(getattr(semantic_embedder, "model_name", specs[0][1]))
                row["embedding_device"] = str(getattr(semantic_embedder, "device", "cpu"))
            component_rows.append(row)
        except Exception as exc:
            errors.append({"phase": "chunking", "variant": strategy, "error": str(exc)})

    sample_chunks = chunk_sets.get("recursive") or next(iter(chunk_sets.values()), [])
    sample_chunks = [chunk for chunk in sample_chunks if chunk.metadata.get("role") != "parent"][:max_embedding_chunks]
    texts = [chunk.text for chunk in sample_chunks]
    if not texts:
        return {"components": component_rows, "errors": errors + [{"phase": "embedding", "error": "Nincs embeddingelhető chunk."}]}

    embedding_cache: dict[tuple[str, str], tuple[object, np.ndarray]] = {}
    for embedding_label, model_name, fallback_embedding in specs:
        for requested_device in embedding_devices:
            try:
                load_started = time.perf_counter()
                provider = create_embedding_provider(
                    model_name,
                    device=requested_device,
                    fallback_to_hashing=fallback_embedding,
                )
                model_load_ms = (time.perf_counter() - load_started) * 1000.0
                # Warm-up outside the timed batch to reduce first-kernel/model initialization noise.
                provider.embed_documents(texts[: min(4, len(texts))])
                started = time.perf_counter()
                vectors = provider.embed_documents(texts)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                actual_device = str(getattr(provider, "device", "cpu"))
                dimensions = int(vectors.shape[1]) if getattr(vectors, "ndim", 0) == 2 and len(vectors) else 0
                component_rows.append(
                    {
                        "component": "embedding",
                        "variant": f"{embedding_label}|{requested_device}",
                        "embedding_mode": embedding_label,
                        "embedding_model": str(getattr(provider, "model_name", model_name)),
                        "requested_device": requested_device,
                        "actual_device": actual_device,
                        "fallback_used": requested_device != actual_device and requested_device != "auto",
                        "model_load_ms": model_load_ms,
                        "total_ms": elapsed_ms,
                        "texts": len(texts),
                        "vector_dimension": dimensions,
                        "mean_ms_per_text": elapsed_ms / max(1, len(texts)),
                        "texts_per_second": len(texts) / max(elapsed_ms / 1000.0, 1e-9),
                        "chars_per_second": sum(map(len, texts)) / max(elapsed_ms / 1000.0, 1e-9),
                        "ram_mb": _rss_mb(),
                        "peak_gpu_memory_mb": _gpu_peak_if_cuda(actual_device),
                    }
                )
                embedding_cache[(embedding_label, requested_device)] = (provider, vectors)
            except Exception as exc:
                errors.append({"phase": "embedding", "variant": f"{embedding_label}|{requested_device}", "error": str(exc)})

    for (embedding_mode, embedding_device), (provider, vectors) in embedding_cache.items():
        if not len(vectors):
            continue
        for requested_vector_device in vector_devices:
            try:
                store = create_vector_store(device=requested_vector_device, fallback_to_numpy=True)
                build_started = time.perf_counter()
                store.add(vectors, sample_chunks[: len(vectors)])
                build_ms = (time.perf_counter() - build_started) * 1000.0
                latencies: list[float] = []
                query_count = min(vector_query_count, len(vectors))
                for query_vector in vectors[:query_count]:
                    started = time.perf_counter()
                    store.search(query_vector, top_k=top_k)
                    latencies.append((time.perf_counter() - started) * 1000.0)
                mean_ms = statistics.fmean(latencies) if latencies else 0.0
                actual_device = str(getattr(store, "device", "cpu"))
                component_rows.append(
                    {
                        "component": "vector-search",
                        "variant": f"{embedding_mode}|{embedding_device}|{requested_vector_device}",
                        "embedding_mode": embedding_mode,
                        "embedding_device": str(getattr(provider, "device", embedding_device)),
                        "requested_vector_device": requested_vector_device,
                        "actual_device": actual_device,
                        "fallback_used": requested_vector_device != actual_device,
                        "vector_backend": str(getattr(store, "backend_name", "unknown")),
                        "vectors": len(vectors),
                        "vector_dimension": int(vectors.shape[1]),
                        "index_build_ms": build_ms,
                        "mean_search_ms": mean_ms,
                        "p50_search_ms": _quantile(latencies, 0.50),
                        "p95_search_ms": _quantile(latencies, 0.95),
                        "p99_search_ms": _quantile(latencies, 0.99),
                        "queries_per_second": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
                        "queries": query_count,
                        "top_k": top_k,
                        "ram_mb": _rss_mb(),
                        "peak_gpu_memory_mb": _gpu_peak_if_cuda(actual_device),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "phase": "vector-search",
                        "variant": f"{embedding_mode}|{embedding_device}|{requested_vector_device}",
                        "error": str(exc),
                    }
                )

    return {"components": component_rows, "errors": errors}
