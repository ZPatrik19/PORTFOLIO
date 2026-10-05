from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from rag_engine.retrieval.context import ContextBuilder
from rag_engine.indexing.embedding_factory import create_embedding_provider
from rag_engine.generation.grounded import GroundedGenerator
from rag_engine.generation.provider_factory import create_llm_provider
from rag_engine.models import ChunkingConfig
from rag_engine.retrieval.query_transform import FollowUpQueryGenerator
from rag_engine.retrieval.query_transform import HyDEGenerator
from rag_engine.retrieval.query_transform import MultiQueryGenerator
from rag_engine.retrieval.query_transform import QueryRewriter
from rag_engine.retrieval.strategies import BaselineRAG
from rag_engine.retrieval.strategies import CompressionRAG
from rag_engine.retrieval.strategies import CorrectiveRAG
from rag_engine.retrieval.strategies import HybridRAG
from rag_engine.retrieval.strategies import HyDERAG
from rag_engine.retrieval.strategies import LexicalRAG
from rag_engine.retrieval.strategies import MultiHopRAG
from rag_engine.retrieval.strategies import MultiQueryRAG
from rag_engine.retrieval.strategies import ParentDocumentRAG
from rag_engine.retrieval.strategies import QueryRewriteRAG
from rag_engine.retrieval.strategies import RerankedRAG
from rag_engine.retrieval.rerank_lexical import LexicalReranker
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.sparse import BM25Retriever
from rag_engine.ingestion.pipeline import IngestionResult, ingest_paths
from rag_engine.indexing.vector_factory import create_vector_store
from rag_engine.indexing.prebuilt import load_prebuilt_medical


@dataclass
class RetrievalResources:
    ingestion: IngestionResult
    embedder: object
    vector_store: object
    dense: DenseRetriever
    sparse: BM25Retriever
    build_trace: dict[str, float | str | bool]


@dataclass
class LabBundle:
    ingestion: IngestionResult
    embedder: object
    vector_store: object
    dense: DenseRetriever
    sparse: BM25Retriever
    hybrid: HybridRetriever
    llm: object
    generator: GroundedGenerator
    build_trace: dict[str, float | str | bool]


def build_retrieval_resources(
    paths: list[Path],
    *,
    chunking: ChunkingConfig,
    embedding_model: str,
    embedding_device: str = "auto",
    vector_device: str = "cpu",
    fallback_embedding: bool = False,
    prefer_prebuilt: bool = True,
) -> RetrievalResources:
    started = time.perf_counter()
    t = time.perf_counter()
    embedder = create_embedding_provider(
        embedding_model,
        device=embedding_device,
        fallback_to_hashing=fallback_embedding,
    )
    embedder_load_ms = (time.perf_counter() - t) * 1000.0

    ingestion = None
    store = None
    prebuilt_used = False
    prebuilt_load_ms = 0.0
    if prefer_prebuilt and chunking.strategy != "semantic":
        t = time.perf_counter()
        prebuilt = load_prebuilt_medical(
            paths,
            chunking,
            embedding_model,
            vector_device=vector_device,
        )
        prebuilt_load_ms = (time.perf_counter() - t) * 1000.0
        if prebuilt is not None:
            ingestion, store = prebuilt
            prebuilt_used = True

    ingest_ms = 0.0
    document_embedding_ms = 0.0
    index_build_ms = 0.0
    if ingestion is None or store is None:
        t = time.perf_counter()
        ingestion = ingest_paths(
            paths,
            chunking,
            embedder=embedder if chunking.strategy == "semantic" else None,
        )
        ingest_ms = (time.perf_counter() - t) * 1000.0
        index_chunks = [chunk for chunk in ingestion.chunks if chunk.metadata.get("role") != "parent"]
        t = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in index_chunks])
        document_embedding_ms = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        store = create_vector_store(device=vector_device, fallback_to_numpy=True)
        store.add(vectors, index_chunks)
        index_build_ms = (time.perf_counter() - t) * 1000.0
    else:
        index_chunks = [chunk for chunk in ingestion.chunks if chunk.metadata.get("role") != "parent"]

    t = time.perf_counter()
    dense = DenseRetriever(embedder, store)
    sparse = BM25Retriever(index_chunks)
    retriever_setup_ms = (time.perf_counter() - t) * 1000.0
    trace: dict[str, float | str | bool] = {
        "prebuilt_index_used": prebuilt_used,
        "embedder_load_ms": embedder_load_ms,
        "prebuilt_load_ms": prebuilt_load_ms,
        "ingestion_ms": ingest_ms,
        "document_embedding_ms": document_embedding_ms,
        "index_build_ms": index_build_ms,
        "retriever_setup_ms": retriever_setup_ms,
        "resource_build_total_ms": (time.perf_counter() - started) * 1000.0,
        "chunks": len(index_chunks),
        "vector_backend": str(getattr(store, "backend_name", "unknown")),
        "vector_device": str(getattr(store, "device", "unknown")),
        "embedding_device": str(getattr(embedder, "device", embedding_device)),
    }
    return RetrievalResources(ingestion, embedder, store, dense, sparse, trace)


def compose_lab(
    resources: RetrievalResources,
    *,
    llm_provider: str = "dummy",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "qwen3:4b",
    fusion: str = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    ollama_connect_timeout: float = 10.0,
    ollama_read_timeout: float = 600.0,
    ollama_max_retries: int = 1,
    ollama_num_predict: int = 384,
    ollama_temperature: float = 0.15,
    ollama_keep_alive: str = "5m",
) -> LabBundle:
    hybrid = HybridRetriever(
        resources.dense,
        resources.sparse,
        fusion=fusion,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
    )
    llm = create_llm_provider(
        llm_provider,
        base_url=ollama_base_url,
        model_name=ollama_model,
        connect_timeout=ollama_connect_timeout,
        read_timeout=ollama_read_timeout,
        max_retries=ollama_max_retries,
        num_predict=ollama_num_predict,
        temperature=ollama_temperature,
        keep_alive=ollama_keep_alive,
    )
    generator = GroundedGenerator(llm)
    return LabBundle(
        resources.ingestion,
        resources.embedder,
        resources.vector_store,
        resources.dense,
        resources.sparse,
        hybrid,
        llm,
        generator,
        dict(resources.build_trace),
    )


def build_lab(
    paths: list[Path],
    *,
    chunking: ChunkingConfig,
    embedding_model: str,
    embedding_device: str = "auto",
    vector_device: str = "cpu",
    llm_provider: str = "dummy",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "qwen3:4b",
    fallback_embedding: bool = False,
    fusion: str = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    ollama_connect_timeout: float = 10.0,
    ollama_read_timeout: float = 600.0,
    ollama_max_retries: int = 1,
    ollama_num_predict: int = 384,
    ollama_temperature: float = 0.15,
    ollama_keep_alive: str = "5m",
) -> LabBundle:
    resources = build_retrieval_resources(
        paths,
        chunking=chunking,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        vector_device=vector_device,
        fallback_embedding=fallback_embedding,
    )
    return compose_lab(
        resources,
        llm_provider=llm_provider,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        fusion=fusion,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        ollama_connect_timeout=ollama_connect_timeout,
        ollama_read_timeout=ollama_read_timeout,
        ollama_max_retries=ollama_max_retries,
        ollama_num_predict=ollama_num_predict,
        ollama_temperature=ollama_temperature,
        ollama_keep_alive=ollama_keep_alive,
    )


def create_rag_pipeline(
    bundle: LabBundle,
    strategy: str,
    *,
    max_context_tokens: int = 1800,
    execution_device: str = "cpu",
    reranker=None,
    top_k: int = 5,
    candidate_count: int = 20,
    context_profile: str = "balanced",
    prompt_profile: str = "professional",
):
    common = dict(
        context_builder=ContextBuilder(max_context_tokens),
        generator=GroundedGenerator(bundle.llm, context_profile=context_profile, prompt_profile=prompt_profile),
        execution_device=execution_device,
        reranker=reranker,
        top_k=top_k,
        candidate_count=candidate_count,
    )
    strategy = strategy.lower()
    if strategy == "baseline":
        return BaselineRAG(retriever=bundle.dense, **common)
    if strategy == "hybrid":
        return HybridRAG(retriever=bundle.hybrid, **common)
    if strategy == "lexical":
        return LexicalRAG(retriever=bundle.sparse, **common)
    if strategy == "reranked":
        common_reranked = {**common, "reranker": reranker or LexicalReranker()}
        return RerankedRAG(retriever=bundle.hybrid, **common_reranked)
    if strategy == "dense-reranked":
        common_reranked = {**common, "reranker": reranker or LexicalReranker()}
        return RerankedRAG(retriever=bundle.dense, **common_reranked)
    if strategy == "hyde":
        return HyDERAG(
            retriever=bundle.dense,
            hyde_generator=HyDEGenerator(bundle.llm),
            **common,
        )
    if strategy == "multi-hop":
        return MultiHopRAG(
            retriever=bundle.hybrid,
            follow_up_generator=FollowUpQueryGenerator(bundle.llm),
            **common,
        )
    if strategy == "multi-query":
        return MultiQueryRAG(
            retriever=bundle.hybrid,
            query_generator=MultiQueryGenerator(bundle.llm),
            **common,
        )
    if strategy == "query-rewrite":
        return QueryRewriteRAG(
            retriever=bundle.hybrid,
            rewriter=QueryRewriter(bundle.llm),
            **common,
        )
    if strategy == "compression":
        return CompressionRAG(retriever=bundle.dense, **common)
    if strategy == "corrective":
        return CorrectiveRAG(
            retriever=bundle.hybrid,
            rewriter=QueryRewriter(bundle.llm),
            **common,
        )
    if strategy == "parent-document":
        parents = {c.chunk_id: c for c in bundle.ingestion.chunks if c.metadata.get("role") == "parent"}
        if not parents:
            raise ValueError("A Parent-Document RAG használatához Parent–Child chunking szükséges.")
        return ParentDocumentRAG(retriever=bundle.dense, parent_chunks=parents, **common)
    raise ValueError(f"Ismeretlen RAG stratégia: {strategy}")
