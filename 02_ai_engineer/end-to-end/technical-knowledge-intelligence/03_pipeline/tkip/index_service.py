"""Knowledge-index lifecycle management.

This module isolates corpus discovery, parsing, chunking, embedding, caching and
index loading from request orchestration. Keeping the index lifecycle separate
makes the request pipeline easier to test and reason about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import EmbeddingCache, ParseCache
from .chunking import chunk_document
from .config import PROJECT_ROOT, resolve_path
from .embeddings import GeminiEmbeddingProvider, LocalHashingEmbeddingProvider
from .exceptions import IndexNotReadyError
from .indexing import NumpyVectorStore, build_indexes
from .ingestion import build_manifest
from .logging_config import get_logger
from .multimodal import enrich_figure_blocks
from .parsing import parse_document
from .quality import build_quality_report
from .retrieval import HybridRetriever

LOGGER = get_logger(__name__)


@dataclass
class IndexState:
    """In-memory index state shared with the application service."""

    chunks: list[Any]
    vectors: np.ndarray
    retriever: HybridRetriever


class IndexService:
    """Build and load the primary knowledge index for the configured corpus."""

    def __init__(self, config: dict[str, Any], embedder: Any) -> None:
        self.config = config
        self.embedder = embedder

    def build(self) -> tuple[IndexState, dict[str, Any]]:
        """Parse the corpus, use caches where possible, and persist primary indexes."""

        documents = build_manifest(self.config)
        chunks: list[Any] = []
        failures: list[dict[str, str]] = []
        chunking_config = self.config["chunking"]
        parse_cache = ParseCache(resolve_path(self.config["paths"]["interim"]) / "parse_cache")
        parse_cache_hits = 0

        for document in documents:
            try:
                blocks = parse_cache.get(document.document_id, document.checksum)
                if blocks is None:
                    blocks = parse_document(document)
                    blocks = enrich_figure_blocks(blocks, self.config, PROJECT_ROOT)
                    parse_cache.put(document.document_id, document.checksum, blocks)
                else:
                    parse_cache_hits += 1

                chunks.extend(
                    chunk_document(
                        document,
                        blocks,
                        chunking_config["strategy"],
                        chunking_config["chunk_size"],
                        chunking_config["overlap"],
                    )
                )
            except Exception as exc:  # document-level fault isolation is intentional
                LOGGER.exception("Document ingestion failed: %s", document.path)
                failures.append({"document_id": document.document_id, "error": str(exc)})

        processed_dir = resolve_path(self.config["paths"]["processed"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "chunks.json").write_text(
            json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        quality_summary, per_document, chunk_frame = build_quality_report(
            documents, chunks, failures
        )
        (processed_dir / "quality_summary.json").write_text(
            json.dumps(quality_summary, indent=2), encoding="utf-8"
        )
        per_document.to_csv(processed_dir / "quality_per_document.csv", index=False)
        chunk_frame.to_csv(processed_dir / "quality_chunks.csv", index=False)

        vectors, embedding_cache_hits, embeddings_computed = self._embed_chunks(chunks)
        vector_store = build_indexes(self.config, chunks, vectors)
        self._persist_index_metadata(documents, vectors)

        retriever = HybridRetriever(chunks, vectors, self.embedder, self.config)
        state = IndexState(chunks=chunks, vectors=vectors, retriever=retriever)
        report = {
            "documents": len(documents),
            "chunks": len(chunks),
            "failures": failures,
            "vector_store": vector_store,
            "parse_cache_hits": parse_cache_hits,
            "embedding_cache_hits": embedding_cache_hits,
            "embeddings_computed": embeddings_computed,
            "quality": quality_summary,
        }
        LOGGER.info(
            "Index build complete: documents=%s chunks=%s failures=%s cache_hits=%s",
            len(documents),
            len(chunks),
            len(failures),
            embedding_cache_hits,
        )
        return state, report

    def load(self) -> IndexState:
        """Load the portable NumPy index and align the query embedding provider."""

        index_dir = resolve_path(self.config["paths"]["indexes"])
        try:
            chunks, vectors = NumpyVectorStore(index_dir / "numpy").load()
        except Exception as exc:
            raise IndexNotReadyError(
                f"Primary NumPy index could not be loaded from {index_dir / 'numpy'}: {exc}"
            ) from exc

        metadata_path = index_dir / "index_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._align_embedder_with_index(metadata, vectors)

        retriever = HybridRetriever(chunks, vectors, self.embedder, self.config)
        LOGGER.info("Loaded primary index: chunks=%s dimensions=%s", len(chunks), vectors.shape)
        return IndexState(chunks=chunks, vectors=vectors, retriever=retriever)

    def _embed_chunks(self, chunks: list[Any]) -> tuple[np.ndarray, int, int]:
        embedding_config = self.config["embedding"]
        effective_model = (
            "local_hashing_blake2b"
            if isinstance(self.embedder, LocalHashingEmbeddingProvider)
            else str(embedding_config.get("model", "unknown"))
        )
        provider_key = "_".join(
            [
                type(self.embedder).__name__,
                effective_model,
                str(embedding_config.get("output_dimensionality", 768)),
            ]
        )
        cache = EmbeddingCache(
            resolve_path(self.config["paths"]["indexes"]) / "embedding_cache",
            provider_key,
        )
        cached_vectors = cache.load()
        missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in cached_vectors]
        if missing_chunks:
            new_vectors = self.embedder.embed_documents([chunk.text for chunk in missing_chunks])
            cached_vectors.update(
                {
                    chunk.chunk_id: vector
                    for chunk, vector in zip(missing_chunks, new_vectors, strict=False)
                }
            )
            cache.save(cached_vectors)

        dimension = int(embedding_config.get("output_dimensionality", 768))
        vectors = (
            np.vstack([cached_vectors[chunk.chunk_id] for chunk in chunks])
            if chunks
            else np.empty((0, dimension), dtype=np.float32)
        )
        return vectors, len(chunks) - len(missing_chunks), len(missing_chunks)

    def _persist_index_metadata(self, documents: list[Any], vectors: np.ndarray) -> None:
        index_dir = resolve_path(self.config["paths"]["indexes"])
        index_dir.mkdir(parents=True, exist_ok=True)
        state = {document.document_id: document.checksum for document in documents}
        (index_dir / "indexed_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

        effective_model = (
            "local_hashing_blake2b"
            if isinstance(self.embedder, LocalHashingEmbeddingProvider)
            else str(self.config["embedding"].get("model", "unknown"))
        )
        dimension = int(
            self.config["embedding"].get(
                "output_dimensionality",
                vectors.shape[1] if vectors.ndim == 2 and vectors.size else 768,
            )
        )
        metadata = {
            "provider_class": type(self.embedder).__name__,
            "model": effective_model,
            "output_dimensionality": dimension,
        }
        (index_dir / "index_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def _align_embedder_with_index(self, metadata: dict[str, Any], vectors: np.ndarray) -> None:
        built_provider = metadata.get("provider_class")
        dimension = int(metadata.get("output_dimensionality", vectors.shape[1]))
        if built_provider == "LocalHashingEmbeddingProvider" and not isinstance(
            self.embedder, LocalHashingEmbeddingProvider
        ):
            self.embedder = LocalHashingEmbeddingProvider(dimension)
        elif built_provider == "GeminiEmbeddingProvider" and not isinstance(
            self.embedder, GeminiEmbeddingProvider
        ):
            raise IndexNotReadyError(
                "This index was built with Gemini embeddings. Configure GEMINI_API_KEY or rebuild "
                "the index with local embeddings before querying it."
            )
