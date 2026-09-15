from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable

import numpy as np

from .cache import EmbeddingCache, ParseCache
from .chunking import chunk_document
from .config import PROJECT_ROOT, resolve_path
from .embeddings import (
    GeminiEmbeddingProvider,
    LocalHashingEmbeddingProvider,
    create_embedding_provider,
)
from .indexing import NumpyVectorStore
from .ingestion import build_manifest
from .logging_config import get_logger
from .multimodal import enrich_figure_blocks
from .parsing import parse_document
from .retrieval import HybridRetriever

LOGGER = get_logger(__name__)

INDEX_STRATEGIES = ("fixed", "recursive", "structure_aware", "semantic")


def _provider_metadata(embedder, cfg, vectors: np.ndarray) -> dict:
    effective_model = "local_hashing_blake2b" if isinstance(embedder, LocalHashingEmbeddingProvider) else str(cfg["embedding"].get("model", "unknown"))
    return {
        "provider_class": type(embedder).__name__,
        "model": effective_model,
        "output_dimensionality": int(vectors.shape[1] if vectors.ndim == 2 and vectors.size else cfg["embedding"].get("output_dimensionality", 768)),
    }


class MultiIndexManager:
    """Build/load independent indexes for chunking experiments.

    The primary index remains untouched. Experimental indexes live under
    ``01_data/indexes/variants/<strategy>/`` so their vectors are never mixed.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = resolve_path(cfg["paths"]["indexes"]) / "variants"
        self.root.mkdir(parents=True, exist_ok=True)

    def available(self) -> list[dict]:
        rows = [{"name": "primary", "ready": self._primary_ready(), "strategy": self.cfg["chunking"].get("strategy", "structure_aware")}]
        for strategy in INDEX_STRATEGIES:
            base = self.root / strategy
            meta = {}
            if (base / "index_metadata.json").exists():
                try:
                    meta = json.loads((base / "index_metadata.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    LOGGER.warning("Ignoring invalid index metadata for %s: %s", strategy, exc)
                    meta = {}
            native_ready = (base / "numpy" / "vectors.npy").exists() and (base / "numpy" / "chunks.json").exists()
            primary_alias = strategy == self.cfg["chunking"].get("strategy") and self._primary_ready() and not native_ready
            rows.append({
                "name": strategy,
                "ready": native_ready or primary_alias,
                "strategy": strategy,
                "alias_of": "primary" if primary_alias else None,
                "chunks": meta.get("chunks"),
                "documents": meta.get("documents"),
                "chunk_size": meta.get("chunk_size"),
                "overlap": meta.get("overlap"),
                "provider_class": meta.get("provider_class"),
                "model": meta.get("model"),
                "built_at": meta.get("built_at"),
            })
        return rows

    def _primary_ready(self) -> bool:
        p = resolve_path(self.cfg["paths"]["indexes"]) / "numpy"
        return (p / "vectors.npy").exists() and (p / "chunks.json").exists()

    def load(self, name: str, embedder=None):
        if name in {"primary", "indexed", "default", ""}:
            base = resolve_path(self.cfg["paths"]["indexes"])
        else:
            if name not in INDEX_STRATEGIES:
                raise ValueError(f"Unknown index variant: {name}")
            candidate = self.root / name
            if not (candidate / "numpy" / "vectors.npy").exists() and name == self.cfg["chunking"].get("strategy") and self._primary_ready():
                base = resolve_path(self.cfg["paths"]["indexes"])
            else:
                base = candidate
        chunks, vectors = NumpyVectorStore(base / "numpy").load()
        meta_path = base / "index_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        dim = int(meta.get("output_dimensionality", vectors.shape[1]))
        built_provider = meta.get("provider_class")
        if built_provider == "LocalHashingEmbeddingProvider":
            embedder = LocalHashingEmbeddingProvider(dim)
        elif embedder is None:
            embedder = create_embedding_provider(self.cfg)
        if built_provider == "GeminiEmbeddingProvider" and not isinstance(embedder, GeminiEmbeddingProvider):
            raise RuntimeError("This variant was built with Gemini embeddings. Configure a Gemini embedding provider or rebuild it locally.")
        return chunks, vectors, HybridRetriever(chunks, vectors, embedder, self.cfg), meta

    def build(self, strategies: Iterable[str], *, chunk_size: int | None = None, overlap: int | None = None, force: bool = False) -> list[dict]:
        strategies = [s for s in dict.fromkeys(strategies) if s in INDEX_STRATEGIES]
        if not strategies:
            return []
        docs = build_manifest(self.cfg)
        parse_cache = ParseCache(resolve_path(self.cfg["paths"]["interim"]) / "parse_cache")
        blocks_by_doc = {}
        parse_failures = []
        for doc in docs:
            try:
                blocks = parse_cache.get(doc.document_id, doc.checksum)
                if blocks is None:
                    blocks = parse_document(doc)
                    blocks = enrich_figure_blocks(blocks, self.cfg, PROJECT_ROOT)
                    parse_cache.put(doc.document_id, doc.checksum, blocks)
                blocks_by_doc[doc.document_id] = blocks
            except Exception as exc:
                parse_failures.append({"document_id": doc.document_id, "error": str(exc)})

        embedder = create_embedding_provider(self.cfg)
        size = int(chunk_size or self.cfg["chunking"].get("chunk_size", 900))
        ov = int(overlap if overlap is not None else self.cfg["chunking"].get("overlap", 120))
        results = []
        for strategy in strategies:
            started = time.perf_counter()
            base = self.root / strategy
            if force and base.exists():
                shutil.rmtree(base)
            if not force and (base / "numpy" / "vectors.npy").exists() and (base / "numpy" / "chunks.json").exists():
                _, _, _, meta = self.load(strategy, embedder)
                results.append({"strategy": strategy, "status": "reused", **meta})
                continue

            chunks = []
            for doc in docs:
                blocks = blocks_by_doc.get(doc.document_id)
                if blocks is None:
                    continue
                chunks.extend(chunk_document(doc, blocks, strategy, size, ov))

            provider_meta = _provider_metadata(embedder, self.cfg, np.empty((0, int(self.cfg["embedding"].get("output_dimensionality", 768))), dtype=np.float32))
            cache_name = f"variant_{strategy}_{provider_meta['provider_class']}_{provider_meta['model']}_{provider_meta['output_dimensionality']}"
            ecache = EmbeddingCache(self.root / "embedding_cache", cache_name)
            cached = ecache.load()
            missing = [c for c in chunks if c.chunk_id not in cached]
            if missing:
                new_vecs = embedder.embed_documents([c.text for c in missing])
                cached.update({c.chunk_id: v for c, v in zip(missing, new_vecs, strict=False)})
                ecache.save(cached)
            vectors = np.vstack([cached[c.chunk_id] for c in chunks]) if chunks else np.empty((0, provider_meta["output_dimensionality"]), dtype=np.float32)
            NumpyVectorStore(base / "numpy").build(chunks, vectors)
            meta = {
                **_provider_metadata(embedder, self.cfg, vectors),
                "strategy": strategy,
                "chunk_size": size,
                "overlap": ov,
                "chunks": len(chunks),
                "documents": len({c.document_id for c in chunks}),
                "parse_failures": parse_failures,
                "embedding_cache_hits": len(chunks) - len(missing),
                "embeddings_computed": len(missing),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "build_seconds": round(time.perf_counter() - started, 3),
            }
            base.mkdir(parents=True, exist_ok=True)
            (base / "index_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append({"strategy": strategy, "status": "built", **meta})
        return results
