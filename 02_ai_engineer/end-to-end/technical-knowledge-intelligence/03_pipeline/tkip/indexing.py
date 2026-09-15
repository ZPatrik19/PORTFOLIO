from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import resolve_path
from .logging_config import get_logger
from .models import Chunk


LOGGER = get_logger(__name__)

class NumpyVectorStore:
    """Portable vector index used by the default local portfolio runtime.

    For this project size NumPy gives a deterministic, dependency-light local
    baseline. BM25 + dense retrieval still runs exactly as before; Qdrant is an
    optional deployment choice rather than a mandatory setup dependency.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir

    def build(self, chunks: list[Chunk], vectors: np.ndarray):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", vectors)
        (self.index_dir / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self):
        chunks = [
            Chunk(**x)
            for x in json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        ]
        vectors = np.load(self.index_dir / "vectors.npy")
        return chunks, vectors


class QdrantVectorStore:
    """Optional Qdrant backend.

    Modes:
    - ``local``: qdrant-client embedded/local mode; suitable only for small demo
      collections.
    - ``server``: a normal Qdrant service (for example Docker on localhost:6333).
    """

    def __init__(self, cfg):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.cfg = cfg
        vcfg = cfg.get("vector_store", {})
        mode = str(vcfg.get("qdrant_mode", "server")).lower()
        if mode == "local":
            path = resolve_path(vcfg.get("qdrant_path", "01_data/indexes/qdrant"))
            self.client = QdrantClient(path=str(path))
        else:
            url = os.getenv("QDRANT_URL") or vcfg.get("qdrant_url", "http://127.0.0.1:6333")
            api_key = os.getenv("QDRANT_API_KEY") or vcfg.get("qdrant_api_key")
            self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = vcfg.get("collection_name", "technical_knowledge")

    def build(self, chunks, vectors):
        m = self.models
        dim = int(vectors.shape[1])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE),
        )
        # Stream batches instead of materialising every PointStruct at once.
        batch_size = int(self.cfg.get("vector_store", {}).get("upsert_batch_size", 128))
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            points = [
                m.PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i].model_dump())
                for i in range(start, end)
            ]
            self.client.upsert(self.collection, points)

    def search(self, query_vector, limit=10):
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        ).points
        return [(Chunk(**p.payload), float(p.score)) for p in result]


def _qdrant_local_is_too_large(cfg, point_count: int) -> bool:
    """Return True when embedded Qdrant local mode should be skipped."""
    vcfg = cfg.get("vector_store", {})
    if str(vcfg.get("qdrant_mode", "server")).lower() != "local":
        return False
    threshold = int(vcfg.get("max_local_points", 20000))
    return point_count > threshold


def build_indexes(cfg, chunks, vectors):
    idx = resolve_path(cfg["paths"]["indexes"])
    idx.mkdir(parents=True, exist_ok=True)

    # Always build the portable NumPy index. It is the authoritative local
    # fallback and the default backend for the 20k+ chunk private-book corpus.
    NumpyVectorStore(idx / "numpy").build(chunks, vectors)
    used = "numpy"

    provider = str(cfg.get("vector_store", {}).get("provider", "numpy")).lower()
    if provider not in {"qdrant", "qdrant+numpy"}:
        return used

    if _qdrant_local_is_too_large(cfg, len(chunks)):
        message = (
            f"Qdrant local mode skipped for {len(chunks)} points. "
            f"The configured local threshold is "
            f"{cfg.get('vector_store', {}).get('max_local_points', 20000)}. "
            "NumPy remains active. For a large Qdrant index use qdrant_mode=server "
            "with Docker/Qdrant Server."
        )
        (idx / "qdrant_skipped.txt").write_text(message, encoding="utf-8")
        print(f"[index] {message}")
        return used

    try:
        QdrantVectorStore(cfg).build(chunks, vectors)
        used = "qdrant+numpy"
        skipped = idx / "qdrant_skipped.txt"
        if skipped.exists():
            skipped.unlink()
    except Exception as exc:  # qdrant-client exposes backend-specific runtime errors
        if not cfg.get("vector_store", {}).get("fallback_to_numpy", True):
            raise
        (idx / "qdrant_fallback.txt").write_text(str(exc), encoding="utf-8")
        LOGGER.warning("Qdrant unavailable; continuing with NumPy index: %s", exc)
    return used
