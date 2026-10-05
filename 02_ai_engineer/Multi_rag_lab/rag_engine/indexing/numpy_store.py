from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_engine.models import Chunk, RetrievedChunk


class NumpyVectorStore:
    """Portable exact-search fallback used when FAISS is unavailable."""

    backend_name = "numpy"
    device = "cpu"

    def __init__(self) -> None:
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) != len(chunks):
            raise ValueError("Vector/chunk count mismatch")
        if not len(vectors):
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        scores = self.vectors @ query
        indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i].chunk_id,
                text=self.chunks[i].text,
                source=str(self.chunks[i].metadata.get("source", "")),
                score=float(scores[i]),
                rank=rank,
                metadata=self.chunks[i].metadata,
            )
            for rank, i in enumerate(indices, start=1)
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "metadata.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorStore":
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        store.chunks = [
            Chunk.model_validate(x) for x in json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        ]
        return store
