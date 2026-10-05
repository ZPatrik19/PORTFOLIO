from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_engine.models import Chunk, RetrievedChunk


class FaissCPUVectorStore:
    backend_name = "faiss"
    device = "cpu"

    def __init__(self, dimension: int | None = None) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is not installed. Run SETUP first or use NumpyVectorStore.") from exc
        self.faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) if dimension else None
        self.chunks: list[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32).copy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = self._normalize(vectors)
        if len(vectors) != len(chunks):
            raise ValueError("Vector/chunk count mismatch")
        if not len(vectors):
            return
        if self.index is None:
            self.dimension = vectors.shape[1]
            self.index = self.faiss.IndexFlatIP(self.dimension)
        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match index")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if self.index is None or not self.chunks:
            return []
        query = self._normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(RetrievedChunk(chunk_id=chunk.chunk_id, text=chunk.text,
                source=str(chunk.metadata.get("source", "")), score=float(score), rank=rank, metadata=chunk.metadata))
        return results

    @staticmethod
    def _write_index_portable(faiss_module, index, target: Path) -> None:
        """Persist a FAISS index without relying on FAISS path handling.

        On Windows, FAISS' native ``write_index`` may fail for otherwise valid
        Unicode paths (for example a directory containing ``é``). Serializing
        the index in memory and letting Python write the bytes keeps persistence
        Unicode-safe and works on Windows/Linux alike.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        serialize = getattr(faiss_module, "serialize_index", None)
        if serialize is None:
            faiss_module.write_index(index, str(target))
            return
        payload = np.asarray(serialize(index), dtype=np.uint8).tobytes()
        target.write_bytes(payload)

    @staticmethod
    def _read_index_portable(faiss_module, source: Path):
        """Load a FAISS index through Python bytes for Unicode-safe paths."""
        deserialize = getattr(faiss_module, "deserialize_index", None)
        if deserialize is None:
            return faiss_module.read_index(str(source))
        payload = np.frombuffer(source.read_bytes(), dtype=np.uint8).copy()
        return deserialize(payload)

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save empty index")
        path.mkdir(parents=True, exist_ok=True)
        self._write_index_portable(self.faiss, self.index, path / "index.faiss")
        (path / "metadata.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (path / "configuration.json").write_text(
            json.dumps({"dimension": self.dimension, "metric": "inner_product_cosine_normalized"}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "FaissCPUVectorStore":
        import faiss

        store = cls()
        store.index = cls._read_index_portable(faiss, path / "index.faiss")
        store.dimension = store.index.d
        store.chunks = [
            Chunk.model_validate(x)
            for x in json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        ]
        return store
