from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_engine.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]: ...
    def save(self, path) -> None: ...
