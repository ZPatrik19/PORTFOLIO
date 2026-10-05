from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingEmbeddingProvider:
    """Deterministic dependency-light provider for tests/offline fallback, not production quality."""

    model_name = "hashing-fallback"
    device = "cpu"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)
