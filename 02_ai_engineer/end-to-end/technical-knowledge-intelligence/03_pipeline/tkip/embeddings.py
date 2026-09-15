"""Embedding provider abstractions for local/offline and Gemini-backed retrieval."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import gemini_api_key
from .logging_config import get_logger
from .utils import batched, tokenize

LOGGER = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Interface implemented by all embedding backends."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed document texts into a two-dimensional float array."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with the same vector space as documents."""

        return self.embed_documents([text])[0]


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-light hashing embedding for offline testing/demo use."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).hexdigest()
            token_hash = int(digest, 16)
            index = token_hash % self.dimension
            sign = 1.0 if ((token_hash >> 8) & 1) == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(text) for text in texts])


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding backend with batching and bounded retries."""

    def __init__(
        self,
        model: str = "gemini-embedding-2",
        dimension: int = 768,
        batch_size: int = 16,
        max_retries: int = 3,
    ) -> None:
        from google import genai

        api_key = gemini_api_key()
        if not api_key:
            raise RuntimeError("Gemini embedding provider requires GEMINI_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        from google.genai import types

        rows: list[list[float]] = []
        for batch in batched(texts, self.batch_size):
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.dimension,
                        ),
                    )
                    embeddings = response.embeddings or []
                    for embedding in embeddings:
                        if embedding.values is None:
                            raise RuntimeError(
                                "Gemini returned an embedding without vector values."
                            )
                        rows.append([float(value) for value in embedding.values])
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    delay_seconds = min(8, 2**attempt)
                    LOGGER.warning(
                        "Gemini embedding attempt %s/%s failed; retrying in %ss: %s",
                        attempt + 1,
                        self.max_retries,
                        delay_seconds,
                        exc,
                    )
                    time.sleep(delay_seconds)

            if last_error is not None:
                raise RuntimeError(
                    f"Gemini embedding failed after {self.max_retries} attempts: {last_error}"
                ) from last_error

        vectors = np.asarray(rows, dtype=np.float32)
        if not len(vectors):
            return np.empty((0, self.dimension), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


def create_embedding_provider(config: dict[str, Any]) -> EmbeddingProvider:
    """Create the configured embedding provider with safe local fallback for ``auto``."""

    embedding_config = config["embedding"]
    provider = embedding_config.get("provider", "auto")
    dimension = int(embedding_config.get("output_dimensionality", 768))

    if provider in {"auto", "gemini"} and gemini_api_key():
        try:
            return GeminiEmbeddingProvider(
                model=embedding_config["model"],
                dimension=dimension,
                batch_size=int(embedding_config.get("batch_size", 16)),
                max_retries=int(embedding_config.get("max_retries", 3)),
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            if provider == "gemini":
                raise
            LOGGER.warning("Gemini embeddings unavailable; using local hashing fallback: %s", exc)

    return LocalHashingEmbeddingProvider(dimension)
