from __future__ import annotations

import logging

from rag_engine.indexing.hashing import HashingEmbeddingProvider
from rag_engine.indexing.sentence_transformer import SentenceTransformerEmbeddingProvider

LOGGER = logging.getLogger(__name__)


def create_embedding_provider(model_name: str, *, device: str | None = "auto", fallback_to_hashing: bool = False):
    if model_name in {"hashing", "__offline_hashing_fallback__"}:
        return HashingEmbeddingProvider()
    normalized_device = device or "auto"
    try:
        return SentenceTransformerEmbeddingProvider(model_name, device=normalized_device)
    except Exception:
        if not fallback_to_hashing:
            raise
        LOGGER.exception("Embedding model unavailable; using hashing fallback")
        return HashingEmbeddingProvider()
