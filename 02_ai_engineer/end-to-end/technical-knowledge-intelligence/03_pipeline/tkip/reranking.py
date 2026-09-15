"""Reranking strategies applied after hybrid candidate retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .logging_config import get_logger
from .models import SearchHit
from .utils import tokenize

LOGGER = get_logger(__name__)

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "how",
    "where",
    "mit",
    "egy",
    "hogy",
    "ami",
    "az",
    "és",
    "vagy",
    "van",
    "hol",
    "könyvek",
    "alapján",
}
MIN_TERM_LENGTH = 3
TITLE_MATCH_BONUS = 0.35


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) >= MIN_TERM_LENGTH and token not in STOP_WORDS
    }


def lightweight_rerank(query: str, hits: Sequence[SearchHit], top_n: int = 8) -> list[SearchHit]:
    """Rerank hits with deterministic lexical and metadata overlap signals."""

    query_tokens = _content_tokens(query)
    for hit in hits:
        metadata_text = " ".join(
            [
                hit.chunk.title or "",
                hit.chunk.chapter or "",
                hit.chunk.section or "",
                " ".join(hit.chunk.keywords or []),
            ]
        )
        body_tokens = _content_tokens(hit.chunk.text)
        metadata_tokens = _content_tokens(metadata_text)
        title_tokens = _content_tokens(hit.chunk.title or "")

        denominator = max(1, len(query_tokens))
        body_overlap = len(query_tokens & body_tokens) / denominator
        metadata_overlap = len(query_tokens & metadata_tokens) / denominator
        title_bonus = TITLE_MATCH_BONUS if query_tokens & title_tokens else 0.0

        hit.reranker_score = (
            0.50 * body_overlap
            + 0.25 * metadata_overlap
            + 0.20 * (hit.hybrid_score or 0.0)
            + title_bonus
        )

    return sorted(hits, key=lambda hit: hit.reranker_score or 0.0, reverse=True)[:top_n]


def cross_encoder_rerank(
    query: str,
    hits: Sequence[SearchHit],
    model_name: str,
    top_n: int = 8,
) -> list[SearchHit]:
    """Rerank candidates with an optional sentence-transformers cross encoder."""

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    scores = model.predict([(query, hit.chunk.text) for hit in hits])
    for hit, score in zip(hits, scores, strict=False):
        hit.reranker_score = float(score)
    return sorted(hits, key=lambda hit: hit.reranker_score or 0.0, reverse=True)[:top_n]


def rerank(query: str, hits: Sequence[SearchHit], config: dict[str, Any]) -> list[SearchHit]:
    """Apply configured reranking with deterministic fallback when optional models fail."""

    reranking_config = config["reranking"]
    top_n = int(reranking_config.get("top_n", 8))
    if not reranking_config.get("enabled", True):
        return list(hits[:top_n])

    if reranking_config.get("provider") == "cross_encoder":
        try:
            return cross_encoder_rerank(
                query,
                hits,
                reranking_config["cross_encoder_model"],
                top_n,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning(
                "Cross-encoder reranking unavailable; using deterministic fallback: %s",
                exc,
            )

    return lightweight_rerank(query, hits, top_n)
