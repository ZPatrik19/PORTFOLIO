"""Lexical, dense and hybrid retrieval primitives."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .models import Chunk, SearchHit
from .utils import tokenize

BM25_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "what", "how", "where",
    "when", "why", "are", "its", "into", "does", "is", "a", "an", "of", "to", "in",
    "on", "or", "be", "can", "do", "mit", "mi", "melyik", "hogyan", "hogy", "egy",
    "az", "és", "vagy", "van", "hol", "mikor", "miért", "erre", "ezt",
}


def query_tokens(query: str) -> list[str]:
    """Tokenize a query and remove low-information stopwords."""

    return [token for token in tokenize(query) if len(token) >= 2 and token not in BM25_STOPWORDS]


class BM25Index:
    """Small in-process field-aware BM25 index for local technical corpora."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [self._lexical_tokens(chunk) for chunk in chunks]
        self.average_document_length = sum(map(len, self.documents)) / max(1, len(self.documents))
        self.document_frequency: Counter[str] = Counter()
        for document_tokens in self.documents:
            self.document_frequency.update(set(document_tokens))
        self.document_count = len(self.documents)

    def _lexical_tokens(self, chunk: Chunk) -> list[str]:
        title = chunk.title or ""
        chapter = chunk.chapter or ""
        section = chunk.section or ""
        keywords = " ".join(chunk.keywords or [])
        # A bounded title prior makes exact book/topic matches visible without a
        # separate search engine while leaving the body text dominant overall.
        lexical_text = f"{title} {title} {title} {chapter} {section} {keywords} {chunk.text}"
        return tokenize(lexical_text)

    def score(self, query: str, document_tokens: list[str]) -> float:
        term_frequency = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0

        for token in query_tokens(query):
            document_frequency = self.document_frequency.get(token, 0)
            inverse_document_frequency = self._idf(document_frequency)
            frequency = term_frequency.get(token, 0)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(1, self.average_document_length)
            )
            if denominator:
                score += inverse_document_frequency * (frequency * (self.k1 + 1)) / denominator
        return score

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_token_set = set(query_tokens(query))
        rows: list[tuple[Chunk, float]] = []

        for chunk, document_tokens in zip(self.chunks, self.documents):
            if not _matches_filters(chunk, filters):
                continue
            score = self.score(query, document_tokens)
            title_tokens = set(tokenize(chunk.title or ""))
            for token in query_token_set & title_tokens:
                score += 2.5 * self._idf(self.document_frequency.get(token, 0))
            rows.append((chunk, score))

        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:k]

    def _idf(self, document_frequency: int) -> float:
        if not self.document_count:
            return 0.0
        return math.log(
            1
            + (self.document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )


def dense_search(
    chunks: list[Chunk],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    k: int = 10,
    filters: dict[str, object] | None = None,
) -> list[tuple[Chunk, float]]:
    """Return the top cosine/dot-product matches from normalized embedding vectors."""

    scores = vectors @ query_vector
    rows = [
        (chunk, float(score))
        for chunk, score in zip(chunks, scores)
        if _matches_filters(chunk, filters)
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:k]


def reciprocal_rank_fusion(
    bm25_results: Iterable[tuple[Chunk, float]],
    dense_results: Iterable[tuple[Chunk, float]],
    k: int = 60,
    limit: int = 24,
) -> list[SearchHit]:
    """Fuse lexical and dense rank lists using Reciprocal Rank Fusion (RRF)."""

    bm25_results = list(bm25_results)
    dense_results = list(dense_results)
    fused_scores: defaultdict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, Chunk] = {}
    bm25_scores = {chunk.chunk_id: score for chunk, score in bm25_results}
    dense_scores = {chunk.chunk_id: score for chunk, score in dense_results}

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        fused_scores[chunk.chunk_id] += 1 / (k + rank)
        chunks_by_id[chunk.chunk_id] = chunk
    for rank, (chunk, _) in enumerate(dense_results, start=1):
        fused_scores[chunk.chunk_id] += 1 / (k + rank)
        chunks_by_id[chunk.chunk_id] = chunk

    ordered_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:limit]
    return [
        SearchHit(
            chunk=chunks_by_id[chunk_id],
            rank=rank,
            bm25_score=bm25_scores.get(chunk_id),
            dense_score=dense_scores.get(chunk_id),
            hybrid_score=fused_scores[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRetriever:
    """Native retrieval engine combining BM25 and dense embedding search."""

    def __init__(self, chunks, vectors, embedding_provider, cfg) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedding_provider
        self.cfg = cfg
        self.bm25 = BM25Index(chunks)

    def search(self, query: str, filters: dict[str, object] | None = None) -> list[SearchHit]:
        retrieval_config = self.cfg["retrieval"]
        bm25_results = self.bm25.search(
            query,
            retrieval_config.get("bm25_k", 20),
            filters,
        )
        query_vector = self.embedder.embed_query(query)
        dense_results = dense_search(
            self.chunks,
            self.vectors,
            query_vector,
            retrieval_config.get("dense_k", 20),
            filters,
        )
        return reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            retrieval_config.get("rrf_k", 60),
            retrieval_config.get("hybrid_k", 24),
        )


def _matches_filters(chunk: Chunk, filters: dict[str, object] | None) -> bool:
    if not filters:
        return True
    return all(value is None or getattr(chunk, key, None) == value for key, value in filters.items())


# Backward-compatible internal alias.
_query_tokens = query_tokens
