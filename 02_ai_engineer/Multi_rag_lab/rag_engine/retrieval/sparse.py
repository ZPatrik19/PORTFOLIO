from __future__ import annotations

import math
import re
from collections import Counter

from rag_engine.models import Chunk, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    """Small BM25 implementation to keep the sparse baseline inspectable and dependency-light."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index title/section metadata together with the body. Medical queries often name
        # the article explicitly (e.g. a disease name), and body-only BM25 could otherwise
        # prefer a generic sentence from an unrelated article.
        self.tokens = [
            _tokenize(
                " ".join(
                    part
                    for part in (
                        str(c.metadata.get("title", "")),
                        str(c.metadata.get("section", "")),
                        c.text,
                    )
                    if part
                )
            )
            for c in chunks
        ]
        self.lengths = [len(x) for x in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.term_freqs = [Counter(x) for x in self.tokens]
        df: Counter[str] = Counter()
        for tokens in self.tokens:
            df.update(set(tokens))
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = _tokenize(query)
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            dl = self.lengths[i]
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores.append((score, i))
        ranked = sorted(scores, reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i].chunk_id,
                text=self.chunks[i].text,
                source=str(self.chunks[i].metadata.get("source", "")),
                score=float(score),
                rank=rank,
                metadata=self.chunks[i].metadata,
            )
            for rank, (score, i) in enumerate(ranked, start=1)
            if score > 0
        ]
