from __future__ import annotations

import re

from rag_engine.retrieval.query_focus import chunk_focus_score


class LexicalReranker:
    """Dependency-free reranker with metadata-aware query focus.

    It remains deterministic/offline, but title and section relevance are deliberately
    included so an exact medical article match is not lost to a generic body-text overlap.
    """

    device = "cpu"
    model_name = "lexical-query-focus"

    def rerank(self, query: str, chunks, top_k: int = 5):
        q = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            searchable = " ".join(
                part
                for part in (
                    str(metadata.get("title", "")),
                    str(metadata.get("section", "")),
                    str(getattr(chunk, "text", "")),
                )
                if part
            )
            terms = set(re.findall(r"\w+", searchable.lower()))
            lexical = len(q & terms) / max(1, len(q))
            focus = chunk_focus_score(query, chunk)
            # Preserve the intuitive 0..1-ish lexical score while adding a bounded
            # query-focus boost for title/section relevance.
            score = lexical + min(1.0, focus.score / 6.0)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            c.model_copy(update={"score": float(s), "rank": rank})
            for rank, (s, c) in enumerate(scored[:top_k], start=1)
        ]
