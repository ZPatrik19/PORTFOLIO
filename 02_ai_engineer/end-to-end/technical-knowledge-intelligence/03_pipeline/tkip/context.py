"""Grounded context selection and prompt assembly."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from .models import SearchHit

SIMILARITY_SAMPLE_CHARS = 1200


def text_similarity(left: str, right: str) -> float:
    """Approximate near-duplicate similarity on bounded text prefixes."""

    return SequenceMatcher(
        None,
        left[:SIMILARITY_SAMPLE_CHARS],
        right[:SIMILARITY_SAMPLE_CHARS],
    ).ratio()


class ContextBuilder:
    """Select diverse, deduplicated evidence under a configurable context budget."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def build(
        self,
        hits: Iterable[SearchHit],
        question: str,
        intent: str,
        tool_results: list[Any] | None = None,
        *,
        max_chars: int | None = None,
        max_chunks_per_document: int | None = None,
    ) -> dict[str, Any]:
        context_config = self.config["context"]
        character_budget = int(max_chars or context_config.get("max_chars", 16000))
        per_document_limit = int(
            max_chunks_per_document or context_config.get("max_chunks_per_document", 3)
        )
        duplicate_threshold = float(context_config.get("deduplicate_threshold", 0.92))

        selected_hits: list[SearchHit] = []
        document_counts: defaultdict[str, int] = defaultdict(int)
        selected_chars = 0

        for hit in hits:
            chunk = hit.chunk
            if document_counts[chunk.document_id] >= per_document_limit:
                continue
            if any(
                text_similarity(chunk.text, selected.chunk.text) >= duplicate_threshold
                for selected in selected_hits
            ):
                continue

            rendered = self._render_source(hit)
            if selected_chars + len(rendered) > character_budget:
                continue

            selected_hits.append(hit)
            document_counts[chunk.document_id] += 1
            selected_chars += len(rendered)

        context = "\n\n".join(self._render_source(hit) for hit in selected_hits)
        tools = "\n".join(str(result) for result in (tool_results or []))
        prompt = (
            f"USER INTENT: {intent}\n\n"
            f"RETRIEVED SOURCES:\n{context}\n\n"
            f"TOOL RESULTS:\n{tools}\n\n"
            f"USER QUESTION:\n{question}"
        )
        return {
            "selected_hits": selected_hits,
            "context": context,
            "chars": len(context),
            "max_chars": character_budget,
            "prompt": prompt,
        }

    @staticmethod
    def _render_source(hit: SearchHit) -> str:
        chunk = hit.chunk
        return (
            f"[SOURCE chunk_id={chunk.chunk_id} document_id={chunk.document_id} "
            f"title={chunk.title!r} page={chunk.page_start} section={chunk.section!r}]\n"
            f"{chunk.text}\n[/SOURCE]"
        )


# Backward-compatible internal alias.
_similar = text_similarity
