from __future__ import annotations

from dataclasses import dataclass

from rag_engine.ingestion.chunking.utils import estimate_tokens, split_sentences
from rag_engine.models import RetrievedChunk
from rag_engine.retrieval.query_focus import rank_chunks_for_query


@dataclass(frozen=True)
class ContextResult:
    text: str
    tokens: int
    included: list[RetrievedChunk]


class ContextBuilder:
    def __init__(self, max_tokens: int = 1800) -> None:
        self.max_tokens = max_tokens

    @staticmethod
    def _header(source_index: int, chunk: RetrievedChunk) -> str:
        source = chunk.metadata.get("source", chunk.source)
        title = chunk.metadata.get("title", "")
        page = chunk.metadata.get("page", "")
        section = chunk.metadata.get("section", "")
        return f"[S{source_index}]\nCím: {title}\nForrás: {source}\nOldal: {page}\nSzakasz: {section}\nTartalom:\n"

    def _fit_highest_ranked(self, header: str, text: str, remaining_tokens: int) -> str | None:
        """Fit evidence by sentence boundaries rather than blind character truncation."""
        chosen: list[str] = []
        for sentence in split_sentences(text):
            candidate = " ".join([*chosen, sentence])
            if estimate_tokens(header + candidate) > remaining_tokens:
                break
            chosen.append(sentence)
        return " ".join(chosen).strip() or None

    def build(self, chunks: list[RetrievedChunk]) -> ContextResult:
        seen: set[str] = set()
        blocks: list[str] = []
        included: list[RetrievedChunk] = []
        used = 0
        for chunk in sorted(chunks, key=lambda item: item.rank):
            fingerprint = chunk.text.strip()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            source_index = len(included) + 1
            header = self._header(source_index, chunk)
            block = header + chunk.text.strip()
            tokens = estimate_tokens(block)
            remaining = self.max_tokens - used
            if tokens > remaining:
                # Preserve some top-ranked evidence if nothing else fits, using sentence boundaries.
                if not included and remaining > estimate_tokens(header):
                    fitted = self._fit_highest_ranked(header, chunk.text, remaining)
                    if fitted:
                        block = header + fitted
                        tokens = estimate_tokens(block)
                    else:
                        continue
                else:
                    continue
            blocks.append(block)
            included.append(chunk)
            used += tokens
        return ContextResult(text="\n\n".join(blocks), tokens=used, included=included)

    def build_for_query(self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None) -> ContextResult:
        """Build context after a conservative query-focus pass.

        Retrieval/reranking remains a separate stage. This pass only orders the already
        retrieved evidence for context construction and strongly prefers an article whose
        title is explicitly named by the query. It prevents generic medical snippets from
        consuming the context budget ahead of the requested topic.
        """
        focused = rank_chunks_for_query(query, chunks, top_k=top_k or len(chunks))
        return self.build(focused)
