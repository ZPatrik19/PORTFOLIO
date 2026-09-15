"""Citation construction and validation against selected retrieval context."""

from __future__ import annotations

from collections.abc import Iterable

from .guardrails import sanitize_quote
from .models import SearchHit, SourceCitation


def citations_from_hits(
    hits: Iterable[SearchHit],
    max_quote_chars: int = 320,
) -> list[SourceCitation]:
    """Create citation objects from trusted context hits."""

    citations: list[SourceCitation] = []
    for hit in hits:
        chunk = hit.chunk
        citations.append(
            SourceCitation(
                document_id=chunk.document_id,
                document_title=chunk.title,
                page=chunk.page_start or None,
                chapter=chunk.chapter,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                quote_or_evidence=sanitize_quote(chunk.text, max_quote_chars),
            )
        )
    return citations


def validate_citations(
    citations: Iterable[SourceCitation],
    context_hits: Iterable[SearchHit],
) -> dict[str, object]:
    """Verify that model-proposed citations resolve to the selected context."""

    allowed_chunks = {hit.chunk.chunk_id: hit.chunk for hit in context_hits}
    errors: list[str] = []

    for citation in citations:
        source_chunk = allowed_chunks.get(citation.chunk_id)
        if source_chunk is None:
            errors.append(f"Unknown or non-context chunk_id: {citation.chunk_id}")
            continue
        if citation.document_id != source_chunk.document_id:
            errors.append(f"Document mismatch for {citation.chunk_id}")
        if (
            citation.page is not None
            and source_chunk.page_start
            and not (source_chunk.page_start <= citation.page <= source_chunk.page_end)
        ):
            errors.append(f"Page mismatch for {citation.chunk_id}")

    return {"valid": not errors, "errors": errors}
