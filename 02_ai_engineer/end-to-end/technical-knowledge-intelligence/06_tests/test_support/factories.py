"""Reusable test-data factories.

The goal of these helpers is to keep individual tests focused on behavior rather
than boilerplate model construction. They intentionally create small, explicit,
deterministic objects suitable for unit/integration tests.
"""

from __future__ import annotations

from pathlib import Path

from tkip.models import Chunk, DocumentRecord, SearchHit, SourceCitation


def make_chunk(
    chunk_id: str = "chunk-1",
    text: str = "Retrieval augmented generation combines retrieval with generation.",
    *,
    document_id: str = "doc-1",
    title: str = "Demo Technical Book",
    source: str = "demo.pdf",
    source_type: str = "demo",
    chunk_type: str = "paragraph",
    page: int = 1,
    chapter: str | None = None,
    section: str | None = None,
    keywords: list[str] | None = None,
) -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        title=title,
        source=source,
        source_type=source_type,
        chunk_type=chunk_type,
        page_start=page,
        page_end=page,
        chapter=chapter,
        section=section,
        keywords=keywords or [],
        text=text,
    )


def make_search_hit(
    chunk: Chunk | None = None,
    *,
    rank: int = 1,
    bm25_score: float = 2.0,
    dense_score: float = 0.7,
    hybrid_score: float = 0.04,
    reranker_score: float = 0.9,
) -> SearchHit:
    return SearchHit(
        chunk=chunk or make_chunk(),
        rank=rank,
        bm25_score=bm25_score,
        dense_score=dense_score,
        hybrid_score=hybrid_score,
        reranker_score=reranker_score,
    )


def make_citation(
    chunk: Chunk | None = None,
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    document_title: str | None = None,
) -> SourceCitation:
    chunk = chunk or make_chunk()
    return SourceCitation(
        document_id=document_id or chunk.document_id,
        document_title=document_title or chunk.title,
        page=chunk.page_start,
        chapter=chunk.chapter,
        section=chunk.section,
        chunk_id=chunk_id or chunk.chunk_id,
        quote_or_evidence=chunk.text[:180],
    )


def make_document_record(
    path: Path,
    *,
    document_id: str = "doc-1",
    title: str = "Demo Document",
    source_type: str = "demo",
    checksum: str = "checksum-test",
) -> DocumentRecord:
    suffix = path.suffix.lower().lstrip(".") or "txt"
    return DocumentRecord(
        document_id=document_id,
        filename=path.name,
        title=title,
        document_type=suffix,
        source=str(path),
        source_type=source_type,
        checksum=checksum,
        path=str(path),
    )
