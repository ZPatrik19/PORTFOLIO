"""Configurable chunking strategies for technical documents."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from .models import Chunk, DocumentRecord, ParsedBlock
from .utils import detect_language, stable_id, tokenize

DEFAULT_KEYWORD_COUNT = 8
SEMANTIC_KEYWORD_COUNT = 12
SEMANTIC_BOUNDARY_THRESHOLD = 0.10
SUPPORTED_FRAMEWORKS = (
    "pytorch",
    "scikit-learn",
    "fastapi",
    "docker",
    "kubernetes",
    "gemini",
    "tensorflow",
    "pandas",
)
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into",
    "egy", "hogy", "ami", "az", "és", "vagy", "van",
}


def extract_keywords(text: str, limit: int = DEFAULT_KEYWORD_COUNT) -> list[str]:
    """Return deterministic high-frequency keywords for metadata and semantic heuristics."""

    candidates = [
        token
        for token in tokenize(text)
        if len(token) > 3 and token not in STOPWORDS
    ]
    return [word for word, _ in Counter(candidates).most_common(limit)]


def detect_framework(text: str) -> str | None:
    """Detect a known technical framework from chunk text."""

    lower_text = text.lower()
    return next((framework for framework in SUPPORTED_FRAMEWORKS if framework in lower_text), None)


def make_chunk(
    document: DocumentRecord,
    text: str,
    blocks: list[ParsedBlock],
    index: int,
    chunk_type: str | None = None,
) -> Chunk:
    """Build a normalized :class:`Chunk` while preserving source metadata."""

    normalized_text = text.strip()
    return Chunk(
        document_id=document.document_id,
        chunk_id=stable_id(document.document_id, str(index), normalized_text[:120], prefix="chk"),
        title=document.title,
        author=document.author,
        chapter=blocks[0].chapter if blocks else None,
        section=blocks[0].section if blocks else None,
        page_start=min((block.page for block in blocks), default=0),
        page_end=max((block.page for block in blocks), default=0),
        chunk_type=chunk_type or _infer_chunk_type(blocks),
        language=detect_language(normalized_text),
        programming_language=None,
        framework=detect_framework(normalized_text),
        keywords=extract_keywords(normalized_text),
        source=document.source,
        source_type=document.source_type,
        asset_path=next((block.asset_path for block in blocks if block.asset_path), None),
        text=normalized_text,
    )


def fixed_chunks(
    document: DocumentRecord,
    blocks: list[ParsedBlock],
    size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Create character-window chunks across the flattened document text."""

    text = "\n\n".join(block.text for block in blocks)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(1, size - overlap)

    while start < len(text):
        segment = text[start : start + size]
        page = blocks[0].page if blocks else 1
        pseudo_block = ParsedBlock(
            document_id=document.document_id,
            page=page,
            block_type="paragraph",
            text=segment,
        )
        chunks.append(make_chunk(document, segment, [pseudo_block], index))
        index += 1
        start += step
    return chunks


def recursive_chunks(
    document: DocumentRecord,
    blocks: list[ParsedBlock],
    size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Split oversized blocks at sentence/paragraph boundaries, then pack them."""

    units: list[ParsedBlock] = []
    for block in blocks:
        if len(block.text) <= size:
            units.append(block)
            continue
        for part in re.split(r"(?<=[.!?])\s+|\n\n+", block.text):
            if not part.strip():
                continue
            units.append(
                ParsedBlock(
                    document_id=block.document_id,
                    page=block.page,
                    block_type=block.block_type,
                    text=part,
                    section=block.section,
                    chapter=block.chapter,
                )
            )
    return _pack_blocks(document, units, size, overlap)


def structure_aware_chunks(
    document: DocumentRecord,
    blocks: list[ParsedBlock],
    size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Preserve headings and code blocks while packing paragraphs by section."""

    del overlap  # structure-aware packing uses semantic block boundaries instead.
    chunks: list[Chunk] = []
    group: list[ParsedBlock] = []
    current_section: str | None = None
    chunk_index = 0

    def flush_group() -> None:
        nonlocal chunk_index
        if not group:
            return
        text = "\n\n".join(block.text for block in group)
        chunks.append(make_chunk(document, text, list(group), chunk_index))
        chunk_index += 1
        group.clear()

    for block in blocks:
        if block.block_type == "header_footer":
            continue
        if block.block_type == "heading":
            flush_group()
            current_section = block.text
            continue
        if block.block_type == "code":
            flush_group()
            chunks.append(make_chunk(document, block.text, [block], chunk_index, "code"))
            chunk_index += 1
            continue
        if current_section and not block.section:
            block.section = current_section

        candidate = "\n\n".join(item.text for item in [*group, block])
        if group and len(candidate) > size:
            flush_group()
        group.append(block)

    flush_group()
    return chunks


def semantic_chunks(
    document: DocumentRecord,
    blocks: list[ParsedBlock],
    size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Use keyword continuity as a lightweight semantic-boundary proxy."""

    del overlap  # semantic boundary detection is content-driven in this lightweight implementation.
    chunks: list[Chunk] = []
    group: list[ParsedBlock] = []
    previous_terms: set[str] | None = None
    chunk_index = 0

    for block in blocks:
        current_terms = set(extract_keywords(block.text, SEMANTIC_KEYWORD_COUNT))
        similarity = _jaccard_similarity(current_terms, previous_terms) if previous_terms else 1.0
        current_length = sum(len(item.text) for item in group)

        if group and (similarity < SEMANTIC_BOUNDARY_THRESHOLD or current_length + len(block.text) > size):
            chunks.append(
                make_chunk(
                    document,
                    "\n\n".join(item.text for item in group),
                    list(group),
                    chunk_index,
                )
            )
            chunk_index += 1
            group.clear()

        group.append(block)
        previous_terms = current_terms

    if group:
        chunks.append(
            make_chunk(
                document,
                "\n\n".join(item.text for item in group),
                list(group),
                chunk_index,
            )
        )
    return chunks


def chunk_document(
    document: DocumentRecord,
    blocks: list[ParsedBlock],
    strategy: str = "structure_aware",
    size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Dispatch document chunking to the configured strategy."""

    strategies: dict[str, Callable[..., list[Chunk]]] = {
        "fixed": fixed_chunks,
        "recursive": recursive_chunks,
        "structure_aware": structure_aware_chunks,
        "semantic": semantic_chunks,
    }
    strategy_function = strategies.get(strategy)
    if strategy_function is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return strategy_function(document, blocks, size, overlap)


def _pack_blocks(
    document: DocumentRecord,
    units: list[ParsedBlock],
    size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    group: list[ParsedBlock] = []
    chunk_index = 0

    for block in units:
        candidate = "\n\n".join(item.text for item in [*group, block])
        if group and len(candidate) > size:
            chunks.append(
                make_chunk(
                    document,
                    "\n\n".join(item.text for item in group),
                    list(group),
                    chunk_index,
                )
            )
            chunk_index += 1
            group = group[-1:] if overlap else []
        group.append(block)

    if group:
        chunks.append(
            make_chunk(
                document,
                "\n\n".join(item.text for item in group),
                list(group),
                chunk_index,
            )
        )
    return chunks


def _infer_chunk_type(blocks: list[ParsedBlock]) -> str:
    if blocks and all(block.block_type == "code" for block in blocks):
        return "code"
    return "mixed"


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


# Backward-compatible aliases for internal helpers used by earlier code/tests.
_keywords = extract_keywords
_framework = detect_framework
_make = make_chunk
_pack = _pack_blocks
