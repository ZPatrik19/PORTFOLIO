from __future__ import annotations

import re
from collections import defaultdict

from .models import Chunk, SearchHit
from .utils import stable_id


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text.strip()]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(text), step):
        seg = text[start : start + size].strip()
        if seg:
            out.append(seg)
        if start + size >= len(text):
            break
    return out


def _recursive(text: str, size: int, overlap: int) -> list[str]:
    units = _sentences(text)
    if not units:
        return _fixed(text, size, overlap)
    out: list[str] = []
    buf: list[str] = []
    for unit in units:
        candidate = " ".join(buf + [unit]).strip()
        if buf and len(candidate) > size:
            out.append(" ".join(buf).strip())
            # Keep the tail sentence as a lightweight semantic overlap.
            buf = buf[-1:] if overlap > 0 else []
        if len(unit) > size:
            out.extend(_fixed(unit, size, overlap))
            buf = []
        else:
            buf.append(unit)
    if buf:
        out.append(" ".join(buf).strip())
    return out


def _structure_aware(text: str, size: int, overlap: int) -> list[str]:
    # Preserve fenced code blocks as atomic units where possible, then pack the
    # surrounding prose recursively. This is intentionally query-time only.
    pieces = re.split(r"(```[\s\S]*?```)", text)
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("```") and piece.endswith("```"):
            if len(piece) <= size:
                out.append(piece)
            else:
                out.extend(_fixed(piece, size, overlap))
        else:
            out.extend(_recursive(piece, size, overlap))
    return out or [text.strip()]


def _semantic(text: str, size: int, overlap: int) -> list[str]:
    # Lightweight semantic proxy: paragraph/sentence groups are broken when
    # vocabulary continuity drops. This avoids an API call during comparison.
    units = _sentences(text)
    if not units:
        return _fixed(text, size, overlap)

    def terms(s: str) -> set[str]:
        return {x.lower() for x in re.findall(r"[A-Za-zÀ-ž0-9_\-]{4,}", s)}

    out: list[str] = []
    buf: list[str] = []
    prev: set[str] | None = None
    for unit in units:
        current = terms(unit)
        continuity = 1.0
        if prev is not None:
            continuity = len(prev & current) / max(1, len(prev | current))
        candidate = " ".join(buf + [unit]).strip()
        if buf and (len(candidate) > size or continuity < 0.06):
            out.append(" ".join(buf).strip())
            buf = buf[-1:] if overlap > 0 else []
        buf.append(unit)
        prev = current
    if buf:
        out.append(" ".join(buf).strip())
    return out


def _split(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    fn = {
        "fixed": _fixed,
        "recursive": _recursive,
        "structure_aware": _structure_aware,
        "semantic": _semantic,
    }.get(strategy, _structure_aware)
    return [x for x in fn(text, size, overlap) if x.strip()]


def _derive_chunk(parent: Chunk, text: str, suffix: str, page_end: int | None = None) -> Chunk:
    return parent.model_copy(
        update={
            "chunk_id": stable_id(parent.chunk_id, suffix, text[:96], prefix="rt"),
            "page_end": page_end if page_end is not None else parent.page_end,
            "text": text,
        }
    )


def rechunk_hits(
    hits: list[SearchHit],
    strategy: str,
    size: int,
    overlap: int,
    max_output_chunks: int = 24,
) -> tuple[list[SearchHit], dict]:
    """Reshape retrieved candidates without rebuilding embeddings/indexes.

    Small target sizes split candidate chunks. Larger target sizes opportunistically
    merge adjacent candidate chunks from the same document/page neighborhood. The
    retrieval scores are retained as ranking priors and the reranker can be run
    again afterwards by the caller.
    """
    if not hits:
        return [], {
            "enabled": True,
            "before": 0,
            "after": 0,
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
        }

    # Merge nearby retrieved chunks when the requested context chunk is larger.
    by_doc: dict[str, list[SearchHit]] = defaultdict(list)
    for h in hits:
        by_doc[h.chunk.document_id].append(h)

    candidates: list[SearchHit] = []
    for doc_hits in by_doc.values():
        doc_hits = sorted(doc_hits, key=lambda h: (h.chunk.page_start, h.rank))
        i = 0
        while i < len(doc_hits):
            base = doc_hits[i]
            text = base.chunk.text
            page_end = base.chunk.page_end
            consumed = [base]
            j = i + 1
            while j < len(doc_hits):
                nxt = doc_hits[j]
                close = nxt.chunk.page_start <= page_end + 1
                proposed = text + "\n\n" + nxt.chunk.text
                if not close or len(proposed) > size:
                    break
                text = proposed
                page_end = max(page_end, nxt.chunk.page_end)
                consumed.append(nxt)
                j += 1
            best = min(consumed, key=lambda h: h.rank)
            merged = _derive_chunk(best.chunk, text, f"merge-{i}-{j}", page_end=page_end)
            candidates.append(
                SearchHit(
                    chunk=merged,
                    rank=best.rank,
                    bm25_score=max((h.bm25_score or 0.0) for h in consumed),
                    dense_score=max((h.dense_score or 0.0) for h in consumed),
                    hybrid_score=max((h.hybrid_score or 0.0) for h in consumed),
                    reranker_score=max((h.reranker_score or 0.0) for h in consumed),
                )
            )
            i = max(j, i + 1)

    # Split merged/base candidates according to the selected strategy.
    out: list[SearchHit] = []
    for h in sorted(candidates, key=lambda x: x.rank):
        segments = _split(h.chunk.text, strategy, size, overlap)
        for idx, seg in enumerate(segments[:6]):
            derived = _derive_chunk(h.chunk, seg, f"split-{idx}")
            out.append(
                SearchHit(
                    chunk=derived,
                    rank=len(out) + 1,
                    bm25_score=h.bm25_score,
                    dense_score=h.dense_score,
                    hybrid_score=h.hybrid_score,
                    reranker_score=h.reranker_score,
                )
            )
            if len(out) >= max_output_chunks:
                break
        if len(out) >= max_output_chunks:
            break

    meta = {
        "enabled": True,
        "before": len(hits),
        "after": len(out),
        "strategy": strategy,
        "size": size,
        "overlap": overlap,
        "note": "Query-time context re-chunking; the persistent vector/BM25 index was not rebuilt.",
    }
    return out, meta
