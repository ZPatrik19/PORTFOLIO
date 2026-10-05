from __future__ import annotations

from collections import defaultdict

from rag_engine.models import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievedChunk]], *, k: int = 60, top_k: int = 5
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, RetrievedChunk] = {}
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] += 1.0 / (k + rank)
            chunks[item.chunk_id] = item
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    return [
        chunks[cid].model_copy(update={"score": scores[cid], "rank": rank}) for rank, cid in enumerate(ordered, start=1)
    ]


def weighted_fusion(
    dense: list[RetrievedChunk], sparse: list[RetrievedChunk], *, dense_weight: float = 0.5, top_k: int = 5
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, RetrievedChunk] = {}
    for weight, results in ((dense_weight, dense), (1 - dense_weight, sparse)):
        if not results:
            continue
        values = [r.score for r in results]
        lo, hi = min(values), max(values)
        span = hi - lo or 1.0
        for r in results:
            scores[r.chunk_id] += weight * ((r.score - lo) / span)
            chunks[r.chunk_id] = r
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    return [
        chunks[cid].model_copy(update={"score": scores[cid], "rank": rank}) for rank, cid in enumerate(ordered, start=1)
    ]
