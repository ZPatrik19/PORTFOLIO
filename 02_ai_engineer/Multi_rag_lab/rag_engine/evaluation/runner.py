from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_engine.evaluation.retrieval import (
    average_precision_at_k,
    f1_at_k,
    first_relevant_rank,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_at_k,
    r_precision,
    context_precision_at_k,
)


@dataclass(frozen=True)
class RetrievalEvaluation:
    recall_at_k: float
    precision_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    mrr: float
    map_at_k: float
    ndcg_at_k: float
    first_relevant_rank: float
    reciprocal_rank_at_k: float
    r_precision: float
    context_precision_at_k: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_retrieval(retrieved: list[str], relevant: set[str], k: int = 5) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        recall_at_k=recall_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        f1_at_k=f1_at_k(retrieved, relevant, k),
        hit_rate_at_k=hit_rate_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant),
        map_at_k=average_precision_at_k(retrieved, relevant, k),
        ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
        first_relevant_rank=first_relevant_rank(retrieved, relevant),
        reciprocal_rank_at_k=reciprocal_rank_at_k(retrieved, relevant, k),
        r_precision=r_precision(retrieved, relevant),
        context_precision_at_k=context_precision_at_k(retrieved, relevant, k),
    )
