from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def first_relevant_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return float(rank)
    return 0.0


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            precision_sum += hits / rank
            seen.add(item)
    return precision_sum / min(len(relevant), k) if hits else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(retrieved[:k], start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0

def reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return reciprocal_rank(retrieved[:k], relevant)


def r_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Precision at R, where R is the number of known relevant items."""
    if not relevant:
        return 0.0
    return precision_at_k(retrieved, relevant, len(relevant))


def context_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Ranking-sensitive context precision for binary relevance labels.

    This is an AP-like deterministic retriever metric: relevant chunks receive more
    credit when they occur earlier in the context. It does not use an LLM judge.
    """
    if not relevant or k <= 0:
        return 0.0
    precision_sum = 0.0
    relevant_in_top_k = 0
    seen: set[str] = set()
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant and item not in seen:
            relevant_in_top_k += 1
            precision_sum += relevant_in_top_k / rank
            seen.add(item)
    return precision_sum / relevant_in_top_k if relevant_in_top_k else 0.0


def recall_profile(retrieved: list[str], relevant: set[str], ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict[int, float]:
    return {k: recall_at_k(retrieved, relevant, k) for k in ks if k > 0}

