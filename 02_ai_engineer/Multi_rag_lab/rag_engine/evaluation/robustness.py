from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QueryVariant:
    name: str
    query: str


@dataclass(frozen=True)
class RobustnessObservation:
    variant: str
    query: str
    result_ids: tuple[str, ...]
    jaccard_at_k: float
    prefix_stability_at_k: float
    first_result_retained: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessSummary:
    variants: int
    mean_jaccard_at_k: float
    worst_jaccard_at_k: float
    mean_prefix_stability_at_k: float
    first_result_retention_rate: float
    observations: tuple[RobustnessObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observations"] = [observation.to_dict() for observation in self.observations]
        return payload


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def generate_query_variants(query: str) -> list[QueryVariant]:
    """Generate deterministic, semantics-preserving-ish stress variants.

    These are not paraphrases. They intentionally probe sensitivity to casing,
    punctuation, whitespace and Hungarian diacritics without invoking another LLM.
    """

    compact = re.sub(r"\s+", " ", query).strip()
    candidates = [
        QueryVariant("original", compact),
        QueryVariant("lowercase", compact.lower()),
        QueryVariant("no-punctuation", re.sub(r"[^\w\sáéíóöőúüűÁÉÍÓÖŐÚÜŰ]", " ", compact, flags=re.UNICODE)),
        QueryVariant("accentless", _strip_accents(compact)),
        QueryVariant("extra-whitespace", re.sub(r"\s+", "   ", compact)),
    ]
    unique: list[QueryVariant] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = re.sub(r"\s+", " ", item.query).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(QueryVariant(item.name, normalized))
    return unique


def jaccard_at_k(reference: list[str], candidate: list[str], k: int) -> float:
    left = set(reference[:k])
    right = set(candidate[:k])
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def prefix_rank_stability(reference: list[str], candidate: list[str], k: int) -> float:
    """Average overlap of every ranking prefix from depth 1..K."""

    if k <= 0:
        return 0.0
    scores: list[float] = []
    for depth in range(1, k + 1):
        left = set(reference[:depth])
        right = set(candidate[:depth])
        scores.append(len(left & right) / depth)
    return sum(scores) / len(scores)


def _retrieve_ids(retriever: Any, query: str, *, top_k: int, candidate_count: int) -> list[str]:
    try:
        results = retriever.retrieve(query, top_k=top_k, candidate_count=max(top_k, candidate_count))
    except TypeError:
        results = retriever.retrieve(query, top_k=top_k)
    return [str(getattr(result, "chunk_id", result)) for result in results[:top_k]]


def evaluate_retriever_robustness(
    retriever: Any,
    query: str,
    *,
    top_k: int = 5,
    candidate_count: int = 20,
) -> RobustnessSummary:
    variants = generate_query_variants(query)
    if not variants:
        return RobustnessSummary(0, 0.0, 0.0, 0.0, 0.0, ())

    reference = _retrieve_ids(retriever, variants[0].query, top_k=top_k, candidate_count=candidate_count)
    observations: list[RobustnessObservation] = []
    for variant in variants[1:]:
        result_ids = _retrieve_ids(retriever, variant.query, top_k=top_k, candidate_count=candidate_count)
        observations.append(
            RobustnessObservation(
                variant=variant.name,
                query=variant.query,
                result_ids=tuple(result_ids),
                jaccard_at_k=jaccard_at_k(reference, result_ids, top_k),
                prefix_stability_at_k=prefix_rank_stability(reference, result_ids, top_k),
                first_result_retained=bool(reference and reference[0] in result_ids[:top_k]),
            )
        )

    if not observations:
        return RobustnessSummary(0, 1.0, 1.0, 1.0, 1.0, ())
    jaccards = [item.jaccard_at_k for item in observations]
    prefix = [item.prefix_stability_at_k for item in observations]
    return RobustnessSummary(
        variants=len(observations),
        mean_jaccard_at_k=sum(jaccards) / len(jaccards),
        worst_jaccard_at_k=min(jaccards),
        mean_prefix_stability_at_k=sum(prefix) / len(prefix),
        first_result_retention_rate=sum(item.first_result_retained for item in observations) / len(observations),
        observations=tuple(observations),
    )
