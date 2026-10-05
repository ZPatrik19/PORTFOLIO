from __future__ import annotations

import re


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def citation_accuracy(answer: str, available_source_count: int) -> float:
    citations = [int(x) for x in _CITATION_PATTERN.findall(answer)]
    if not citations:
        return 0.0
    valid = sum(1 for c in citations if 1 <= c <= available_source_count)
    return valid / len(citations)


def citation_coverage(answer: str) -> float:
    """Fraction of non-empty answer sentences that contain at least one [Sx] citation."""
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(answer) if part.strip()]
    if not sentences:
        return 0.0
    cited = sum(bool(_CITATION_PATTERN.search(sentence)) for sentence in sentences)
    return cited / len(sentences)


def citation_source_coverage(answer: str, available_source_count: int) -> float:
    if available_source_count <= 0:
        return 0.0
    cited = {int(x) for x in _CITATION_PATTERN.findall(answer) if 1 <= int(x) <= available_source_count}
    return len(cited) / available_source_count


def citation_density(answer: str) -> float:
    words = re.findall(r"\w+", answer)
    if not words:
        return 0.0
    return len(_CITATION_PATTERN.findall(answer)) / len(words) * 100.0


def context_utilization(answer: str, context: str) -> float:
    """Lexical proxy, explicitly not a semantic faithfulness judge."""
    answer_terms = set(re.findall(r"\w+", answer.lower()))
    context_terms = set(re.findall(r"\w+", context.lower()))
    if not answer_terms:
        return 0.0
    return len(answer_terms & context_terms) / len(answer_terms)


def answer_completeness(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    lower = answer.lower()
    return sum(1 for term in expected_keywords if term.lower() in lower) / len(expected_keywords)


def answer_token_estimate(answer: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", answer))) if answer.strip() else 0


def answer_redundancy_proxy(answer: str) -> float:
    """Repeated-sentence fraction. 0 is less repetitive, 1 is fully repetitive."""
    sentences = [re.sub(r"\s+", " ", part.strip().lower()) for part in _SENTENCE_SPLIT.split(answer) if part.strip()]
    if len(sentences) <= 1:
        return 0.0
    unique = len(set(sentences))
    return 1.0 - unique / len(sentences)
