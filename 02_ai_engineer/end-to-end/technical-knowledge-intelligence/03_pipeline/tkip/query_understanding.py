from __future__ import annotations

from .utils import detect_language


def classify_intent(query: str) -> str:
    """Classify a user query into a deterministic routing intent."""

    normalized = query.lower()
    if any(token in normalized for token in ["compare", "hasonlíts", "különbség"]):
        return "COMPARISON"
    if any(
        token in normalized for token in ["code", "kód", "implement", "dataset example", "példa"]
    ):
        return "CODE_SEARCH"
    if any(
        token in normalized for token in ["teach me", "taníts", "learning plan", "tanulási terv"]
    ):
        return "LEARNING"
    if any(
        token in normalized
        for token in ["which document", "melyik dokument", "hol talál", "where can i find"]
    ):
        return "METADATA_SEARCH"
    if any(token in normalized for token in ["official documentation", "hivatalos dokumentáció"]):
        return "PUBLIC_DOC_COMPARISON"
    if any(token in normalized for token in ["what is", "mi az", "magyarázd", "explain"]):
        return "CONCEPTUAL"
    return "FACTUAL"


def rewrite_query(query: str, intent: str) -> str:
    """Add retrieval hints while preserving the original factual query."""

    if intent == "CODE_SEARCH" and "example" not in query.lower():
        return query + " implementation example code"
    if intent == "LEARNING":
        return query + " intuition concepts mathematics example common mistakes"
    return query


def understand_query(query: str) -> dict[str, str]:
    """Return routing intent, detected language and deterministic rewrite."""

    intent = classify_intent(query)
    return {
        "intent": intent,
        "language": detect_language(" " + query + " "),
        "rewritten_query": rewrite_query(query, intent),
    }
