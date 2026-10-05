from __future__ import annotations

import re

from rag_engine.retrieval.query_focus import INTENT_STEMS, detect_query_intents, normalize_for_match


ENGLISH_META_PATTERNS = (
    r"\bokay[,!]?\s+(?:let(?:'s| us)|i(?:'ll| will))",
    r"\blet(?:'s| us)\s+(?:tackle|analy[sz]e|figure|break)",
    r"\bthe user\b",
    r"\bfirst[, ]+looking at\b",
    r"\bproblem statement\b",
    r"\bprovided content\b",
    r"\bi need to\b",
    r"\bso[, ]+the user\b",
)

HUNGARIAN_FUNCTION_WORDS = {
    "a", "az", "és", "hogy", "nem", "egy", "is", "ha", "mert", "ami", "amely",
    "vagy", "illetve", "ezt", "azt", "kell", "lehet", "orvos", "betegség", "tünet",
    "kockázat", "kezelés", "vizsgálat", "szükséges", "vérnyomás", "forrás", "alapján",
}
ENGLISH_FUNCTION_WORDS = {
    "the", "and", "that", "this", "with", "from", "user", "first", "then", "because",
    "answer", "provided", "content", "problem", "section", "probably", "want", "need",
    "looking", "going", "here", "let", "let's",
}

GENERIC_FAILURE_PATTERNS = (
    "a lokális modell nem adott",
    "nem adott megbízható magyar",
    "ellenőrizd a visszakeresett",
    "futtasd újra a lekérdezést",
)


def looks_like_english_meta_answer(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    return any(re.search(pattern, lowered) for pattern in ENGLISH_META_PATTERNS)


def is_clearly_english(text: str) -> bool:
    """Conservative language check suitable for Hungarian medical text."""
    words = re.findall(r"[a-záéíóöőúüű']+", text.lower())
    if len(words) < 8:
        return False
    hu = sum(word in HUNGARIAN_FUNCTION_WORDS for word in words)
    en = sum(word in ENGLISH_FUNCTION_WORDS for word in words)
    accented_words = sum(any(ch in word for ch in "áéíóöőúüű") for word in words)
    hu_signal = hu + accented_words * 0.45
    return en >= 4 and en > hu_signal * 1.8


def answer_requires_hungarian_retry(text: str) -> bool:
    if not text.strip():
        return True
    return looks_like_english_meta_answer(text) or is_clearly_english(text)


def answer_requires_quality_retry(query: str, text: str, context: str = "") -> bool:
    """Reject superficially Hungarian but unusable RAG answers.

    The guard is intentionally conservative. It only enforces properties directly implied
    by the user question: enough substance, citations when evidence exists, and coverage of
    explicit query intents such as risk + medical evaluation.
    """
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.casefold()
    if any(pattern in lowered for pattern in GENERIC_FAILURE_PATTERNS):
        return True

    words = re.findall(r"[\wáéíóöőúüű-]+", stripped.lower(), flags=re.UNICODE)
    intents = detect_query_intents(query)
    if len(intents) >= 2:
        # A multi-part question should not collapse into one short bullet for one intent.
        # Tighten the minimum only when the supplied evidence is rich enough to support
        # a longer answer; sparse contexts keep the conservative historical threshold.
        context_words = re.findall(r"[\wáéíóöőúüű-]+", context.lower(), flags=re.UNICODE) if context else []
        min_words = 35
        if len(context_words) >= 180:
            min_words = 90
        if len(context_words) >= 450:
            min_words = 120
        if len(words) < min_words:
            return True
    if context and "[S" in context and not re.search(r"\[S\d+\]", stripped):
        return True

    normalized_answer = normalize_for_match(stripped)
    for intent in intents:
        stems = INTENT_STEMS[intent]
        if not any(stem in normalized_answer for stem in stems):
            return True
    return False
