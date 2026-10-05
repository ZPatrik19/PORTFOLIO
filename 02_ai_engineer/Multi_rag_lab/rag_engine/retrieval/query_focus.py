from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class QueryIntent(StrEnum):
    RISK = "risk"
    MEDICAL_EVALUATION = "medical_evaluation"
    SYMPTOMS = "symptoms"
    CAUSES = "causes"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    COMPLICATIONS = "complications"


STOPWORDS = {
    "a",
    "az",
    "egy",
    "es",
    "vagy",
    "hogy",
    "ha",
    "mikor",
    "mi",
    "mik",
    "mely",
    "melyek",
    "fo",
    "fobb",
    "is",
    "de",
    "nem",
    "van",
    "vannak",
    "lehet",
    "kell",
    "szukseges",
    "ennek",
    "annak",
    "illetve",
    "soran",
    "alapjan",
    "kapcsan",
    "eseten",
    "olyan",
    "ami",
    "amely",
    "mit",
    "hogyan",
    "milyen",
    "melyik",
    "valamint",
    "vagyis",
    "adott",
    "szerint",
    "tudni",
    "erdemes",
}

INTENT_STEMS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.RISK: ("kockaz", "riziko", "veszely", "hajlamos", "kovetkez"),
    QueryIntent.MEDICAL_EVALUATION: (
        "orvos",
        "kivizsg",
        "vizsgal",
        "ellatas",
        "surgos",
        "mento",
        "szakrendel",
        "haziorvos",
    ),
    QueryIntent.SYMPTOMS: ("tunet", "panasz", "jel", "eszlel"),
    QueryIntent.CAUSES: ("ok", "kivalto", "eredet", "hatter"),
    QueryIntent.DIAGNOSIS: ("diagnoz", "diagnoszt", "felismer", "kimutat"),
    QueryIntent.TREATMENT: ("kezel", "terap", "gyogyszer", "mutet"),
    QueryIntent.PREVENTION: ("megeloz", "prevenc", "elkerul"),
    QueryIntent.COMPLICATIONS: ("szovod", "kovetkez", "karosod"),
}


@dataclass(frozen=True)
class ChunkFocus:
    score: float
    title_score: float
    section_score: float
    text_score: float
    intent_hits: int


def normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in normalize_for_match(text).split() if token and token not in STOPWORDS]


def _stem(token: str) -> str:
    # Lightweight Hungarian-friendly matching. We deliberately avoid pretending this is
    # a linguistic stemmer; the prefix only makes common suffix variants comparable.
    if len(token) >= 9:
        return token[:7]
    if len(token) >= 7:
        return token[:6]
    if len(token) >= 5:
        return token[:5]
    return token


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in _tokens(text)}


def _overlap(left: str, right: str) -> float:
    a, b = _stems(left), _stems(right)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def detect_query_intents(query: str) -> list[QueryIntent]:
    normalized = normalize_for_match(query)
    intents: list[QueryIntent] = []
    for intent, stems in INTENT_STEMS.items():
        if any(stem in normalized for stem in stems):
            intents.append(intent)
    return intents


def title_match_score(query: str, title: str) -> float:
    q = normalize_for_match(query)
    t = normalize_for_match(title)
    if not t:
        return 0.0
    if t in q:
        return 1.0
    title_tokens = _stems(t)
    query_tokens = _stems(q)
    if not title_tokens:
        return 0.0
    coverage = len(title_tokens & query_tokens) / len(title_tokens)
    # Require a meaningful amount of title coverage before calling it a strong match.
    return min(1.0, coverage)


def _intent_hits(query: str, candidate: str) -> int:
    candidate_n = normalize_for_match(candidate)
    hits = 0
    for intent in detect_query_intents(query):
        if any(stem in candidate_n for stem in INTENT_STEMS[intent]):
            hits += 1
    return hits


def chunk_focus_score(query: str, chunk) -> ChunkFocus:
    metadata = getattr(chunk, "metadata", {}) or {}
    title = str(metadata.get("title") or "")
    section = str(metadata.get("section") or "")
    text = str(getattr(chunk, "text", "") or "")

    title_score = title_match_score(query, title)
    section_score = _overlap(query, section)
    text_score = _overlap(query, text)
    intent_hits = _intent_hits(query, f"{section}\n{text}")

    # Title match is intentionally strong for article-centric questions. It prevents a
    # generic sentence containing "orvosi vizsgálat" from outranking the requested disease.
    score = 5.0 * title_score + 1.8 * section_score + 1.0 * text_score + 0.55 * intent_hits
    return ChunkFocus(score, title_score, section_score, text_score, intent_hits)


def rank_chunks_for_query(query: str, chunks: Iterable, *, top_k: int | None = None) -> list:
    items = list(chunks)
    if not items:
        return []

    focused = [(chunk_focus_score(query, item), item) for item in items]
    strong_titles = [focus.title_score for focus, _ in focused if focus.title_score >= 0.72]
    has_strong_title = bool(strong_titles)

    def sort_key(pair):
        focus, item = pair
        original_rank = int(getattr(item, "rank", 999999) or 999999)
        # If an article title clearly matches the query, prefer that article before
        # generic cross-document lexical matches.
        preferred = 1 if has_strong_title and focus.title_score >= 0.72 else 0
        return (preferred, focus.score, -original_rank)

    ordered = sorted(focused, key=sort_key, reverse=True)
    if has_strong_title:
        preferred = [pair for pair in ordered if pair[0].title_score >= 0.72]
        others = [pair for pair in ordered if pair[0].title_score < 0.72]
        # Keep the exact-topic evidence first. A small number of other chunks can remain
        # as supporting evidence if there are not enough topic chunks.
        minimum_topic = min(len(preferred), max(2, top_k or len(items)))
        ordered = preferred[:minimum_topic] + others

    if top_k is not None:
        ordered = ordered[:top_k]

    ranked = []
    for rank, (focus, item) in enumerate(ordered, start=1):
        metadata = dict(getattr(item, "metadata", {}) or {})
        metadata["query_focus_score"] = round(float(focus.score), 6)
        metadata["query_title_match"] = round(float(focus.title_score), 6)
        if hasattr(item, "model_copy"):
            ranked.append(item.model_copy(update={"rank": rank, "metadata": metadata}))
        else:
            ranked.append(item)
    return ranked
