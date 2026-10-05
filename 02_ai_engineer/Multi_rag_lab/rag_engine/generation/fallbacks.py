from __future__ import annotations

import re
from dataclasses import dataclass

from rag_engine.retrieval.query_focus import (
    INTENT_STEMS,
    QueryIntent,
    detect_query_intents,
    normalize_for_match,
    title_match_score,
)


STOPWORDS = {
    "a", "az", "egy", "és", "vagy", "hogy", "ha", "mikor", "mi", "mik", "mely", "melyek",
    "fő", "főbb", "is", "de", "nem", "van", "vannak", "lehet", "kell", "szükséges", "ennek",
    "annak", "betegség", "betegseg", "betegségnek", "illetve", "során", "alapján", "kapcsán",
}

QUERY_EXPANSIONS = {
    "kockázat": {"kockázat", "kockázati", "rizikó", "szövődmény", "veszély", "károsodás"},
    "orvos": {"orvos", "orvosi", "kivizsgálás", "kivizsgálni", "vizsgálat", "fordulni", "ellátás", "háziorvos"},
    "tünet": {"tünet", "panasz", "jel"},
    "kezelés": {"kezelés", "terápia", "gyógyszer"},
}

# Sentence-level matching is intentionally stricter than query intent detection.
# Broad stems such as ``következ`` are useful for recognizing a user's intent,
# but they also match ordinary phrases like ``következtében`` and would promote
# unrelated source sentences inside the safety fallback.
FALLBACK_INTENT_STEMS = {
    **INTENT_STEMS,
    QueryIntent.RISK: (
        "kockaz", "riziko", "veszely", "karos", "kart", "szovod", "sulyos",
        "lappang", "keringesi", "sokaig semmifele tunete", "stroke", "infarkt",
        "elegtelenseg", "aortareped",
    ),
    QueryIntent.MEDICAL_EVALUATION: (
        "orvos", "kivizsg", "ellatas", "surgos", "mento", "forduljon", "haziorvos",
        "szakorvos", "112", "180 120", "meghaladja",
    ),
}

BOILERPLATE_MARKERS = (
    "kapcsolódó tartalmak",
    "készült az efop",
    "az oldalt működteti",
    "minden jog fenntartva",
    "együttműködő partner",
)


@dataclass(frozen=True)
class EvidenceBlock:
    citation: str
    title: str
    section: str
    source: str
    content: str


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-záéíóöőúüű0-9-]+", text.lower())) - STOPWORDS
    expanded = set(words)
    for trigger, additions in QUERY_EXPANSIONS.items():
        if any(trigger in word for word in words):
            expanded.update(additions)
    return expanded


def _parse_field(raw: str, field: str) -> str:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(.*)$", raw)
    return match.group(1).strip() if match else ""


def _evidence_blocks(context: str) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for raw in re.split(r"(?=\[S\d+\])", context):
        raw = raw.strip()
        if not raw:
            continue
        match = re.match(r"(\[S\d+\])", raw)
        if not match:
            continue
        citation = match.group(1)
        content = raw.split("Tartalom:\n", 1)[-1].strip()
        blocks.append(
            EvidenceBlock(
                citation=citation,
                title=_parse_field(raw, "Cím"),
                section=_parse_field(raw, "Szakasz"),
                source=_parse_field(raw, "Forrás"),
                content=content,
            )
        )
    return blocks


def _sentence_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    # HTML parsing can place inline links/strong tags on separate lines.  Treat
    # those line breaks as whitespace before sentence segmentation, otherwise a
    # perfectly valid sentence may become fragments such as ``(hipotónia) ...``.
    # Preserve a sentence boundary around Markdown headings before collapsing
    # line breaks. Without this, a table row immediately followed by a heading
    # can merge into the first prose sentence under that heading.
    text = re.sub(r"\n\s*#{1,6}\s+[^\n]+\s*\n", "\n.\n", text)
    normalized = re.sub(r"\s*\n+\s*", " ", text).strip()
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        sentence = re.sub(r"^#{1,6}\s*", "", sentence.strip(" -•\t"))
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = re.sub(r"\s+([.,;:!?])", r"\1", sentence)
        sentence = re.sub(r"\(\s+", "(", sentence)
        sentence = re.sub(r"\s+\)", ")", sentence)
        if len(sentence) < 35 or len(sentence.split()) < 5:
            continue
        if sentence.count("(") > sentence.count(")"):
            continue
        if sentence.startswith(("Forrás:", "Cím:", "Oldal:", "Szakasz:")):
            continue
        lowered = sentence.casefold()
        if any(marker in lowered for marker in BOILERPLATE_MARKERS):
            continue
        # Safety fallback must never expose a visibly truncated source fragment.
        if not sentence.endswith((".", "!", "?")):
            continue
        # Drop obvious fragments such as "orvosi vizsgálat esetén)."
        if re.match(r"^[a-záéíóöőúüű].{0,35}\)\.?$", sentence):
            continue
        candidates.append(sentence)
    return candidates


def _intent_match_count(sentence: str, intent: QueryIntent) -> int:
    normalized = normalize_for_match(sentence)
    return sum(1 for stem in FALLBACK_INTENT_STEMS[intent] if stem in normalized)


def _contains_intent(sentence: str, intent: QueryIntent) -> bool:
    return _intent_match_count(sentence, intent) > 0


def _intent_action_bonus(sentence: str, intent: QueryIntent | None) -> float:
    if intent is None:
        return 0.0
    normalized = normalize_for_match(sentence)
    if intent == QueryIntent.MEDICAL_EVALUATION:
        actionable = ("fordul", "surgos", "mento", "112", "kivizsgalasra van szukseg", "meghaladja")
        return 2.4 if any(term in normalized for term in actionable) else 0.0
    if intent == QueryIntent.RISK:
        direct = ("kockaz", "karos", "szovod", "sulyos", "veszely")
        return 1.6 if any(term in normalized for term in direct) else 0.0
    return 0.0


def _intent_heading(intent: QueryIntent) -> str:
    return {
        QueryIntent.RISK: "Fő kockázatok",
        QueryIntent.MEDICAL_EVALUATION: "Mikor indokolt orvosi kivizsgálás?",
        QueryIntent.SYMPTOMS: "Jellemző tünetek",
        QueryIntent.CAUSES: "Lehetséges okok",
        QueryIntent.DIAGNOSIS: "Kivizsgálás és diagnózis",
        QueryIntent.TREATMENT: "Kezelés",
        QueryIntent.PREVENTION: "Megelőzés",
        QueryIntent.COMPLICATIONS: "Lehetséges szövődmények",
    }[intent]


def _preferred_blocks(query: str, blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
    if not blocks:
        return []
    title_scores = [(title_match_score(query, block.title), block) for block in blocks]
    strong = [block for score, block in title_scores if score >= 0.72]
    # If the user explicitly named an article/topic, do not silently answer from a different
    # disease just because it contains a generic phrase like "orvosi vizsgálat".
    return strong or blocks


def _focus_contrast_clause(sentence: str, block: EvidenceBlock) -> str:
    """Keep the title-relevant side of an explicit contrast when it is self-contained."""
    parts = re.split(r",\s+(?:ám|azonban|viszont)\s+", sentence, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2 or not block.title:
        return sentence
    left, right = parts
    title_tokens = [token for token in normalize_for_match(block.title).split() if len(token) >= 5]
    left_compact = normalize_for_match(left).replace(" ", "")
    right_compact = normalize_for_match(right).replace(" ", "")
    right_matches = any(token.replace(" ", "") in right_compact for token in title_tokens)
    left_matches = any(token.replace(" ", "") in left_compact for token in title_tokens)
    if right_matches and not left_matches:
        focused = right.strip()
        return focused[:1].upper() + focused[1:] if focused else sentence
    return sentence


def _score_sentence(query: str, block: EvidenceBlock, sentence: str, intent: QueryIntent | None) -> float:
    query_tokens = _tokens(query)
    sentence_tokens = _tokens(sentence)
    overlap = len(query_tokens & sentence_tokens) / max(1.0, len(query_tokens) ** 0.5)
    title_boost = 3.5 * title_match_score(query, block.title)
    section_text = f"{block.section} {sentence}"
    intent_hits = _intent_match_count(section_text, intent) if intent is not None else 0
    # Intent matching is a gate/boost, not a term-frequency score: overlapping
    # stems such as ``orvos`` and ``szakorvos`` must not overpower a much more
    # actionable sentence that better matches the user's wording.
    intent_boost = 2.2 if intent_hits else 0.0
    section_overlap = len(query_tokens & _tokens(block.section)) * 0.35
    completeness = 0.25 if sentence.endswith((".", "!", "?")) else 0.0
    return overlap + title_boost + intent_boost + _intent_action_bonus(sentence, intent) + section_overlap + completeness


def _select_sentences(
    query: str,
    blocks: list[EvidenceBlock],
    *,
    intent: QueryIntent | None,
    limit: int,
) -> list[tuple[str, str]]:
    strict: list[tuple[float, int, str, str]] = []
    weak: list[tuple[float, int, str, str]] = []
    order = 0
    for block in blocks:
        for sentence in _sentence_candidates(block.content):
            sentence = _focus_contrast_clause(sentence, block)
            intent_match = intent is None or _contains_intent(f"{block.section} {sentence}", intent)
            lexical_overlap = len(_tokens(query) & _tokens(sentence))
            if not intent_match and lexical_overlap < 2:
                continue
            score = _score_sentence(query, block, sentence, intent)
            if score <= 0:
                continue
            target = strict if intent_match else weak
            target.append((score, -order, sentence, block.citation))
            order += 1

    # If direct evidence for the requested subquestion exists, do not dilute it with
    # generic sentences merely because they share the disease name.
    scored = strict or weak
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, _, sentence, citation in sorted(scored, reverse=True):
        fingerprint = normalize_for_match(sentence)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append((sentence, citation))
        if len(selected) >= limit:
            break
    return selected


def build_evidence_fallback_answer(query: str, context: str, *, max_sentences: int = 10) -> str:
    """Create a readable Hungarian source-grounded answer without inventing facts.

    This is the final safety path when the local LLM still produces unusable output after
    repair. It is query-aware, title-aware and intent-aware, so it does not simply dump the
    lexically closest sentences from unrelated medical articles.
    """
    all_blocks = _evidence_blocks(context)
    blocks = _preferred_blocks(query, all_blocks)
    if not blocks:
        return "A rendelkezésre álló dokumentumok nem tartalmaznak elegendő információt a kérdés megválaszolásához."

    intents = detect_query_intents(query)
    # Keep the response focused: when the question has explicit subquestions, answer each
    # under its own heading instead of producing a generic evidence dump.
    sections: list[tuple[QueryIntent, list[tuple[str, str]]]] = []
    per_intent = max(2, min(5, max_sentences // max(1, len(intents))))
    for intent in intents:
        selected = _select_sentences(query, blocks, intent=intent, limit=per_intent)
        if selected:
            sections.append((intent, selected))

    if sections:
        lines: list[str] = []
        for intent, selected in sections:
            lines.append(f"### {_intent_heading(intent)}")
            # Keep the final safety path deterministic and source-grounded, but render
            # several complete evidence sentences as a readable paragraph instead of an
            # ultra-short quote list. This preserves fidelity while giving the user enough
            # context when the local LLM/repair path cannot be trusted.
            paragraph = " ".join(f"{sentence} {citation}" for sentence, citation in selected)
            lines.append(paragraph)
            lines.append("")
        return "\n".join(lines).strip()

    selected = _select_sentences(query, blocks, intent=None, limit=max_sentences)
    if not selected:
        return "A rendelkezésre álló dokumentumok nem tartalmaznak elegendő, közvetlenül releváns információt a kérdés megválaszolásához."

    lines = ["### A források alapján"]
    lines.append(" ".join(f"{sentence} {citation}" for sentence, citation in selected))
    return "\n".join(lines)
