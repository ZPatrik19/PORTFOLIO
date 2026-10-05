from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, Field

from rag_engine.models import Chunk


class MedicalEvaluationItem(BaseModel):
    id: str
    query: str
    question_type: str
    article_title: str
    source_id: str
    source_url: str
    section: str
    evidence_text: str
    expected_key_facts: list[str] = Field(default_factory=list)
    language: str = "hu"


class MedicalEvaluationDataset(BaseModel):
    version: str = "1.0"
    corpus_id: str
    source_name: str
    source_documents: int
    requested_questions: int
    items: list[MedicalEvaluationItem]
    methodology: str = (
        "Extractive, source-grounded evaluation dataset. Questions and expected key facts are "
        "derived from locally downloaded source articles; no medical facts are invented by the builder."
    )


@dataclass(frozen=True)
class _SectionCandidate:
    article_title: str
    source_id: str
    source_url: str
    section: str
    question_type: str
    text: str


QUESTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("symptoms", ("tünet", "panasz")),
    ("causes", ("okai", "oka", "kialakul", "kockázati", "rizikó")),
    ("diagnosis", ("diagnózis", "diagnoszt", "felismer", "vizsgálat")),
    ("treatment", ("kezelés", "kezelése", "terápia", "gyógykezel")),
    ("prevention", ("megelőzés", "megelőzhető", "prevenció")),
    ("doctor", ("mikor forduljon orvoshoz", "orvoshoz fordul", "sürgősség")),
    ("complications", ("szövődmény", "következmény")),
)

QUESTION_TEMPLATES = {
    "overview": "Mit érdemes tudni a(z) {title} témáról a forráscikk alapján?",
    "symptoms": "Melyek a(z) {title} jellemző tünetei vagy panaszai?",
    "causes": "Milyen okok vagy kockázati tényezők kapcsolódnak a(z) {title} kialakulásához?",
    "diagnosis": "Hogyan történik a(z) {title} felismerése vagy diagnosztikája?",
    "treatment": "Hogyan kezelhető a(z) {title}?",
    "prevention": "Hogyan előzhető meg vagy csökkenthető a(z) {title} kockázata?",
    "doctor": "Mikor indokolt orvoshoz fordulni a(z) {title} kapcsán?",
    "complications": "Milyen szövődmények vagy következmények kapcsolódhatnak a(z) {title} állapothoz?",
}

HU_STOPWORDS = {
    "a", "az", "és", "hogy", "ha", "de", "vagy", "is", "egy", "egyik", "mely", "milyen",
    "mi", "mit", "mikor", "hogyan", "ez", "ezt", "ezen", "arra", "annak", "ami", "amely",
    "mint", "már", "még", "nem", "igen", "van", "volt", "lehet", "kell", "illetve", "kapcsán",
    "során", "esetén", "alapján", "között", "után", "előtt", "nélkül", "továbbá", "által",
}


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wáéíóöőúüű]+", _normalise(text), flags=re.UNICODE)
        if len(token) > 2 and token not in HU_STOPWORDS
    }


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    raw = re.split(r"(?<=[.!?])\s+", compact)
    return [sentence.strip() for sentence in raw if 35 <= len(sentence.strip()) <= 360]


def _clean_article_dom(path: Path) -> tuple[BeautifulSoup, Tag]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return soup, root


def _heading_text(heading: Tag) -> str:
    return re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()


def _section_text(heading: Tag) -> str:
    pieces: list[str] = []
    for node in heading.next_elements:
        if node is heading:
            continue
        if isinstance(node, Tag) and node.name in {"h1", "h2", "h3"}:
            break
        if isinstance(node, NavigableString):
            value = re.sub(r"\s+", " ", str(node)).strip()
            if value and value not in pieces:
                pieces.append(value)
    return " ".join(pieces).strip()


def _overview_text(root: Tag) -> str:
    paragraphs: list[str] = []
    for tag in root.find_all(["p", "li"], limit=16):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if len(text) >= 50:
            paragraphs.append(text)
        if sum(len(item) for item in paragraphs) >= 1400:
            break
    return " ".join(paragraphs)


def _question_type(section_name: str) -> str | None:
    lowered = _normalise(section_name)
    for question_type, patterns in QUESTION_RULES:
        if any(pattern in lowered for pattern in patterns):
            return question_type
    return None


def _facts(text: str, max_facts: int = 4) -> list[str]:
    facts = _sentences(text)
    if not facts:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) >= 45:
            return [compact[:320]]
    return facts[:max_facts]


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Az orvosi korpusz manifest nem található: {manifest_path}. Előbb futtasd a medical corpus letöltést."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _candidate_sections(raw_dir: Path, manifest: dict[str, object]) -> list[_SectionCandidate]:
    candidates: list[_SectionCandidate] = []
    documents = manifest.get("documents", [])
    if not isinstance(documents, list):
        return candidates
    for record in documents:
        if not isinstance(record, dict):
            continue
        path = raw_dir / str(record.get("filename", ""))
        if not path.exists() or path.suffix.lower() not in {".html", ".htm"}:
            continue
        article_title = str(record.get("title") or path.stem)
        source_id = str(record.get("source_id") or "")
        source_url = str(record.get("url") or "")
        _, root = _clean_article_dom(path)

        overview = _overview_text(root)
        if len(overview) < 100:
            overview = re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()[:2400]
        if len(overview) >= 100:
            candidates.append(
                _SectionCandidate(article_title, source_id, source_url, "Áttekintés", "overview", overview)
            )

        for heading in root.find_all(["h2", "h3"]):
            section_name = _heading_text(heading)
            question_type = _question_type(section_name)
            if question_type is None:
                continue
            text = _section_text(heading)
            if len(text) < 90:
                continue
            candidates.append(
                _SectionCandidate(article_title, source_id, source_url, section_name, question_type, text)
            )
    return candidates


def _select_balanced(candidates: list[_SectionCandidate], target_questions: int) -> list[_SectionCandidate]:
    if target_questions <= 0:
        return []
    by_type: dict[str, deque[_SectionCandidate]] = defaultdict(deque)
    seen_keys: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.source_id, candidate.question_type, candidate.section.casefold())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        by_type[candidate.question_type].append(candidate)

    selected: list[_SectionCandidate] = []
    used_articles: set[str] = set()
    type_order = ["symptoms", "causes", "diagnosis", "treatment", "prevention", "doctor", "complications", "overview"]

    # First pass: maximise article diversity.
    while len(selected) < target_questions:
        progress = False
        for question_type in type_order:
            queue = by_type.get(question_type, deque())
            deferred: deque[_SectionCandidate] = deque()
            chosen = None
            while queue:
                candidate = queue.popleft()
                if candidate.source_id not in used_articles:
                    chosen = candidate
                    break
                deferred.append(candidate)
            queue.extend(deferred)
            if chosen is not None:
                selected.append(chosen)
                used_articles.add(chosen.source_id)
                progress = True
                if len(selected) >= target_questions:
                    break
        if not progress:
            break

    # Second pass: if the requested dataset is larger than source diversity, allow a second question/article.
    if len(selected) < target_questions:
        for question_type in type_order:
            queue = by_type.get(question_type, deque())
            while queue and len(selected) < target_questions:
                selected.append(queue.popleft())

    return selected[:target_questions]


def build_medical_evaluation_dataset(
    *,
    raw_dir: Path,
    manifest_path: Path,
    output_path: Path,
    target_questions: int = 80,
) -> MedicalEvaluationDataset:
    manifest = _load_manifest(manifest_path)
    candidates = _candidate_sections(raw_dir, manifest)
    selected = _select_balanced(candidates, target_questions)
    if len(selected) < target_questions:
        raise RuntimeError(
            f"Csak {len(selected)} forrásolt evaluation kérdés készíthető a kért {target_questions} helyett. "
            "Tölts le több vagy teljesebb cikket, vagy csökkentsd a target_questions értéket."
        )

    items: list[MedicalEvaluationItem] = []
    for index, candidate in enumerate(selected, start=1):
        facts = _facts(candidate.text)
        if not facts:
            continue
        template = QUESTION_TEMPLATES[candidate.question_type]
        items.append(
            MedicalEvaluationItem(
                id=f"medical-eval-{index:03d}",
                query=template.format(title=candidate.article_title),
                question_type=candidate.question_type,
                article_title=candidate.article_title,
                source_id=candidate.source_id,
                source_url=candidate.source_url,
                section=candidate.section,
                evidence_text=candidate.text[:2400],
                expected_key_facts=facts,
            )
        )

    dataset = MedicalEvaluationDataset(
        corpus_id=str(manifest.get("corpus_id", "hungarian_medical")),
        source_name=str(manifest.get("display_name", "Egészségvonal Egészség A–Z")),
        source_documents=int(manifest.get("available_documents", 0) or 0),
        requested_questions=target_questions,
        items=items,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(item.model_dump(), ensure_ascii=False) for item in dataset.items) + "\n",
        encoding="utf-8",
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "version": dataset.version,
                "corpus_id": dataset.corpus_id,
                "source_name": dataset.source_name,
                "source_documents": dataset.source_documents,
                "questions": len(dataset.items),
                "question_types": dict(_count_types(dataset.items)),
                "methodology": dataset.methodology,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset


def _count_types(items: Iterable[MedicalEvaluationItem]) -> Iterable[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.question_type] += 1
    return sorted(counts.items())


def load_medical_evaluation_dataset(path: Path) -> list[MedicalEvaluationItem]:
    if not path.exists():
        return []
    items: list[MedicalEvaluationItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(MedicalEvaluationItem.model_validate_json(line))
    return items


def resolve_relevant_chunk_ids(
    item: MedicalEvaluationItem,
    chunks: list[Chunk],
    *,
    max_relevant: int = 4,
    min_overlap: float = 0.10,
) -> set[str]:
    evidence_terms = _tokens(" ".join(item.expected_key_facts) or item.evidence_text)
    if not evidence_terms:
        evidence_terms = _tokens(item.evidence_text)

    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        source_id = str(chunk.metadata.get("source_id") or "")
        source_url = str(chunk.metadata.get("source") or "")
        if item.source_id and source_id != item.source_id and source_url != item.source_url:
            continue
        chunk_terms = _tokens(chunk.text)
        if not chunk_terms:
            continue
        recall_overlap = len(evidence_terms & chunk_terms) / max(1, len(evidence_terms))
        precision_overlap = len(evidence_terms & chunk_terms) / max(1, len(chunk_terms))
        score = 0.75 * recall_overlap + 0.25 * precision_overlap
        scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    relevant = {chunk.chunk_id for score, chunk in scored[:max_relevant] if score >= min_overlap}
    if not relevant and scored:
        relevant.add(scored[0][1].chunk_id)
    return relevant


def key_fact_coverage(answer: str, facts: list[str], *, threshold: float = 0.30) -> float:
    if not facts:
        return 0.0
    answer_terms = _tokens(answer)
    covered = 0
    for fact in facts:
        fact_terms = _tokens(fact)
        if not fact_terms:
            continue
        overlap = len(answer_terms & fact_terms) / len(fact_terms)
        if overlap >= threshold:
            covered += 1
    return covered / len(facts)
