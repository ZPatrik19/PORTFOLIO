from __future__ import annotations

from rag_engine.retrieval.context import ContextBuilder
from rag_engine.models import RetrievedChunk
from rag_engine.retrieval.query_focus import QueryIntent, detect_query_intents, rank_chunks_for_query, title_match_score
from rag_engine.retrieval.rerank_lexical import LexicalReranker


QUERY = "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?"


def _chunk(chunk_id: str, title: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=f"https://example.test/{chunk_id}",
        score=1.0 / rank,
        rank=rank,
        metadata={"title": title},
    )


def test_detects_two_requested_medical_intents() -> None:
    intents = detect_query_intents(QUERY)
    assert QueryIntent.RISK in intents
    assert QueryIntent.MEDICAL_EVALUATION in intents
    assert QueryIntent.DIAGNOSIS not in intents


def test_exact_article_title_gets_strong_match() -> None:
    assert title_match_score(QUERY, "Magasvérnyomás-betegség") == 1.0
    assert title_match_score(QUERY, "Vaginizmus") == 0.0


def test_query_focus_prefers_named_article_over_generic_orvosi_sentence() -> None:
    chunks = [
        _chunk("wrong", "Rabdomiolízis", "Azonnali orvosi kivizsgálás szükséges rosszullét esetén.", 1),
        _chunk("right", "Magasvérnyomás-betegség", "A tartósan magas vérnyomás szövődmények kockázatát növeli.", 2),
    ]
    ranked = rank_chunks_for_query(QUERY, chunks, top_k=2)
    assert ranked[0].chunk_id == "right"
    assert ranked[0].metadata["query_title_match"] == 1.0


def test_lexical_reranker_uses_title_and_query_focus() -> None:
    chunks = [
        _chunk("wrong", "Rabdomiolízis", "Orvosi kivizsgálás szükséges és kockázat áll fenn.", 1),
        _chunk("right", "Magasvérnyomás-betegség", "A vérnyomás tartós emelkedése szövődményekhez vezethet.", 2),
    ]
    reranked = LexicalReranker().rerank(QUERY, chunks, top_k=2)
    assert reranked[0].chunk_id == "right"


def test_context_builder_prioritizes_named_article() -> None:
    chunks = [
        _chunk("wrong", "Vaginizmus", "Orvosi vizsgálat javasolt bizonyos panaszok esetén.", 1),
        _chunk(
            "right", "Magasvérnyomás-betegség", "Ismételten magas vérnyomás esetén orvosi kivizsgálás szükséges.", 2
        ),
    ]
    context = ContextBuilder(max_tokens=500).build_for_query(QUERY, chunks, top_k=1)
    assert [item.chunk_id for item in context.included] == ["right"]
    assert "Vaginizmus" not in context.text
