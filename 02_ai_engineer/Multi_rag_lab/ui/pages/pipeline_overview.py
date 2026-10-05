from __future__ import annotations

import html

import streamlit as st

from rag_engine.platform.config import load_settings
from rag_engine.presets import CHUNKING_STRATEGIES, RAG_STRATEGIES
from ui.components.education import kpi_cards, note_box, page_intro, section_intro


def _stage_card(title: str, subtitle: str, items: list[str], *, accent: str) -> None:
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    st.markdown(
        f"""
<div style="border:1px solid rgba(148,163,184,.24);border-top:3px solid {accent};border-radius:14px;
            padding:14px 15px 12px 15px;min-height:190px;background:rgba(15,23,42,.18);">
  <div style="font-size:1.02rem;font-weight:700;margin-bottom:.2rem">{html.escape(title)}</div>
  <div style="font-size:.88rem;opacity:.78;margin-bottom:.65rem">{html.escape(subtitle)}</div>
  <ul style="margin:.2rem 0 0 1.1rem;padding:0;line-height:1.55;font-size:.88rem">{bullets}</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def _tier(title: str, description: str, stages: list[tuple[str, str, list[str]]], *, accent: str) -> None:
    section_intro(title, description)
    for start in range(0, len(stages), 4):
        cols = st.columns(min(4, len(stages) - start))
        for col, (stage_title, subtitle, items) in zip(cols, stages[start : start + 4], strict=False):
            with col:
                _stage_card(stage_title, subtitle, items, accent=accent)
        if start + 4 < len(stages):
            st.markdown(
                '<div style="text-align:center;font-size:1.4rem;opacity:.55;margin:.1rem 0 .45rem 0">↓</div>',
                unsafe_allow_html=True,
            )


def render() -> None:
    page_intro(
        "RAG pipeline térkép",
        "A projekt teljes adatútja három mérnöki szinten: offline indexelés, online retrieval + grounded generation, majd evaluation és üzemeltetés. A kártyák a ténylegesen implementált stratégiákat mutatják.",
        eyebrow="3-TIER ENGINEERING PIPELINE · INGESTION → RETRIEVAL → GENERATION → EVALUATION",
    )

    settings = load_settings()
    kpi_cards(
        [
            ("Chunking", f"{len(CHUNKING_STRATEGIES)} stratégia", "Fixed, recursive, semantic, parent-child és további variánsok."),
            ("RAG", f"{len(RAG_STRATEGIES)} stratégia", "Baseline, hybrid, reranked, HyDE, multi-query, multi-hop stb."),
            ("Embedding", st.session_state.get("embedding_mode", "e5-small"), f"Device: {st.session_state.get('embedding_device', 'auto')}."),
            ("Aktív RAG", st.session_state.get("rag_strategy", "hybrid"), f"LLM: {st.session_state.get('llm_provider', settings.llm_provider)}."),
        ],
        columns=4,
    )

    _tier(
        "1. Offline / Indexing pipeline",
        "A drága dokumentumfeldolgozási műveletek egyszer futnak, majd a perzisztens index újrahasznosítható az interaktív RAG-ban.",
        [
            ("Dokumentumforrás", "HTML · PDF · DOCX · TXT · Markdown", ["forrás URL + metadata", "SHA-256 / provenance", "100 cikkes magyar orvosi korpusz"]),
            ("Parsing + Cleaning", "nyers fájlból tiszta dokumentum", ["HTML/PDF/DOCX parser", "navigation/footer zajszűrés", "Unicode és whitespace normalizálás"]),
            ("Chunking", "8 választható stratégia", ["fixed / fixed-token / recursive", "sentence / paragraph / structure-aware", "semantic / parent-child"]),
            ("Embedding", "szöveg → numerikus reprezentáció", ["Multilingual E5 / MiniLM", "CPU vagy CUDA", "batch + local HF cache"]),
            ("Vector index", "perzisztens nearest-neighbour index", ["FAISS CPU", "FAISS GPU Linux/WSL esetén", "NumPy exact fallback"]),
            ("Persistált asset", "újraindítás után is használható", ["chunks JSON", "FAISS index", "index fingerprint + build state"]),
        ],
        accent="#38bdf8",
    )

    st.markdown('<div style="text-align:center;font-size:1.6rem;opacity:.6;margin:.55rem 0">↓</div>', unsafe_allow_html=True)

    _tier(
        "2. Online / Query + Answering pipeline",
        "A kérdés futásidejű útvonala. Itt történik a retrieval, opcionális reranking, context engineering és az Ollama/Qwen grounded válaszgenerálás.",
        [
            ("Query processing", "keresésbarát kérdés", ["normalizálás", "query rewrite", "multi-query / HyDE / multi-hop"]),
            ("Retrieval", "candidate evidence", ["dense vector search", "BM25 sparse search", "hybrid RRF / weighted fusion"]),
            ("Reranking", "candidate → jobb sorrend", ["none", "lexical CPU", "Cross-Encoder CPU/CUDA"]),
            ("Context building", "Top-K evidence → LLM context", ["deduplikáció", "source diversity", "token budget + context profile"]),
            ("Grounded generation", "evidence-alapú válasz", ["Ollama / Qwen", "prompt profile", "repair + deterministic fallback"]),
            ("Citations", "ellenőrizhető állítások", ["[S1], [S2]…", "citation coverage", "source binding + evidence trace"]),
        ],
        accent="#a78bfa",
    )

    st.markdown('<div style="text-align:center;font-size:1.6rem;opacity:.6;margin:.55rem 0">↓</div>', unsafe_allow_html=True)

    _tier(
        "3. Evaluation / Performance / Operations",
        "A pipeline nem csak válaszol: quality, latency, robustness és hardver-trade-off szinten is mérhető és összehasonlítható.",
        [
            ("Retrieval quality", "rangsorolási minőség", ["Recall@K / Precision@K / F1@K", "MRR / MAP / nDCG", "no-hit / late-hit / diversity"]),
            ("Generation quality", "grounded válaszminőség", ["citation accuracy / coverage", "key-fact coverage", "context utilization / fallback rate"]),
            ("Performance", "serving és komponens latency", ["P50 / P95 / P99", "TTFT / token/s / QPS", "CPU vs CUDA"]),
            ("Robustness", "stabilitás kérdésváltozatokra", ["query perturbation", "Top-K overlap", "ranking stability"]),
            ("Experiment registry", "reprodukálható összehasonlítás", ["run_id + config hash", "matrix results", "exportálható JSON/SQLite"]),
            ("CI / Runtime", "üzemeltetési guardrail", ["unit + integration + UI smoke", "repository contract", "CUDA self-hosted validation"]),
        ],
        accent="#f59e0b",
    )

    note_box(
        "Hogyan olvasd a pipeline-t?",
        "Az offline szintet csak akkor kell újraépíteni, ha a dokumentumok, chunking, embedding modell vagy vector index konfiguráció változik. A query/retrieval/generation beállítások többsége ezután már a perzisztens indexre épül. A benchmark oldalak ugyanezeket a rétegeket külön és end-to-end módon is mérik.",
    )
