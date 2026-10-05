from __future__ import annotations

from collections import Counter

import streamlit as st

from rag_engine.knowledge import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_ORDER,
    all_reference_entries,
    filter_reference_entries,
    reference_markdown,
    reference_stats,
)
from ui.components.education import kpi_cards, note_box, page_intro, section_intro
from ui.components.knowledge import render_direction_legend, render_entry_grid, render_topic_cards


def _category_entries(entries, category: str):
    return [item for item in entries if item.category == category]


def render() -> None:
    entries = all_reference_entries()
    stats = reference_stats(entries)

    page_intro(
        "Fogalomtár és metrikák",
        "A teljes Multi-RAG Engineering Lab központi tudástára: ingestion, darabolás, embedding, FAISS, retrieval, reranking, grounded RAG, kiértékelési metrikák, teljesítmény, experiment tracking és infrastruktúra – rövid definícióval, értelmezéssel, képlettel és projektkapcsolattal.",
        eyebrow="TUDÁSTÁR · DEFINÍCIÓK · KÉPLETEK · METRIKÁK · STRATÉGIÁK",
    )

    kpi_cards(
        [
            ("Fogalomtári tételek", str(stats["entries"]), "A projektben ténylegesen használt vagy mért fogalmak."),
            ("Metrikák", str(stats["metrics"]), "Retrieval, groundedness, latency és erőforrás-mutatók."),
            ("Képletek", str(stats["formulas"]), "Ahol értelmes, közvetlen matematikai definícióval."),
            ("Stratégiák", str(stats["strategies"]), "Darabolási, retrieval/reranking és RAG stratégiák."),
            ("Témakörök", str(stats["categories"]), "Az end-to-end pipeline logikája szerint csoportosítva."),
        ],
        columns=5,
    )

    render_direction_legend()
    tabs = st.tabs(["Témák", "Keresés", "Metrikák", "Képletek", "Stratégiák és profilok", "Letöltés"])

    with tabs[0]:
        section_intro(
            "Témakörök szerint",
            "Válassz egy pipeline-területet. A definíciók nem alfabetikus listában, hanem az AI Engineering workflow logikája szerint vannak rendezve.",
        )
        render_topic_cards(entries)
        category = st.selectbox("Megnyitott témakör", CATEGORY_ORDER, index=CATEGORY_ORDER.index("Retrieval metrikák"), key="reference_topic")
        st.caption(CATEGORY_DESCRIPTIONS[category])
        render_entry_grid(_category_entries(entries, category), expanded_first=True)

    with tabs[1]:
        section_intro("Kereshető fogalomtár", "Keress névre, magyarázatra, projektoldalra vagy kapcsolódó kifejezésre.")
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            query = st.text_input("Keresés", placeholder="pl. MRR, CUDA, grounded prompt, overlap, latency...", key="reference_search")
        with c2:
            categories = st.multiselect("Témakör", CATEGORY_ORDER, key="reference_categories")
        kinds = sorted({item.kind for item in entries})
        with c3:
            selected_kinds = st.multiselect("Típus", kinds, key="reference_kinds")
        formulas_only = st.toggle("Csak képlettel rendelkező tételek", value=False, key="reference_formula_only")
        filtered = filter_reference_entries(
            entries,
            query=query,
            categories=set(categories) if categories else None,
            kinds=set(selected_kinds) if selected_kinds else None,
            formulas_only=formulas_only,
        )
        st.caption(f"{len(filtered)} találat")
        render_entry_grid(filtered)

    with tabs[2]:
        section_intro(
            "Metrikák egy helyen",
            "A retrieval-, RAG-, teljesítmény- és erőforrás-metrikák együtt. A nyíl az általánosan kedvező irányt jelzi; a kontextusfüggő metrikákat külön jelöljük.",
        )
        metric_entries = [item for item in entries if item.kind in {"Metrika", "Összetett metrika"}]
        metric_categories = [category for category in CATEGORY_ORDER if any(item.category == category for item in metric_entries)]
        selected_metric_category = st.selectbox("Metrikacsoport", metric_categories, key="metric_reference_category")
        render_entry_grid(_category_entries(metric_entries, selected_metric_category), expanded_first=True)
        note_box(
            "Fontos",
            "Az összesített és hatékonysági pontszám projekt-specifikus diagnosztikai aggregátum. A részmetrikák – például Recall@K, MRR, nDCG@K, TTFT vagy Citation accuracy – továbbra is elsődlegesek.",
        )

    with tabs[3]:
        section_intro("Képletgyűjtemény", "A projektben használt fő matematikai definíciók és scoring formulák.")
        formula_entries = [item for item in entries if item.formula]
        formula_group = st.selectbox(
            "Képletek témaköre",
            [category for category in CATEGORY_ORDER if any(item.category == category for item in formula_entries)],
            key="formula_category",
        )
        render_entry_grid(_category_entries(formula_entries, formula_group), expanded_first=True)

    with tabs[4]:
        section_intro(
            "Stratégiák és profilok",
            "A darabolási és RAG stratégiák mellett a context- és promptprofilok is ugyanitt kereshetők, röviden leírva, mikor és miért használjuk őket.",
        )
        strategy_entries = [item for item in entries if item.kind in {"Stratégia", "Profil"}]
        counts = Counter(item.category for item in strategy_entries)
        available = [category for category in CATEGORY_ORDER if counts.get(category, 0)]
        chosen = st.selectbox("Stratégia/profil csoport", available, key="strategy_reference_category")
        render_entry_grid(_category_entries(strategy_entries, chosen), expanded_first=True)

    with tabs[5]:
        section_intro(
            "Letölthető tudástár",
            "A fogalomtár egyetlen Markdown fájlba exportálható, így a repó dokumentációjában vagy interjúfelkészüléshez külön is használható.",
        )
        markdown = reference_markdown(entries)
        st.download_button(
            "Fogalomtár letöltése Markdownként",
            data=markdown.encode("utf-8"),
            file_name="multi_rag_fogalomtar_es_metrikak.md",
            mime="text/markdown",
            width="stretch",
        )
        with st.expander("Markdown előnézet", expanded=False):
            st.code(markdown[:12000], language="markdown")
