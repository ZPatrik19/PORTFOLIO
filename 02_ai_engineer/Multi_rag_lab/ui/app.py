from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ui.components.common import ensure_state, sidebar_runtime_controls
from ui.components.theme import apply_theme
from ui.pages import (
    chunking_lab,
    documents,
    embedding_lab,
    evaluation,
    experiments,
    guide,
    knowledge_base,
    measurement_lab,
    overview,
    pipeline_overview,
    performance_lab,
    pipeline_matrix,
    rag_comparison,
    rag_playground,
    runtime_lab,
    vector_search,
)

st.set_page_config(
    page_title="Multi-RAG Engineering Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
ensure_state()

# V21: primary navigation reflects the actual engineering workflow instead of exposing
# every implementation page as one long flat menu.
NAV_GROUPS = {
    "Áttekintés": [
        ("Irányítópult", "overview", overview.render, None),
        ("Pipeline térkép", "pipeline_overview", pipeline_overview.render, None),
        ("Használati útmutató", "guide", guide.render, None),
    ],
    "Tudástár": [
        ("Fogalomtár és metrikák", "knowledge_base", knowledge_base.render, None),
    ],
    "Építés": [
        ("Dokumentumok", "documents", documents.render, "data"),
        ("Darabolás", "chunking_lab", chunking_lab.render, "chunk"),
        ("Beágyazás", "embedding_lab", embedding_lab.render, "embed"),
        ("Visszakeresés", "vector_search", vector_search.render, "retrieve"),
    ],
    "Futtatás": [
        ("RAG játszótér", "rag_playground", rag_playground.render, "generate"),
        ("RAG összehasonlítás", "rag_comparison", rag_comparison.render, "generate"),
    ],
    "Kiértékelés": [
        ("Teljesítmény", "performance_lab", performance_lab.render, "evaluate"),
        ("Kiértékelés", "evaluation", evaluation.render, "evaluate"),
        ("Teljes pipeline benchmark", "pipeline_matrix", pipeline_matrix.render, "evaluate"),
        ("Mérési labor", "measurement_lab", measurement_lab.render, "evaluate"),
        ("Kísérletek", "experiments", experiments.render, "evaluate"),
    ],
    "Rendszer": [
        ("Infrastruktúra", "runtime_lab", runtime_lab.render, None),
    ],
}

SLUG_INDEX: dict[str, tuple[str, str, object, str | None]] = {}
for group_name, pages in NAV_GROUPS.items():
    for label, slug, renderer, stage in pages:
        SLUG_INDEX[slug] = (group_name, label, renderer, stage)

query_page = st.query_params.get("page", "overview")
if query_page not in SLUG_INDEX:
    query_page = "overview"

default_group, default_label, _, _ = SLUG_INDEX[query_page]
st.session_state.setdefault("nav_group", default_group)
st.session_state.setdefault("active_page_slug", query_page)

if st.session_state.get("active_page_slug") not in SLUG_INDEX:
    st.session_state["active_page_slug"] = "overview"
    st.session_state["nav_group"] = "Áttekintés"

active_group_from_slug = SLUG_INDEX[st.session_state["active_page_slug"]][0]
if st.session_state.get("nav_group") != active_group_from_slug:
    st.session_state["nav_group"] = active_group_from_slug

st.sidebar.markdown("## Multi-RAG Lab")
st.sidebar.caption(
    "Építés → Futtatás → Kiértékelés. A részletes runtime-beállítások külön, lenyitható blokkban maradnak."
)

selected_group = st.sidebar.selectbox(
    "Terület",
    list(NAV_GROUPS),
    index=list(NAV_GROUPS).index(st.session_state["nav_group"]),
    key="sidebar_nav_group",
)

pages = NAV_GROUPS[selected_group]
page_slugs = [page[1] for page in pages]
current_slug = st.session_state.get("active_page_slug")
if current_slug not in page_slugs:
    current_slug = page_slugs[0]

selected_slug = st.sidebar.radio(
    "Oldal",
    page_slugs,
    index=page_slugs.index(current_slug),
    format_func=lambda slug: SLUG_INDEX[slug][1],
    label_visibility="collapsed",
    key=f"sidebar_page_{selected_group}",
)

st.session_state["nav_group"] = selected_group
st.session_state["active_page_slug"] = selected_slug
st.query_params["page"] = selected_slug

st.sidebar.divider()
st.sidebar.toggle(
    "Prezentációs mód",
    key="presentation_mode",
    help="Elrejti a részletes runtime-kontrollokat, hogy a dashboard és a benchmark eredmények interjún/prezentációban tisztábban látszódjanak.",
)
if st.session_state.get("presentation_mode", False):
    st.sidebar.markdown(
        '<div class="rag-presentation-note">Prezentációs mód aktív · a részletes runtime-kontrollok rejtve vannak.</div>',
        unsafe_allow_html=True,
    )
else:
    sidebar_runtime_controls()

_, _, render, _ = SLUG_INDEX[selected_slug]
render()
