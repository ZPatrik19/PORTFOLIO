from __future__ import annotations

import html
from collections import Counter

import streamlit as st

from rag_engine.knowledge import CATEGORY_DESCRIPTIONS, CATEGORY_ORDER, ReferenceEntry


_DIRECTION = {
    "up": ("↑", "Nagyobb általában kedvezőbb", "up"),
    "down": ("↓", "Kisebb általában kedvezőbb", "down"),
    "context": ("↔", "Kontextusfüggő – nincs univerzális optimum", "context"),
}


def _chips(values: tuple[str, ...]) -> str:
    return "".join(f'<span class="rag-reference-chip">{html.escape(value)}</span>' for value in values)


def render_topic_cards(entries: list[ReferenceEntry]) -> None:
    counts = Counter(item.category for item in entries)
    for start in range(0, len(CATEGORY_ORDER), 3):
        columns = st.columns(3)
        for col, category in zip(columns, CATEGORY_ORDER[start : start + 3], strict=False):
            count = counts.get(category, 0)
            col.markdown(
                f"""
<div class="rag-reference-topic">
  <div class="rag-reference-topic-count">{count} tétel</div>
  <div class="rag-reference-topic-title">{html.escape(category)}</div>
  <div class="rag-reference-topic-text">{html.escape(CATEGORY_DESCRIPTIONS.get(category, ""))}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_entry(entry: ReferenceEntry, *, expanded: bool = False) -> None:
    arrow, direction_text, state = _DIRECTION.get(entry.direction, _DIRECTION["context"])
    header = f"{entry.name} · {entry.kind}"
    with st.expander(header, expanded=expanded):
        st.markdown(
            f"""
<div class="rag-reference-detail-head">
  <span class="rag-reference-kind">{html.escape(entry.kind)}</span>
  <span class="rag-reference-direction {state}">{arrow} {html.escape(direction_text)}</span>
</div>
<div class="rag-reference-definition">{html.escape(entry.definition)}</div>
""",
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown("**Miért fontos?**")
            st.write(entry.why_it_matters)
            if entry.interpretation:
                st.markdown("**Értelmezés / működés**")
                st.write(entry.interpretation)
            if entry.caveat:
                st.warning(entry.caveat)
        with right:
            if entry.formula:
                st.markdown("**Képlet**")
                st.latex(entry.formula)
            if entry.unit:
                st.markdown(f"**Egység / skála:** `{entry.unit}`")
            if entry.project_area:
                st.markdown("**Hol jelenik meg a projektben?**")
                st.markdown(_chips(entry.project_area), unsafe_allow_html=True)
            if entry.related:
                st.markdown("**Kapcsolódó fogalmak**")
                st.markdown(_chips(entry.related), unsafe_allow_html=True)


def render_entry_grid(entries: list[ReferenceEntry], *, expanded_first: bool = False) -> None:
    if not entries:
        st.info("A kiválasztott szűrőkkel nincs megjeleníthető definíció.")
        return
    columns = st.columns(2)
    for index, entry in enumerate(entries):
        with columns[index % 2]:
            render_entry(entry, expanded=expanded_first and index == 0)


def render_direction_legend() -> None:
    st.markdown(
        """
<div class="rag-reference-legend">
  <span><b>↑</b> nagyobb érték általában kedvezőbb</span>
  <span><b>↓</b> kisebb érték általában kedvezőbb</span>
  <span><b>↔</b> kontextusfüggő, nincs univerzális optimum</span>
</div>
""",
        unsafe_allow_html=True,
    )
