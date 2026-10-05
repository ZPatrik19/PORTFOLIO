from __future__ import annotations

import streamlit as st

from ui.components.education import evidence_cards


def render_answer_evidence(result, *, key_prefix: str = "rag") -> None:
    evidence_cards(result.retrieved_chunks, answer=result.answer, columns=2)
    with st.expander("Nyers bizonyítékok és metadata", expanded=False):
        for chunk in result.retrieved_chunks:
            title = chunk.metadata.get("title") or chunk.source or "Ismeretlen forrás"
            st.markdown(f"**#{chunk.rank} · {title} · pontszám {chunk.score:.5f}**")
            st.write(chunk.text)
            st.json(chunk.metadata)
