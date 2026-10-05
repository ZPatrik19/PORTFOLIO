from __future__ import annotations

import streamlit as st

from ui.components.education import note_box


def render_grounded_prompt(result, *, key_prefix: str = "grounded") -> None:
    parts = dict(getattr(result, "grounded_prompt_parts", {}) or {})
    final_prompt = str(getattr(result, "grounded_prompt", "") or "")
    if not final_prompt:
        note_box("Prompt nem érhető el", "A futás nem tárolta a ténylegesen elküldött grounded promptot. Futtasd újra a választ az új pipeline-verzióval.")
        return

    note_box(
        "Mit látsz itt?",
        "Ez a tényleges forrásokra támaszkodó prompt, amely a felhasználói kérdést, a retrievalből összeállított bizonyítékokat és a válaszadási szabályokat egyesíti. Ez a nézet segít a grounding és prompt-debugging ellenőrzésében.",
    )
    tabs = st.tabs(["Rendszerutasítás", "Kérdés", "Bizonyítékok", "Teljes prompt"])
    with tabs[0]:
        st.code(str(parts.get("instructions", "")), language=None)
    with tabs[1]:
        st.code(str(parts.get("query", getattr(result, "query", ""))), language=None)
    with tabs[2]:
        evidence = str(parts.get("context", getattr(result, "context_text", "")))
        st.code(evidence, language=None)
    with tabs[3]:
        st.code(final_prompt, language=None)
        st.download_button(
            "Grounded prompt mentése TXT-be",
            data=final_prompt,
            file_name="grounded_prompt.txt",
            mime="text/plain",
            key=f"{key_prefix}_download_prompt",
        )
