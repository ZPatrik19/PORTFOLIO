from __future__ import annotations

import html

import streamlit as st

from rag_engine.presets import PAGE_METRICS
from ui.components.i18n import hu_label


CARD_KICKERS = ["Mit csinál?", "Miért fontos?", "Mikor hasznos?", "Mit figyelj?", "Mérési fókusz", "Best practice"]



def page_intro(title: str, subtitle: str, *, eyebrow: str | None = None) -> None:
    eyebrow_html = f'<div class="rag-small">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f"""
<div class="rag-hero">
{eyebrow_html}
<h2>{html.escape(title)}</h2>
<p>{html.escape(subtitle)}</p>
</div>
""",
        unsafe_allow_html=True,
    )



def section_intro(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="rag-section-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="rag-section-title">{html.escape(title)}</div>
{subtitle_html}
""",
        unsafe_allow_html=True,
    )



def info_cards(cards: list[tuple[str, str]], *, columns: int = 3) -> None:
    for start in range(0, len(cards), columns):
        cols = st.columns(columns)
        for idx, (col, (title, body)) in enumerate(zip(cols, cards[start : start + columns], strict=False), start=start):
            kicker = CARD_KICKERS[idx % len(CARD_KICKERS)]
            col.markdown(
                f'''
<div class="rag-card">
  <div class="rag-kicker">{html.escape(kicker)}</div>
  <h4>{html.escape(title)}</h4>
  <p>{html.escape(body)}</p>
</div>
''',
                unsafe_allow_html=True,
            )



def kpi_cards(items: list[tuple[str, str, str]], *, columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (label, value, subtext) in zip(cols, items[start : start + columns], strict=False):
            col.markdown(
                f'''
<div class="rag-kpi">
  <div class="label">{html.escape(label)}</div>
  <div class="value">{html.escape(value)}</div>
  <div class="sub">{html.escape(subtext)}</div>
</div>
''',
                unsafe_allow_html=True,
            )



def note_box(title: str, text: str) -> None:
    st.markdown(
        f'<div class="rag-note"><strong>{html.escape(title)}</strong><br>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )



def status_cards(items: list[tuple[str, str, str, str]], *, columns: int = 2) -> None:
    """Render readable runtime/status cards.

    Each item is (label, value, detail, status), where status is ok/warn/off/info.
    """
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[start : start + columns], strict=False):
            label, value, detail, status = item
            state = status if status in {"ok", "warn", "off", "info"} else "info"
            html_block = (
                '<div class="rag-status-card">'
                f'<div class="rag-status-head"><span class="rag-status-dot {state}"></span>{html.escape(label)}</div>'
                f'<div class="rag-status-value">{html.escape(value)}</div>'
                f'<div class="rag-status-detail">{html.escape(detail)}</div>'
                '</div>'
            )
            col.markdown(html_block, unsafe_allow_html=True)


def metrics_reference(page: str) -> None:
    metrics = PAGE_METRICS.get(page, [])
    if not metrics:
        return
    with st.expander("Mit jelentenek az ezen az oldalon használt metrikák?", expanded=False):
        for name, explanation in metrics:
            st.markdown(f"**{hu_label(name)}** — {explanation}")



def methodology_note(title: str, text: str) -> None:
    st.info(f"**{title}:** {text}")


def empty_state(title: str, text: str, *, hint: str | None = None) -> None:
    hint_html = f'<div class="rag-empty-hint">{html.escape(hint)}</div>' if hint else ""
    st.markdown(
        f"""
<div class="rag-empty">
  <div class="rag-empty-icon">◇</div>
  <div>
    <div class="rag-empty-title">{html.escape(title)}</div>
    <div class="rag-empty-text">{html.escape(text)}</div>
    {hint_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def live_run_card(*, current: int, total: int, label: str, elapsed: str, eta: str, predicted: str) -> str:
    percent = 0 if total <= 0 else min(100, max(0, round(current / total * 100)))
    return f"""
<div class="rag-live-run">
  <div class="rag-live-run-head">
    <span class="rag-live-dot"></span>
    <strong>Benchmark fut</strong>
    <span class="rag-live-percent">{percent}%</span>
  </div>
  <div class="rag-live-current">{html.escape(label)}</div>
  <div class="rag-live-grid">
    <div><span>Blokk</span><b>{current:,} / {total:,}</b></div>
    <div><span>Eltelt</span><b>{html.escape(elapsed)}</b></div>
    <div><span>Élő ETA</span><b>{html.escape(eta)}</b></div>
    <div><span>Becsült tartomány</span><b>{html.escape(predicted)}</b></div>
  </div>
</div>
"""


def evidence_cards(items, *, answer: str | None = None, columns: int = 2) -> None:
    if not items:
        empty_state("Nincs bizonyíték", "A visszakeresés nem adott vissza megjeleníthető forrásrészletet.")
        return
    max_score = max(abs(float(getattr(item, "score", 0.0) or 0.0)) for item in items) or 1.0
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[start : start + columns], strict=False):
            rank = int(getattr(item, "rank", 0) or 0)
            score = float(getattr(item, "score", 0.0) or 0.0)
            meta = getattr(item, "metadata", {}) or {}
            source = str(meta.get("title") or getattr(item, "source", None) or "Ismeretlen forrás")
            page = meta.get("page") or "—"
            chunk_id = str(getattr(item, "chunk_id", "—"))
            cited = bool(answer and f"[S{rank}]" in answer)
            pct = max(8, min(100, round(abs(score) / max_score * 100)))
            status = "cited" if cited else "uncited"
            badge = "Hivatkozva" if cited else "Nincs hivatkozva"
            excerpt = str(getattr(item, "text", "")).strip().replace("\n", " ")
            if len(excerpt) > 260:
                excerpt = excerpt[:257] + "..."
            col.markdown(
                f"""
<div class="rag-evidence-card">
  <div class="rag-evidence-top">
    <span class="rag-rank-chip">#{rank}</span>
    <span class="rag-evidence-status {status}">{badge}</span>
  </div>
  <div class="rag-evidence-title">{html.escape(source)}</div>
  <div class="rag-evidence-meta">szövegrész {html.escape(chunk_id)} · oldal {html.escape(str(page))} · pontszám {score:.5f}</div>
  <div class="rag-score-track"><div class="rag-score-fill" style="width:{pct}%"></div></div>
  <div class="rag-evidence-excerpt">{html.escape(excerpt)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
