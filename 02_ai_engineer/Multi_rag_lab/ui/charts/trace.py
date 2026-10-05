from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ui.components.tokens import SEMANTIC_COLORS


def rag_trace_graph(trace: dict[str, float | str], *, key: str, reranker_active: bool = True) -> None:
    stages = [
        ("Lekérdezés", 0.0, SEMANTIC_COLORS["data"]),
        ("Visszakeresés", float(trace.get("retrieval_ms", 0.0) or 0.0), SEMANTIC_COLORS["retrieval"]),
        ("Újrarangsorolás", float(trace.get("reranking_ms", 0.0) or 0.0), SEMANTIC_COLORS["reranking"]),
        ("Kontextusépítés", float(trace.get("context_ms", 0.0) or 0.0), SEMANTIC_COLORS["chunking"]),
        ("LLM-generálás", float(trace.get("generation_ms", 0.0) or 0.0), SEMANTIC_COLORS["generation"]),
        ("Válasz", float(trace.get("total_ms", 0.0) or 0.0), SEMANTIC_COLORS["evaluation"]),
    ]
    if not reranker_active:
        stages[2] = ("Újrarangsorolás · kihagyva", 0.0, SEMANTIC_COLORS["muted"])

    fig = go.Figure()
    x_positions = [0.08, 0.25, 0.42, 0.59, 0.76, 0.93]
    box_w = 0.135
    y = 0.54
    for idx, ((label, latency, color), x) in enumerate(zip(stages, x_positions, strict=False)):
        fig.add_shape(
            type="rect", x0=x-box_w/2, x1=x+box_w/2, y0=y-0.16, y1=y+0.16,
            line=dict(color=color, width=2), fillcolor="rgba(18,27,43,.88)",
        )
        subtitle = "bemenet" if idx == 0 else (f"{latency:.0f} ms" if idx < len(stages)-1 else f"összesen {latency:.0f} ms")
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b><br><span style='font-size:11px'>{subtitle}</span>", showarrow=False)
        if idx < len(stages)-1:
            next_x = x_positions[idx+1]
            fig.add_annotation(
                x=next_x-box_w/2+0.005, y=y, ax=x+box_w/2-0.005, ay=y,
                xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(170,190,220,.55)", text="",
            )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        title="RAG futási trace · lépések és késleltetés",
        height=300,
        margin=dict(l=12, r=12, t=60, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4ff", family="Inter, Segoe UI, Arial, sans-serif"),
    )
    st.plotly_chart(fig, width="stretch", key=key)
