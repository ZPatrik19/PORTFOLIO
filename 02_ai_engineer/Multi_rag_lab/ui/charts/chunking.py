from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ui.components.tokens import CHART_PALETTE


def chunk_document_strip_chart(chunks, *, key: str, title: str = "Dokumentumszalag · chunk-határok") -> None:
    """Sequence-oriented chunk strip.

    The base chunk models do not retain exact character offsets for every strategy,
    therefore the strip deliberately visualizes relative chunk span/sequence rather
    than pretending to show exact source offsets.
    """
    if not chunks:
        return
    lengths = [max(1, len(chunk.text)) for chunk in chunks]
    total = sum(lengths)
    cursor = 0.0
    fig = go.Figure()
    for idx, (chunk, length) in enumerate(zip(chunks, lengths, strict=False), start=1):
        start = cursor / total
        cursor += length
        end = cursor / total
        color = CHART_PALETTE[(idx - 1) % len(CHART_PALETTE)]
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=0.25,
            y1=0.75,
            line=dict(color=color, width=1.4),
            fillcolor=color,
            opacity=0.62,
        )
        if len(chunks) <= 18 or idx in {1, len(chunks)}:
            fig.add_annotation(
                x=(start + end) / 2, y=0.5, text=f"C{idx}", showarrow=False, font=dict(size=10, color="#ffffff")
            )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Relatív dokumentumpozíció", showgrid=False)
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        title=title,
        height=230,
        margin=dict(l=18, r=18, t=65, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4ff", family="Inter, Segoe UI, Arial, sans-serif"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", key=key)
