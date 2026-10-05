from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from rag_engine.evaluation.projection import pca_2d


def embedding_projection_chart(
    vectors: np.ndarray,
    chunks,
    *,
    key: str,
    query_vector: np.ndarray | None = None,
    query_label: str = "Lekérdezés",
) -> None:
    if vectors.ndim != 2 or len(vectors) < 2:
        st.info("A 2D vetítéshez legalább két beágyazási vektor szükséges.")
        return

    combined = vectors
    query_idx = None
    if query_vector is not None:
        q = np.asarray(query_vector).reshape(1, -1)
        if q.shape[1] == vectors.shape[1]:
            query_idx = len(vectors)
            combined = np.vstack([vectors, q])

    coords = pca_2d(combined)
    rows = []
    for idx, chunk in enumerate(chunks):
        meta = getattr(chunk, "metadata", {}) or {}
        rows.append(
            {
                "PCA 1": coords[idx, 0],
                "PCA 2": coords[idx, 1],
                "Típus": "Szövegrész",
                "Forrás": str(meta.get("title") or meta.get("source") or "Ismeretlen"),
                "Szövegrész": str(getattr(chunk, "chunk_id", f"C{idx + 1}")),
                "Részlet": str(getattr(chunk, "text", ""))[:180].replace("\n", " "),
            }
        )
    if query_idx is not None:
        rows.append(
            {
                "PCA 1": coords[query_idx, 0],
                "PCA 2": coords[query_idx, 1],
                "Típus": query_label,
                "Forrás": "—",
                "Szövegrész": "LEKÉRDEZÉS",
                "Részlet": query_label,
            }
        )

    frame = pd.DataFrame(rows)
    fig = px.scatter(
        frame,
        x="PCA 1",
        y="PCA 2",
        color="Típus",
        symbol="Típus",
        hover_data=["Forrás", "Szövegrész", "Részlet"],
        title="Beágyazási tér · 2D PCA-vetítés",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.82, line=dict(width=0.6, color="rgba(255,255,255,.25)")))
    fig.update_layout(
        height=460,
        margin=dict(l=24, r=24, t=76, b=100),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch", key=key)
