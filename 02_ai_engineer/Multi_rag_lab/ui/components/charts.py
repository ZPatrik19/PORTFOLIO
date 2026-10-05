from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components.tokens import CHART_PALETTE, SEMANTIC_COLORS


CHART_COLORS = CHART_PALETTE
px.defaults.color_discrete_sequence = CHART_COLORS

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    margin=dict(l=24, r=24, t=78, b=106),
    legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=10), tracegroupgap=4),
    hoverlabel=dict(bgcolor="rgba(18,27,43,0.98)", font=dict(color="#f5f9ff")),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#edf4ff", size=12, family="Inter, Segoe UI, Arial, sans-serif"),
    title_font=dict(size=16, color="#f5f9ff"),
)



def _apply_layout(fig: go.Figure, *, title: str | None = None, height: int = 340) -> go.Figure:
    # Keep Plotly Express / caller-provided titles unless this helper explicitly receives one.
    # Do not pass `title` twice: PLOTLY_LAYOUT contains only title styling, not title content.
    layout = dict(PLOTLY_LAYOUT)
    layout["height"] = height
    if title is not None:
        layout["title"] = title
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(170,190,220,0.12)",
        zeroline=False,
        linecolor="rgba(170,190,220,0.18)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(170,190,220,0.12)",
        zeroline=False,
        linecolor="rgba(170,190,220,0.18)",
    )
    return fig


def chunk_length_chart(chunks, *, key: str) -> None:
    frame = pd.DataFrame(
        {
            "Karakter": [len(c.text) for c in chunks],
            "Token becslés": [max(1, len(c.text) // 4) for c in chunks],
        }
    )
    if frame.empty:
        st.info("Nincs megjeleníthető chunk.")
        return
    fig = px.histogram(
        frame,
        x="Karakter",
        nbins=min(30, max(8, len(frame) // 2)),
        title="Szövegrész-hossz eloszlása",
        color_discrete_sequence=["#3979d3"],
    )
    fig.update_traces(hovertemplate="Karakter: %{x}<br>Darabszám: %{y}<extra></extra>")
    fig.update_layout(yaxis_title="Szövegrészek száma", bargap=0.08)
    _apply_layout(fig, height=320)
    st.plotly_chart(fig, width="stretch", key=key)



def chunk_sequence_chart(chunks, *, key: str) -> None:
    frame = pd.DataFrame(
        {
            "Szövegrész sorszáma": list(range(1, len(chunks) + 1)),
            "Karakter": [len(c.text) for c in chunks],
            "Forrás": [str(c.metadata.get("title") or c.metadata.get("source") or "") for c in chunks],
            "Oldal": [c.metadata.get("page") for c in chunks],
        }
    )
    if frame.empty:
        return
    fig = px.line(
        frame,
        x="Szövegrész sorszáma",
        y="Karakter",
        markers=True,
        hover_data=["Forrás", "Oldal"],
        title="Szövegrész-méret a dokumentum sorrendjében",
    )
    fig.update_traces(line=dict(width=2.6, color="#7d57c1"), marker=dict(size=7, color="#3979d3"))
    _apply_layout(fig, height=320)
    st.plotly_chart(fig, width="stretch", key=key)



def chunk_boundary_chart(chunks, *, selected_index: int | None = None, key: str) -> None:
    if not chunks:
        return
    labels = [f"#{i + 1}" for i in range(len(chunks))]
    colors = ["#dfe7f3"] * len(chunks)
    if selected_index is not None and 0 <= selected_index < len(chunks):
        colors[selected_index] = "#3979d3"
    frame = pd.DataFrame(
        {
            "Szövegrész": labels,
            "Karakter": [len(c.text) for c in chunks],
            "Oldal": [c.metadata.get("page") for c in chunks],
            "Szekció": [str(c.metadata.get("section") or "—") for c in chunks],
            "Szín": colors,
        }
    )
    fig = go.Figure(
        go.Bar(
            x=frame["Szövegrész"],
            y=frame["Karakter"],
            marker_color=frame["Szín"],
            customdata=frame[["Oldal", "Szekció"]],
            hovertemplate="%{x}<br>Karakter: %{y}<br>Oldal: %{customdata[0]}<br>Szekció: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Interaktív darabolási vizualizáció",
        xaxis_title="Szövegrészek sorrendje",
        yaxis_title="Karakterhossz",
        showlegend=False,
    )
    _apply_layout(fig, height=340)
    st.plotly_chart(fig, width="stretch", key=key)



def strategy_comparison_chart(records: list[dict], *, key: str) -> None:
    if not records:
        return
    frame = pd.DataFrame(records)
    fig = px.bar(
        frame,
        x="Stratégia",
        y="Szövegrészek",
        color="Átlagos hossz",
        hover_data=["Medián", "P95", "Idő (ms)"],
        title="Darabolási stratégiák összehasonlítása",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis_title="Létrehozott szövegrészek")
    _apply_layout(fig, height=360)
    st.plotly_chart(fig, width="stretch", key=key)



def latency_bar(records: list[dict], *, key: str, title: str) -> None:
    if not records:
        return
    frame = pd.DataFrame(records)
    fig = px.bar(
        frame,
        x="component",
        y="total_ms",
        color="device",
        barmode="group",
        title=title,
        color_discrete_sequence=["#3979d3", "#7d57c1", "#27a87d"],
    )
    fig.update_layout(xaxis_title="Komponens", yaxis_title="ms")
    _apply_layout(fig, height=380)
    st.plotly_chart(fig, width="stretch", key=key)



def retrieval_scores_chart(results, *, key: str, title: str = "Top-K pontszámok") -> None:
    if not results:
        return
    frame = pd.DataFrame(
        {
            "Rang": [item.rank for item in results],
            "Pontszám": [item.score for item in results],
            "Szövegrész": [item.chunk_id for item in results],
            "Forrás": [str(item.metadata.get("title") or item.source or "") for item in results],
        }
    )
    fig = px.bar(
        frame,
        x="Rang",
        y="Pontszám",
        hover_data=["Szövegrész", "Forrás"],
        title=title,
        color="Pontszám",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    _apply_layout(fig, height=340)
    st.plotly_chart(fig, width="stretch", key=key)



def pipeline_trace_chart(trace: dict[str, float | str], *, key: str) -> None:
    mapping = [
        ("retrieval_ms", "Visszakeresés"),
        ("reranking_ms", "Újrarangsorolás"),
        ("context_ms", "Kontextusépítés"),
        ("generation_ms", "Generálás"),
    ]
    labels: list[str] = []
    values: list[float] = []
    for raw_key, label in mapping:
        value = trace.get(raw_key, 0.0)
        if isinstance(value, (int, float)):
            labels.append(label)
            values.append(float(value))
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=["#3979d3", "#5a6de0", "#27a87d", "#7d57c1"]),
            text=[f"{value:.1f} ms" for value in values],
            textposition="auto",
        )
    )
    fig.update_layout(title="Pipeline késleltetés bontása", xaxis_title="ms", yaxis_title="")
    _apply_layout(fig, height=320)
    st.plotly_chart(fig, width="stretch", key=key)



def context_profile_chart(profile_rows: list[dict] | list[object], *, key: str, title: str = "Kontextusprofil és forráseloszlás") -> None:
    if not profile_rows:
        return

    first = profile_rows[0]
    # Accept both pre-aggregated dict rows and retrieved chunk objects.
    if isinstance(first, dict):
        frame = pd.DataFrame(profile_rows)
        y_columns = [column for column in ["Kontextus token", "Források", "Forrásdiverzitás"] if column in frame.columns]
        if not y_columns or "Profil" not in frame.columns:
            return
        fig = px.bar(
            frame,
            x="Profil",
            y=y_columns,
            barmode="group",
            title=title,
            color_discrete_sequence=["#3979d3", "#27a87d", "#7d57c1"],
        )
    else:
        grouped: dict[str, dict[str, float | int | str]] = {}
        for idx, chunk in enumerate(profile_rows, start=1):
            meta = getattr(chunk, "metadata", {}) or {}
            source = str(meta.get("title") or getattr(chunk, "source", None) or f"Forrás {idx}")
            row = grouped.setdefault(source, {"Forrás": source, "Kontextus token": 0, "Chunkok": 0, "Átlagscore": 0.0})
            row["Kontextus token"] = int(row["Kontextus token"]) + max(1, len(getattr(chunk, "text", "")) // 4)
            row["Chunkok"] = int(row["Chunkok"]) + 1
            row["Átlagscore"] = float(row["Átlagscore"]) + float(getattr(chunk, "score", 0.0) or 0.0)
        rows = []
        for row in grouped.values():
            chunk_count = max(1, int(row["Chunkok"]))
            row["Átlagscore"] = float(row["Átlagscore"]) / chunk_count
            rows.append(row)
        frame = pd.DataFrame(rows).sort_values(["Kontextus token", "Chunkok"], ascending=[False, False]).head(8)
        fig = px.bar(
            frame,
            x="Forrás",
            y="Kontextus token",
            color="Chunkok",
            text="Chunkok",
            hover_data=["Átlagscore"],
            title=title,
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            xaxis_title="Forrás",
            yaxis_title="Kontextus token (becslés)",
            coloraxis_colorbar_title="Chunk",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
    _apply_layout(fig, height=360)
    st.plotly_chart(fig, width="stretch", key=key)



def evaluation_heatmap(frame: pd.DataFrame, *, key: str, title: str) -> None:
    if frame.empty:
        return
    numeric = frame.set_index("Stratégia")
    fig = px.imshow(
        numeric,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title=title,
    )
    fig.update_xaxes(side="top")
    _apply_layout(fig, height=max(320, 70 + 60 * max(1, len(numeric))))
    st.plotly_chart(fig, width="stretch", key=key)


def quality_latency_scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    hover: list[str],
    key: str,
    title: str,
    size: str | None = None,
) -> None:
    if frame.empty or x not in frame or y not in frame:
        return
    kwargs = dict(data_frame=frame, x=x, y=y, color=color, hover_data=hover, title=title)
    if size and size in frame and (frame[size].fillna(0) >= 0).all():
        kwargs["size"] = size
    fig = px.scatter(**kwargs)
    fig.update_traces(marker=dict(opacity=0.90, line=dict(width=0.7, color="rgba(255,255,255,0.32)")))
    fig.update_layout(hovermode="closest")
    _apply_layout(fig, height=460)
    st.plotly_chart(fig, width="stretch", key=key)


def stacked_latency_chart(
    frame: pd.DataFrame,
    *,
    strategy_col: str,
    stage_columns: list[str],
    key: str,
    title: str,
) -> None:
    if frame.empty:
        return
    available = [column for column in stage_columns if column in frame.columns]
    if not available:
        return
    long = frame[[strategy_col, *available]].melt(
        id_vars=strategy_col,
        value_vars=available,
        var_name="Pipeline szakasz",
        value_name="Késleltetés ms",
    )
    fig = px.bar(
        long,
        x=strategy_col,
        y="Késleltetés ms",
        color="Pipeline szakasz",
        barmode="stack",
        title=title,
    )
    _apply_layout(fig, height=460)
    st.plotly_chart(fig, width="stretch", key=key)


def radar_metrics_chart(
    frame: pd.DataFrame,
    *,
    label_col: str,
    metrics: list[str],
    key: str,
    title: str,
    max_series: int = 6,
) -> None:
    if frame.empty:
        return
    metrics = [metric for metric in metrics if metric in frame.columns]
    if len(metrics) < 2:
        return
    plot_frame = frame.head(max_series).copy()
    fig = go.Figure()
    for _, row in plot_frame.iterrows():
        values = []
        for metric in metrics:
            value = float(row.get(metric, 0.0) or 0.0)
            values.append(max(0.0, min(1.0, value)))
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=str(row[label_col]),
                opacity=0.55,
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(170,190,220,0.14)", tickfont=dict(size=10)),
            angularaxis=dict(gridcolor="rgba(170,190,220,0.12)"),
        ),
        title=title,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=9)),
    )
    _apply_layout(fig, height=460)
    st.plotly_chart(fig, width="stretch", key=key)


def device_metric_bars(
    frame: pd.DataFrame,
    *,
    category: str,
    metric: str,
    device: str,
    key: str,
    title: str,
) -> None:
    if frame.empty or metric not in frame.columns:
        return
    fig = px.bar(frame, x=category, y=metric, color=device, barmode="group", title=title)
    _apply_layout(fig, height=440)
    st.plotly_chart(fig, width="stretch", key=key)


def matrix_heatmap(
    frame: pd.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    key: str,
    title: str,
    agg: str = "mean",
) -> None:
    if frame.empty or any(name not in frame.columns for name in (row, column, value)):
        return
    pivot = frame.pivot_table(index=row, columns=column, values=value, aggfunc=agg)
    if pivot.empty:
        return
    fig = px.imshow(
        pivot,
        text_auto=".3f",
        color_continuous_scale="Viridis",
        aspect="auto",
        title=title,
    )
    fig.update_xaxes(side="top")
    _apply_layout(fig, height=max(360, 90 + 48 * len(pivot)))
    st.plotly_chart(fig, width="stretch", key=key)


def metric_bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    key: str,
    title: str,
    color: str | None = None,
    orientation: str = "v",
) -> None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return
    auto_horizontal = orientation == "v" and len(frame) >= 6
    if auto_horizontal:
        kwargs = dict(data_frame=frame, x=y, y=x, title=title, orientation="h")
    else:
        kwargs = dict(data_frame=frame, x=x, y=y, title=title, orientation=orientation)
    if color and color in frame.columns:
        kwargs["color"] = color
    fig = px.bar(**kwargs)
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.18)")
    if color == x or (auto_horizontal and color == x):
        fig.update_layout(showlegend=False)
    if auto_horizontal:
        fig.update_yaxes(categoryorder="total ascending", automargin=True, title=None)
        fig.update_xaxes(automargin=True)
    elif orientation == "v" and len(frame) > 8:
        fig.update_xaxes(tickangle=-28, automargin=True)
    _apply_layout(fig, height=460)
    st.plotly_chart(fig, width="stretch", key=key)


def percentile_latency_chart(
    frame: pd.DataFrame,
    *,
    category: str,
    percentiles: list[str],
    key: str,
    title: str,
    value_label: str = "Késleltetés ms",
) -> None:
    if frame.empty or category not in frame.columns:
        return
    available = [column for column in percentiles if column in frame.columns]
    if not available:
        return
    long = frame[[category, *available]].melt(
        id_vars=category,
        value_vars=available,
        var_name="Percentilis",
        value_name=value_label,
    )
    if len(frame) >= 6:
        fig = px.bar(long, x=value_label, y=category, color="Percentilis", barmode="group", title=title, orientation="h")
        fig.update_yaxes(automargin=True, title=None)
    else:
        fig = px.bar(long, x=category, y=value_label, color="Percentilis", barmode="group", title=title)
        fig.update_xaxes(automargin=True)
    _apply_layout(fig, height=460)
    st.plotly_chart(fig, width="stretch", key=key)


def retrieval_rank_flow_chart(
    dense_results,
    bm25_results,
    final_results,
    *,
    key: str,
    title: str = "Jelöltek rangmozgása · Dense → BM25 → végső rang",
) -> None:
    """Show how candidate ranks move across retrieval stages.

    The chart uses only chunk IDs and ranks, so it works with the existing retrieval
    result model without requiring extra score semantics.
    """
    if not final_results:
        return

    def rank_map(items):
        return {str(item.chunk_id): int(item.rank) for item in items or []}

    dense_map = rank_map(dense_results)
    bm25_map = rank_map(bm25_results)
    final_map = rank_map(final_results)
    selected = [str(item.chunk_id) for item in final_results[: min(8, len(final_results))]]
    missing_rank = max(
        [*dense_map.values(), *bm25_map.values(), *final_map.values(), 1],
    ) + 2

    fig = go.Figure()
    stages = ["Dense", "BM25", "Végső"]
    for chunk_id in selected:
        ranks = [
            dense_map.get(chunk_id, missing_rank),
            bm25_map.get(chunk_id, missing_rank),
            final_map.get(chunk_id, missing_rank),
        ]
        fig.add_trace(
            go.Scatter(
                x=stages,
                y=ranks,
                mode="lines+markers",
                name=chunk_id,
                hovertemplate=(
                    f"<b>{chunk_id}</b><br>Stage: %{{x}}<br>Rank: %{{y}}<extra></extra>"
                ),
                line=dict(width=2.2),
                marker=dict(size=8),
            )
        )
    fig.update_yaxes(autorange="reversed", title="Rang · kisebb = jobb")
    fig.update_xaxes(title="Visszakeresési szakasz")
    fig.update_layout(showlegend=True, legend_title_text="Chunk")
    _apply_layout(fig, title=title, height=430)
    st.plotly_chart(fig, width="stretch", key=key)


def delta_bar_chart(
    frame: pd.DataFrame,
    *,
    metric_col: str,
    delta_col: str,
    key: str,
    title: str,
    label_col: str | None = None,
) -> None:
    """Render a compact diverging A/B delta chart.

    Positive values are improvements after metric direction has already been
    normalized by the caller; negative values represent regressions.
    """
    if frame.empty or metric_col not in frame.columns or delta_col not in frame.columns:
        return
    data = frame.copy()
    data["Irány"] = data[delta_col].apply(lambda value: "Javulás" if float(value) >= 0 else "Romlás")
    fig = px.bar(
        data,
        x=delta_col,
        y=metric_col,
        orientation="h",
        color="Irány",
        color_discrete_map={
            "Javulás": SEMANTIC_COLORS["success"],
            "Romlás": SEMANTIC_COLORS["error"],
        },
        text=data[delta_col].map(lambda value: f"{float(value):+.3f}"),
        hover_data=[label_col] if label_col and label_col in data.columns else None,
        title=title,
    )
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="rgba(210,220,235,.55)")
    fig.update_layout(xaxis_title="B hatása A-hoz képest", yaxis_title="", showlegend=False)
    fig.update_traces(textposition="outside", cliponaxis=False)
    _apply_layout(fig, height=max(320, 72 + 46 * len(data)))
    st.plotly_chart(fig, width="stretch", key=key)
