from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from i18n import pick

PRIMARY = "#60a5fa"
SECONDARY = "#94a3b8"
ACCENT = "#818cf8"
MUTED = "#64748b"
GRID = "rgba(148,163,184,0.15)"
TEXT = "#cbd5e1"
PAPER = "rgba(0,0,0,0)"
PLOT = "rgba(15,23,42,0.18)"


def _base(fig: go.Figure, *, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=18, t=48, b=28),
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        font=dict(color=TEXT, size=11),
        hoverlabel=dict(bgcolor="#111827", bordercolor="#475569", font_color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        modebar=dict(bgcolor="rgba(15,23,42,.75)", color="#94a3b8", activecolor="#e5e7eb"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def retrieval_quality_figure(summary: pd.DataFrame, ui_lang: str) -> go.Figure:
    frame = summary.copy()
    metrics = [c for c in ["recall@5", "mrr", "ndcg@5"] if c in frame.columns]
    long = frame.melt(id_vars=["method"], value_vars=metrics, var_name="metric", value_name="score")
    label_map = {"recall@5": "Recall@5", "mrr": "MRR", "ndcg@5": "nDCG@5"}
    long["metric"] = long["metric"].map(label_map)
    fig = px.bar(
        long,
        x="method",
        y="score",
        color="metric",
        barmode="group",
        color_discrete_sequence=[PRIMARY, ACCENT, SECONDARY],
        labels={"method": pick(ui_lang, "Módszer", "Method"), "score": pick(ui_lang, "Pontszám", "Score"), "metric": pick(ui_lang, "Metrika", "Metric")},
        title=pick(ui_lang, "Retrieval minőség módszerenként", "Retrieval quality by method"),
    )
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>")
    return _base(fig, height=350)


def retrieval_latency_figure(summary: pd.DataFrame, ui_lang: str) -> go.Figure:
    frame = summary.copy()
    metrics = [c for c in ["p50_latency_ms", "p95_latency_ms"] if c in frame.columns]
    long = frame.melt(id_vars=["method"], value_vars=metrics, var_name="metric", value_name="latency_ms")
    long["metric"] = long["metric"].map({"p50_latency_ms": "P50", "p95_latency_ms": "P95"})
    fig = px.line(
        long,
        x="method",
        y="latency_ms",
        color="metric",
        markers=True,
        color_discrete_sequence=[PRIMARY, SECONDARY],
        labels={"method": pick(ui_lang, "Módszer", "Method"), "latency_ms": pick(ui_lang, "Késleltetés (ms)", "Latency (ms)"), "metric": pick(ui_lang, "Percentilis", "Percentile")},
        title=pick(ui_lang, "Késleltetés módszerenként", "Latency by method"),
    )
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f} ms<extra></extra>")
    return _base(fig, height=350)


def quality_latency_scatter(summary: pd.DataFrame, ui_lang: str) -> go.Figure:
    frame = summary.copy()
    y_col = "ndcg@5" if "ndcg@5" in frame.columns else "recall@5"
    x_col = "p95_latency_ms" if "p95_latency_ms" in frame.columns else "p50_latency_ms"
    fig = px.scatter(
        frame,
        x=x_col,
        y=y_col,
        text="method",
        size="recall@5" if "recall@5" in frame.columns else None,
        color_discrete_sequence=[PRIMARY],
        labels={x_col: pick(ui_lang, "Késleltetés (ms)", "Latency (ms)"), y_col: "nDCG@5" if y_col == "ndcg@5" else "Recall@5"},
        title=pick(ui_lang, "Minőség–késleltetés kompromisszum", "Quality–latency trade-off"),
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(color="#cbd5e1", width=1), opacity=.9), hovertemplate="%{text}<br>x=%{x:.2f} ms<br>quality=%{y:.3f}<extra></extra>")
    return _base(fig, height=350)


def retrieval_radar(summary: pd.DataFrame, ui_lang: str) -> go.Figure:
    metrics = [c for c in ["recall@5", "mrr", "ndcg@5", "hit_rate"] if c in summary.columns]
    labels = {"recall@5": "Recall@5", "mrr": "MRR", "ndcg@5": "nDCG@5", "hit_rate": "Hit Rate"}
    fig = go.Figure()
    palette = [PRIMARY, ACCENT, SECONDARY, MUTED, "#a5b4fc"]
    for idx, (_, row) in enumerate(summary.head(5).iterrows()):
        r = [float(row[m]) if pd.notna(row[m]) else 0.0 for m in metrics]
        theta = [labels[m] for m in metrics]
        fig.add_trace(go.Scatterpolar(r=r + [r[0]], theta=theta + [theta[0]], fill="toself", opacity=.20, name=str(row.get("method", idx)), line=dict(color=palette[idx % len(palette)], width=2)))
    fig.update_layout(
        polar=dict(bgcolor=PLOT, radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID, tickfont=dict(color=TEXT)), angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT))),
        title=pick(ui_lang, "Retrieval profil", "Retrieval profile"),
    )
    return _base(fig, height=390)


def deterministic_rag_figure(frame: pd.DataFrame, ui_lang: str) -> go.Figure:
    columns = [c for c in ["citation_correctness", "citation_completeness", "no_answer_accuracy", "structured_output_validity", "faithfulness", "answer_correctness"] if c in frame.columns]
    means = frame[columns].mean(numeric_only=True).dropna()
    fig = go.Figure(go.Bar(x=means.index, y=means.values, marker_color=PRIMARY, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_layout(title=pick(ui_lang, "Deterministic RAG validáció", "Deterministic RAG validation"), showlegend=False)
    fig.update_yaxes(range=[0, 1.05])
    return _base(fig, height=330)


def prompt_heatmap(frame: pd.DataFrame, ui_lang: str) -> go.Figure:
    metrics = [c for c in ["prompt_score", "output_score", "grounding_score", "context_score", "tool_score"] if c in frame.columns]
    pivot = frame.groupby("profile")[metrics].mean(numeric_only=True)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=[[0, "#0f172a"], [1, PRIMARY]], zmin=0, zmax=1, colorbar=dict(title=pick(ui_lang, "Pontszám", "Score"))))
    fig.update_layout(title=pick(ui_lang, "Promptprofil teljesítménymátrix", "Prompt profile performance matrix"))
    return _base(fig, height=max(330, 80 + 42 * len(pivot)))


def prompt_footprint_figure(frame: pd.DataFrame, ui_lang: str) -> go.Figure:
    grouped = frame.groupby("profile")[[c for c in ["input_tokens", "output_tokens", "latency_ms", "cost_usd"] if c in frame.columns]].mean(numeric_only=True).reset_index()
    fig = go.Figure()
    if "latency_ms" in grouped.columns:
        fig.add_trace(go.Bar(x=grouped["profile"], y=grouped["latency_ms"], name=pick(ui_lang, "Késleltetés", "Latency"), marker_color=PRIMARY))
    if "cost_usd" in grouped.columns:
        fig.add_trace(go.Scatter(x=grouped["profile"], y=grouped["cost_usd"], name=pick(ui_lang, "Költség (USD)", "Cost (USD)"), mode="lines+markers", yaxis="y2", line=dict(color=ACCENT, width=2)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="USD", showgrid=False))
    fig.update_layout(title=pick(ui_lang, "Promptprofil: késleltetés és költség", "Prompt profile: latency and cost"), barmode="group")
    return _base(fig, height=350)


def tool_telemetry_figure(frame: pd.DataFrame, ui_lang: str) -> go.Figure:
    fig = go.Figure()
    if "tool_score" in frame.columns:
        fig.add_trace(go.Scatter(x=frame.index, y=pd.to_numeric(frame["tool_score"], errors="coerce"), mode="lines+markers", name=pick(ui_lang, "Tool score", "Tool score"), line=dict(color=PRIMARY, width=2)))
    if "tool_latency_ms" in frame.columns:
        fig.add_trace(go.Bar(x=frame.index, y=pd.to_numeric(frame["tool_latency_ms"], errors="coerce"), name=pick(ui_lang, "Tool latency", "Tool latency"), marker_color=SECONDARY, opacity=.45, yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="ms", showgrid=False))
    fig.update_layout(title=pick(ui_lang, "Eszközhívási telemetria", "Tool-calling telemetry"))
    return _base(fig, height=330)


def regression_figure(artifact: dict, ui_lang: str) -> go.Figure:
    metrics = ["recall@5", "mrr", "hit_rate", "precision@5"]
    names = [m for m in metrics if m in artifact]
    values = [float(artifact[m]) for m in names]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=[PRIMARY, ACCENT, SECONDARY, MUTED][: len(names)], hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(title=pick(ui_lang, "Legutóbbi regressziós retrieval eredmény", "Latest retrieval regression result"), showlegend=False)
    fig.update_yaxes(range=[0, max(1.05, max(values, default=1) * 1.1)])
    return _base(fig, height=320)


def catalog_status_figure(frame: pd.DataFrame, ui_lang: str) -> go.Figure:
    if frame.empty or "status" not in frame.columns:
        return go.Figure()
    counts = frame["status"].astype(str).value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = px.bar(counts, x="status", y="count", color_discrete_sequence=[PRIMARY], title=pick(ui_lang, "Benchmark családok státusz szerint", "Benchmark families by status"))
    return _base(fig, height=280)


def ab_footprint_figure(metrics_a: dict, metrics_b: dict, ui_lang: str) -> go.Figure:
    labels = [
        pick(ui_lang, "Bemeneti tokenek", "Input tokens"),
        pick(ui_lang, "Kimeneti tokenek", "Output tokens"),
        pick(ui_lang, "Késleltetés (ms)", "Latency (ms)"),
    ]
    a = [float(metrics_a.get("input_tokens", 0) or 0), float(metrics_a.get("output_tokens", 0) or 0), float(metrics_a.get("latency_ms", 0) or 0)]
    b = [float(metrics_b.get("input_tokens", 0) or 0), float(metrics_b.get("output_tokens", 0) or 0), float(metrics_b.get("latency_ms", 0) or 0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=a, name="A", marker_color=PRIMARY))
    fig.add_trace(go.Bar(x=labels, y=b, name="B", marker_color=SECONDARY))
    fig.update_layout(title=pick(ui_lang, "A/B erőforrás-lábnyom", "A/B resource footprint"), barmode="group")
    return _base(fig, height=320)


def ab_quality_figure(metrics_a: dict, metrics_b: dict, ui_lang: str) -> go.Figure:
    labels = [
        pick(ui_lang, "Válasz QA", "Output QA"),
        pick(ui_lang, "Grounding", "Grounding"),
        pick(ui_lang, "Kontextus QA", "Context QA"),
        pick(ui_lang, "Tool QA", "Tool QA"),
    ]
    keys = ["output_QA", "grounding_QA", "context_QA", "tool_QA"]
    a = [float(metrics_a.get(key, 0) or 0) for key in keys]
    b = [float(metrics_b.get(key, 0) or 0) for key in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=a, name="A", marker_color=PRIMARY))
    fig.add_trace(go.Bar(x=labels, y=b, name="B", marker_color=SECONDARY))
    fig.update_layout(title=pick(ui_lang, "A/B minőségi összehasonlítás", "A/B quality comparison"), barmode="group")
    fig.update_yaxes(range=[0, max(1.05, max(a + b, default=1) * 1.1)])
    return _base(fig, height=330)
