from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from i18n import pick


@dataclass(frozen=True)
class WorkflowNode:
    key: str
    x: float
    y: float
    title_hu: str
    title_en: str
    type_name: str
    detail_hu: str
    detail_en: str
    symbol: str = "square"


PHASES = [
    (0.0, 3.55, "1", "Indexelés", "Indexing", "Dokumentumok → kereshető tudásréteg"),
    (3.55, 6.20, "2", "Kérdésértés és routing", "Query understanding & routing", "Kérdés → irányítás"),
    (6.20, 11.05, "3", "Párhuzamos retrieval", "Parallel retrieval", "BM25 + dense → fusion"),
    (11.05, 14.15, "4", "Generálás és reflexió", "Generation & reflection", "Tools + model + validation"),
    (14.15, 18.0, "5", "Értékelés és human loop", "Evaluation & human loop", "Review → feedback → improvement"),
]

NODES = [
    WorkflowNode("docs", 0.55, 3.00, "Dokumentumok", "Documents", "DocumentLoader", "PDF · DOCX · MD · HTML", "PDF · DOCX · MD · HTML", symbol="square"),
    WorkflowNode("parse", 1.55, 3.00, "Parsing", "Parsing", "DocumentTransformer", "struktúra · metaadat · QA", "structure · metadata · QA", symbol="hexagon"),
    WorkflowNode("chunk", 2.55, 3.00, "Chunking", "Chunking", "TextSplitter", "fixed · recursive · semantic", "fixed · recursive · semantic", symbol="diamond"),
    WorkflowNode("embed", 3.20, 2.00, "Embedding", "Embeddings", "Embeddings", "cache · batching · provider", "cache · batching · provider", symbol="pentagon"),
    WorkflowNode("index", 3.20, 4.05, "Indexek", "Indexes", "VectorStore + BM25", "primary · fixed · semantic", "primary · fixed · semantic", symbol="hexagon2"),
    WorkflowNode("question", 4.00, 3.00, "Felhasználói kérdés", "User question", "RunnableInput", "HU/EN · filterek · preset", "HU/EN · filters · preset", symbol="circle"),
    WorkflowNode("guard", 4.95, 3.00, "Guardrails", "Guardrails", "RunnableLambda", "injection · validáció", "injection · validation", symbol="triangle-up"),
    WorkflowNode("prompt", 5.82, 3.00, "Prompttervezés", "Prompt engineering", "PromptTemplate", "16 profil · optimalizálás", "16 profiles · optimization", symbol="star-square"),
    WorkflowNode("router", 6.55, 3.00, "Router", "Router", "RunnableBranch", "query type · filters · tools", "query type · filters · tools", symbol="diamond"),
    WorkflowNode("bm25", 7.65, 4.15, "BM25", "BM25", "Retriever", "lexikális keresés", "lexical search", symbol="circle"),
    WorkflowNode("dense", 7.65, 1.85, "Dense", "Dense", "Retriever", "szemantikus keresés", "semantic search", symbol="circle"),
    WorkflowNode("fusion", 8.90, 3.00, "RRF / Fusion", "RRF / Fusion", "RunnableParallel", "rangsorok egyesítése", "merge ranked lists", symbol="bowtie"),
    WorkflowNode("rerank", 9.95, 3.00, "Reranker", "Reranker", "Reranker", "jelöltek újrapontozása", "candidate rescoring", symbol="triangle-right"),
    WorkflowNode("context", 10.75, 3.00, "Context Builder", "Context Builder", "ContextBuilder", "dedup · diversity · budget", "dedup · diversity · budget", symbol="hexagon"),
    WorkflowNode("tools", 11.85, 4.10, "Tool Node", "Tool Node", "ToolNode", "allowlist · max steps", "allowlist · max steps", symbol="cross"),
    WorkflowNode("model", 12.55, 2.75, "Gemini / Chat Model", "Gemini / Chat Model", "ChatModel", "grounded synthesis", "grounded synthesis", symbol="star"),
    WorkflowNode("validate", 13.45, 2.75, "Structured Output", "Structured Output", "Pydantic", "JSON · citations · schema", "JSON · citations · schema", symbol="square"),
    WorkflowNode("evaluate", 14.85, 3.00, "Evaluate", "Evaluate", "Evaluator", "quality · latency · cost", "quality · latency · cost", symbol="circle"),
    WorkflowNode("review", 16.00, 3.00, "Review", "Review", "Human-in-the-loop", "emberi ellenőrzés", "human review", symbol="hexagon"),
    WorkflowNode("send", 17.10, 4.15, "Válasz", "Send", "Response", "jóváhagyott válasz", "approved answer", symbol="triangle-right"),
    WorkflowNode("feedback", 17.10, 1.85, "Feedback", "Feedback", "RegressionSuite", "failure → regression sample", "failure → regression sample", symbol="circle-open"),
]

EDGES = [
    ("docs", "parse", ""), ("parse", "chunk", ""), ("chunk", "embed", ""), ("embed", "index", ""),
    ("index", "question", "ready index"), ("question", "guard", ""), ("guard", "prompt", ""), ("prompt", "router", ""),
    ("router", "bm25", "lexical"), ("router", "dense", "semantic"), ("bm25", "fusion", ""), ("dense", "fusion", ""),
    ("fusion", "rerank", ""), ("rerank", "context", ""), ("context", "model", "context"),
    ("context", "tools", "tools"), ("tools", "model", "result"), ("model", "validate", ""), ("validate", "evaluate", ""),
    ("evaluate", "review", ""), ("review", "send", "approved"), ("review", "feedback", "rejected"),
    ("feedback", "prompt", "feedback loop"),
]


def _node_map() -> dict[str, WorkflowNode]:
    return {node.key: node for node in NODES}


def _add_phase_backgrounds(fig: go.Figure, ui_lang: str) -> None:
    fills = [
        "rgba(20,184,166,0.075)",
        "rgba(59,130,246,0.075)",
        "rgba(99,102,241,0.075)",
        "rgba(245,158,11,0.060)",
        "rgba(34,197,94,0.060)",
    ]
    phase_colors = ["#5eead4", "#93c5fd", "#a5b4fc", "#fbbf24", "#86efac"]
    for idx, (x0, x1, number, hu, en, _detail_hu) in enumerate(PHASES):
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0.45,
            y1=5.55,
            fillcolor=fills[idx],
            line=dict(color="rgba(148,163,184,0.20)", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=5.25,
            text=f"<b>{number}. {pick(ui_lang, hu, en)}</b>",
            showarrow=False,
            font=dict(size=13, color=phase_colors[idx]),
            align="center",
        )


def _add_edges(fig: go.Figure, ui_lang: str) -> None:
    nodes = _node_map()
    for source, target, label in EDGES:
        a = nodes[source]
        b = nodes[target]
        color = "#64748b"
        dash = "solid"
        width = 1.7
        if source == "router" and target == "bm25":
            color = "#60a5fa"
        elif (source == "router" and target == "dense") or (
            source in {"bm25", "dense"} and target == "fusion"
        ):
            color = "#818cf8"
        elif source == "context" and target == "tools":
            color = "#f59e0b"
            dash = "dot"
        elif source == "tools" and target == "model":
            color = "#f59e0b"
        elif source == "review" and target == "send":
            color = "#4ade80"
        elif source == "review" and target == "feedback":
            color = "#c084fc"
            dash = "dot"
        elif source == "feedback":
            color = "#c084fc"
            dash = "dot"
            width = 1.9

        # Orthogonal-ish routing gives the graph a workflow-editor feel.
        if abs(a.y - b.y) > 0.2:
            mid_x = (a.x + b.x) / 2
            fig.add_shape(type="line", x0=a.x, y0=a.y, x1=mid_x, y1=a.y, line=dict(color=color, width=width, dash=dash), layer="below")
            fig.add_shape(type="line", x0=mid_x, y0=a.y, x1=mid_x, y1=b.y, line=dict(color=color, width=width, dash=dash), layer="below")
            fig.add_annotation(x=b.x, y=b.y, ax=mid_x, ay=b.y, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=width, arrowcolor=color)
        else:
            fig.add_annotation(x=b.x, y=b.y, ax=a.x, ay=a.y, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=width, arrowcolor=color)

        if label:
            fig.add_annotation(
                x=(a.x + b.x) / 2,
                y=(a.y + b.y) / 2 + (0.22 if source != "feedback" else -0.25),
                text=label,
                showarrow=False,
                font=dict(size=8, color="#94a3b8"),
                bgcolor="rgba(15,23,42,0.72)",
                borderpad=2,
            )


def _node_label(node: WorkflowNode, ui_lang: str) -> str:
    title = pick(ui_lang, node.title_hu, node.title_en)
    return f"<b>{title}</b><br><span style='font-size:9px;color:#94a3b8'>{node.type_name}</span>"


def _node_style(node: WorkflowNode) -> tuple[str, str, int]:
    """Return restrained phase color and node size for the orchestration graph."""
    if node.key in {"docs", "parse", "chunk", "embed", "index"}:
        return "#123337", "#2dd4bf", 48
    if node.key in {"question", "guard", "prompt", "router"}:
        return "#172554", "#60a5fa", 48
    if node.key in {"bm25", "dense", "fusion", "rerank", "context"}:
        return "#25235a", "#818cf8", 48
    if node.key in {"tools", "model", "validate"}:
        return "#3a2d17", "#f59e0b", 49
    if node.key in {"evaluate", "review", "send"}:
        return "#163326", "#4ade80", 48
    return "#2e2047", "#c084fc", 48


def _add_nodes(fig: go.Figure, ui_lang: str) -> None:
    for node in NODES:
        title = pick(ui_lang, node.title_hu, node.title_en)
        detail = pick(ui_lang, node.detail_hu, node.detail_en)
        fill_color, border_color, marker_size = _node_style(node)
        fig.add_trace(
            go.Scatter(
                x=[node.x],
                y=[node.y],
                mode="markers+text",
                marker=dict(
                    symbol=node.symbol,
                    size=marker_size,
                    color=fill_color,
                    line=dict(color=border_color, width=2.0),
                ),
                text=[_node_label(node, ui_lang)],
                textposition="bottom center" if node.key not in {"bm25", "dense", "tools", "send", "feedback"} else "middle right",
                textfont=dict(size=10, color="#e5e7eb"),
                hovertemplate=(
                    f"<b>{title}</b><br>"
                    f"{node.type_name}<br>"
                    f"{detail}<extra></extra>"
                ),
                showlegend=False,
                cliponaxis=False,
            )
        )


def orchestration_figure(kp, ui_lang: str) -> go.Figure:
    fig = go.Figure()
    _add_phase_backgrounds(fig, ui_lang)
    _add_edges(fig, ui_lang)
    _add_nodes(fig, ui_lang)

    documents = len({chunk.document_id for chunk in kp.chunks})
    fig.add_annotation(
        x=0.15,
        y=0.72,
        xanchor="left",
        text=pick(
            ui_lang,
            f"Dokumentumok: {documents} · Chunkok: {len(kp.chunks):,} · Hover = részletek · Scroll/zoom engedélyezett",
            f"Documents: {documents} · Chunks: {len(kp.chunks):,} · Hover for details · Scroll/zoom enabled",
        ),
        showarrow=False,
        font=dict(size=9, color="#94a3b8"),
    )

    fig.update_xaxes(visible=False, range=[-0.15, 18.2], fixedrange=False)
    fig.update_yaxes(visible=False, range=[0.25, 5.75], fixedrange=False)
    fig.update_layout(
        height=690,
        margin=dict(l=8, r=8, t=12, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        hoverlabel=dict(bgcolor="#111827", font_color="#e5e7eb", bordercolor="#475569"),
        dragmode="pan",
        modebar=dict(bgcolor="rgba(15,23,42,.75)", color="#94a3b8", activecolor="#e5e7eb"),
    )
    return fig


def render_orchestration_graph(kp, ui_lang: str) -> None:
    st.plotly_chart(
        orchestration_figure(kp, ui_lang),
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
        key="orchestration_graph_plotly",
    )
