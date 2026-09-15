from __future__ import annotations

import html
import json
import re
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import graphviz as _graphviz
except ImportError:
    _graphviz = None

from i18n import column_label, localize_columns, localize_value, pick, stage_label, tr
from tkip.config import PROJECT_ROOT
from tkip.multi_index import MultiIndexManager
from tkip.workflow_graph import workflow_dot, workflow_rows


def fmt_score(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def safe_label(text: str, limit: int = 52) -> str:
    return re.sub(r"[^A-Za-z0-9_À-ž .:/()\-]", "", str(text or ""))[:limit]


def score_normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum, maximum = float(numeric.min()), float(numeric.max())
    if maximum <= minimum:
        return pd.Series([1.0 if maximum > 0 else 0.0] * len(numeric), index=numeric.index)
    return (numeric - minimum) / (maximum - minimum)


def stream_words(text: str):
    for token in re.split(r"(\s+)", text):
        yield token
        if token.strip():
            time.sleep(0.004)


def render_zoomable_dot(
    dot: str,
    *,
    key: str,
    ui_lang: str,
    title: str | None = None,
    default_zoom: int = 130,
) -> None:
    if title:
        st.markdown(f"#### {title}")
    zoom = st.slider(
        tr(ui_lang, "zoom"),
        70,
        240,
        default_zoom,
        10,
        key=f"{key}_zoom",
        label_visibility="collapsed",
    )
    try:
        if _graphviz is None:
            raise RuntimeError("python-graphviz package is not installed")
        svg_text = _graphviz.Source(dot).pipe(format="svg").decode("utf-8")
        svg_text = re.sub(r"^<\?xml[^>]*>\s*", "", svg_text)
        height = min(1600, max(720, int(780 * max(1.0, zoom / 100.0))))
        components.html(
            f"""
            <div style='background:#FFFFFF;border:1px solid #CBD5E1;border-radius:18px;padding:12px;'>
              <style>
                .tki-wf-wrap {{ overflow:auto; background:#FFFFFF; border-radius:12px; }}
                .tki-wf-wrap svg {{ width:{zoom}% !important; height:auto !important; }}
              </style>
              <div class='tki-wf-wrap'>{svg_text}</div>
            </div>
            """,
            height=height,
            scrolling=True,
        )
    except Exception:
        st.graphviz_chart(dot, use_container_width=True)


def render_header(kp, ui_lang: str, gemini_available: bool) -> None:
    documents = {chunk.document_id for chunk in kp.chunks}
    private_documents = {
        chunk.document_id for chunk in kp.chunks if chunk.source_type == "private"
    }
    gemini_label = tr(ui_lang, "gemini_verified") if gemini_available else pick(
        ui_lang,
        "Gemini nincs ellenőrizve",
        "Gemini not verified",
    )
    dot_class = "" if gemini_available else " off"
    st.markdown(
        f"""
<div class="tki-hero">
  <div class="tki-title">{html.escape(tr(ui_lang, 'title'))}</div>
  <div class="status-row">
    <span class="status-pill"><span class="status-dot{dot_class}"></span>{html.escape(gemini_label)}</span>
    <span class="status-pill">📚 {len(documents)} {html.escape(tr(ui_lang, 'documents'))}</span>
    <span class="status-pill">🔒 {len(private_documents)} {html.escape(tr(ui_lang, 'private'))}</span>
    <span class="status-pill">🧩 {len(kp.chunks):,} {html.escape(tr(ui_lang, 'indexed_chunks'))}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_pipeline(diag: dict, ui_lang: str) -> None:
    steps = diag.get("pipeline_steps", [])
    if not steps:
        st.info(tr(ui_lang, "no_pipeline_telemetry"))
        return

    render_zoomable_dot(
        workflow_dot("query", pipeline_steps=steps, language=ui_lang),
        key="request_execution_workflow",
        ui_lang=ui_lang,
        title=tr(ui_lang, "visual_execution"),
        default_zoom=140,
    )

    latency_frame = pd.DataFrame(
        [
            {
                pick(ui_lang, "lépés", "stage"): _stage_label(ui_lang, str(step.get("stage") or "")),
                "latency_ms": float(step.get("duration_ms") or 0),
            }
            for step in steps
            if float(step.get("duration_ms") or 0) > 0
        ]
    )
    left, right = st.columns([1.55, 1])
    with left:
        if not latency_frame.empty:
            st.markdown(f"#### {tr(ui_lang, 'stage_latency')}")
            st.bar_chart(latency_frame.set_index(latency_frame.columns[0]), height=330)
    with right:
        analysis = diag.get("query_analysis") or {}
        st.markdown(f"#### {tr(ui_lang, 'request_interpretation')}")
        st.metric(tr(ui_lang, "intent"), localize_value(ui_lang, diag.get("intent", "—")))
        st.write(f"**{tr(ui_lang, 'topics')}:** " + (", ".join(analysis.get("topics") or []) or "—"))
        st.write(
            f"**{tr(ui_lang, 'context_strategy')}:** "
            f"{localize_value(ui_lang, analysis.get('context_strategy') or '—')}"
        )
        st.write(
            f"**{tr(ui_lang, 'recommended_tools')}:** "
            + (", ".join(analysis.get("recommended_tools") or []) or tr(ui_lang, "none"))
        )
        st.write(
            f"**{tr(ui_lang, 'context')}:** {diag.get('context_selected', 0)} "
            f"{pick(ui_lang, 'darab', 'chunk')} · {diag.get('context_chars', 0):,} {pick(ui_lang, 'karakter', 'chars')}"
        )

    with st.expander(tr(ui_lang, "step_telemetry"), expanded=False):
        rows = []
        for index, step in enumerate(steps, 1):
            row = {
                "#": index,
                pick(ui_lang, "lépés", "stage"): _stage_label(ui_lang, str(step.get("stage") or "")),
                pick(ui_lang, "állapot", "status"): _status_label(ui_lang, str(step.get("status") or "")),
                pick(ui_lang, "késleltetés (ms)", "latency (ms)"): step.get("duration_ms"),
                pick(ui_lang, "részlet", "detail"): _stage_detail(ui_lang, step),
            }
            for key, value in (step.get("metrics") or {}).items():
                if not isinstance(value, (dict, list)):
                    row[column_label(ui_lang, key)] = value
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=min(640, 100 + 34 * len(rows)),
        )


def render_ranking(
    diag: dict,
    ui_lang: str,
    kp=None,
    previous_diag: dict | None = None,
) -> None:
    ranking = diag.get("ranking", [])
    if not ranking:
        st.info(pick(ui_lang, "Nincs ranking diagnosztika.", "No ranking diagnostics available."))
        return

    frame = pd.DataFrame(ranking)
    for column in ["bm25_score", "dense_score", "hybrid_score", "reranker_score"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    top = frame.head(10).copy()
    st.markdown(f"#### {tr(ui_lang, 'ranking_score_profile')}")
    score_columns = [
        column
        for column in ["bm25_score", "dense_score", "hybrid_score", "reranker_score"]
        if column in top.columns
    ]
    normalized = pd.DataFrame(index=top["final_rank"].astype(int))
    labels = {
        "bm25_score": "BM25",
        "dense_score": "Dense",
        "hybrid_score": "RRF Hybrid",
        "reranker_score": "Reranker",
    }
    for column in score_columns:
        normalized[labels[column]] = score_normalize(top[column]).values
    normalized.index.name = pick(ui_lang, "végső rang", "final rank")
    st.bar_chart(normalized, height=330)

    st.markdown(f"#### {tr(ui_lang, 'rank_movement')}")
    movement: dict[str, list[float]] = {}
    stages = ["BM25", "Dense", "Hybrid", pick(ui_lang, "Végső", "Final")]
    for _, row in top.head(min(6, len(top))).iterrows():
        name = f"#{int(row['final_rank'])} {str(row['document'])[:24]}"
        ranks = [row.get("bm25_rank"), row.get("dense_rank"), row.get("hybrid_rank"), row.get("final_rank")]
        movement[name] = [1.0 / max(1, float(value)) if pd.notna(value) else 0.0 for value in ranks]
    if movement:
        st.line_chart(pd.DataFrame(movement, index=stages), height=300)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"#### {tr(ui_lang, 'final_ranking_table')}")
        columns = [
            "final_rank",
            "bm25_rank",
            "dense_rank",
            "hybrid_rank",
            "document",
            "page",
            "chapter",
            "section",
            "chunk_type",
            "bm25_score",
            "dense_score",
            "hybrid_score",
            "reranker_score",
            "chunk_id",
            "snippet",
        ]
        shown = frame[[column for column in columns if column in frame.columns]].copy()
        for column in ["bm25_score", "dense_score", "hybrid_score", "reranker_score"]:
            if column in shown:
                shown[column] = shown[column].map(fmt_score)
        shown = shown.rename(columns=_ranking_column_labels(ui_lang))
        st.dataframe(shown, use_container_width=True, hide_index=True, height=min(580, 95 + 40 * len(shown)))
    with right:
        st.markdown(f"#### {tr(ui_lang, 'evidence_distribution')}")
        st.bar_chart(frame["document"].value_counts().head(10), height=300)
        context_frame = pd.DataFrame(diag.get("context_chunks", []))
        if not context_frame.empty and "chars" in context_frame:
            st.markdown(f"#### {tr(ui_lang, 'context_chunk_sizes')}")
            st.bar_chart(context_frame.set_index("rank")[["chars"]], height=250)

    with st.expander(tr(ui_lang, "ranking_method"), expanded=False):
        if ui_lang == "hu":
            st.markdown(
                "1. **BM25**: lexikális egyezések.\n"
                "2. **Szemantikus visszakeresés**: jelentésbeli hasonlóság.\n"
                "3. **RRF**: a két rangsor összevonása.\n"
                "4. **Újrarangsorolás**: a legerősebb jelöltek újrapontozása.\n"
                "5. **Kontextustervezés**: deduplikáció, diverzitás és kontextuskeret."
            )
        else:
            st.markdown(
                "1. **BM25**: lexical matching.\n"
                "2. **Dense retrieval**: semantic similarity.\n"
                "3. **RRF**: rank-list fusion.\n"
                "4. **Reranker**: rescoring of the strongest candidates.\n"
                "5. **Context engineering**: deduplication, diversity and context budget."
            )

    if kp is not None:
        st.markdown(f"#### {tr(ui_lang, 'persistent_index_comparison')}")
        manager = MultiIndexManager(kp.cfg)
        ready_variants = [item["name"] for item in manager.available() if item.get("ready")]
        query = (
            diag.get("retrieval_query")
            or diag.get("optimized_query")
            or diag.get("corrected_query")
            or diag.get("original_query")
        )
        if len(ready_variants) <= 1:
            st.info(
                pick(
                    ui_lang,
                    "Az összehasonlításhoz építs további indexváltozatokat a Könyvtár oldalon.",
                    "Build additional index variants on the Library page to compare them.",
                )
            )
        elif query and st.button(
            tr(ui_lang, "run_cross_index"),
            key=f"compare_indexes_{diag.get('request_id', 'run')}",
        ):
            rows: list[dict] = []
            document_ranks: dict[str, dict[str, float]] = {}
            for variant in ready_variants:
                try:
                    started = time.perf_counter()
                    if variant == "primary":
                        retriever = kp.retriever
                        chunks = kp.chunks
                    else:
                        chunks, _, retriever, _ = manager.load(variant, kp.embedder)
                    hits = retriever.search(query)[:8]
                    elapsed = (time.perf_counter() - started) * 1000
                    rows.append(
                        {
                            "index": variant,
                            "chunks": len(chunks),
                            "latency_ms": round(elapsed, 2),
                            "top1_document": hits[0].chunk.title if hits else None,
                            "top1_hybrid_score": hits[0].hybrid_score if hits else None,
                            "unique_docs_top8": len({hit.chunk.document_id for hit in hits}),
                        }
                    )
                    for rank, hit in enumerate(hits[:5], 1):
                        document_ranks.setdefault(hit.chunk.title[:36], {})[variant] = 1.0 / rank
                except Exception as exc:
                    rows.append({"index": variant, "error": str(exc)})
            comparison_frame = pd.DataFrame(rows)
            comparison_frame = comparison_frame.rename(
                columns=localize_columns(ui_lang, list(comparison_frame.columns))
            )
            st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            if document_ranks:
                st.bar_chart(pd.DataFrame(document_ranks).fillna(0.0).T, height=360)

    if previous_diag:
        current = diag.get("runtime_chunking") or {}
        previous = previous_diag.get("runtime_chunking") or {}
        if current != previous:
            st.markdown(f"#### {tr(ui_lang, 'chunk_experiment_compare')}")
            comparison = pd.DataFrame(
                [
                    {
                        pick(ui_lang, "futás", "run"): tr(ui_lang, "previous"),
                        "strategy": previous.get("strategy"),
                        "size": previous.get("size"),
                        "overlap": previous.get("overlap"),
                        "context_chunks": previous_diag.get("context_selected"),
                        "context_chars": previous_diag.get("context_chars"),
                    },
                    {
                        pick(ui_lang, "futás", "run"): tr(ui_lang, "current"),
                        "strategy": current.get("strategy"),
                        "size": current.get("size"),
                        "overlap": current.get("overlap"),
                        "context_chunks": diag.get("context_selected"),
                        "context_chars": diag.get("context_chars"),
                    },
                ]
            )
            comparison = comparison.rename(
                columns={
                    "strategy": pick(ui_lang, "stratégia", "strategy"),
                    "size": pick(ui_lang, "méret", "size"),
                    "overlap": pick(ui_lang, "átfedés", "overlap"),
                    "context_chunks": pick(ui_lang, "kontextusdarabok", "context chunks"),
                    "context_chars": pick(ui_lang, "kontextuskarakterek", "context characters"),
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)


def diagram_dot(diagram) -> str:
    lines = [
        "digraph G {",
        "rankdir=LR;",
        'graph [pad="0.2", nodesep="0.45", ranksep="0.65"];',
        'node [shape=box, style="rounded,filled", fillcolor="#f4f5f7", color="#8b93a1", fontname="Arial"];',
        'edge [color="#7b8492", fontname="Arial", fontsize=10];',
    ]
    known = set()
    for node in diagram.nodes[:12]:
        node_id = re.sub(r"[^A-Za-z0-9_]", "_", node.id) or "node"
        known.add(node.id)
        label = safe_label(node.label, 80).replace('"', "'")
        lines.append(f'{node_id} [label="{label}"];')
    id_map = {
        node.id: re.sub(r"[^A-Za-z0-9_]", "_", node.id) or "node"
        for node in diagram.nodes[:12]
    }
    for edge in diagram.edges[:18]:
        if edge.source not in known or edge.target not in known:
            continue
        label = safe_label(edge.label or "", 44).replace('"', "'")
        attribute = f' [label="{label}"]' if label else ""
        lines.append(f'{id_map[edge.source]} -> {id_map[edge.target]}{attribute};')
    lines.append("}")
    return "\n".join(lines)


def render_evidence(answer, diag: dict, ui_lang: str) -> None:
    st.markdown(f"#### {tr(ui_lang, 'source_citations')}")
    validation = diag.get("citation_validation") or {}
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric(tr(ui_lang, "citation_validator"), "PASS" if validation.get("valid") else "CHECK")
    col_2.metric(tr(ui_lang, "citations"), len(answer.sources))
    document_count = len(answer.used_documents or {source.document_title for source in answer.sources})
    col_3.metric(tr(ui_lang, "documents_used"), document_count)

    for index, source in enumerate(answer.sources, 1):
        location = " · ".join(
            value
            for value in [
                f"p. {source.page}" if source.page else None,
                source.chapter,
                source.section,
                f"chunk {source.chunk_id}",
            ]
            if value
        )
        st.markdown(
            f'<div class="evidence-card"><div class="evidence-title">[{index}] {html.escape(source.document_title)}</div>'
            f'<div class="evidence-meta">{html.escape(location)}</div>'
            f'<div class="evidence-text">{html.escape(source.quote_or_evidence)}</div></div>',
            unsafe_allow_html=True,
        )

    if getattr(answer, "visuals", None):
        st.markdown(f"#### {tr(ui_lang, 'grounded_source_image')}")
        for visual in answer.visuals[:2]:
            path = PROJECT_ROOT / visual.asset_path
            if path.exists():
                st.image(
                    str(path),
                    caption=visual.caption or f"{visual.document_title} · p. {visual.page}",
                    use_container_width=True,
                )

    st.markdown(f"#### {tr(ui_lang, 'tool_calling')}")
    analysis = diag.get("query_analysis") or {}
    expected = analysis.get("recommended_tools") or []
    if expected:
        st.write(f"**{tr(ui_lang, 'tool_router_recommended')}:** {', '.join(expected)}")
    tool_results = diag.get("tool_results", [])
    if not tool_results:
        st.caption(tr(ui_lang, "no_tools_selected"))
    for index, tool in enumerate(tool_results, 1):
        name = tool.get("tool", "tool") if isinstance(tool, dict) else "tool"
        status = tool.get("status", "unknown") if isinstance(tool, dict) else "unknown"
        with st.expander(f"{index}. {name} · {status}", expanded=False):
            if isinstance(tool, dict):
                st.write(f"**{tr(ui_lang, 'arguments')}**")
                st.json(tool.get("arguments", {}))
                result = tool.get("result", tool.get("tool_error"))
                raw = json.dumps(result, ensure_ascii=False, default=str, indent=2)
                st.code(raw[:9000] + ("\n…" if len(raw) > 9000 else ""), language="json")

    if answer.diagram and answer.diagram.nodes:
        st.markdown(f"#### {tr(ui_lang, 'answer_diagram')}")
        try:
            st.graphviz_chart(diagram_dot(answer.diagram), use_container_width=True)
        except Exception:
            st.code(diagram_dot(answer.diagram), language="dot")


def render_quality(answer, diag: dict, ui_lang: str) -> None:
    review = diag.get("quality_review") or {}
    query_review = diag.get("query_review") or {}
    pipeline_evaluation = diag.get("pipeline_evaluation") or {}
    query_analysis = diag.get("query_analysis") or {}

    st.markdown(f"#### {tr(ui_lang, 'quality_scorecard')}")
    score_rows = [
        (tr(ui_lang, "prompt_quality"), review.get("prompt_score")),
        (tr(ui_lang, "output_quality"), review.get("output_score")),
        (tr(ui_lang, "language_quality"), review.get("language_score")),
        (tr(ui_lang, "grounding"), review.get("grounding_score")),
        (tr(ui_lang, "topic_analysis"), pipeline_evaluation.get("topic_analysis_score")),
        (tr(ui_lang, "context_analysis"), pipeline_evaluation.get("context_analysis_score")),
        (tr(ui_lang, "tool_calling"), pipeline_evaluation.get("tool_calling_score")),
        (tr(ui_lang, "retrieval_quality"), pipeline_evaluation.get("retrieval_quality_score")),
        (tr(ui_lang, "citation_validity"), pipeline_evaluation.get("citation_score")),
    ]
    score_frame = pd.DataFrame(
        [{"dimension": label, "score": score} for label, score in score_rows if score is not None]
    )
    if not score_frame.empty:
        metric_columns = st.columns(min(5, len(score_frame)))
        for index, row in score_frame.head(5).iterrows():
            metric_columns[index % len(metric_columns)].metric(row["dimension"], f"{int(row['score'])}/100")
        st.bar_chart(score_frame.set_index("dimension"), height=330)

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {tr(ui_lang, 'topic_intent')}")
        st.write(
            f"**{tr(ui_lang, 'intent')}:** "
            f"{localize_value(ui_lang, diag.get('intent', '—'))}"
        )
        st.write(f"**{tr(ui_lang, 'topics')}:** " + (", ".join(query_analysis.get("topics") or []) or "—"))
        st.write(f"**{pick(ui_lang, 'Keretrendszerek', 'Frameworks')}:** " + (", ".join(query_analysis.get("frameworks") or []) or "—"))
        st.write(
            f"**{tr(ui_lang, 'complexity')}:** "
            f"{localize_value(ui_lang, query_analysis.get('complexity', '—'))}"
        )
        st.write(
            f"**{tr(ui_lang, 'context_strategy')}:** "
            f"{localize_value(ui_lang, query_analysis.get('context_strategy', '—'))}"
        )
    with right:
        st.markdown(f"#### {tr(ui_lang, 'context_tool_analysis')}")
        st.write(f"**{tr(ui_lang, 'context_documents')}:** {pipeline_evaluation.get('context_documents', '—')}")
        st.write(f"**{tr(ui_lang, 'context_chars_label')}:** {pipeline_evaluation.get('context_chars', '—')}")
        st.write(
            f"**{tr(ui_lang, 'expected_tools')}:** "
            + (", ".join(pipeline_evaluation.get("expected_tools") or []) or tr(ui_lang, "none"))
        )
        st.write(
            f"**{tr(ui_lang, 'actual_tools')}:** "
            + (", ".join(pipeline_evaluation.get("actual_tools") or []) or tr(ui_lang, "none"))
        )
        st.write(f"**{tr(ui_lang, 'avg_reranker')}:** {pipeline_evaluation.get('avg_reranker_score', '—')}")

    st.markdown(f"#### {tr(ui_lang, 'query_language_review')}")
    original = query_review.get("original", diag.get("original_query", ""))
    corrected = query_review.get("corrected", diag.get("corrected_query", ""))
    left, right = st.columns(2)
    left.text_area(tr(ui_lang, "original_query"), value=original or "", height=100, disabled=True)
    right.text_area(tr(ui_lang, "corrected_query"), value=corrected or "", height=100, disabled=True)

    prompt_optimization = diag.get("prompt_optimization") or {}
    if prompt_optimization and prompt_optimization.get("profile") != "none":
        st.markdown(f"#### {tr(ui_lang, 'prompt_transformation')}")
        left, right = st.columns(2)
        with left:
            st.caption(tr(ui_lang, "original_corrected_task"))
            st.code(str(prompt_optimization.get("original") or corrected or original), language="text")
        with right:
            st.caption(tr(ui_lang, "optimized"))
            st.code(
                str(prompt_optimization.get("optimized") or diag.get("optimized_query") or corrected),
                language="text",
            )

    issues = query_review.get("issues") or []
    if issues:
        st.warning(" · ".join(str(issue) for issue in issues))
    elif query_review:
        st.success(tr(ui_lang, "no_language_issue"))

    groups = [
        (pick(ui_lang, "Promptproblémák", "Prompt issues"), review.get("prompt_issues", [])),
        (pick(ui_lang, "Nyelvi problémák", "Language issues"), review.get("language_issues", [])),
        (pick(ui_lang, "Válaszproblémák", "Output issues"), review.get("output_issues", [])),
        (pick(ui_lang, "Javaslatok", "Recommendations"), review.get("recommendations", [])),
    ]
    for title, items in groups:
        if items:
            st.markdown(f"**{title}**")
            for item in items:
                st.write(f"- {item}")

    st.markdown(f"#### {tr(ui_lang, 'output_validation')}")
    validation = diag.get("citation_validation") or {}
    checks = {
        pick(ui_lang, "Strukturált Pydantic kimenet", "Structured Pydantic output"): True,
        pick(ui_lang, "Hivatkozás-validáció", "Citation validation"): bool(validation.get("valid")),
        pick(ui_lang, "Nincs-bizonyíték viselkedés", "Insufficient-evidence behavior"): (
            True if not answer.insufficient_evidence else len(answer.sources) == 0
        ),
        pick(ui_lang, "Tool allowlist végrehajtás", "Tool allowlist execution"): not any(
            item.get("tool_error")
            for item in diag.get("tool_results", [])
            if isinstance(item, dict)
        ),
        pick(ui_lang, "Validált forrásvizuál", "Trusted source visual"): bool(getattr(answer, "visuals", [])) or not answer.sources,
    }
    check_key = pick(ui_lang, "ellenőrzés", "check")
    status_key = pick(ui_lang, "állapot", "status")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    check_key: label,
                    status_key: pick(ui_lang, "MEGFELEL", "PASS")
                    if passed
                    else pick(ui_lang, "ELLENŐRIZD", "CHECK"),
                }
                for label, passed in checks.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(tr(ui_lang, "system_prompt"), expanded=False):
        st.code(diag.get("system_prompt", "—"), language="text")
    with st.expander(tr(ui_lang, "final_model_prompt"), expanded=False):
        prompt = diag.get("final_prompt", "—")
        st.code(prompt, language="text")
        st.caption(f"{tr(ui_lang, 'prompt_characters')}: {len(prompt):,}")


def render_system_workflow(ui_lang: str) -> None:
    tabs = st.tabs(
        [
            tr(ui_lang, "full_lifecycle"),
            tr(ui_lang, "query_path"),
            tr(ui_lang, "indexing_path"),
        ]
    )
    for tab, kind, zoom in [
        (tabs[0], "full", 135),
        (tabs[1], "query", 150),
        (tabs[2], "indexing", 145),
    ]:
        with tab:
            render_zoomable_dot(
                workflow_dot(kind, language=ui_lang),
                key=f"system_workflow_{kind}",
                ui_lang=ui_lang,
                default_zoom=zoom,
            )
            rows = pd.DataFrame(workflow_rows(kind, language=ui_lang))
            if not rows.empty:
                rows = rows.rename(
                    columns={
                        "number": "#",
                        "label": pick(ui_lang, "lépés", "step"),
                        "detail": pick(ui_lang, "művelet", "what happens"),
                        "group": pick(ui_lang, "fázis", "phase"),
                    }
                )
                columns = [
                    "#",
                    pick(ui_lang, "fázis", "phase"),
                    pick(ui_lang, "lépés", "step"),
                    pick(ui_lang, "művelet", "what happens"),
                ]
                st.dataframe(rows[columns], use_container_width=True, hide_index=True)


def render_answer(
    answer,
    ui_lang: str,
    is_new: bool,
    kp,
    previous_answer=None,
) -> None:
    tabs = st.tabs(
        [
            tr(ui_lang, "answer"),
            tr(ui_lang, "sources_tools"),
            tr(ui_lang, "retrieval_lab"),
            tr(ui_lang, "pipeline"),
            tr(ui_lang, "qa"),
        ]
    )
    diagnostics = answer.diagnostics or {}

    with tabs[0]:
        metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
        metric_1.metric(tr(ui_lang, "confidence"), f"{answer.confidence:.2f}")
        metric_2.metric(tr(ui_lang, "latency"), f"{answer.latency_ms or 0:,.0f} ms")
        metric_3.metric(tr(ui_lang, "citations"), len(answer.sources))
        metric_4.metric(tr(ui_lang, "tools"), len(answer.used_tools))
        metric_5.metric(tr(ui_lang, "cost"), f"${float(diagnostics.get('estimated_cost_usd') or 0):.5f}")

        if answer.insufficient_evidence:
            st.warning(tr(ui_lang, "insufficient_evidence"))
        with st.chat_message("assistant", avatar="🧠"):
            if is_new:
                st.write_stream(stream_words(answer.answer))
            else:
                st.markdown(answer.answer)

        if getattr(answer, "visuals", None) and answer.answer_type != "retrieval_only":
            visual = answer.visuals[0]
            path = PROJECT_ROOT / visual.asset_path
            if path.exists():
                st.markdown(f"#### {tr(ui_lang, 'grounded_source_image')}")
                st.image(
                    str(path),
                    caption=visual.caption or f"{visual.document_title} · p. {visual.page}",
                    use_container_width=True,
                )

        if answer.diagram and answer.diagram.nodes:
            st.markdown(f"#### {tr(ui_lang, 'visual_summary')}")
            try:
                st.graphviz_chart(diagram_dot(answer.diagram), use_container_width=True)
            except Exception:
                st.code(diagram_dot(answer.diagram), language="dot")

        if answer.follow_up_questions:
            st.markdown(f"#### {tr(ui_lang, 'followups')}")
            for follow_up in answer.follow_up_questions[:4]:
                st.write(f"- {follow_up}")

        if previous_answer is not None:
            with st.expander(tr(ui_lang, "compare_previous"), expanded=False):
                left, right = st.columns(2)
                with left:
                    st.caption(tr(ui_lang, "previous"))
                    st.markdown(previous_answer.answer)
                with right:
                    st.caption(tr(ui_lang, "current"))
                    st.markdown(answer.answer)

        feedback_1, feedback_2, trace_column = st.columns([1, 1, 3])
        if feedback_1.button(tr(ui_lang, "helpful"), key=f"helpful_{answer.request_id}", use_container_width=True):
            kp.telemetry.feedback(answer.request_id, True)
            st.toast(tr(ui_lang, "feedback_saved"))
        if feedback_2.button(tr(ui_lang, "not_helpful"), key=f"not_helpful_{answer.request_id}", use_container_width=True):
            kp.telemetry.feedback(answer.request_id, False)
            st.toast(tr(ui_lang, "feedback_saved"))
        trace_column.caption(f"request_id: {answer.request_id} · trace_id: {answer.trace_id}")

    with tabs[1]:
        render_evidence(answer, diagnostics, ui_lang)
    with tabs[2]:
        render_ranking(
            diagnostics,
            ui_lang,
            kp,
            previous_answer.diagnostics if previous_answer is not None else None,
        )
    with tabs[3]:
        render_pipeline(diagnostics, ui_lang)
    with tabs[4]:
        render_quality(answer, diagnostics, ui_lang)


def _status_label(ui_lang: str, status: str) -> str:
    return localize_value(ui_lang, status)


def _stage_label(ui_lang: str, stage: str) -> str:
    return stage_label(ui_lang, stage)


def _stage_detail(ui_lang: str, step: dict) -> str:
    """Return concise stage detail without leaking the other UI language."""
    detail = str(step.get("detail") or "")
    if ui_lang != "hu":
        return detail
    stage = str(step.get("stage") or "")
    concise = {
        "Input guardrails": "Bemenet ellenőrizve.",
        "Knowledge index": "A tudásindex készen áll.",
        "Index strategy": "Aktív index kiválasztva.",
        "Prompt language check": "Nyelvi ellenőrzés kész.",
        "Advanced prompt engineering": "Promptprofil alkalmazva.",
        "Query understanding": "Szándék és lekérdezés értelmezve.",
        "Topic & context analysis": "Téma- és kontextusigény meghatározva.",
        "Hybrid retrieval": "Hibrid jelöltek visszakeresve.",
        "Metadata filters": "Metadataszűrők alkalmazva.",
        "Reranker": "Jelöltek újrarangsorolva.",
        "Document diversity": "Dokumentumdiverzitás alkalmazva.",
        "Dynamic chunking": "Lekérdezéskori darabolás alkalmazva.",
        "Context engineering": "Végső kontextus összeállítva.",
        "Tool calling": "Eszközhívási lépés lezárva.",
        "Context refresh": "Kontextus frissítve.",
        "Gemini generation": "Forrásalapú válasz generálva.",
        "Citation validator": "Hivatkozások ellenőrizve.",
        "Source visual": "Forrásvizuál ellenőrizve.",
        "Prompt & output QA": "Kimeneti ellenőrzés lezárva.",
        "Response ready": "Válasz elkészült.",
    }
    return concise.get(stage, "")


def _ranking_column_labels(ui_lang: str) -> dict[str, str]:
    columns = [
        "final_rank",
        "bm25_rank",
        "dense_rank",
        "hybrid_rank",
        "document",
        "page",
        "chapter",
        "section",
        "chunk_type",
        "chunk_id",
        "snippet",
        "bm25_score",
        "dense_score",
        "hybrid_score",
        "reranker_score",
    ]
    return {name: column_label(ui_lang, name) for name in columns}
