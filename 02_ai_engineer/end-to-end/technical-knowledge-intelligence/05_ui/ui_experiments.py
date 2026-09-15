from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from i18n import (
    benchmark_family_label,
    benchmark_term_label,
    budget_label,
    chunk_label,
    localize_columns,
    localize_value,
    normalize_language,
    pick,
    profile_label,
    tr,
)
from tkip.config import PROJECT_ROOT
from tkip.gemini_service import GeminiService
from tkip.models import AskRequest
from tkip.multi_index import MultiIndexManager
from tkip.presets import ANSWER_PRESETS, CHUNK_PRESETS
from tkip.prompt_engineering import PROFILE_KEYS, PROFILES, local_optimize
from ui_charts import (
    ab_footprint_figure,
    ab_quality_figure,
    catalog_status_figure,
    deterministic_rag_figure,
    prompt_footprint_figure,
    prompt_heatmap,
    quality_latency_scatter,
    regression_figure,
    retrieval_latency_figure,
    retrieval_quality_figure,
    retrieval_radar,
    tool_telemetry_figure,
)


def render_benchmarking(
    kp,
    ui_lang: str,
    effective_key: str | None,
    gemini_verified: bool,
    questions: dict[str, list[str]],
) -> None:
    ui_lang = normalize_language(ui_lang)
    st.markdown(f"### {tr(ui_lang, 'benchmark_title')}")
    _render_benchmark_snapshot(ui_lang)
    tabs = st.tabs(
        [
            tr(ui_lang, "retrieval_chunking"),
            tr(ui_lang, "prompt_engineering"),
            tr(ui_lang, "rag_structured"),
            tr(ui_lang, "tool_calling"),
            tr(ui_lang, "robustness_regression"),
            tr(ui_lang, "benchmark_catalog"),
            tr(ui_lang, "experiment_traces"),
        ]
    )

    with tabs[0]:
        _render_section_header(
            ui_lang,
            pick(
                ui_lang, "Visszakeresés és chunking benchmark", "Retrieval and chunking benchmark"
            ),
            pick(
                ui_lang,
                "Az oldal előre betöltött offline benchmarkeredményeket is megmutat, így akkor is látszanak a fő metrikák, ha most nem futtatsz új mérést.",
                "This page also loads precomputed offline benchmark results, so the main metrics are visible even when you do not run a new experiment now.",
            ),
        )
        overview = _default_retrieval_summary()
        if isinstance(overview, pd.DataFrame) and not overview.empty:
            _render_retrieval_overview_cards(overview, ui_lang)
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(
                    retrieval_quality_figure(overview, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="retrieval_overview_quality",
                )
            with chart_right:
                st.plotly_chart(
                    retrieval_latency_figure(overview, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="retrieval_overview_latency",
                )
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(
                    quality_latency_scatter(overview, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="retrieval_overview_quality_latency",
                )
            with chart_right:
                st.plotly_chart(
                    retrieval_radar(overview, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="retrieval_overview_radar",
                )

        with st.expander(tr(ui_lang, "metric_dictionary"), expanded=False):
            _render_metric_guide(_retrieval_metric_rows(ui_lang), ui_lang)

        manager = MultiIndexManager(kp.cfg)
        ready_variants = [item["name"] for item in manager.available() if item.get("ready")]
        col_a, col_b = st.columns([1, 1.25])
        sample_count = col_a.slider(tr(ui_lang, "evaluation_sample_count"), 50, 400, 150, 25)
        selected_variants = col_b.multiselect(
            tr(ui_lang, "persistent_variants"),
            ready_variants,
            default=ready_variants,
        )

        if st.button(
            tr(ui_lang, "run_retrieval_chunk_benchmark"),
            type="primary",
            use_container_width=True,
            disabled=not selected_variants,
        ):
            from tkip.evaluation import benchmark_index_variants, generate_eval_dataset

            with st.spinner(tr(ui_lang, "running_cross_index")):
                samples = generate_eval_dataset(kp.chunks, sample_count)
                raw, summary, variant_meta = benchmark_index_variants(
                    kp.retriever,
                    manager,
                    kp.embedder,
                    samples,
                    kp.cfg,
                    selected_variants,
                )
                st.session_state["benchmark_samples"] = samples
                st.session_state["benchmark_raw"] = raw
                st.session_state["benchmark_summary"] = summary
                st.session_state["benchmark_variant_meta"] = variant_meta

        summary = st.session_state.get("benchmark_summary")
        if not isinstance(summary, pd.DataFrame) or summary.empty:
            summary = _default_retrieval_summary()
        raw = st.session_state.get("benchmark_raw")
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raw = _default_retrieval_raw()
        variant_meta = st.session_state.get("benchmark_variant_meta")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            st.markdown(f"#### {tr(ui_lang, 'cross_index_summary')}")
            display_columns = [
                "index_variant",
                "method",
                "samples",
                "recall@1",
                "recall@3",
                "recall@5",
                "recall@10",
                "precision@5",
                "mrr",
                "ndcg@5",
                "hit_rate",
                "p50_latency_ms",
                "p95_latency_ms",
                "chunk_count",
                "avg_chunk_chars",
                "median_chunk_chars",
            ]
            display_columns = [column for column in display_columns if column in summary.columns]
            summary_display = summary[display_columns].rename(
                columns=localize_columns(ui_lang, display_columns)
            )
            st.dataframe(
                summary_display,
                use_container_width=True,
                hide_index=True,
                height=min(720, 130 + 34 * len(summary)),
            )

            sort_columns = [
                column for column in ["recall@5", "mrr", "ndcg@5"] if column in summary.columns
            ]
            best = (
                summary.sort_values(sort_columns, ascending=False).iloc[0]
                if sort_columns
                else summary.iloc[0]
            )
            fastest = (
                summary.sort_values("p50_latency_ms").iloc[0]
                if "p50_latency_ms" in summary.columns
                else summary.iloc[0]
            )
            m1, m2, m3, m4 = st.columns(4)
            if "recall@5" in summary.columns:
                m1.metric(tr(ui_lang, "best_recall"), f"{100 * float(best['recall@5']):.1f}%")
            if "mrr" in summary.columns:
                m2.metric(tr(ui_lang, "best_mrr"), f"{float(best['mrr']):.3f}")
            if "ndcg@5" in summary.columns:
                m3.metric(tr(ui_lang, "best_ndcg"), f"{float(best.get('ndcg@5', 0)):.3f}")
            if "p50_latency_ms" in summary.columns:
                m4.metric(tr(ui_lang, "fastest_p50"), f"{float(fastest['p50_latency_ms']):.1f} ms")

            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    retrieval_quality_figure(summary, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="retrieval_live_quality",
                )
            with right:
                st.plotly_chart(
                    retrieval_latency_figure(summary, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="retrieval_live_latency",
                )

            if variant_meta:
                st.markdown(f"#### {tr(ui_lang, 'chunk_structural_profile')}")
                metadata_frame = pd.DataFrame(
                    [{"index_variant": name, **metadata} for name, metadata in variant_meta.items()]
                )
                metadata_display = metadata_frame.rename(
                    columns=localize_columns(ui_lang, list(metadata_frame.columns))
                )
                st.dataframe(metadata_display, use_container_width=True, hide_index=True)

        if isinstance(raw, pd.DataFrame) and not raw.empty:
            with st.expander(tr(ui_lang, "raw_benchmark_rows"), expanded=False):
                raw_display = raw.rename(columns=localize_columns(ui_lang, list(raw.columns)))
                st.dataframe(raw_display, use_container_width=True, hide_index=True, height=430)

    with tabs[1]:
        _render_section_header(
            ui_lang,
            pick(ui_lang, "Prompttervezési benchmark", "Prompt engineering benchmark"),
            pick(
                ui_lang,
                "Itt látható, hogyan változik a kérdés átfogalmazása, a grounded válasz minősége és a token/költség lábnyom a különböző promptprofilok között.",
                "This section shows how question rewriting, grounded answer quality and token/cost footprint change across prompt profiles.",
            ),
        )
        pcols = st.columns(4)
        pcols[0].metric(tr(ui_lang, "prompt_profiles"), len(PROFILE_KEYS))
        pcols[1].metric(
            pick(ui_lang, "Próbakérdések", "Question bank"),
            len(questions[normalize_language(ui_lang)]),
        )
        pcols[2].metric(pick(ui_lang, "Alapértelmezett profilok", "Default profiles"), 4)
        pcols[3].metric(
            pick(ui_lang, "Élő modellmérés", "Live model benchmark"),
            pick(ui_lang, "Gemini", "Gemini"),
        )
        with st.expander(tr(ui_lang, "metric_dictionary"), expanded=False):
            _render_metric_guide(_prompt_metric_rows(ui_lang), ui_lang)
        if not effective_key:
            st.warning(tr(ui_lang, "prompt_benchmark_key_required"))
        elif not gemini_verified:
            st.warning(tr(ui_lang, "prompt_benchmark_verify"))

        profile_map = _profile_options(ui_lang)
        default_profiles = {
            "rag_grounded",
            "technical_deep_dive",
            "costar",
            "crispe",
        }
        selected_profiles = st.multiselect(
            tr(ui_lang, "prompt_profiles"),
            list(profile_map),
            default=[label for label, key in profile_map.items() if key in default_profiles],
            max_selections=8,
        )
        question_bank = questions[normalize_language(ui_lang)]
        selected_questions = st.multiselect(
            tr(ui_lang, "benchmark_questions"),
            question_bank,
            default=question_bank[:3],
            max_selections=8,
        )

        budget_map = _budget_label_map(ui_lang)
        chunk_map = _chunk_label_map(ui_lang)
        budget_col, chunk_col = st.columns(2)
        prompt_budget_label = budget_col.selectbox(
            tr(ui_lang, "answer_budget"),
            list(budget_map),
            index=1,
            key="prompt_bench_budget",
        )
        prompt_chunk_label = chunk_col.selectbox(
            tr(ui_lang, "chunk_profile"),
            list(chunk_map),
            index=1,
            key="prompt_bench_chunk",
        )

        preview_question = selected_questions[0] if selected_questions else question_bank[0]
        preview_rows = []
        for label in selected_profiles[:4]:
            profile_key = profile_map[label]
            transformed = local_optimize(preview_question, profile_key, ui_lang)
            preview_rows.append(
                {
                    pick(ui_lang, "Profil", "Profile"): label,
                    pick(ui_lang, "Eredeti kérdés", "Original question"): preview_question,
                    pick(ui_lang, "Átalakított prompt", "Transformed prompt"): transformed.get(
                        "optimized"
                    ),
                }
            )
        if preview_rows:
            st.markdown(
                pick(ui_lang, "#### Mintaprompt-átalakítások", "#### Sample transformed prompts")
            )
            st.dataframe(
                pd.DataFrame(preview_rows), use_container_width=True, hide_index=True, height=250
            )
            st.markdown(
                pick(
                    ui_lang,
                    "#### Példa prompt-transzformáció JSON",
                    "#### Example prompt-transformation JSON",
                )
            )
            st.code(
                json.dumps(
                    local_optimize(preview_question, profile_map[selected_profiles[0]], ui_lang),
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )

        can_run = bool(effective_key and selected_profiles and selected_questions)
        if st.button(
            tr(ui_lang, "run_prompt_benchmark"),
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        ):
            probe = GeminiService(kp.cfg, api_key=effective_key)
            with st.spinner(tr(ui_lang, "validating_before_benchmark")):
                api_ok, api_message = probe.test_connection()
            if not api_ok:
                st.error(f"{tr(ui_lang, 'benchmark_validation_failed')}\n\n{api_message}")
                return
            _run_prompt_benchmark(
                kp=kp,
                ui_lang=ui_lang,
                effective_key=effective_key,
                selected_profiles=selected_profiles,
                selected_questions=selected_questions,
                profile_map=profile_map,
                budget_key=budget_map[prompt_budget_label],
                chunk_key=chunk_map[prompt_chunk_label],
            )

        prompt_frame = st.session_state.get("prompt_benchmark_df")
        if isinstance(prompt_frame, pd.DataFrame) and not prompt_frame.empty:
            prompt_display = prompt_frame.rename(
                columns=localize_columns(ui_lang, list(prompt_frame.columns))
            )
            st.dataframe(prompt_display, use_container_width=True, hide_index=True, height=420)
            successful = prompt_frame[prompt_frame["status"] == "success"].copy()
            if not successful.empty:
                quality_columns = [
                    column
                    for column in [
                        "prompt_score",
                        "output_score",
                        "grounding_score",
                        "context_score",
                        "tool_score",
                    ]
                    if column in successful.columns
                ]
                if quality_columns:
                    st.markdown(f"#### {tr(ui_lang, 'quality_by_prompt')}")
                    st.plotly_chart(
                        prompt_heatmap(successful, ui_lang),
                        use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": True},
                        key="prompt_benchmark_heatmap",
                    )
                footprint_columns = [
                    column
                    for column in ["input_tokens", "output_tokens", "latency_ms", "cost_usd"]
                    if column in successful.columns
                ]
                if footprint_columns:
                    st.markdown(f"#### {tr(ui_lang, 'token_latency_cost')}")
                    st.plotly_chart(
                        prompt_footprint_figure(successful, ui_lang),
                        use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": True},
                        key="prompt_benchmark_footprint",
                    )
                    footprint = (
                        successful.groupby("profile")[footprint_columns].mean().reset_index()
                    )
                    footprint = footprint.rename(
                        columns=localize_columns(ui_lang, list(footprint.columns))
                    )
                    with st.expander(
                        pick(ui_lang, "Nyers prompt footprint adatok", "Raw prompt footprint data"),
                        expanded=False,
                    ):
                        st.dataframe(footprint, use_container_width=True, hide_index=True)

    with tabs[2]:
        _render_section_header(
            ui_lang,
            pick(ui_lang, "RAG és strukturált kimenet", "RAG and structured output"),
            pick(
                ui_lang,
                "A retrieval minősége önmagában nem elég: itt együtt látszik a grounded válasz, a hivatkozások, a structured output és a rendszer-lábnyom.",
                "Retrieval quality alone is not enough: this section combines grounded answer quality, citations, structured output and system footprint.",
            ),
        )
        deterministic_path = (
            PROJECT_ROOT / "07_results" / "evaluation" / "generation_deterministic_metrics.csv"
        )
        deterministic = (
            pd.read_csv(deterministic_path) if deterministic_path.exists() else pd.DataFrame()
        )
        if not deterministic.empty:
            means = deterministic.mean(numeric_only=True)
            rcols = st.columns(4)
            rcols[0].metric(
                pick(ui_lang, "Hivatkozáshelyesség", "Citation correctness"),
                f"{100 * float(means.get('citation_correctness', 0)):.0f}%",
            )
            rcols[1].metric(
                pick(ui_lang, "Hivatkozáslefedettség", "Citation completeness"),
                f"{100 * float(means.get('citation_completeness', 0)):.0f}%",
            )
            rcols[2].metric(
                pick(ui_lang, "Nincs-válasz pontosság", "No-answer accuracy"),
                f"{100 * float(means.get('no_answer_accuracy', 0)):.0f}%",
            )
            rcols[3].metric(
                pick(ui_lang, "Strukturált kimenet", "Structured output"),
                f"{100 * float(means.get('structured_output_validity', 0)):.0f}%",
            )
        st.dataframe(_rag_metric_frame(ui_lang), use_container_width=True, hide_index=True)
        with st.expander(tr(ui_lang, "metric_dictionary"), expanded=False):
            _render_metric_guide(_rag_metric_rows(ui_lang), ui_lang)
        if deterministic_path.exists():
            st.markdown(f"#### {tr(ui_lang, 'existing_deterministic_validation')}")
            deterministic_display = deterministic.head(100).rename(
                columns=localize_columns(ui_lang, list(deterministic.columns))
            )
            st.dataframe(
                deterministic_display, use_container_width=True, hide_index=True, height=340
            )
            score_columns = [
                column
                for column in [
                    "citation_correctness",
                    "citation_completeness",
                    "no_answer_accuracy",
                    "structured_output_validity",
                    "faithfulness",
                    "answer_correctness",
                    "hallucination_rate",
                ]
                if column in deterministic.columns
            ]
            if score_columns:
                st.plotly_chart(
                    deterministic_rag_figure(deterministic, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="rag_deterministic_metrics",
                )
        demo_answer = _load_json_example("ask_demo.json")
        if demo_answer:
            st.markdown(pick(ui_lang, "#### Példa válaszartifact", "#### Example answer artifact"))
            st.code(json.dumps(demo_answer, ensure_ascii=False, indent=2)[:4000], language="json")
        st.caption(
            pick(
                ui_lang,
                "A faithfulness, answer correctness és hallucination metrikák csak címkézett golden set, emberi review vagy explicit judge mellett értelmezhetők igazán.",
                "Faithfulness, answer correctness and hallucination metrics are most meaningful with a labeled golden set, human review or an explicit judge.",
            )
        )

    with tabs[3]:
        _render_section_header(
            ui_lang,
            pick(ui_lang, "Eszközhívási benchmark", "Tool-calling benchmark"),
            pick(
                ui_lang,
                "Itt az a kérdés, hogy a rendszer jó eszközt választ-e, jó argumentumokkal, felesleges hívások nélkül, és ezek után valóban megoldja-e a feladatot.",
                "The focus here is whether the system chooses the right tool, with the right arguments, without unnecessary calls, and then actually completes the task.",
            ),
        )
        tcols = st.columns(4)
        tcols[0].metric(pick(ui_lang, "Metrikacsalád", "Metric family"), 8)
        tcols[1].metric(
            pick(ui_lang, "Sémavalidáció", "Schema validation"), pick(ui_lang, "aktív", "enabled")
        )
        tcols[2].metric(
            pick(ui_lang, "Tool allowlist", "Tool allowlist"), pick(ui_lang, "aktív", "enabled")
        )
        tcols[3].metric(
            pick(ui_lang, "Max. tool step", "Max tool steps"),
            int(kp.cfg.get("tools", {}).get("max_calls", 2) or 2),
        )
        st.dataframe(_tool_metric_frame(ui_lang), use_container_width=True, hide_index=True)
        with st.expander(tr(ui_lang, "metric_dictionary"), expanded=False):
            _render_metric_guide(_tool_metric_rows(ui_lang), ui_lang)
        telemetry_rows = kp.telemetry.recent(500)
        tool_rows = []
        for row in telemetry_rows:
            if row.get("tool_calls") or row.get("tool_calling_score") is not None:
                tool_rows.append(
                    {
                        "timestamp": row.get("timestamp"),
                        "query_type": row.get("query_type"),
                        "tool_calls": row.get("tool_calls"),
                        "tool_score": row.get("tool_calling_score"),
                        "tool_latency_ms": row.get("tool_latency_ms"),
                        "cost": row.get("estimated_cost_usd"),
                    }
                )
        if tool_rows:
            st.markdown(f"#### {tr(ui_lang, 'recent_tool_telemetry')}")
            tool_frame = pd.DataFrame(tool_rows)
            st.plotly_chart(
                tool_telemetry_figure(tool_frame, ui_lang),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
                key="tool_telemetry_chart",
            )
            display_tool_frame = tool_frame.copy()
            if "query_type" in display_tool_frame.columns:
                display_tool_frame["query_type"] = display_tool_frame["query_type"].map(
                    lambda value: localize_value(ui_lang, value)
                )
            display_tool_frame = display_tool_frame.rename(
                columns=localize_columns(ui_lang, list(display_tool_frame.columns))
            )
            with st.expander(
                pick(ui_lang, "Nyers eszközhívási telemetria", "Raw tool-calling telemetry"),
                expanded=False,
            ):
                st.dataframe(
                    display_tool_frame, use_container_width=True, hide_index=True, height=340
                )
        else:
            st.info(tr(ui_lang, "no_tool_telemetry"))
        st.markdown(pick(ui_lang, "#### Példa eszközhívási JSON", "#### Example tool-calling JSON"))
        st.code(
            json.dumps(
                {
                    "tool_name": "search_library",
                    "arguments": {
                        "query": "BM25 vs dense retrieval",
                        "top_k": 5,
                        "source_type": "private",
                    },
                    "expected_checks": ["schema_valid", "tool_allowed", "results_non_empty"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )

    with tabs[4]:
        _render_section_header(
            ui_lang,
            pick(
                ui_lang,
                "Hallucináció, promptinjekció és regresszió",
                "Hallucination, prompt injection and regression",
            ),
            pick(
                ui_lang,
                "Ezek a tesztek azt mutatják meg, mennyire biztonságos és stabil a rendszer, valamint hogy egy módosítás után romlott-e valami a korábbi működéshez képest.",
                "These tests show how safe and stable the system is, and whether any change regressed previous behavior.",
            ),
        )
        regression_path = PROJECT_ROOT / "07_results" / "benchmarks" / "regression_console.json"
        regression_artifact = None
        if regression_path.exists():
            try:
                regression_artifact = json.loads(regression_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                regression_artifact = None
        if isinstance(regression_artifact, dict):
            hcols = st.columns(4)
            hcols[0].metric(
                "Recall@5", f"{100 * float(regression_artifact.get('recall@5', 0)):.1f}%"
            )
            hcols[1].metric("MRR", f"{float(regression_artifact.get('mrr', 0)):.3f}")
            hcols[2].metric(
                "Hit Rate", f"{100 * float(regression_artifact.get('hit_rate', 0)):.0f}%"
            )
            hcols[3].metric(
                pick(ui_lang, "Retrieval latency", "Retrieval latency"),
                f"{float(regression_artifact.get('latency_ms', 0)):.2f} ms",
            )
        st.dataframe(_robustness_frame(ui_lang), use_container_width=True, hide_index=True)
        with st.expander(tr(ui_lang, "metric_dictionary"), expanded=False):
            _render_metric_guide(_robustness_metric_rows(ui_lang), ui_lang)
        if isinstance(regression_artifact, dict):
            st.plotly_chart(
                regression_figure(regression_artifact, ui_lang),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
                key="regression_metrics_chart",
            )
            with st.expander(
                pick(ui_lang, "Regressziós artifact · JSON", "Regression artifact · JSON"),
                expanded=False,
            ):
                st.json(regression_artifact)

    with tabs[5]:
        from tkip.benchmark_catalog import catalog_rows

        _render_section_header(
            ui_lang,
            pick(ui_lang, "Teljes AI/NLP mérési katalógus", "Complete AI/NLP benchmark catalog"),
            pick(
                ui_lang,
                "Itt tematikusan, családonként rendezve látszanak a lehetséges benchmarkok, módszerek és mérőszámok.",
                "This page organizes benchmark families, methods and metrics into a more readable catalog.",
            ),
        )
        catalog = pd.DataFrame(catalog_rows())
        if "benchmark_family" in catalog.columns:
            catalog["benchmark_family"] = catalog["benchmark_family"].map(
                lambda value: benchmark_family_label(ui_lang, value)
            )
        if "status" in catalog.columns:
            catalog["status"] = catalog["status"].map(lambda value: localize_value(ui_lang, value))
        for column in ("methods", "metrics"):
            if column in catalog.columns:
                catalog[column] = catalog[column].map(
                    lambda value: [
                        benchmark_term_label(ui_lang, item.strip())
                        for item in str(value).split(" | ")
                    ]
                )
        families = catalog["benchmark_family"].tolist() if not catalog.empty else []
        if not catalog.empty:
            ccols = st.columns(4)
            ccols[0].metric(pick(ui_lang, "Mérési családok", "Benchmark families"), len(catalog))
            ccols[1].metric(
                pick(ui_lang, "Összes módszer", "Total methods"),
                sum(len(items) for items in catalog["methods"]),
            )
            ccols[2].metric(
                pick(ui_lang, "Összes metrika", "Total metrics"),
                sum(len(items) for items in catalog["metrics"]),
            )
            ccols[3].metric(
                pick(ui_lang, "Implementált / részleges", "Implemented / partial"),
                int(catalog["status"].astype(str).str.contains("implement", case=False).sum()),
            )
            st.plotly_chart(
                catalog_status_figure(catalog, ui_lang),
                use_container_width=True,
                config={"displaylogo": False},
                key="catalog_status_chart",
            )
        selected_family = (
            st.selectbox(tr(ui_lang, "inspect_family"), families) if families else None
        )
        if selected_family:
            row = catalog[catalog["benchmark_family"] == selected_family].iloc[0]
            left, right = st.columns(2)
            with left:
                st.markdown(
                    pick(ui_lang, "#### Módszerek és változatok", "#### Methods and variants")
                )
                for item in row.get("methods", []):
                    st.write(f"- {item}")
            with right:
                st.markdown(pick(ui_lang, "#### Metrikák", "#### Metrics"))
                for item in row.get("metrics", []):
                    st.write(f"- {item}")
            meta = pd.DataFrame(
                [
                    {
                        pick(ui_lang, "Mérési család", "Benchmark family"): row.get(
                            "benchmark_family"
                        ),
                        pick(ui_lang, "Állapot", "Status"): row.get("status"),
                        pick(ui_lang, "Módszerek száma", "Method count"): len(
                            row.get("methods", [])
                        ),
                        pick(ui_lang, "Metrikák száma", "Metric count"): len(
                            row.get("metrics", [])
                        ),
                    }
                ]
            )
            st.dataframe(meta, use_container_width=True, hide_index=True)
        with st.expander(
            pick(ui_lang, "Teljes táblázat megnyitása", "Open full table"), expanded=False
        ):
            display = catalog.copy()
            display["methods"] = display["methods"].map(lambda items: " | ".join(items))
            display["metrics"] = display["metrics"].map(lambda items: " | ".join(items))
            display = display.rename(columns=localize_columns(ui_lang, list(display.columns)))
            st.dataframe(display, use_container_width=True, hide_index=True, height=620)

    with tabs[6]:
        from tkip.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker(PROJECT_ROOT)
        traces = tracker.recent(300)
        st.markdown(f"#### {tr(ui_lang, 'experiment_tracking')}")
        if traces:
            summary_rows = []
            for trace in traces:
                metrics = trace.get("metrics") or {}
                usage = trace.get("usage") or {}
                summary_rows.append(
                    {
                        "experiment_id": trace.get("experiment_id"),
                        "timestamp": trace.get("timestamp"),
                        "family": trace.get("benchmark_family"),
                        "prompt": trace.get("prompt_name"),
                        "status": trace.get("status"),
                        "output_score": metrics.get("output_score"),
                        "grounding": metrics.get("grounding_score"),
                        "latency_ms": usage.get("latency_ms"),
                        "cost_usd": usage.get("cost_usd"),
                    }
                )
            trace_frame = pd.DataFrame(summary_rows)
            trace_frame = trace_frame.rename(
                columns=localize_columns(ui_lang, list(trace_frame.columns))
            )
            st.dataframe(trace_frame, use_container_width=True, hide_index=True, height=480)
            selected_id = st.selectbox(
                tr(ui_lang, "open_experiment"),
                [trace.get("experiment_id") for trace in reversed(traces)],
            )
            selected = next(
                (trace for trace in traces if trace.get("experiment_id") == selected_id),
                None,
            )
            if selected:
                st.json(selected)
        else:
            st.info(tr(ui_lang, "no_experiment_trace"))


def _run_prompt_benchmark(
    *,
    kp,
    ui_lang: str,
    effective_key: str,
    selected_profiles: list[str],
    selected_questions: list[str],
    profile_map: dict[str, str],
    budget_key: str,
    chunk_key: str,
) -> None:
    from tkip.experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker(PROJECT_ROOT)
    rows: list[dict] = []
    total = len(selected_profiles) * len(selected_questions)
    progress = st.progress(0, text=tr(ui_lang, "running_prompt_benchmark"))
    completed = 0

    for label in selected_profiles:
        profile_key = profile_map[label]
        for question in selected_questions:
            request = _request_from_settings(
                question=question,
                language=ui_lang,
                budget_key=budget_key,
                prompt_profile=profile_key,
                chunk_key=chunk_key,
                prompt_check=True,
                quality_review=True,
                include_source_visual=False,
                gemini_optimize_prompt=True,
            )
            try:
                answer = kp.ask(request, gemini_api_key=effective_key)
                diagnostics = answer.diagnostics or {}
                quality = diagnostics.get("quality_review") or {}
                pipeline = diagnostics.get("pipeline_evaluation") or {}
                row = {
                    "profile": label,
                    "question": question,
                    "status": "success",
                    "prompt_score": quality.get("prompt_score"),
                    "output_score": quality.get("output_score"),
                    "language_score": quality.get("language_score"),
                    "grounding_score": quality.get("grounding_score"),
                    "context_score": pipeline.get("context_analysis_score"),
                    "tool_score": pipeline.get("tool_calling_score"),
                    "citation_valid": bool(
                        (diagnostics.get("citation_validation") or {}).get("valid")
                    ),
                    "structured_output_valid": True,
                    "latency_ms": answer.latency_ms,
                    "input_tokens": diagnostics.get("input_tokens") or 0,
                    "output_tokens": diagnostics.get("output_tokens") or 0,
                    "cost_usd": diagnostics.get("estimated_cost_usd") or 0,
                    "citations": len(answer.sources),
                    "tools": len(answer.used_tools),
                }
                tracker.append(
                    {
                        "benchmark_family": "Prompt Engineering",
                        "model": kp.cfg["gemini"]["model"],
                        "prompt_name": profile_key,
                        "prompt_version": "v1",
                        "user_prompt": question,
                        "optimized_prompt": (diagnostics.get("prompt_optimization") or {}).get(
                            "optimized"
                        ),
                        "system_prompt": diagnostics.get("system_prompt"),
                        "index_variant": diagnostics.get("index_variant"),
                        "retrieval": {
                            "ranking": (diagnostics.get("ranking") or [])[:10],
                            "context_chars": diagnostics.get("context_chars"),
                        },
                        "tool_calls": diagnostics.get("tool_results") or [],
                        "parsed_response": answer.model_dump(),
                        "metrics": row,
                        "usage": {
                            "input_tokens": row["input_tokens"],
                            "output_tokens": row["output_tokens"],
                            "latency_ms": row["latency_ms"],
                            "cost_usd": row["cost_usd"],
                        },
                        "status": "success",
                    }
                )
            except Exception as exc:
                row = {
                    "profile": label,
                    "question": question,
                    "status": "error",
                    "error": str(exc),
                }
                tracker.append(
                    {
                        "benchmark_family": "Prompt Engineering",
                        "prompt_name": profile_key,
                        "user_prompt": question,
                        "status": "error",
                        "error": str(exc),
                    }
                )
            rows.append(row)
            completed += 1
            progress.progress(int(100 * completed / total), text=f"{completed}/{total} · {label}")

    progress.empty()
    st.session_state["prompt_benchmark_df"] = pd.DataFrame(rows)


def _render_section_header(ui_lang: str, title: str, description: str) -> None:
    st.markdown(
        f"""<div style="margin:.2rem 0 .85rem"><div class="section-title" style="font-size:1.18rem">{title}</div><div class="small-muted">{description}</div></div>""",
        unsafe_allow_html=True,
    )


def _render_metric_guide(rows: list[dict[str, str]], ui_lang: str) -> None:
    if not rows:
        return
    cols = st.columns(2)
    for idx, row in enumerate(rows):
        values = list(row.values())
        title = values[0] if values else "Metric"
        meaning = values[1] if len(values) > 1 else ""
        extra = values[2] if len(values) > 2 else ""
        extra_label = pick(ui_lang, "Értelmezés", "Interpretation")
        cols[idx % 2].markdown(
            f"""<div class="evidence-card" style="min-height:142px;border-top:3px solid #60a5fa">
<div class="evidence-title">{title}</div>
<div class="evidence-text">{meaning}</div>
<div class="small-muted" style="margin-top:.55rem"><b>{extra_label}:</b> {extra}</div>
</div>""",
            unsafe_allow_html=True,
        )


def _render_benchmark_snapshot(ui_lang: str) -> None:
    retrieval = _default_retrieval_summary()
    deterministic_path = (
        PROJECT_ROOT / "07_results" / "evaluation" / "generation_deterministic_metrics.csv"
    )
    deterministic = (
        pd.read_csv(deterministic_path) if deterministic_path.exists() else pd.DataFrame()
    )
    best = (
        retrieval.sort_values("recall@5", ascending=False).iloc[0]
        if not retrieval.empty and "recall@5" in retrieval.columns
        else None
    )
    means = (
        deterministic.mean(numeric_only=True) if not deterministic.empty else pd.Series(dtype=float)
    )
    st.markdown(
        f"""<div style="margin:.2rem 0 .65rem"><div class="small-muted">{pick(ui_lang, "Előre betöltött offline baseline · új mérés nélkül is látható", "Preloaded offline baseline · visible without running a new benchmark")}</div></div>""",
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    cols[0].metric(
        "Recall@5", f"{100 * float(best.get('recall@5', 0)):.1f}%" if best is not None else "—"
    )
    cols[1].metric("MRR", f"{float(best.get('mrr', 0)):.3f}" if best is not None else "—")
    cols[2].metric("nDCG@5", f"{float(best.get('ndcg@5', 0)):.3f}" if best is not None else "—")
    cols[3].metric(
        pick(ui_lang, "Hivatkozáshelyesség", "Citation correctness"),
        f"{100 * float(means.get('citation_correctness', float('nan'))):.0f}%"
        if "citation_correctness" in means
        else "—",
    )
    cols[4].metric(
        pick(ui_lang, "Nincs-válasz pontosság", "No-answer accuracy"),
        f"{100 * float(means.get('no_answer_accuracy', float('nan'))):.0f}%"
        if "no_answer_accuracy" in means
        else "—",
    )
    cols[5].metric(
        pick(ui_lang, "Strukturált kimenet", "Structured output"),
        f"{100 * float(means.get('structured_output_validity', float('nan'))):.0f}%"
        if "structured_output_validity" in means
        else "—",
    )

    if not retrieval.empty:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(
                retrieval_quality_figure(retrieval, ui_lang),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
                key="benchmark_header_retrieval_quality",
            )
        with right:
            if not deterministic.empty:
                st.plotly_chart(
                    deterministic_rag_figure(deterministic, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="benchmark_header_rag_validation",
                )
            else:
                st.plotly_chart(
                    quality_latency_scatter(retrieval, ui_lang),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    key="benchmark_header_quality_latency",
                )


def _default_retrieval_summary() -> pd.DataFrame:
    path = PROJECT_ROOT / "07_results" / "benchmarks" / "retrieval_methods_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rename_map = {
        "Method": "method",
        "MRR": "mrr",
        "Hit Rate": "hit_rate",
        "Recall@5": "recall@5",
        "nDCG@5": "ndcg@5",
        "P50 latency": "p50_latency_ms",
        "P95 latency": "p95_latency_ms",
    }
    frame = frame.rename(columns=rename_map)
    for legacy in ["recall@1", "recall@3", "recall@10", "precision@5", "samples"]:
        if legacy not in frame.columns:
            frame[legacy] = None
    frame["index_variant"] = "primary"
    frame["method"] = frame["method"].fillna("unknown")
    if "p50_latency_ms" in frame.columns:
        frame["p50_latency_ms"] = pd.to_numeric(frame["p50_latency_ms"], errors="coerce")
    if "p95_latency_ms" in frame.columns:
        frame["p95_latency_ms"] = pd.to_numeric(frame["p95_latency_ms"], errors="coerce")
    return frame


def _default_retrieval_raw() -> pd.DataFrame:
    path = PROJECT_ROOT / "07_results" / "benchmarks" / "retrieval_benchmark.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _render_retrieval_overview_cards(frame: pd.DataFrame, ui_lang: str) -> None:
    if frame.empty:
        return
    best_recall = (
        frame.sort_values("recall@5", ascending=False).iloc[0]
        if "recall@5" in frame.columns
        else frame.iloc[0]
    )
    best_mrr = (
        frame.sort_values("mrr", ascending=False).iloc[0]
        if "mrr" in frame.columns
        else frame.iloc[0]
    )
    fastest = (
        frame.sort_values("p50_latency_ms", ascending=True).iloc[0]
        if "p50_latency_ms" in frame.columns
        else frame.iloc[0]
    )
    cols = st.columns(4)
    cols[0].metric(pick(ui_lang, "Összes módszer", "Methods"), int(frame["method"].nunique()))
    cols[1].metric(
        tr(ui_lang, "best_recall"), f"{100 * float(best_recall.get('recall@5', 0) or 0):.1f}%"
    )
    cols[2].metric(tr(ui_lang, "best_mrr"), f"{float(best_mrr.get('mrr', 0) or 0):.3f}")
    cols[3].metric(
        tr(ui_lang, "fastest_p50"), f"{float(fastest.get('p50_latency_ms', 0) or 0):.1f} ms"
    )


def _load_json_example(name: str) -> dict | list | None:
    path = PROJECT_ROOT / "07_results" / "examples" / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _retrieval_metric_rows(ui_lang: str) -> list[dict[str, str]]:
    if ui_lang == "hu":
        return [
            {
                "Metrika": "Recall@K",
                "Mit mér?": "A releváns chunkok mekkora része kerül be a top-K találatok közé.",
                "Mikor jó?": "A lehető legtöbb releváns forrás már kis K mellett megjelenik.",
            },
            {
                "Metrika": "Precision@K",
                "Mit mér?": "A top-K találatok mekkora része valóban releváns.",
                "Mikor jó?": "Kevés a zajos, rossz vagy tévesen felhozott chunk.",
            },
            {
                "Metrika": "MRR",
                "Mit mér?": "Milyen korán jelenik meg az első releváns találat.",
                "Mikor jó?": "A legjobb találat már az első helyek egyikén van.",
            },
            {
                "Metrika": "nDCG@K",
                "Mit mér?": "A rangsor egészét, pozícióérzékenyen értékeli.",
                "Mikor jó?": "A releváns találatok nemcsak bent vannak, hanem jó sorrendben is.",
            },
            {
                "Metrika": "Hit Rate",
                "Mit mér?": "Volt-e legalább egy releváns találat.",
                "Mikor jó?": "A rendszer ritkán bukik teljesen üresen vagy rossz irányba.",
            },
            {
                "Metrika": "P50 / P95 latency",
                "Mit mér?": "Tipikus és szélső késleltetés.",
                "Mikor jó?": "A rendszer egyszerre gyors és stabil, nincsenek nagy tüskék.",
            },
        ]
    return [
        {
            "Metric": "Recall@K",
            "What does it measure?": "How much of the relevant evidence is recovered in top-K.",
            "What is good?": "Most relevant evidence appears already at small K.",
        },
        {
            "Metric": "Precision@K",
            "What does it measure?": "How much of top-K is truly relevant.",
            "What is good?": "Little noisy or irrelevant evidence is returned.",
        },
        {
            "Metric": "MRR",
            "What does it measure?": "How early the first relevant result appears.",
            "What is good?": "The first relevant hit is near rank 1.",
        },
        {
            "Metric": "nDCG@K",
            "What does it measure?": "Position-aware ranking quality across the list.",
            "What is good?": "Relevant evidence is not only present but well ordered.",
        },
        {
            "Metric": "Hit Rate",
            "What does it measure?": "Whether at least one relevant hit is returned.",
            "What is good?": "The system rarely misses completely.",
        },
        {
            "Metric": "P50 / P95 latency",
            "What does it measure?": "Typical and tail latency.",
            "What is good?": "The system is both fast and stable.",
        },
    ]


def _prompt_metric_rows(ui_lang: str) -> list[dict[str, str]]:
    if ui_lang == "hu":
        return [
            {
                "Metrika": "Prompt score",
                "Mit mér?": "Mennyire tiszta, célzott és végrehajtható az átalakított prompt.",
                "Mikor hasznos?": "Összehasonlíthatóvá teszi a promptprofilokat.",
            },
            {
                "Metrika": "Output score",
                "Mit mér?": "A válasz szerkezetét, hasznosságát és teljességét.",
                "Mikor hasznos?": "Kiderül, melyik prompt ad jobb végső választ.",
            },
            {
                "Metrika": "Grounding score",
                "Mit mér?": "Mennyire támaszkodik a válasz valódi forrásokra.",
                "Mikor hasznos?": "Hallucinációk és laza állítások csökkentésére.",
            },
            {
                "Metrika": "Context score",
                "Mit mér?": "Mennyire volt jó a kiválasztott kontextus.",
                "Mikor hasznos?": "Megmutatja, hogy a prompt és a retrieval együtt mennyire működik.",
            },
            {
                "Metrika": "Tool score",
                "Mit mér?": "Hasznosak és helyesek voltak-e az eszközhívások.",
                "Mikor hasznos?": "Agentes vagy toolos feladatoknál kulcsfontosságú.",
            },
            {
                "Metrika": "Input / output tokens, latency, cost",
                "Mit mér?": "Erőforrás-lábnyom.",
                "Mikor hasznos?": "Minőség–költség–sebesség kompromisszumokhoz.",
            },
        ]
    return [
        {
            "Metric": "Prompt score",
            "What does it measure?": "How clear, targeted and executable the transformed prompt is.",
            "When useful?": "It helps compare prompt profiles.",
        },
        {
            "Metric": "Output score",
            "What does it measure?": "Answer structure, usefulness and completeness.",
            "When useful?": "Shows which prompt creates better final answers.",
        },
        {
            "Metric": "Grounding score",
            "What does it measure?": "How strongly the answer relies on real sources.",
            "When useful?": "Helpful for reducing hallucinations.",
        },
        {
            "Metric": "Context score",
            "What does it measure?": "How suitable the selected context was.",
            "When useful?": "Shows how prompt design and retrieval interact.",
        },
        {
            "Metric": "Tool score",
            "What does it measure?": "Whether tool usage was useful and correct.",
            "When useful?": "Important for agentic or tool-based tasks.",
        },
        {
            "Metric": "Input / output tokens, latency, cost",
            "What does it measure?": "Resource footprint.",
            "When useful?": "Useful for quality-cost-speed trade-offs.",
        },
    ]


def _rag_metric_rows(ui_lang: str) -> list[dict[str, str]]:
    if ui_lang == "hu":
        return [
            {
                "Réteg": "Retrieval",
                "Metrika": "Recall@K / MRR / nDCG@K",
                "Miért fontos?": "A generálás csak annyira jó, amennyire jó a felhozott bizonyíték.",
            },
            {
                "Réteg": "Generálás",
                "Metrika": "Answer correctness / faithfulness / relevance",
                "Miért fontos?": "A válasz legyen helyes, hasznos és forrásalapú.",
            },
            {
                "Réteg": "Kontextus",
                "Metrika": "Context precision / recall / utilization",
                "Miért fontos?": "A túl sok vagy túl zajos kontextus lerontja a választ.",
            },
            {
                "Réteg": "Hivatkozás",
                "Metrika": "Citation correctness / completeness",
                "Miért fontos?": "A felhasználó vissza tudjon menni a forráshoz.",
            },
            {
                "Réteg": "Structured output",
                "Metrika": "Valid JSON / schema compliance",
                "Miért fontos?": "Gépi feldolgozásnál a válasz legyen stabilan parse-olható.",
            },
            {
                "Réteg": "Rendszer",
                "Metrika": "Latency / tokens / cost / failure rate",
                "Miért fontos?": "Production kompromisszumok és skálázhatóság.",
            },
        ]
    return [
        {
            "Layer": "Retrieval",
            "Metric": "Recall@K / MRR / nDCG@K",
            "Why it matters": "Generation quality depends on retrieved evidence.",
        },
        {
            "Layer": "Generation",
            "Metric": "Answer correctness / faithfulness / relevance",
            "Why it matters": "The answer must be useful, correct and source-grounded.",
        },
        {
            "Layer": "Context",
            "Metric": "Context precision / recall / utilization",
            "Why it matters": "Too much or too noisy context hurts answer quality.",
        },
        {
            "Layer": "Citation",
            "Metric": "Citation correctness / completeness",
            "Why it matters": "Users must be able to trace claims back to sources.",
        },
        {
            "Layer": "Structured output",
            "Metric": "Valid JSON / schema compliance",
            "Why it matters": "Machine consumption requires stable parseable outputs.",
        },
        {
            "Layer": "System",
            "Metric": "Latency / tokens / cost / failure rate",
            "Why it matters": "Production trade-offs and scalability.",
        },
    ]


def _rag_metric_frame(ui_lang: str) -> pd.DataFrame:
    if ui_lang == "hu":
        rows = [
            {
                "Réteg": "Visszakeresés",
                "Metrikák": "Recall@K · Precision@K · MRR · nDCG@K · Hit Rate",
            },
            {
                "Réteg": "Generálás",
                "Metrikák": "Válaszhelyesség · Relevancia · Hűség · Forrásalapúság",
            },
            {
                "Réteg": "Kontextus",
                "Metrikák": "Kontextus-precision · Kontextus-recall · Kontextuskihasználás",
            },
            {
                "Réteg": "Hivatkozás",
                "Metrikák": "Hivatkozási precision · recall · helyesség · téves hivatkozások aránya",
            },
            {
                "Réteg": "Strukturált kimenet",
                "Metrikák": "Érvényes JSON · sémamegfelelés · feldolgozási hibaarány",
            },
            {"Réteg": "Rendszer", "Metrikák": "Késleltetés · tokenek · költség · hibaarány"},
        ]
    else:
        rows = [
            {"Layer": "Retrieval", "Metrics": "Recall@K · Precision@K · MRR · nDCG@K · Hit Rate"},
            {
                "Layer": "Generation",
                "Metrics": "Answer Correctness · Relevance · Faithfulness · Groundedness",
            },
            {
                "Layer": "Context",
                "Metrics": "Context Precision · Context Recall · Context utilization",
            },
            {
                "Layer": "Citation",
                "Metrics": "Citation Precision · Recall · Correctness · False Citation Rate",
            },
            {
                "Layer": "Structured output",
                "Metrics": "Valid JSON · Schema Validation · Parse Failure Rate",
            },
            {"Layer": "System", "Metrics": "Latency · Tokens · Cost · Failure Rate"},
        ]
    return pd.DataFrame(rows)


def _tool_metric_rows(ui_lang: str) -> list[dict[str, str]]:
    if ui_lang == "hu":
        return [
            {
                "Metrika": "Eszközválasztási pontosság",
                "Mit mér?": "A modell a megfelelő eszközt választotta-e.",
                "Miért fontos?": "Rosszul választott eszközből rossz válasz lesz.",
            },
            {
                "Metrika": "Eszközhívási precision / recall",
                "Mit mér?": "A szükséges és felesleges hívások aránya.",
                "Miért fontos?": "Mutatja az agent hatékonyságát.",
            },
            {
                "Metrika": "Argumentumpontosság",
                "Mit mér?": "Helyesek-e az argumentumok.",
                "Miért fontos?": "Jó tool rossz paraméterekkel ugyanúgy hibás lehet.",
            },
            {
                "Metrika": "Sémamegfelelés",
                "Mit mér?": "A tool-argumentum JSON megfelel-e a sémának.",
                "Miért fontos?": "A backend stabil végrehajtásához kell.",
            },
            {
                "Metrika": "Feladatteljesítési arány",
                "Mit mér?": "A teljes user-cél végül megoldódott-e.",
                "Miért fontos?": "Ez a legközvetlenebb üzleti mérőszám.",
            },
            {
                "Metrika": "Eszközhallucinációs arány",
                "Mit mér?": "Nem létező vagy nem engedélyezett eszközök használata.",
                "Miért fontos?": "Biztonság és megbízhatóság.",
            },
        ]
    return [
        {
            "Metric": "Tool selection accuracy",
            "What does it measure?": "Whether the model chose the correct tool.",
            "Why important?": "A wrong tool often means a wrong answer.",
        },
        {
            "Metric": "Tool call precision / recall",
            "What does it measure?": "Needed versus unnecessary calls.",
            "Why important?": "Shows agent efficiency.",
        },
        {
            "Metric": "Argument accuracy",
            "What does it measure?": "Whether the arguments are correct.",
            "Why important?": "A good tool with bad parameters still fails.",
        },
        {
            "Metric": "Schema validity",
            "What does it measure?": "Whether tool-call JSON conforms to schema.",
            "Why important?": "Needed for stable backend execution.",
        },
        {
            "Metric": "Task completion rate",
            "What does it measure?": "Whether the full user goal was achieved.",
            "Why important?": "The most direct business metric.",
        },
        {
            "Metric": "Tool hallucination rate",
            "What does it measure?": "Use of non-existing or forbidden tools.",
            "Why important?": "Critical for safety and reliability.",
        },
    ]


def _tool_metric_frame(ui_lang: str) -> pd.DataFrame:
    if ui_lang == "hu":
        pairs = [
            ("Eszközválasztási pontosság", "A rendszer a megfelelő eszközt választotta-e."),
            (
                "Eszközhívási pontosság / lefedettség",
                "A szükséges és felesleges eszközhívások aránya.",
            ),
            ("Argumentumpontosság", "Az eszköz helyes paramétereket kapott-e."),
            ("Sémamegfelelés", "Az eszközargumentumok megfelelnek-e az előírt sémának."),
            ("Végrehajtási sikeresség", "A kiszolgálóoldali eszközhívás sikeresen lefutott-e."),
            ("Feladatteljesítési arány", "A teljes felhasználói feladat sikeresen teljesült-e."),
            ("Felesleges eszközhívások aránya", "Mennyi szükségtelen eszközhívás történt."),
            (
                "Eszközhallucinációs arány",
                "Nem létező vagy nem engedélyezett eszközhívások aránya.",
            ),
        ]
    else:
        pairs = [
            ("Tool Selection Accuracy", "Correct tool selected."),
            ("Tool Call Precision / Recall", "Needed versus unnecessary calls."),
            ("Argument Accuracy", "Correct function arguments."),
            ("Schema Validity", "Arguments match the schema."),
            ("Execution Success Rate", "Backend execution success."),
            ("Task Completion Rate", "Whether the whole task completed."),
            ("Unnecessary Tool Rate", "Rate of unnecessary calls."),
            ("Tool Hallucination Rate", "Calls outside the allowlist."),
        ]
    key_a, key_b = ("metrika", "jelentés") if ui_lang == "hu" else ("metric", "meaning")
    return pd.DataFrame([{key_a: metric, key_b: meaning} for metric, meaning in pairs])


def _robustness_metric_rows(ui_lang: str) -> list[dict[str, str]]:
    if ui_lang == "hu":
        return [
            {
                "Család": "Hallucináció",
                "Mit néz?": "Helyes válasz vagy helyes tartózkodás történik-e különböző nehézségi helyzetekben.",
                "Kulcsmetrikák": "Correct answer · correct abstention · unsupported claims · false citations",
            },
            {
                "Család": "Promptinjekció",
                "Mit néz?": "Ellenáll-e a rendszer az utasítás-felülírásnak és a kiszivárogtatási próbáknak.",
                "Kulcsmetrikák": "attack rejection · injection success · unauthorized tool · data leakage",
            },
            {
                "Család": "Regresszió",
                "Mit néz?": "Egy változtatás után romlottak-e a fő metrikák.",
                "Kulcsmetrikák": "Recall@5 delta · MRR delta · citation delta · latency delta · cost delta",
            },
        ]
    return [
        {
            "Family": "Hallucination",
            "What does it check?": "Whether the system answers correctly or abstains correctly across different situations.",
            "Key metrics": "correct answer · correct abstention · unsupported claims · false citations",
        },
        {
            "Family": "Prompt injection",
            "What does it check?": "Whether the system resists instruction override and leakage attempts.",
            "Key metrics": "attack rejection · injection success · unauthorized tool · data leakage",
        },
        {
            "Family": "Regression",
            "What does it check?": "Whether key metrics got worse after a change.",
            "Key metrics": "Recall@5 delta · MRR delta · citation delta · latency delta · cost delta",
        },
    ]


def _robustness_frame(ui_lang: str) -> pd.DataFrame:
    if ui_lang == "hu":
        return pd.DataFrame(
            [
                {
                    "teszt": "Hallucináció",
                    "esetek": "megválaszolható · nincs válasz · részleges · ellentmondó · elavult",
                    "metrikák": "Helyes válasz · helyes tartózkodás · hallucináció · nem támogatott állítások · téves hivatkozások",
                },
                {
                    "teszt": "Promptinjekció",
                    "esetek": "utasítás-felülírás · rendszerprompt · jogosulatlan eszköz · kontextuskiszivárogtatás",
                    "metrikák": "Támadáselutasítás · injekciós sikeresség · jogosulatlan eszköz · promptkiszivárgás · adatszivárgás",
                },
                {
                    "teszt": "Regresszió",
                    "esetek": "prompt · beágyazás · visszakereső · modellverzió",
                    "metrikák": "Recall@5 változás · MRR változás · hivatkozásváltozás · késleltetésváltozás · költségváltozás",
                },
            ]
        )
    return pd.DataFrame(
        [
            {
                "suite": "Hallucination",
                "cases": "answerable · no-answer · partial · contradictory · outdated",
                "metrics": "Correct Answer · Correct Abstention · Hallucination · Unsupported Claims · False Citations",
            },
            {
                "suite": "Prompt injection",
                "cases": "instruction override · system prompt · unauthorized tool · context exfiltration",
                "metrics": "Attack Rejection · Injection Success · Unauthorized Tool · Prompt Leakage · Data Leakage",
            },
            {
                "suite": "Regression",
                "cases": "prompt · embedding · retriever · model version",
                "metrics": "Recall@5 delta · MRR delta · Citation delta · Latency delta · Cost delta",
            },
        ]
    )


def _mode_for_profile(profile: str) -> str:
    if profile in {"structured_tutor", "socratic_tutor"}:
        return "learning"
    if profile in {"comparison_matrix", "research_synthesis", "decision_tradeoff"}:
        return "compare"
    if profile in {"code_first", "debug_root_cause"}:
        return "code"
    return "ask"


def _prompt_style_for_profile(profile: str) -> str:
    if profile in {"structured_tutor", "socratic_tutor", "mathematical_derivation"}:
        return "teacher"
    if profile in {
        "comparison_matrix",
        "research_synthesis",
        "decision_tradeoff",
        "evidence_verification",
    }:
        return "comparison"
    if profile in {"code_first", "debug_root_cause"}:
        return "code_first"
    return "grounded"


def _request_from_settings(
    *,
    question: str,
    language: str,
    budget_key: str,
    prompt_profile: str,
    chunk_key: str,
    prompt_check: bool = True,
    quality_review: bool | None = None,
    custom_instruction: str | None = None,
    temperature: float | None = None,
    context_max_chars: int | None = None,
    max_output_tokens: int | None = None,
    max_tool_calls: int | None = None,
    retrieval_final_k: int | None = None,
    include_source_visual: bool = True,
    gemini_optimize_prompt: bool = True,
    selected_docs: list[str] | None = None,
    source_type: str | None = None,
    chunk_type: str | None = None,
    topic: str | None = None,
    runtime_chunking: bool = False,
    runtime_chunk_size: int | None = None,
    runtime_chunk_overlap: int | None = None,
) -> AskRequest:
    budget = ANSWER_PRESETS[budget_key]
    chunk = CHUNK_PRESETS[chunk_key]
    return AskRequest(
        question=question,
        language=language,
        mode=_mode_for_profile(prompt_profile),
        document_ids=selected_docs or [],
        topic=topic or None,
        source_type=source_type,
        chunk_type=chunk_type,
        debug=True,
        prompt_style=_prompt_style_for_profile(prompt_profile),
        custom_instructions=custom_instruction or None,
        temperature=float(temperature if temperature is not None else budget["temperature"]),
        prompt_language_check=prompt_check,
        quality_review=bool(budget["quality_review"] if quality_review is None else quality_review),
        prompt_optimization=prompt_profile,
        prompt_optimization_use_gemini=gemini_optimize_prompt,
        index_variant=chunk["index_variant"],
        runtime_chunking=runtime_chunking,
        runtime_chunk_strategy=chunk["runtime_strategy"],
        runtime_chunk_size=int(runtime_chunk_size or chunk["chunk_size"]),
        runtime_chunk_overlap=int(
            runtime_chunk_overlap if runtime_chunk_overlap is not None else chunk["overlap"]
        ),
        answer_preset=budget_key,
        context_max_chars=int(context_max_chars or budget["context_max_chars"]),
        retrieval_final_k=int(retrieval_final_k or budget["retrieval_final_k"]),
        max_output_tokens=int(max_output_tokens or budget["max_output_tokens"]),
        max_tool_calls=int(
            max_tool_calls if max_tool_calls is not None else budget["max_tool_calls"]
        ),
        include_source_visual=include_source_visual,
    )


def _profile_options(ui_lang: str) -> dict[str, str]:
    return {profile_label(ui_lang, key, PROFILES[key]["label"]): key for key in PROFILE_KEYS}


def _budget_label_map(ui_lang: str) -> dict[str, str]:
    return {
        budget_label(ui_lang, key, value["label"]): key for key, value in ANSWER_PRESETS.items()
    }


def _chunk_label_map(ui_lang: str) -> dict[str, str]:
    return {chunk_label(ui_lang, key, value["label"]): key for key, value in CHUNK_PRESETS.items()}


def render_prompt_preview(question: str, profile_key: str, ui_lang: str) -> None:
    ui_lang = normalize_language(ui_lang)
    if not question.strip():
        st.caption(
            pick(
                ui_lang,
                "Írj vagy válassz kérdést a prompt előnézetéhez.",
                "Write or select a question to preview the prompt.",
            )
        )
        return
    transformed = local_optimize(question, profile_key, ui_lang)
    left, right = st.columns(2)
    with left:
        st.caption(tr(ui_lang, "original_question"))
        st.code(question, language="text")
    with right:
        st.caption(tr(ui_lang, "transformed"))
        st.code(transformed["optimized"], language="text")


def _answer_metrics(answer) -> dict:
    diagnostics = answer.diagnostics or {}
    quality = diagnostics.get("quality_review") or {}
    pipeline = diagnostics.get("pipeline_evaluation") or {}
    return {
        "preset": diagnostics.get("answer_preset"),
        "prompt_profile": (diagnostics.get("prompt_optimization") or {}).get("profile"),
        "index": diagnostics.get("index_variant"),
        "tokens": diagnostics.get("total_tokens") or 0,
        "input_tokens": diagnostics.get("input_tokens") or 0,
        "output_tokens": diagnostics.get("output_tokens") or 0,
        "estimated_cost_usd": round(float(diagnostics.get("estimated_cost_usd") or 0), 6),
        "latency_ms": round(float(answer.latency_ms or 0), 1),
        "context_chars": diagnostics.get("context_chars") or 0,
        "citations": len(answer.sources),
        "tools": len(answer.used_tools),
        "confidence": round(float(answer.confidence), 3),
        "output_QA": quality.get("output_score"),
        "grounding_QA": quality.get("grounding_score"),
        "context_QA": pipeline.get("context_analysis_score"),
        "tool_QA": pipeline.get("tool_calling_score"),
    }


def render_ab_results(a_answer, b_answer, ui_lang: str) -> None:
    ui_lang = normalize_language(ui_lang)
    st.markdown(f"### {tr(ui_lang, 'ab_result')}")
    metrics_a = _answer_metrics(a_answer)
    metrics_b = _answer_metrics(b_answer)
    metric_labels = {
        "preset": pick(ui_lang, "Válaszkeret", "Answer preset"),
        "prompt_profile": tr(ui_lang, "prompt_profile"),
        "index": pick(ui_lang, "Index", "Index"),
        "tokens": pick(ui_lang, "Összes token", "Total tokens"),
        "input_tokens": pick(ui_lang, "Bemeneti tokenek", "Input tokens"),
        "output_tokens": pick(ui_lang, "Kimeneti tokenek", "Output tokens"),
        "estimated_cost_usd": pick(ui_lang, "Becsült költség (USD)", "Estimated cost (USD)"),
        "latency_ms": pick(ui_lang, "Késleltetés (ms)", "Latency (ms)"),
        "context_chars": pick(ui_lang, "Kontextuskarakterek", "Context characters"),
        "citations": tr(ui_lang, "citations"),
        "tools": tr(ui_lang, "tools"),
        "confidence": tr(ui_lang, "confidence"),
        "output_QA": pick(ui_lang, "Válaszellenőrzés", "Output QA"),
        "grounding_QA": tr(ui_lang, "grounding"),
        "context_QA": pick(ui_lang, "Kontextusellenőrzés", "Context QA"),
        "tool_QA": pick(ui_lang, "Eszközhívási ellenőrzés", "Tool QA"),
    }
    table = pd.DataFrame(
        [
            {
                pick(ui_lang, "Metrika", "Metric"): metric_labels.get(key, key),
                "A": metrics_a.get(key),
                "B": metrics_b.get(key),
            }
            for key in metrics_a
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown(f"#### {tr(ui_lang, 'cost_latency')}")
    st.plotly_chart(
        ab_footprint_figure(metrics_a, metrics_b, ui_lang),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
        key="ab_footprint_chart",
    )

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {tr(ui_lang, 'variant_a_answer')}")
        st.markdown(a_answer.answer)
        st.caption(
            f"{tr(ui_lang, 'sources')}: {len(a_answer.sources)} · {tr(ui_lang, 'tools')}: "
            f"{', '.join(a_answer.used_tools) or tr(ui_lang, 'none')} · {tr(ui_lang, 'cost')}: "
            f"${metrics_a.get('estimated_cost_usd', 0):.6f}"
        )
    with right:
        st.markdown(f"#### {tr(ui_lang, 'variant_b_answer')}")
        st.markdown(b_answer.answer)
        st.caption(
            f"{tr(ui_lang, 'sources')}: {len(b_answer.sources)} · {tr(ui_lang, 'tools')}: "
            f"{', '.join(b_answer.used_tools) or tr(ui_lang, 'none')} · {tr(ui_lang, 'cost')}: "
            f"${metrics_b.get('estimated_cost_usd', 0):.6f}"
        )

    st.markdown(f"#### {tr(ui_lang, 'quality_comparison')}")
    st.plotly_chart(
        ab_quality_figure(metrics_a, metrics_b, ui_lang),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
        key="ab_quality_chart",
    )

    st.markdown(f"#### {tr(ui_lang, 'retrieval_comparison')}")
    ranking_a = pd.DataFrame((a_answer.diagnostics or {}).get("ranking", [])[:8])
    ranking_b = pd.DataFrame((b_answer.diagnostics or {}).get("ranking", [])[:8])
    left, right = st.columns(2)
    with left:
        if not ranking_a.empty:
            st.caption(tr(ui_lang, "top_evidence_a"))
            columns = [
                column
                for column in ["final_rank", "document", "page", "reranker_score"]
                if column in ranking_a
            ]
            ranking_display = ranking_a[columns].rename(columns=localize_columns(ui_lang, columns))
            st.dataframe(ranking_display, use_container_width=True, hide_index=True)
    with right:
        if not ranking_b.empty:
            st.caption(tr(ui_lang, "top_evidence_b"))
            columns = [
                column
                for column in ["final_rank", "document", "page", "reranker_score"]
                if column in ranking_b
            ]
            ranking_display = ranking_b[columns].rename(columns=localize_columns(ui_lang, columns))
            st.dataframe(ranking_display, use_container_width=True, hide_index=True)

    with st.expander(tr(ui_lang, "compare_prompts"), expanded=False):
        left, right = st.columns(2)
        left.code(
            (a_answer.diagnostics.get("prompt_optimization") or {}).get("optimized", ""),
            language="text",
        )
        right.code(
            (b_answer.diagnostics.get("prompt_optimization") or {}).get("optimized", ""),
            language="text",
        )
