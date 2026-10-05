from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_engine.presets import HUNGARIAN_QUERY_PRESETS, RAG_STRATEGIES
from rag_engine.evaluation.generation import (
    answer_redundancy_proxy,
    answer_token_estimate,
    citation_accuracy,
    citation_coverage,
    citation_source_coverage,
    context_utilization,
)
from rag_engine.service import create_rag_pipeline
from ui.components.charts import (
    metric_bar_chart,
    percentile_latency_chart,
    quality_latency_scatter,
    radar_metrics_chart,
    stacked_latency_chart,
)
from ui.components.common import get_lab
from ui.components.education import evidence_cards, info_cards, kpi_cards, metrics_reference, note_box, page_intro
from ui.components.grounded_prompt import render_grounded_prompt
from ui.components.exports import render_dataframe_exports
from ui.components.reranker import build_selected_reranker, reranker_status_text
from ui.components.i18n import LABELS, rag_strategy_label
from ui.components.tables import safe_dataframe


def render() -> None:
    page_intro(
        "RAG stratégiák összehasonlítása",
        "Ugyanazt a magyar kérdést több RAG-architektúrán futtatjuk ugyanazon korpuszon. A nézet külön méri a visszakeresést, újrarangsorolást, generálást, TTFT-t, hivatkozásokat és a kontextus kihasználását.",
        eyebrow="TÖBB RAG STRATÉGIA · MINŐSÉG · KÉSLELTETÉS · TTFT · HIVATKOZÁSOK",
    )
    metrics_reference("rag")
    info_cards(
        [
            ("Több RAG stratégia", "Dense, BM25, hibrid, újrarangsorolt, HyDE, Multi-Query, Query-Rewrite, Multi-Hop, Parent-Document, compression és corrective variánsok."),
            ("Hibatűrő összehasonlítás", "Egyetlen hibás stratégia nem állítja le a teljes összehasonlítást; a kimaradt futások külön jelennek meg."),
            ("LLM telemetria", "Ollama streaming esetén TTFT, kimeneti token és token/s is mérhető."),
            ("Többdimenziós értékelés", "A minőséget, válaszidőt és bizonyítékhasználatot együtt érdemes értelmezni."),
        ],
        columns=4,
    )

    preset = st.selectbox(
        "Magyar tesztkérdés",
        HUNGARIAN_QUERY_PRESETS,
        format_func=lambda item: f"{item.label} · {item.topic}",
        key="rag_compare_preset",
    )
    query = st.text_area("Összehasonlító kérdés", value=preset.query, height=90, key=f"rag_compare_query_{preset.label}")
    strategies = st.multiselect(
        "Összehasonlítandó stratégiák",
        list(RAG_STRATEGIES),
        default=["baseline", "lexical", "hybrid", "reranked", "compression", "corrective"],
        format_func=lambda key: rag_strategy_label(key, RAG_STRATEGIES[key]["name"]),
    )

    with st.expander("Mit csinálnak a kiválasztott stratégiák?", expanded=False):
        for key in strategies:
            meta = RAG_STRATEGIES[key]
            st.markdown(f"**{rag_strategy_label(key, meta['name'])}** — {meta['summary']}  \n`{meta['flow']}`  \n*Ajánlott:* {meta['when']}")

    _, preview_reranker = build_selected_reranker("reranked")
    st.caption(f"Globális újrarangsoroló-beállítás: **{reranker_status_text(preview_reranker)}**. Csak az explicit rerankingot tartalmazó stratégiák alkalmazzák.")

    if st.session_state.get("llm_provider") == "ollama":
        st.info("Ollama módban a HyDE, több lekérdezéses, átíró, többlépéses és korrekciós stratégiák több LLM-hívást használhatnak, ezért lényegesen lassabbak lehetnek.")

    if st.button("Összehasonlítás futtatása", type="primary", disabled=not strategies):
        rows: list[dict[str, object]] = []
        details: dict[str, object] = {}
        errors: list[dict[str, str]] = []
        progress = st.progress(0.0, text="RAG stratégiák előkészítése...")
        base_lab = None
        for index, strategy in enumerate(strategies, start=1):
            strategy_name = rag_strategy_label(strategy, RAG_STRATEGIES[strategy]["name"])
            progress.progress((index - 1) / max(1, len(strategies)), text=f"{strategy_name} fut...")
            try:
                if strategy == "parent-document":
                    lab = get_lab(strategy="parent-child")
                else:
                    if base_lab is None:
                        base_lab = get_lab()
                    lab = base_lab
                reranker, reranker_status = build_selected_reranker(strategy, allow_inactive=True)
                pipeline = create_rag_pipeline(
                    lab,
                    strategy,
                    max_context_tokens=int(st.session_state.get("context_budget", 1800)),
                    execution_device=str(getattr(lab.embedder, "device", "cpu")),
                    reranker=reranker,
                    top_k=int(st.session_state.get("top_k", 5)),
                    candidate_count=int(st.session_state.get("candidate_count", 20)),
                    context_profile=st.session_state.get("context_profile", "balanced"),
                    prompt_profile=st.session_state.get("prompt_profile", "professional"),
                )
                result = pipeline.answer(query)
                context = result.context_text or "\n".join(chunk.text for chunk in result.retrieved_chunks)
                source_count = len(result.retrieved_chunks)
                source_diversity = len({str(c.metadata.get("title") or c.source or "") for c in result.retrieved_chunks if c.metadata.get("title") or c.source}) / max(1, source_count)
                rows.append(
                    {
                        "Stratégia": strategy_name,
                        "Kulcs": strategy,
                        "Újrarangsoroló": reranker_status_text(reranker_status),
                        "Visszakeresés ms": round(result.retrieval_latency_ms, 1),
                        "Újrarangsorolás ms": round(result.reranking_latency_ms or 0.0, 1),
                        "Generálás ms": round(result.generation_latency_ms, 1),
                        "TTFT ms": round(float(result.generation_ttft_ms or 0.0), 1),
                        "Teljes ms": round(result.total_latency_ms, 1),
                        "Token/s": round(float(result.tokens_per_second or 0.0), 2),
                        "Kimeneti token": int(result.output_tokens or 0),
                        "Kontextus token": result.context_tokens,
                        "Bizonyíték": source_count,
                        LABELS["source_diversity"]: round(source_diversity, 3),
                        LABELS["citation_accuracy"]: round(citation_accuracy(result.answer, source_count), 3),
                        LABELS["citation_coverage"]: round(citation_coverage(result.answer), 3),
                        LABELS["citation_source_coverage"]: round(citation_source_coverage(result.answer, source_count), 3),
                        LABELS["context_utilization"]: round(context_utilization(result.answer, context), 3),
                        "Redundancia proxy": round(answer_redundancy_proxy(result.answer), 3),
                        "Választoken becslés": answer_token_estimate(result.answer),
                    }
                )
                details[strategy] = result
            except Exception as exc:
                errors.append({"Stratégia": strategy_name, "Hiba": str(exc)})
        progress.progress(1.0, text="Összehasonlítás kész")
        st.session_state["rag_comparison_rows"] = rows
        st.session_state["rag_comparison_details"] = details
        st.session_state["rag_comparison_errors"] = errors

    rows = st.session_state.get("rag_comparison_rows", [])
    details = st.session_state.get("rag_comparison_details", {})
    errors = st.session_state.get("rag_comparison_errors", [])
    if errors:
        with st.expander(f"Hibás / kimaradt stratégiák ({len(errors)})", expanded=False):
            safe_dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
    if not rows:
        return

    frame = pd.DataFrame(rows)
    frame[LABELS["overall_quality_score"]] = (
        0.28 * frame[LABELS["citation_accuracy"]].clip(0, 1)
        + 0.20 * frame[LABELS["citation_coverage"]].clip(0, 1)
        + 0.16 * frame[LABELS["citation_source_coverage"]].clip(0, 1)
        + 0.24 * frame[LABELS["context_utilization"]].clip(0, 1)
        + 0.12 * frame[LABELS["source_diversity"]].clip(0, 1)
    ).round(3)

    score_col = LABELS["overall_quality_score"]
    best = frame.sort_values([score_col, LABELS["citation_accuracy"]], ascending=False).iloc[0]
    fastest = frame.sort_values("Teljes ms").iloc[0]
    kpi_cards(
        [
            ("Legnagyobb összpontszám", str(best["Stratégia"]), f"Pontszám: {float(best[score_col]):.3f}"),
            ("Legkisebb válaszidő", str(fastest["Stratégia"]), f"{float(fastest['Teljes ms']):.0f} ms"),
            ("Legjobb hivatkozási pontosság", f"{frame[LABELS['citation_accuracy']].max():.2f}", "Érvényes [Sx] hivatkozások aránya."),
            ("Legnagyobb generálási sebesség", f"{frame['Token/s'].max():.1f} token/s", "Lokális generálási áteresztőképesség."),
        ],
        columns=4,
    )
    with st.expander("Nyers összehasonlítási eredmények", expanded=False):
        safe_dataframe(frame.sort_values([score_col, LABELS["citation_accuracy"]], ascending=False), width="stretch", hide_index=True)
    render_dataframe_exports(frame, stem="rag_osszehasonlitas", key_prefix="rag_compare_export")

    c1, c2 = st.columns(2)
    with c1:
        quality_latency_scatter(
            frame,
            x="Teljes ms",
            y=score_col,
            color="Stratégia",
            hover=[LABELS["citation_accuracy"], LABELS["citation_coverage"], "TTFT ms", "Token/s", "Kontextus token", "Újrarangsoroló"],
            size="Bizonyíték",
            key="rag_compare_quality_latency",
            title="Minőség–késleltetés egyensúly",
        )
    with c2:
        metric_bar_chart(
            frame.sort_values([score_col, LABELS["citation_accuracy"]], ascending=False),
            x="Stratégia",
            y=score_col,
            color="Stratégia",
            key="rag_compare_overall_bar",
            title="RAG stratégiák · összesített minőségi pontszám",
        )

    c3, c4 = st.columns(2)
    with c3:
        radar_metrics_chart(
            frame.sort_values([score_col, LABELS["citation_accuracy"]], ascending=False),
            label_col="Stratégia",
            metrics=[LABELS["citation_accuracy"], LABELS["citation_coverage"], LABELS["citation_source_coverage"], LABELS["context_utilization"], LABELS["source_diversity"]],
            key="rag_compare_radar",
            title="RAG stratégiák minőségi profilja",
            max_series=5,
        )
    with c4:
        percentile_latency_chart(
            frame,
            category="Stratégia",
            percentiles=["TTFT ms", "Generálás ms", "Teljes ms"],
            key="rag_compare_latency_triplet",
            title="Válaszidő profil: első token / generálás / teljes idő",
            value_label="ms",
        )

    t1, t2 = st.columns(2)
    with t1:
        metric_bar_chart(
            frame.sort_values("Token/s", ascending=False),
            x="Stratégia",
            y="Token/s",
            color="Stratégia",
            key="rag_compare_token_speed",
            title="Generálási sebesség stratégiánként",
        )
    with t2:
        token_frame = frame[["Stratégia", "Kimeneti token", "Kontextus token"]].melt(
            id_vars="Stratégia", var_name="Token típus", value_name="Token"
        )
        metric_bar_chart(
            token_frame,
            x="Stratégia",
            y="Token",
            color="Token típus",
            key="rag_compare_token_volume",
            title="Kimeneti és kontextustokenek",
        )

    stacked_latency_chart(
        frame,
        strategy_col="Stratégia",
        stage_columns=["Visszakeresés ms", "Újrarangsorolás ms", "Generálás ms"],
        key="rag_compare_stage_latency",
        title="Pipeline késleltetés bontása stratégiánként",
    )

    st.subheader("Válaszok és forrásbizonyítékok")
    tabs = st.tabs([rag_strategy_label(key, RAG_STRATEGIES[key]["name"]) for key in details])
    for tab, (key, result) in zip(tabs, details.items(), strict=False):
        with tab:
            st.markdown(result.answer)
            telemetry = {
                "TTFT ms": result.generation_ttft_ms,
                "Kimeneti token": result.output_tokens,
                "Token/s": result.tokens_per_second,
                "Kontextus token": result.context_tokens,
                "Teljes idő ms": result.total_latency_ms,
            }
            safe_dataframe(pd.DataFrame([telemetry]), width="stretch", hide_index=True)
            if result.transformed_queries:
                st.caption("Lekérdezés-transzformáció: " + " | ".join(result.transformed_queries))
            evidence_cards(result.retrieved_chunks, answer=result.answer, columns=2)
            with st.expander("Forrásokra támaszkodó prompt", expanded=False):
                render_grounded_prompt(result, key_prefix=f"comparison_{key}")

    note_box(
        "Értelmezés",
        "Az egykérdéses összehasonlítás diagnosztikai nézet. Reprodukálható, több kérdéses következtetéshez használd a Kiértékelés és a Teljes pipeline benchmark oldalakat.",
    )
