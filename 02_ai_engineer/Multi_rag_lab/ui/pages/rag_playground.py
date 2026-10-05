from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_engine.presets import CONTEXT_PROFILES, HUNGARIAN_QUERY_PRESETS, PROMPT_PROFILES, RAG_STRATEGIES
from rag_engine.evaluation.generation import (
    answer_redundancy_proxy,
    answer_token_estimate,
    citation_accuracy,
    citation_coverage,
    citation_source_coverage,
    context_utilization,
)
from rag_engine.generation.provider_ollama import OllamaGenerationError
from rag_engine.service import create_rag_pipeline
from ui.charts.trace import rag_trace_graph
from ui.components.charts import context_profile_chart, retrieval_scores_chart
from ui.components.common import get_lab
from ui.components.education import empty_state, info_cards, kpi_cards, metrics_reference, note_box, page_intro
from ui.components.evidence import render_answer_evidence
from ui.components.grounded_prompt import render_grounded_prompt
from ui.components.exports import render_text_export
from ui.components.reranker import build_selected_reranker, reranker_status_text
from ui.components.i18n import LABELS, rag_strategy_label
from ui.components.tables import safe_dataframe


def render() -> None:
    strategy = st.session_state.get("rag_strategy", "hybrid")
    details = RAG_STRATEGIES[strategy]
    context_profile = st.session_state.get("context_profile", "balanced")
    prompt_profile = st.session_state.get("prompt_profile", "professional")
    _, preview_reranker_status = build_selected_reranker(strategy, allow_inactive=True)

    page_intro(
        "RAG játszótér",
        "A teljes RAG folyamat egy kérdésen: visszakeresés, opcionális lekérdezés-transzformáció és újrarangsorolás, kontextusépítés, lokális generálás, hivatkozások és a tényleges grounded prompt ellenőrzése.",
        eyebrow="VISSZAKERESÉS · ÚJRARANGSOROLÁS · KONTEXTUS · LLM · BIZONYÍTÉKOK",
    )
    metrics_reference("rag")
    info_cards(
        [
            (rag_strategy_label(strategy, details["name"]), details["summary"]),
            ("Folyamat", details["flow"]),
            ("Aktív újrarangsorolás", f"{reranker_status_text(preview_reranker_status)}. {preview_reranker_status.reason}"),
            ("Kontextusprofil", str(CONTEXT_PROFILES[context_profile]["summary"])),
        ],
        columns=4,
    )

    preset = st.selectbox(
        "Kérdésminta a magyar korpuszhoz",
        HUNGARIAN_QUERY_PRESETS,
        format_func=lambda item: f"{item.label} · {item.topic}",
        key="rag_query_preset",
    )
    query = st.text_area("Kérdés", value=preset.query, height=110, key=f"rag_query_{preset.label}")

    top_k = int(st.session_state.get("top_k", 5))
    candidate_count = int(st.session_state.get("candidate_count", max(20, top_k)))
    context_budget = int(st.session_state.get("context_budget", 1800))
    st.markdown(
        f"**Aktív futás:** `{rag_strategy_label(strategy, details['name'])}` · Top-K=`{top_k}` · jelöltek=`{candidate_count}` · "
        f"kontextuskeret=`{context_budget}` · LLM=`{st.session_state.get('llm_provider', 'dummy')}` · "
        f"újrarangsorolás=`{reranker_status_text(preview_reranker_status)}`"
    )

    if strategy in {"multi-query", "query-rewrite", "corrective", "hyde", "multi-hop"} and st.session_state.get("llm_provider") == "dummy":
        st.info("Ez a stratégia lekérdezés-transzformációt használ. Dummy módban determinisztikus demonstráció fut; valódi nyelvi transzformációhoz válts Ollamára.")

    if st.button("RAG folyamat futtatása", type="primary"):
        try:
            chunking_override = "parent-child" if strategy == "parent-document" else None
            with st.spinner("RAG pipeline fut: visszakeresés → kontextus → lokális generálás..."):
                lab = get_lab(strategy=chunking_override)
                reranker, reranker_status = build_selected_reranker(strategy, allow_inactive=True)
                pipeline = create_rag_pipeline(
                    lab,
                    strategy,
                    max_context_tokens=context_budget,
                    execution_device=getattr(lab.embedder, "device", "cpu"),
                    reranker=reranker,
                    top_k=top_k,
                    candidate_count=candidate_count,
                    context_profile=context_profile,
                    prompt_profile=prompt_profile,
                )
                result = pipeline.answer(query)
            st.session_state["last_rag_result"] = result
            st.session_state["last_rag_runtime"] = {
                "Beágyazási modell": getattr(lab.embedder, "model_name", "ismeretlen"),
                "Beágyazási eszköz": getattr(lab.embedder, "device", "ismeretlen"),
                "Vektorbackend": getattr(lab.vector_store, "backend_name", "ismeretlen"),
                "Vektor eszköz": getattr(lab.vector_store, "device", "ismeretlen"),
                "Újrarangsoroló": reranker_status_text(reranker_status),
                "Újrarangsoroló megjegyzés": reranker_status.reason,
                "LLM szolgáltató": getattr(lab.llm, "name", "ismeretlen"),
                "LLM modell": getattr(lab.llm, "model_name", "ismeretlen"),
                "RAG stratégia": rag_strategy_label(strategy, strategy),
                "Index forrás": "perzisztens, előre épített" if bool(getattr(lab, "build_trace", {}).get("prebuilt_index_used")) else "futás közben épített",
                "Retrieval erőforrás build": f"{float(getattr(lab, 'build_trace', {}).get('resource_build_total_ms', 0.0)):.0f} ms",
                "Dokumentum embedding build": f"{float(getattr(lab, 'build_trace', {}).get('document_embedding_ms', 0.0)):.0f} ms",
                "Kontextusprofil": CONTEXT_PROFILES[context_profile]["name"],
                "Promptprofil": PROMPT_PROFILES[prompt_profile]["name"],
                "reranker_active": reranker_status.active,
            }
        except OllamaGenerationError as exc:
            st.error(str(exc))
            st.warning("4 GB VRAM mellett próbáld a Qwen3 Low Memory profilt és a 900–1200 tokenes kontextuskeretet. A visszakeresési eredmények ettől nem vesznek el.")
        except Exception as exc:
            st.error(f"A RAG futás sikertelen: {exc}")

    result = st.session_state.get("last_rag_result")
    runtime = st.session_state.get("last_rag_runtime", {})
    if result is None:
        empty_state(
            "Még nincs RAG futás",
            "Futtass egy kérdést, hogy megjelenjen a válasz, a pipeline trace, a kontextusprofil, a forrásbizonyíték és a tényleges grounded prompt.",
            hint="Első próbához a Hibrid RAG + lexikális újrarangsorolás + kiegyensúlyozott kontextus jó kiindulópont.",
        )
        return

    context_text = result.context_text or "\n".join(chunk.text for chunk in result.retrieved_chunks)
    source_count = len(result.retrieved_chunks)
    citation_acc = citation_accuracy(result.answer, source_count)
    citation_cov = citation_coverage(result.answer)
    source_cov = citation_source_coverage(result.answer, source_count)
    context_score = context_utilization(result.answer, context_text)

    kpi_cards(
        [
            ("Teljes válaszidő", f"{result.total_latency_ms:.0f} ms", f"Visszakeresés {result.retrieval_latency_ms:.0f} · generálás {result.generation_latency_ms:.0f} ms"),
            ("TTFT", f"{float(result.generation_ttft_ms or 0):.0f} ms", "Az első generált tokenig eltelt idő."),
            ("Generálási sebesség", f"{float(result.tokens_per_second or 0):.1f} token/s", f"Kimeneti token: {int(result.output_tokens or 0)}"),
            ("Kontextus", f"{result.context_tokens} token", f"{source_count} bizonyíték / forrásrészlet"),
            (LABELS["citation_accuracy"], f"{citation_acc:.2f}", "Az [Sx] hivatkozások érvényessége."),
            (LABELS["citation_coverage"], f"{citation_cov:.2f}", f"Forráslefedettség: {source_cov:.2f}"),
            (LABELS["context_utilization"], f"{context_score:.2f}", "Lexikális groundedness proxy."),
            ("Válaszhossz", f"{answer_token_estimate(result.answer)} token", f"Redundancia proxy: {answer_redundancy_proxy(result.answer):.2f}"),
        ],
        columns=4,
    )

    tabs = st.tabs(["Válasz", "Futási nyomvonal", "Kontextus", "Bizonyítékok", "LLM telemetria", "Forrásokra támaszkodó prompt"])

    with tabs[0]:
        st.subheader("Válasz")
        generation_mode = str(result.trace.get("generation_mode", "standard"))
        st.markdown(result.answer)
        if generation_mode != "standard":
            st.caption("Válaszgenerálási mód: " + {
                "grounded-repair": "automatikusan javított, forrásokra támaszkodó válasz",
                "source-synthesis-fallback": "forrásmondatokból összeállított biztonsági válasz",
            }.get(generation_mode, generation_mode))
        st.caption("Hivatkozások: " + ", ".join(result.citations or ["nincs"]))
        export_text = f"Kérdés:\n{result.query}\n\nVálasz:\n{result.answer}\n\nHivatkozások: {', '.join(result.citations or ['nincs'])}\n"
        render_text_export(export_text, label="Válasz letöltése TXT-ként", filename="rag_valasz.txt", key=f"rag_answer_export_{strategy}")

    with tabs[1]:
        rag_trace_graph(
            result.trace,
            key=f"rag_graph_{strategy}_{len(result.retrieved_chunks)}",
            reranker_active=bool(runtime.get("reranker_active", False)),
        )
        st.markdown("**Tényleges futtatási környezet**")
        safe_dataframe(
            pd.DataFrame([{"Komponens": key, "Érték": value} for key, value in runtime.items() if key != "reranker_active"]),
            width="stretch",
            hide_index=True,
        )

    with tabs[2]:
        context_profile_chart(
            result.retrieved_chunks,
            key=f"rag_context_{strategy}_{result.context_tokens}",
            title="Kontextusprofil · források és tokenarányok",
        )
        note_box(
            "Kontextus értelmezése",
            "A kontextus-kihasználtság lexikális proxy: azt méri, hogy a válasz szókészletének mekkora része található meg a visszakeresett kontextusban. Nem helyettesít klinikai vagy szemantikus faithfulness értékelést.",
        )

    with tabs[3]:
        render_answer_evidence(result, key_prefix=f"rag_evidence_{strategy}")
        retrieval_scores_chart(
            result.retrieved_chunks,
            key=f"rag_scores_{strategy}_{len(result.retrieved_chunks)}",
            title="Bizonyítékpontszámok profilja",
        )

    with tabs[4]:
        telemetry = {
            "TTFT ms": result.generation_ttft_ms,
            "Generálás ms": result.generation_latency_ms,
            "Kimeneti token": result.output_tokens,
            "Token/s": result.tokens_per_second,
            "Teljes idő ms": result.total_latency_ms,
            "Kontextus token": result.context_tokens,
        }
        safe_dataframe(pd.DataFrame([telemetry]), width="stretch", hide_index=True)
        with st.expander("Teljes folyamat nyomvonala", expanded=False):
            st.json(result.trace)

    with tabs[5]:
        render_grounded_prompt(result, key_prefix=f"grounded_{strategy}")
