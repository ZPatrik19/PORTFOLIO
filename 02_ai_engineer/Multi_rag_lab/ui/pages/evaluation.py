from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine.presets import HUNGARIAN_QUERY_PRESETS, RAG_STRATEGIES
from rag_engine.evaluation.generation import answer_completeness, citation_accuracy, context_utilization
from rag_engine.evaluation.medical_benchmark import (
    corpus_paths_from_manifest,
    rows_to_dicts,
    run_rag_benchmark,
    run_retrieval_benchmark,
)
from rag_engine.evaluation.medical_dataset import (
    build_medical_evaluation_dataset,
    load_medical_evaluation_dataset,
)
from rag_engine.evaluation.runner import evaluate_retrieval
from rag_engine.platform.registry import ExperimentRegistry
from rag_engine.service import create_rag_pipeline
from ui.components.charts import (
    metric_bar_chart,
    percentile_latency_chart,
    quality_latency_scatter,
    radar_metrics_chart,
    stacked_latency_chart,
)
from ui.components.common import ROOT, get_lab, get_selected_retriever, selected_embedding_model
from ui.components.education import info_cards, kpi_cards, metrics_reference, note_box, page_intro, section_intro
from ui.components.reranker import build_selected_reranker, reranker_status_text
from ui.components.exports import render_dataframe_exports
from ui.components.evaluation_view_models import (
    build_rag_display_frame,
    build_retrieval_display_frame,
    rag_summary,
    retrieval_summary,
)
from ui.components.tables import safe_dataframe, safe_data_editor


EVAL_PATH = ROOT / "artifacts" / "evaluations" / "medical_rag_eval.jsonl"
RAW_DIR = ROOT / "data" / "raw" / "hungarian_medical"
MANIFEST_PATH = RAW_DIR / "manifest.json"
RETRIEVAL_OUTPUT = ROOT / "artifacts" / "evaluations" / "medical_retrieval_benchmark.json"
RAG_OUTPUT = ROOT / "artifacts" / "evaluations" / "medical_rag_benchmark.json"
REGISTRY_PATH = ROOT / "artifacts" / "experiments" / "experiments.sqlite3"

QUESTION_TYPE_HU = {
    "overview": "Áttekintés",
    "symptoms": "Tünetek",
    "causes": "Okok / kockázatok",
    "diagnosis": "Diagnózis",
    "treatment": "Kezelés",
    "prevention": "Megelőzés",
    "doctor": "Mikor forduljon orvoshoz?",
    "complications": "Szövődmények",
}

TOPIC_KEYWORDS = {
    "Kardiológia": ["vérnyomás", "szív", "kockázat", "orvos"],
    "Anyagcsere": ["cukorbetegség", "vércukor", "inzulin", "tünet"],
    "Pulmonológia": ["asztma", "légzés", "köhögés", "zihálás"],
    "Allergológia": ["anafilaxia", "allergia", "sürgős", "tünet"],
    "Neurológia": ["agyrázkódás", "fejfájás", "tünet", "orvos"],
    "Mentális egészség": ["depresszió", "hangulat", "tünet", "segítség"],
    "Mozgásszervi egészség": ["csontritkulás", "csont", "kockázat", "megelőzés"],
    "Szemészet": ["zöldhályog", "szem", "látás", "felismerés"],
}


def _dataset_frame(items) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": item.id,
                "Típus": QUESTION_TYPE_HU.get(item.question_type, item.question_type),
                "Kérdés": item.query,
                "Forráscikk": item.article_title,
                "Szekció": item.section,
                "Expected factek": " | ".join(item.expected_key_facts[:3]),
                "Forrás URL": item.source_url,
            }
            for item in items
        ]
    )


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render() -> None:
    page_intro(
        "RAG kiértékelés",
        "A kiértékelés most már ugyanabból a magyar orvosi korpuszból épül, mint a RAG. A forrásolt dataset 50–100 kérdést, source article hivatkozást, bizonyíték-részletet és expected key facteket tartalmaz; erre futtatható retrieval- és teljes RAG-benchmark.",
        eyebrow="SOURCE-GROUNDED DATASET · RETRIEVAL EVAL · RAG EVAL · REPRODUCIBLE BENCHMARK",
    )
    metrics_reference("evaluation")
    info_cards(
        [
            ("Recall@K", "A forrásolt bizonyíték-hez illeszkedő releváns chunkok mekkora részét hozza vissza a Top-K."),
            (
                "Ranking metrikák",
                "MRR, MAP@K és nDCG@K együtt mutatják az első releváns találatot és a teljes Top-K rangsor minőségét.",
            ),
            (
                "Diverzitás és zaj",
                "F1@K, source diversity, duplicate ratio és labeling coverage segít megmutatni, mennyire hasznos a retrieved halmaz.",
            ),
            (
                "Generation quality",
                "Kulcstény-lefedettség, citation accuracy/coverage/source coverage, context utilization és redundancia proxy.",
            ),
        ],
        columns=4,
    )

    items = load_medical_evaluation_dataset(EVAL_PATH)
    unique_sources = len({item.source_id for item in items}) if items else 0
    kpi_cards(
        [
            ("Evaluation kérdések", str(len(items)), "Alapértelmezett cél: 80, konfigurálható 50–100 között."),
            ("Forráscikkek", str(unique_sources), "A kérdések tényleges Egészségvonal-cikkekhez kötődnek."),
            (
                "Visszakeresési metrikák",
                "10+ minőségi/teljesítmény metrika",
                "Recall, Precision, F1, Hit Rate, MRR, MAP, nDCG, rang, diverzitás és késleltetés.",
            ),
            ("RAG metrikák", "15+ E2E", "Citation, key facts, context, redundancy, TTFT, token/s és stage latency."),
        ],
        columns=4,
    )

    dataset_tab, retrieval_tab, rag_tab, manual_tab = st.tabs(
        ["1. Forrásolt dataset", "2. Visszakeresési benchmark", "3. Teljes RAG benchmark", "4. Manuális ellenőrzés"]
    )

    with dataset_tab:
        section_intro(
            "Magyar orvosi evaluation dataset",
            "A builder nem talál ki orvosi tényeket: a kérdések, bizonyíték-részletek és expected key factek a lokálisan letöltött cikkekből készülnek.",
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            target_questions = st.slider("Dataset mérete", 50, 100, 80, step=5, key="medical_eval_target")
            if st.button("Evaluation dataset felépítése / frissítése", type="primary", width="stretch"):
                try:
                    with st.spinner("Forráscikkek elemzése és kérdések készítése..."):
                        dataset = build_medical_evaluation_dataset(
                            raw_dir=RAW_DIR,
                            manifest_path=MANIFEST_PATH,
                            output_path=EVAL_PATH,
                            target_questions=target_questions,
                        )
                    st.success(f"{len(dataset.items)} forrásolt kérdés elkészült.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"A dataset építése sikertelen: {exc}")
            note_box(
                "Módszertani megjegyzés",
                "A releváns chunk-ID nem statikusan van beégetve. Benchmark futáskor az bizonyíték és a forrásazonosító alapján oldjuk fel, ezért ugyanaz a kérdéskészlet több chunking stratégiával is használható.",
            )
        with c2:
            if items:
                type_counts = pd.Series(
                    [QUESTION_TYPE_HU.get(item.question_type, item.question_type) for item in items]
                ).value_counts()
                st.bar_chart(type_counts)
            else:
                st.info(
                    "Még nincs evaluation dataset. Előbb töltsd le a 100 cikkes orvosi korpuszt, majd építsd fel a datasetet."
                )

        if items:
            safe_dataframe(_dataset_frame(items), width="stretch", hide_index=True)
            with st.expander("Egy evaluation rekord teljes tartalma", expanded=False):
                preview = st.selectbox("Rekord", items, format_func=lambda item: f"{item.id} · {item.query}")
                st.json(preview.model_dump())

    with retrieval_tab:
        section_intro(
            "Darabolás × visszakeresés × újrarangsorolás benchmark",
            "Azonos kérdéskészleten hasonlíthatod össze a chunking stratégiákat, Dense/BM25/Hybrid retrievalt és az opcionális rerankingot.",
        )
        if not items:
            st.warning("A benchmarkhoz előbb építsd fel az evaluation datasetet az első tabon.")
        else:
            q_count = st.slider(
                "Benchmark kérdések", 5, len(items), min(20, len(items)), step=5, key="retrieval_eval_qcount"
            )
            chunkings = st.multiselect(
                "Chunking stratégiák",
                ["fixed", "recursive", "sentence", "paragraph", "structure-aware", "parent-child", "semantic"],
                default=["recursive", "sentence", "paragraph"],
            )
            retrieval_modes = st.multiselect(
                "Visszakeresési módok",
                ["dense", "bm25", "hybrid-rrf", "hybrid-weighted"],
                default=["dense", "bm25", "hybrid-rrf", "hybrid-weighted"],
            )
            rerankers = st.multiselect(
                "Újrarangsoroló", ["none", "lexical", "cross-encoder"], default=["none", "lexical"]
            )
            st.caption(
                "Cross-Encoder benchmark jelentősen lassabb lehet. Semantic chunking szintén embeddinget használ már a daraboláskor."
            )
            if st.button(
                "Visszakeresési benchmark futtatása",
                type="primary",
                disabled=not chunkings or not retrieval_modes or not rerankers,
            ):
                registry = ExperimentRegistry(REGISTRY_PATH)
                run_id = None
                started = time.perf_counter()
                try:
                    paths = corpus_paths_from_manifest(MANIFEST_PATH, RAW_DIR)
                    model_name, fallback = selected_embedding_model()
                    benchmark_config = {
                        "benchmark_type": "retrieval",
                        "questions": q_count,
                        "chunking": chunkings,
                        "retrieval_modes": retrieval_modes,
                        "rerankers": rerankers,
                        "chunk_size": int(st.session_state.get("chunk_size", 700)),
                        "overlap": int(st.session_state.get("chunk_overlap", 100)),
                        "semantic_threshold": float(st.session_state.get("semantic_threshold", 0.72)),
                        "embedding_model": model_name,
                        "embedding_device": st.session_state.get("embedding_device", "auto"),
                        "vector_device": st.session_state.get("vector_device", "cpu"),
                        "top_k": int(st.session_state.get("top_k", 5)),
                        "candidate_count": int(st.session_state.get("candidate_count", 20)),
                        "fusion": st.session_state.get("fusion_mode", "rrf"),
                        "rrf_k": int(st.session_state.get("rrf_k", 60)),
                        "dense_weight": float(st.session_state.get("dense_weight", 0.5)),
                        "reranker_device": st.session_state.get("reranker_device", "auto"),
                    }
                    run_id, config_hash = registry.create_run(
                        benchmark_type="retrieval",
                        config=benchmark_config,
                        dataset_path=EVAL_PATH,
                        questions=q_count,
                        notes="Streamlit retrieval benchmark",
                    )
                    with st.spinner(
                        "Visszakeresési benchmark fut: chunking → embedding → index → queryk → metrikák..."
                    ):
                        rows = run_retrieval_benchmark(
                            paths=paths,
                            items=items[:q_count],
                            chunking_strategies=chunkings,
                            retrieval_modes=retrieval_modes,
                            rerankers=rerankers,
                            chunk_size=benchmark_config["chunk_size"],
                            overlap=benchmark_config["overlap"],
                            semantic_threshold=benchmark_config["semantic_threshold"],
                            embedding_model=model_name,
                            embedding_device=benchmark_config["embedding_device"],
                            vector_device=benchmark_config["vector_device"],
                            top_k=benchmark_config["top_k"],
                            candidate_count=benchmark_config["candidate_count"],
                            fusion=benchmark_config["fusion"],
                            rrf_k=benchmark_config["rrf_k"],
                            dense_weight=benchmark_config["dense_weight"],
                            fallback_embedding=fallback,
                            reranker_device=benchmark_config["reranker_device"],
                        )
                    registry.add_retrieval_results(run_id, rows, embedding_model=model_name)
                    registry.complete_run(run_id, duration_ms=(time.perf_counter() - started) * 1000)
                    payload = rows_to_dicts(rows)
                    st.session_state["medical_retrieval_eval"] = payload
                    st.session_state["last_experiment_run_id"] = run_id
                    _save_json(
                        RETRIEVAL_OUTPUT,
                        {"run_id": run_id, "config_hash": config_hash, "questions": q_count, "results": payload},
                    )
                    st.success(
                        f"Benchmark elmentve az Experiment Registry-be. run_id: {run_id} · config: {config_hash[:12]}"
                    )
                except Exception as exc:
                    if run_id is not None:
                        registry.fail_run(run_id, exc, duration_ms=(time.perf_counter() - started) * 1000)
                    st.error(f"A retrieval benchmark sikertelen: {exc}")

            retrieval_results = st.session_state.get("medical_retrieval_eval", [])
            if retrieval_results:
                display = build_retrieval_display_frame(retrieval_results)
                summary = retrieval_summary(display)
                with st.expander("Nyers retrieval benchmark eredmények", expanded=False):
                    safe_dataframe(
                        display.sort_values(["Összesített pontszám", "nDCG@K"], ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                kpi_cards(
                    [
                        (
                            "Legjobb visszakeresési konfiguráció",
                            summary["best_configuration"],
                            "Súlyozott összesített minőségi pontszám alapján.",
                        ),
                        (
                            "Legjobb összesített pontszám",
                            f"{summary['best_overall_score']:.3f}",
                            "Recall, MRR, nDCG, F1, forrásdiverzitás és címkézési lefedettség kombinációja.",
                        ),
                        (
                            "Legjobb hatékonysági pontszám",
                            f"{summary['best_efficiency_score']:.3f}",
                            "Minőség + késleltetés egyensúlyozott mutató.",
                        ),
                        (
                            "Leggyorsabb konfiguráció",
                            f"{summary['fastest_latency_ms']:.1f} ms",
                            "Az átlagos késleltetés minimuma az aktuális benchmarkon.",
                        ),
                    ],
                    columns=4,
                )
                c1, c2 = st.columns(2)
                with c1:
                    quality_latency_scatter(
                        display,
                        x="Átlagos késleltetés ms",
                        y="Összesített pontszám",
                        color="retriever",
                        hover=["chunking", "reranker", "Recall@K", "MRR", "QPS", "nDCG@K"],
                        size="QPS",
                        key="medical_retrieval_quality_latency",
                        title="Visszakeresési minőség–latency trade-off",
                    )
                with c2:
                    metric_bar_chart(
                        display.sort_values(["Összesített pontszám", "nDCG@K"], ascending=False).head(12),
                        x="Stratégia",
                        y="Összesített pontszám",
                        color="retriever",
                        key="medical_retrieval_overall_bar",
                        title="Top retrieval konfigurációk · összesített minőségi pontszám",
                    )
                c3, c4 = st.columns(2)
                with c3:
                    radar_metrics_chart(
                        display.sort_values(["Összesített pontszám", "nDCG@K"], ascending=False),
                        label_col="Stratégia",
                        metrics=["Recall@K", "Precision@K", "F1@K", "MRR", "nDCG@K"],
                        key="medical_retrieval_radar",
                        title="Top retrieval konfigurációk · minőségi profil",
                        max_series=5,
                    )
                with c4:
                    metric_bar_chart(
                        display.groupby("retriever", as_index=False)[["Összesített pontszám", "nDCG@K", "MRR"]]
                        .mean()
                        .sort_values("Összesített pontszám", ascending=False),
                        x="retriever",
                        y="Összesített pontszám",
                        color="retriever",
                        key="medical_retrieval_family_bar",
                        title="Visszakeresési családok átlaga · összesített minőségi pontszám",
                    )
                percentile_latency_chart(
                    display.sort_values("Összesített pontszám", ascending=False).head(12),
                    category="Stratégia",
                    percentiles=["P50 késleltetés ms", "P95 késleltetés ms", "P99 késleltetés ms"],
                    key="medical_retrieval_percentiles_v2",
                    title="Visszakeresési késleltetés percentilisei · top konfigurációk",
                    value_label="Késleltetés ms",
                )
                note_box(
                    "Mit jelent az összesített pontszám?",
                    "Az összesített visszakeresési pontszám súlyozott összeg: Recall@K, MRR, nDCG@K, F1@K, forrásdiverzitás és label coverage. Diagnosztikai célú, ezért nem helyettesíti a részmetrikák külön értelmezését.",
                )

    with rag_tab:
        section_intro(
            "Teljes RAG pipeline benchmark",
            "A retrieval után a context builder és az LLM is fut. A generated answer a dataset expected key factjeihez és a hivatkozásokhoz mérhető.",
        )
        if not items:
            st.warning("A benchmarkhoz előbb építsd fel az evaluation datasetet.")
        else:
            q_count = st.slider(
                "RAG benchmark kérdések", 5, len(items), min(10, len(items)), step=5, key="rag_eval_qcount"
            )
            strategies = st.multiselect(
                "RAG stratégiák",
                list(RAG_STRATEGIES),
                default=["baseline", "hybrid", "reranked", "compression", "corrective"],
                format_func=lambda x: RAG_STRATEGIES[x]["name"],
            )
            st.caption(
                f"Aktív LLM szolgáltató: **{st.session_state.get('llm_provider', 'dummy')}**. Ollama esetén valódi lokális generálás történik."
            )
            _, eval_reranker_status = build_selected_reranker("reranked")
            st.caption(
                f"Az újrarangsorolt RAG stratégiák ezt használják: **{reranker_status_text(eval_reranker_status)}**."
            )
            if st.button("Teljes RAG benchmark futtatása", type="primary", disabled=not strategies):
                registry = ExperimentRegistry(REGISTRY_PATH)
                run_id = None
                started = time.perf_counter()
                try:
                    paths = corpus_paths_from_manifest(MANIFEST_PATH, RAW_DIR)
                    model_name, fallback = selected_embedding_model()
                    eval_reranker, eval_reranker_status = build_selected_reranker("reranked")
                    benchmark_config = {
                        "benchmark_type": "rag",
                        "questions": q_count,
                        "rag_strategies": strategies,
                        "chunking": st.session_state.get("chunking_strategy", "recursive"),
                        "chunk_size": int(st.session_state.get("chunk_size", 700)),
                        "overlap": int(st.session_state.get("chunk_overlap", 100)),
                        "semantic_threshold": float(st.session_state.get("semantic_threshold", 0.72)),
                        "embedding_model": model_name,
                        "embedding_device": st.session_state.get("embedding_device", "auto"),
                        "vector_device": st.session_state.get("vector_device", "cpu"),
                        "llm_provider": st.session_state.get("llm_provider", "dummy"),
                        "top_k": int(st.session_state.get("top_k", 5)),
                        "candidate_count": int(st.session_state.get("candidate_count", 20)),
                        "context_budget": int(st.session_state.get("context_budget", 1800)),
                        "context_profile": st.session_state.get("context_profile", "balanced"),
                        "prompt_profile": st.session_state.get("prompt_profile", "professional"),
                        "reranker": eval_reranker_status.applied,
                        "reranker_device": eval_reranker_status.device,
                    }
                    run_id, config_hash = registry.create_run(
                        benchmark_type="rag",
                        config=benchmark_config,
                        dataset_path=EVAL_PATH,
                        questions=q_count,
                        notes="Streamlit end-to-end RAG benchmark",
                    )
                    with st.spinner("RAG stratégiák futtatása a forrásolt dataseten..."):
                        rows = run_rag_benchmark(
                            paths=paths,
                            items=items[:q_count],
                            rag_strategies=strategies,
                            chunking_strategy=benchmark_config["chunking"],
                            chunk_size=benchmark_config["chunk_size"],
                            overlap=benchmark_config["overlap"],
                            semantic_threshold=benchmark_config["semantic_threshold"],
                            embedding_model=model_name,
                            embedding_device=benchmark_config["embedding_device"],
                            vector_device=benchmark_config["vector_device"],
                            llm_provider=benchmark_config["llm_provider"],
                            ollama_profile=st.session_state.get("ollama_profile", "balanced"),
                            top_k=benchmark_config["top_k"],
                            candidate_count=benchmark_config["candidate_count"],
                            max_context_tokens=benchmark_config["context_budget"],
                            fallback_embedding=fallback,
                            context_profile=benchmark_config["context_profile"],
                            prompt_profile=benchmark_config["prompt_profile"],
                            reranker=eval_reranker,
                        )
                    registry.add_rag_results(
                        run_id,
                        rows,
                        chunking=benchmark_config["chunking"],
                        embedding_model=model_name,
                        context_budget=benchmark_config["context_budget"],
                        vector_device=benchmark_config["vector_device"],
                    )
                    registry.complete_run(run_id, duration_ms=(time.perf_counter() - started) * 1000)
                    payload = rows_to_dicts(rows)
                    st.session_state["medical_rag_eval"] = payload
                    st.session_state["last_experiment_run_id"] = run_id
                    _save_json(
                        RAG_OUTPUT,
                        {"run_id": run_id, "config_hash": config_hash, "questions": q_count, "results": payload},
                    )
                    st.success(
                        f"RAG benchmark elmentve az Experiment Registry-be. run_id: {run_id} · config: {config_hash[:12]}"
                    )
                except Exception as exc:
                    if run_id is not None:
                        registry.fail_run(run_id, exc, duration_ms=(time.perf_counter() - started) * 1000)
                    st.error(f"A RAG benchmark sikertelen: {exc}")

            rag_results = st.session_state.get("medical_rag_eval", [])
            if rag_results:
                frame = build_rag_display_frame(rag_results)
                summary = rag_summary(frame)
                with st.expander("Nyers RAG benchmark eredmények", expanded=False):
                    safe_dataframe(
                        frame.sort_values(["Összesített pontszám", "Kulcstény-lefedettség"], ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                kpi_cards(
                    [
                        (
                            "Legjobb RAG stratégia",
                            summary["best_strategy"],
                            "Hivatkozási pontosság, kulcstény-lefedettség és kontextus-kihasználtság alapján.",
                        ),
                        (
                            "Legjobb összesített pontszám",
                            f"{summary['best_overall_score']:.3f}",
                            "Súlyozott végpont a fő minőségi mutatókból.",
                        ),
                        (
                            "Legjobb hatékonysági pontszám",
                            f"{summary['best_efficiency_score']:.3f}",
                            "Minőség és teljes késleltetés egyensúlya.",
                        ),
                        (
                            "Legjobb TTFT",
                            f"{summary['best_ttft_ms']:.0f} ms",
                            "Az első tokenig mért minimum a futott stratégiák között.",
                        ),
                    ],
                    columns=4,
                )
                c1, c2 = st.columns(2)
                with c1:
                    quality_latency_scatter(
                        frame,
                        x="Átlagos teljes idő ms",
                        y="Összesített pontszám",
                        color="Stratégia",
                        hover=[
                            "Hivatkozási pontosság",
                            "Hivatkozási lefedettség",
                            "TTFT ms",
                            "Token/s",
                            "P95 teljes idő ms",
                            "Kulcstény-lefedettség",
                        ],
                        size="Átlagos kontextustoken",
                        key="medical_rag_quality_latency",
                        title="RAG minőség–késleltetés egyensúly",
                    )
                with c2:
                    metric_bar_chart(
                        frame.sort_values(["Összesített pontszám", "Kulcstény-lefedettség"], ascending=False).head(10),
                        x="Stratégia",
                        y="Összesített pontszám",
                        color="Stratégia",
                        key="medical_rag_overall_bar",
                        title="RAG stratégiák · összesített minőségi pontszám",
                    )
                c3, c4 = st.columns(2)
                with c3:
                    radar_metrics_chart(
                        frame.sort_values(["Összesített pontszám", "Kulcstény-lefedettség"], ascending=False),
                        label_col="Stratégia",
                        metrics=[
                            "Hivatkozási pontosság",
                            "Hivatkozási lefedettség",
                            "Forráslefedettség",
                            "Kulcstény-lefedettség",
                            "Kontextus-kihasználtság",
                        ],
                        key="medical_rag_radar",
                        title="RAG quality profil · top stratégiák",
                        max_series=6,
                    )
                with c4:
                    metric_bar_chart(
                        frame[["Stratégia", "Token/s", "TTFT ms", "Átlagos teljes idő ms"]].sort_values(
                            "Token/s", ascending=False
                        ),
                        x="Stratégia",
                        y="Token/s",
                        color="Stratégia",
                        key="medical_rag_tokenspeed",
                        title="RAG stratégiák · generálási sebesség (token/s)",
                    )
                stacked_latency_chart(
                    frame,
                    strategy_col="Stratégia",
                    stage_columns=["Visszakeresés ms", "Újrarangsorolás ms", "Generálás ms"],
                    key="medical_rag_stage_latency",
                    title="End-to-end latency bontás",
                )
                percentile_latency_chart(
                    frame,
                    category="Stratégia",
                    percentiles=["P50 teljes idő ms", "P95 teljes idő ms", "P99 teljes idő ms"],
                    key="medical_rag_percentiles_v2",
                    title="RAG teljes késleltetés · P50 / P95 / P99",
                    value_label="Késleltetés ms",
                )
                note_box(
                    "Mit jelent az összesített pontszám?",
                    "A RAG összesített pontszám súlyozott metrika: hivatkozási pontosság, hivatkozási lefedettség, forráslefedettség, kulcstény-lefedettség és kontextus-kihasználtság. Összegző nézet, de a klinikai helyesség külön ellenőrzést igényel.",
                )

    with manual_tab:
        section_intro(
            "Manuális retrieval címkézés",
            "A source-grounded dataset mellett megmarad a kézi ellenőrzés is, mert a retrieval relevance végső validációjához emberi review továbbra is értékes.",
        )
        preset = st.selectbox(
            "Kiértékelési kérdés", HUNGARIAN_QUERY_PRESETS, format_func=lambda x: f"{x.label} · {x.topic}"
        )
        query = st.text_area("Lekérdezés", value=preset.query, height=85, key=f"eval_query_{preset.label}")
        k = st.slider("K", 1, 20, int(st.session_state.get("top_k", 5)), key="eval_k")
        pool_size = st.slider(
            "Címkézési jelöltkészlet mérete",
            k,
            60,
            max(k, int(st.session_state.get("candidate_count", 20))),
            key="eval_pool_size",
        )

        if st.button("Jelöltek lekérése manuális címkézéshez"):
            try:
                lab = get_lab()
                retriever = get_selected_retriever(lab)
                try:
                    candidates = retriever.retrieve(query, top_k=pool_size, candidate_count=pool_size)
                except TypeError:
                    candidates = retriever.retrieve(query, top_k=pool_size)
                st.session_state["eval_candidates"] = candidates
            except Exception as exc:
                st.error(f"A visszakeresés sikertelen: {exc}")

        candidates = st.session_state.get("eval_candidates", [])
        if candidates:
            rows = [
                {
                    "Releváns": False,
                    "Rang": item.rank,
                    "Szövegrész ID": item.chunk_id,
                    "Forrás": item.metadata.get("title") or item.source,
                    "Oldal": item.metadata.get("page"),
                    "Pontszám": float(item.score),
                    "Részlet": item.text[:220].replace("\n", " "),
                }
                for item in candidates
            ]
            edited = safe_data_editor(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
                disabled=["Rang", "Szövegrész ID", "Forrás", "Oldal", "Pontszám", "Részlet"],
                key="eval_label_editor",
            )
            labels = set(edited.loc[edited["Releváns"], "Szövegrész ID"].tolist())
            metrics = evaluate_retrieval([x.chunk_id for x in candidates], labels, k)
            metric_rows = [
                {"Metrika": "Recall@K", "Érték": metrics.recall_at_k},
                {"Metrika": "Precision@K", "Érték": metrics.precision_at_k},
                {"Metrika": "F1@K", "Érték": metrics.f1_at_k},
                {"Metrika": "Hit Rate@K", "Érték": metrics.hit_rate_at_k},
                {"Metrika": "MRR", "Érték": metrics.mrr},
                {"Metrika": "MAP@K", "Érték": metrics.map_at_k},
                {"Metrika": "nDCG@K", "Érték": metrics.ndcg_at_k},
                {"Metrika": "Első releváns rang", "Érték": metrics.first_relevant_rank},
            ]
            safe_dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

        st.divider()
        st.markdown("**Egyedi válaszgenerálási proxyk**")
        expected = st.text_input(
            "Elvárt kulcsszavak vesszővel", ",".join(TOPIC_KEYWORDS.get(preset.topic, ["információ", "forrás"]))
        )
        if st.button("RAG válasz és proxyk futtatása"):
            try:
                lab = get_lab(
                    strategy="parent-child" if st.session_state.get("rag_strategy") == "parent-document" else None
                )
                pipeline = create_rag_pipeline(
                    lab,
                    st.session_state.get("rag_strategy", "hybrid"),
                    max_context_tokens=int(st.session_state.get("context_budget", 1800)),
                    top_k=k,
                    candidate_count=int(st.session_state.get("candidate_count", 20)),
                    context_profile=st.session_state.get("context_profile", "balanced"),
                    prompt_profile=st.session_state.get("prompt_profile", "professional"),
                )
                result = pipeline.answer(query)
                context = "\n".join(chunk.text for chunk in result.retrieved_chunks)
                keywords = [x.strip() for x in expected.split(",") if x.strip()]
                st.markdown(result.answer)
                cols = st.columns(3)
                cols[0].metric("Completeness proxy", f"{answer_completeness(result.answer, keywords):.3f}")
                cols[1].metric(
                    "Citation proxy", f"{citation_accuracy(result.answer, len(result.retrieved_chunks)):.3f}"
                )
                cols[2].metric("Context-utilization proxy", f"{context_utilization(result.answer, context):.3f}")
            except Exception as exc:
                st.error(f"A generálási ellenőrzés sikertelen: {exc}")
