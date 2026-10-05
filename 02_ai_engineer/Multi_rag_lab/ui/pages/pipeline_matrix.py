from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine.presets import RAG_STRATEGIES
from rag_engine.evaluation.medical_benchmark import corpus_paths_from_manifest
from rag_engine.evaluation.medical_dataset import build_medical_evaluation_dataset, load_medical_evaluation_dataset
from rag_engine.evaluation.pipeline_matrix import build_matrix_plan, run_pipeline_matrix
from rag_engine.evaluation.runtime_estimator import estimate_matrix_runtime
from rag_engine.platform.registry import ExperimentRegistry
from rag_engine.platform.device import cuda_available
from rag_engine.platform.hardware import get_hardware_profile
from ui.components.charts import (
    device_metric_bars,
    metric_bar_chart,
    percentile_latency_chart,
    quality_latency_scatter,
    radar_metrics_chart,
    stacked_latency_chart,
)
from ui.components.common import ROOT
from ui.components.education import info_cards, kpi_cards, live_run_card, note_box, page_intro, section_intro, status_cards
from ui.components.exports import render_dataframe_exports
from ui.components.tables import safe_dataframe


EVAL_PATH = ROOT / "artifacts" / "evaluations" / "medical_rag_eval.jsonl"
RAW_DIR = ROOT / "data" / "raw" / "hungarian_medical"
MANIFEST_PATH = RAW_DIR / "manifest.json"
REGISTRY_PATH = ROOT / "artifacts" / "experiments" / "experiments.sqlite3"
OUTPUT_PATH = ROOT / "artifacts" / "evaluations" / "pipeline_matrix.json"


PRESETS = {
    "Gyors ellenőrzés": {
        "questions": 5,
        "embeddings": ["multilingual"],
        "chunkings": ["recursive", "sentence"],
        "retrieval": ["dense", "hybrid-rrf"],
        "rerankers": ["none", "lexical"],
        "rag": ["baseline", "hybrid", "reranked"],
        "component_docs": 8,
        "component_chunks": 96,
    },
    "Portfólió benchmark": {
        "questions": 20,
        "embeddings": ["multilingual", "e5-small", "english"],
        "chunkings": ["fixed", "recursive", "sentence", "paragraph", "structure-aware"],
        "retrieval": ["dense", "bm25", "hybrid-rrf", "hybrid-weighted"],
        "rerankers": ["none", "lexical"],
        "rag": ["baseline", "lexical", "hybrid", "reranked", "compression", "corrective"],
        "component_docs": 20,
        "component_chunks": 256,
    },
    "Teljes benchmark": {
        "questions": 80,
        "embeddings": ["multilingual", "e5-small", "english", "hashing"],
        "chunkings": ["fixed", "fixed-token", "recursive", "sentence", "paragraph", "structure-aware", "parent-child", "semantic"],
        "retrieval": ["dense", "bm25", "hybrid-rrf", "hybrid-weighted"],
        "rerankers": ["none", "lexical", "cross-encoder"],
        "rag": list(RAG_STRATEGIES),
        "component_docs": 40,
        "component_chunks": 512,
    },
}


def _save(payload: dict[str, object]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _retrieval_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["Konfiguráció"] = (
        frame["embedding_mode"].astype(str)
        + " · E="
        + frame["requested_embedding_device"].astype(str)
        + " · V="
        + frame["requested_vector_device"].astype(str)
        + " · R="
        + frame["requested_reranker_device"].astype(str)
        + " · "
        + frame["chunking"].astype(str)
        + " · "
        + frame["retriever"].astype(str)
        + " · "
        + frame["reranker"].astype(str)
    )
    frame["ID"] = [f"R{index:03d}" for index in range(1, len(frame) + 1)]
    return frame


def _rag_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    reranker_text = frame["reranker"].astype(str) if "reranker" in frame.columns else pd.Series(["none"] * len(frame), index=frame.index)
    reranker_device = frame["requested_reranker_device"].astype(str) if "requested_reranker_device" in frame.columns else pd.Series(["—"] * len(frame), index=frame.index)
    frame["Konfiguráció"] = (
        frame["embedding_mode"].astype(str)
        + " · E="
        + frame["requested_embedding_device"].astype(str)
        + " · V="
        + frame["requested_vector_device"].astype(str)
        + " · "
        + frame["chunking"].astype(str)
        + " · "
        + frame["rag_strategy"].astype(str)
        + " · RR="
        + reranker_text
        + "/"
        + reranker_device
    )
    frame["ID"] = [f"G{index:03d}" for index in range(1, len(frame) + 1)]
    return frame


def _clip01(series: pd.Series) -> pd.Series:
    return series.fillna(0).clip(lower=0, upper=1)


def _scale_inverse(series: pd.Series) -> pd.Series:
    filled = series.fillna(series.median() if not series.dropna().empty else 0.0)
    lo, hi = float(filled.min()), float(filled.max())
    if hi - lo < 1e-9:
        return pd.Series([1.0] * len(filled), index=filled.index)
    return 1 - ((filled - lo) / (hi - lo))


def _retrieval_overall_score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.24 * _clip01(frame["recall_at_k"])
        + 0.20 * _clip01(frame["mrr"])
        + 0.20 * _clip01(frame["ndcg_at_k"])
        + 0.14 * _clip01(frame["f1_at_k"])
        + 0.12 * _clip01(frame["source_diversity_at_k"])
        + 0.10 * _clip01(frame["labeling_coverage"])
    ).round(3)


def _rag_overall_score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.28 * _clip01(frame["citation_accuracy"])
        + 0.20 * _clip01(frame["citation_coverage"])
        + 0.16 * _clip01(frame["citation_source_coverage"])
        + 0.24 * _clip01(frame["key_fact_coverage"])
        + 0.12 * _clip01(frame["context_utilization"])
    ).round(3)


def _human_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} ó {m:02d} p"
    if m:
        return f"{m} p {s:02d} mp"
    return f"{s} mp"


def _component_frames(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    frame = pd.DataFrame(rows)
    parse = frame[frame["component"] == "parsing-cleaning"].copy()
    chunking = frame[frame["component"] == "chunking"].copy()
    embedding = frame[frame["component"] == "embedding"].copy()
    vector = frame[frame["component"] == "vector-search"].copy()
    return parse, chunking, embedding, vector



def _safe_load_eval_items() -> list:
    if not EVAL_PATH.exists():
        return []
    try:
        return load_medical_evaluation_dataset(EVAL_PATH)
    except Exception:
        return []


def _corpus_paths() -> list[Path]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        return corpus_paths_from_manifest(MANIFEST_PATH, RAW_DIR)
    except Exception:
        return []


def _prepare_full_benchmark_assets() -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ROOT / "scripts" / "prepare_medical_corpus.py")]
    return subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        check=False,
    )


def _rebuild_eval_dataset() -> int:
    dataset = build_medical_evaluation_dataset(
        raw_dir=RAW_DIR,
        manifest_path=MANIFEST_PATH,
        output_path=EVAL_PATH,
        target_questions=80,
    )
    return len(dataset.items)

def render() -> None:
    page_intro(
        "Teljes pipeline benchmark",
        "A teljes RAG rendszer kontrollált benchmarkja egy helyen: parsing/cleaning, chunking, embedding, CPU/CUDA, FAISS/vector backend, retrieval, reranking és RAG stratégiák. A cél a teljes quality–latency–resource trade-off reprodukálható feltérképezése ugyanazon forrásolt dataseten.",
        eyebrow="FULL-PIPELINE MATRIX · CPU/CUDA · COMPONENT PROFILING · RETRIEVAL · RAG · PLOTLY",
    )

    items = _safe_load_eval_items()
    corpus_paths = _corpus_paths()
    corpus_ready = len(corpus_paths) >= 100
    eval_ready = len(items) >= 5

    status_cards(
        [
            ("Orvosi korpusz", "Kész" if corpus_ready else "Hiányos", f"{len(corpus_paths)} / 100 forrásdokumentum", "ok" if corpus_ready else "warn"),
            ("Evaluation dataset", "Kész" if eval_ready else "Hiányzik", f"{len(items)} forrásolt kérdés", "ok" if eval_ready else "warn"),
            ("CUDA", "Elérhető" if cuda_available() else "CPU mód", "Embedding és Cross-Encoder gyorsítás külön mérhető.", "ok" if cuda_available() else "info"),
            ("Benchmark mód", "Reprodukálható", "A quality benchmark ugyanazon corpus + question set páron fut.", "info"),
        ],
        columns=4,
    )

    if not corpus_ready or not eval_ready:
        section_intro(
            "Benchmark előkészítése",
            "Az oldal most nem áll meg egy figyelmeztetésnél: innen helyben létrehozható vagy javítható a szükséges corpus és evaluation dataset.",
        )
        if corpus_paths and MANIFEST_PATH.exists() and not eval_ready:
            if st.button("Evaluation dataset újraépítése a meglévő korpuszból", type="primary", width="stretch"):
                try:
                    with st.spinner("Forrásolt evaluation kérdéskészlet építése..."):
                        count = _rebuild_eval_dataset()
                    st.success(f"Evaluation dataset elkészült: {count} kérdés.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Evaluation dataset építése sikertelen: {exc}")
        else:
            if st.button("100 cikkes korpusz + evaluation dataset előkészítése", type="primary", width="stretch"):
                try:
                    with st.spinner("Korpusz letöltése, evaluation dataset és alapindex építése. Első futáskor ez több perc is lehet..."):
                        result = _prepare_full_benchmark_assets()
                    if result.returncode == 0:
                        st.success("Benchmark assetek elkészültek.")
                        st.rerun()
                    else:
                        st.error(f"Az előkészítés hibakóddal állt le: {result.returncode}")
                        with st.expander("Előkészítési napló", expanded=True):
                            st.code((result.stdout + "\n" + result.stderr)[-12000:], language=None)
                except subprocess.TimeoutExpired:
                    st.error("Az előkészítés 60 perc után sem fejeződött be. Futtasd terminálból: python scripts/prepare_medical_corpus.py")
                except Exception as exc:
                    st.error(f"Az előkészítés nem indítható: {exc}")
        note_box(
            "CLI alternatíva",
            "Terminálból ugyanaz a folyamat: `python scripts/prepare_medical_corpus.py`. A letöltő retry/resume logikát használ, ezért megszakadás után biztonságosan újrafuttatható.",
        )
        return

    kpi_cards(
        [
            ("Evaluation dataset", str(len(items)), "Ugyanaz a source-grounded kérdéskészlet minden variánshoz."),
            ("CUDA", "elérhető" if cuda_available() else "nem elérhető", "A beágyazás és a Cross-Encoder tényleges CUDA futtatás esetén mérhető."),
            ("Chunking", "8 stratégia", "Fixed, recursive, sentence, paragraph, structure-aware, parent-child és semantic."),
            ("RAG", f"{len(RAG_STRATEGIES)} stratégia", "Baseline, lexical, hybrid, reranked, HyDE, Multi-Hop és további variánsok."),
        ],
        columns=4,
    )

    info_cards(
        [
            ("Komponensprofil", "Parsing/cleaning, chunking latency és chunk-statisztika, embedding load/throughput, vector index build és search percentilisek."),
            ("Visszakeresési minőség", "Recall@K, Precision@K, F1@K, Hit Rate, MRR, MAP@K, nDCG@K, first relevant rank, diversity és duplicate ratio."),
            ("Generálás", "Hivatkozási pontosság/coverage/source coverage, key-fact coverage, context utilization, answer redundancy, TTFT és token/s."),
            ("Reprodukálhatóság", "Minden teljes mátrix run_id-val, config hash-sel és teljes JSON payload-dal bekerül az Experiment Registry-be."),
        ],
        columns=4,
    )

    section_intro(
        "Benchmark-tér definiálása",
        "A nagy mátrix valódi méréseket futtat. Az Exhaustive preset ezért szándékosan drága; lokális 4 GB-os GPU-n először a Portfólió benchmarkkal érdemes validálni az egész workflow-t.",
    )

    preset_name = st.selectbox("Benchmark előbeállítás", list(PRESETS), index=1, key="matrix_preset")
    preset = PRESETS[preset_name]
    cuda_devices = ["cpu", "cuda"] if cuda_available() else ["cpu"]

    info_cards(
        [
            ("Preset", preset_name),
            ("Kérdésszám", f"{min(int(preset['questions']), len(items))} alapérték"),
            ("Beágyazási tér", f"{len(preset['embeddings'])} modell · {'CPU + CUDA' if cuda_available() else 'CPU'}"),
            ("RAG tér", f"{len(preset['rag'])} stratégia"),
        ],
        columns=4,
    )

    with st.expander("Benchmark konfiguráció · részletes beállítások", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            question_count = st.slider("Evaluation kérdések", 5, len(items), min(int(preset["questions"]), len(items)), step=5, key=f"matrix_q_{preset_name}")
            embedding_modes = st.multiselect(
                "Beágyazási modellek",
                ["multilingual", "e5-small", "english", "hashing"],
                default=list(preset["embeddings"]),
                key=f"matrix_embed_{preset_name}",
            )
            embedding_devices = st.multiselect(
                "Beágyazási eszköz",
                ["cpu", "cuda"],
                default=cuda_devices,
                key=f"matrix_embed_devices_{preset_name}",
            )
        with c2:
            chunkings = st.multiselect(
                "Chunking stratégiák",
                ["fixed", "fixed-token", "recursive", "sentence", "paragraph", "structure-aware", "parent-child", "semantic"],
                default=list(preset["chunkings"]),
                key=f"matrix_chunk_{preset_name}",
            )
            vector_devices = st.multiselect(
                "Vector backend device",
                ["cpu", "cuda"],
                default=["cpu"],
                help="Windows faiss-cpu esetén a CUDA kérés valós CPU fallbackként fog látszani, nem hamis GPU eredményként.",
                key=f"matrix_vector_devices_{preset_name}",
            )
            include_components = st.checkbox("Komponens microbenchmark", value=True, key=f"matrix_components_{preset_name}")
        with c3:
            retrieval_modes = st.multiselect(
                "Visszakeresési módok",
                ["dense", "bm25", "hybrid-rrf", "hybrid-weighted"],
                default=list(preset["retrieval"]),
                key=f"matrix_retrieval_{preset_name}",
            )
            rerankers = st.multiselect(
                "Újrarangsorolóek",
                ["none", "lexical", "cross-encoder"],
                default=list(preset["rerankers"]),
                key=f"matrix_rerank_{preset_name}",
            )
            reranker_devices = st.multiselect(
                "Cross-Encoder device",
                ["cpu", "cuda"],
                default=cuda_devices if "cross-encoder" in rerankers else ["cpu"],
                key=f"matrix_reranker_devices_{preset_name}",
            )
        with c4:
            include_rag = st.checkbox("End-to-end RAG is fusson", value=preset_name != "Teljes benchmark", key=f"matrix_include_rag_{preset_name}")
            rag_strategies = st.multiselect(
                "RAG stratégiák",
                list(RAG_STRATEGIES),
                default=list(preset["rag"]),
                format_func=lambda key: RAG_STRATEGIES[key]["name"],
                key=f"matrix_rag_{preset_name}",
                disabled=not include_rag,
            )
            llm_provider = st.selectbox(
                "Mátrix LLM",
                ["dummy", "ollama"],
                index=0,
                format_func=lambda x: "Dummy – gyors teljes mátrix" if x == "dummy" else "Ollama – valódi lokális Qwen",
                key="matrix_llm_provider",
                disabled=not include_rag,
            )
            ollama_profile = st.selectbox(
                "Ollama profil",
                ["low_memory", "balanced", "extended_context"],
                index=0 if llm_provider == "ollama" else 1,
                key="matrix_ollama_profile",
                disabled=not include_rag or llm_provider != "ollama",
            )

        st.markdown("**Komponens benchmark részletesség**")
        d1, d2 = st.columns(2)
        with d1:
            component_documents = st.slider(
                "Chunking/parsing mintadokumentumok",
                5,
                100,
                min(int(preset["component_docs"]), 100),
                step=5,
                key=f"matrix_component_docs_{preset_name}",
                disabled=not include_components,
            )
        with d2:
            component_chunks = st.slider(
                "Beágyazási / vektoros benchmark chunkminta",
                32,
                1024,
                int(preset["component_chunks"]),
                step=32,
                key=f"matrix_component_chunks_{preset_name}",
                disabled=not include_components,
            )
        st.caption("A microbenchmark külön méri a pipeline komponenseket; ez nem helyettesíti a forrásolt retrieval/RAG quality benchmarkot.")

    plan = build_matrix_plan(
        embedding_modes=embedding_modes,
        embedding_devices=embedding_devices,
        vector_devices=vector_devices,
        reranker_devices=reranker_devices,
        chunkings=chunkings,
        retrieval_modes=retrieval_modes,
        rerankers=rerankers,
        rag_strategies=rag_strategies,
        include_rag=include_rag,
        include_components=include_components,
    )

    kpi_cards(
        [
            ("Komponens variáns", f"{plan.component_configurations:,}", "Parsing/chunking/embedding/vector microbenchmark becsült konfigurációszám."),
            ("Visszakeresési variáns", f"{plan.retrieval_configurations:,}", "Beágyazás × device × vector backend × chunking × retrieval × tényleges reranker-variáns. A lexical/none nem duplázódik CUDA device szerint."),
            ("RAG variáns", f"{plan.rag_configurations:,}", "Beágyazás × eszköz × vektortár × darabolás × RAG stratégia; az újrarangsorolt stratégiáknál az újrarangsoroló és eszköze is külön variáns."),
            ("Összes konfiguráció", f"{plan.total_configurations:,}", "A kérdésszám ezen felül megszorozza a query-szintű futások számát."),
        ],
        columns=4,
    )

    hardware = get_hardware_profile("auto")
    runtime_estimate = estimate_matrix_runtime(
        plan=plan,
        question_count=question_count,
        component_documents=component_documents,
        component_chunks=component_chunks,
        include_components=include_components,
        include_rag=include_rag,
        llm_provider=llm_provider,
        ollama_profile=ollama_profile,
        cuda_selected=("cuda" in embedding_devices) or ("cuda" in reranker_devices),
        cuda_available=cuda_available(),
        rerankers=rerankers,
        gpu_vram_gb=(hardware.gpu_total_memory_mb / 1024) if hardware.gpu_total_memory_mb else None,
    )
    eta_low_s = runtime_estimate.low_seconds
    eta_high_s = runtime_estimate.high_seconds
    kpi_cards(
        [
            ("Becsült futási idő", f"{_human_seconds(runtime_estimate.expected_seconds)}", "Konfigurációszám, kérdésszám, LLM provider és CPU/CUDA választás alapján számolt becslés."),
            ("Reális időtartomány", f"{_human_seconds(eta_low_s)} – {_human_seconds(eta_high_s)}", "A gép terhelése, cold start és Ollama válaszidő miatt természetes szórással."),
            ("Lekérdezésszintű futások", f"{runtime_estimate.retrieval_executions:,}", "Visszakeresési részterhelés a kiválasztott benchmark-térben."),
            ("End-to-end generálások", f"{runtime_estimate.rag_generations:,}", "Csak akkor számít, ha az end-to-end RAG be van kapcsolva."),
        ],
        columns=4,
    )
    note_box(
        "Hogyan számoljuk a benchmark időt?",
        f"A becslés a tényleges workload-méretből számol: {runtime_estimate.component_runs} komponensfutás, {runtime_estimate.retrieval_executions} retrieval query és {runtime_estimate.rag_generations} end-to-end generálás. Modell: {runtime_estimate.basis}. Ez továbbra is becslés; az Experiment Registry később historikus futásokkal kalibrálható.",
    )

    if plan.cuda_requested and not plan.cuda_available:
        st.warning("CUDA ki van választva, de a PyTorch runtime-ban nem elérhető. Az embedding/reranker CUDA sorok kimaradnak; becsült GPU eredményt nem generálunk.")
    if "cuda" in vector_devices:
        st.info("A FAISS GPU külön képesség a PyTorch CUDA-tól. Ha a Windows build csak faiss-cpu, a kért CUDA vector backend tényleges CPU fallbackként és külön actual_device mezővel jelenik meg.")
    if llm_provider == "ollama" and plan.rag_configurations > 30:
        st.warning("A valódi Ollama generálás több tucat RAG variánson hosszú lehet. A teljes retrieval/component mátrixhoz Dummy módot, majd célzott Ollama RAG futást használj.")
    if plan.total_configurations > 500:
        st.warning(f"Nagy benchmark tér: {plan.total_configurations:,} konfiguráció. Ez több index/model futást és jelentős futási időt jelent.")

    run_disabled = not all([embedding_modes, embedding_devices, vector_devices, reranker_devices, chunkings, retrieval_modes, rerankers]) or (include_rag and not rag_strategies)
    if st.button("TELJES PIPELINE MÁTRIX FUTTATÁSA", type="primary", width="stretch", disabled=run_disabled):
        registry = ExperimentRegistry(REGISTRY_PATH)
        run_id = None
        started = time.perf_counter()
        progress_bar = st.progress(0.0, text="Mátrix előkészítése...")
        status = st.empty()
        try:
            paths = corpus_paths
            config = {
                "benchmark_type": "pipeline_matrix",
                "preset": preset_name,
                "questions": question_count,
                "embedding_modes": embedding_modes,
                "embedding_devices": embedding_devices,
                "vector_devices": vector_devices,
                "reranker_devices": reranker_devices,
                "chunkings": chunkings,
                "retrieval_modes": retrieval_modes,
                "rerankers": rerankers,
                "rag_strategies": rag_strategies if include_rag else [],
                "llm_provider": llm_provider if include_rag else None,
                "ollama_profile": ollama_profile if include_rag and llm_provider == "ollama" else None,
                "include_components": include_components,
                "component_documents": component_documents,
                "component_chunks": component_chunks,
                "chunk_size": int(st.session_state.get("chunk_size", 700)),
                "overlap": int(st.session_state.get("chunk_overlap", 100)),
                "semantic_threshold": float(st.session_state.get("semantic_threshold", 0.72)),
                "top_k": int(st.session_state.get("top_k", 5)),
                "candidate_count": int(st.session_state.get("candidate_count", 20)),
                "context_budget": int(st.session_state.get("context_budget", 1800)),
                "fusion": st.session_state.get("fusion_mode", "rrf"),
                "rrf_k": int(st.session_state.get("rrf_k", 60)),
                "dense_weight": float(st.session_state.get("dense_weight", 0.5)),
                "context_profile": st.session_state.get("context_profile", "balanced"),
                "prompt_profile": st.session_state.get("prompt_profile", "professional"),
            }
            run_id, config_hash = registry.create_run(
                benchmark_type="pipeline_matrix",
                config=config,
                dataset_path=EVAL_PATH,
                questions=question_count,
                notes=f"Streamlit full pipeline matrix · {preset_name}",
            )

            def update_progress(current: int, total: int, label: str) -> None:
                fraction = min(1.0, current / max(total, 1))
                elapsed = max(0.0, time.perf_counter() - started)
                eta = (elapsed / current) * max(0, total - current) if current > 0 else ((eta_low_s + eta_high_s) / 2)
                progress_bar.progress(fraction, text=label)
                status.markdown(
                    live_run_card(
                        current=current,
                        total=total,
                        label=label,
                        elapsed=_human_seconds(elapsed),
                        eta=_human_seconds(eta),
                        predicted=f"{_human_seconds(eta_low_s)} – {_human_seconds(eta_high_s)}",
                    ),
                    unsafe_allow_html=True,
                )

            result = run_pipeline_matrix(
                paths=paths,
                items=items,
                embedding_modes=embedding_modes,
                embedding_devices=embedding_devices,
                vector_devices=vector_devices,
                reranker_devices=reranker_devices,
                chunkings=chunkings,
                retrieval_modes=retrieval_modes,
                rerankers=rerankers,
                rag_strategies=rag_strategies if include_rag else [],
                llm_provider=llm_provider if include_rag else "dummy",
                ollama_profile=ollama_profile,
                questions=question_count,
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                semantic_threshold=config["semantic_threshold"],
                top_k=config["top_k"],
                candidate_count=config["candidate_count"],
                context_budget=config["context_budget"],
                fusion=config["fusion"],
                rrf_k=config["rrf_k"],
                dense_weight=config["dense_weight"],
                context_profile=config["context_profile"],
                prompt_profile=config["prompt_profile"],
                include_rag=include_rag,
                include_components=include_components,
                component_documents=component_documents,
                component_chunks=component_chunks,
                progress=update_progress,
            )
            registry.add_matrix_results(run_id, result["components"], result_type="pipeline_matrix_component")
            registry.add_matrix_results(run_id, result["retrieval"], result_type="pipeline_matrix_retrieval")
            registry.add_matrix_results(run_id, result["rag"], result_type="pipeline_matrix_rag")
            registry.complete_run(run_id, duration_ms=(time.perf_counter() - started) * 1000)
            payload = {"run_id": run_id, "config_hash": config_hash, "config": config, **result}
            _save(payload)
            st.session_state["pipeline_matrix_result"] = payload
            st.session_state["last_experiment_run_id"] = run_id
            progress_bar.progress(1.0, text="Mátrix kész")
            st.success(f"Pipeline mátrix kész. run_id: {run_id} · config: {config_hash[:12]}")
        except Exception as exc:
            if run_id is not None:
                registry.fail_run(run_id, exc, duration_ms=(time.perf_counter() - started) * 1000)
            st.error(f"A pipeline mátrix futása sikertelen: {exc}")

    payload = st.session_state.get("pipeline_matrix_result")
    if not payload and OUTPUT_PATH.exists():
        try:
            payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    if not payload:
        return

    components = payload.get("components", [])
    retrieval = _retrieval_frame(payload.get("retrieval", []))
    rag = _rag_frame(payload.get("rag", []))
    errors = payload.get("errors", [])
    parse_frame, chunk_frame, embedding_frame, vector_frame = _component_frames(components)

    st.divider()
    section_intro(
        "Mátrix eredmények",
        "Öt nézet különíti el a komponensprofilozást, retrieval qualityt, end-to-end generálást, CPU/CUDA trade-offokat és a kihagyott konfigurációkat.",
    )

    component_tab, retrieval_tab, rag_tab, device_tab, errors_tab = st.tabs(
        ["Komponens profiling", "Visszakeresési minőség", "End-to-end RAG", "CPU/CUDA és latency", "Hibák / fallback"]
    )

    with component_tab:
        if not parse_frame.empty:
            st.markdown("#### Parsing + cleaning")
            safe_dataframe(parse_frame, width="stretch", hide_index=True)
        if not chunk_frame.empty:
            st.markdown("#### Chunking benchmark")
            safe_dataframe(chunk_frame, width="stretch", hide_index=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                metric_bar_chart(chunk_frame, x="chunking", y="total_ms", key="matrix_chunk_latency", title="Chunking teljes latency")
            with ch2:
                metric_bar_chart(chunk_frame, x="chunking", y="documents_per_second", key="matrix_chunk_throughput", title="Chunking throughput · dokumentum/s")
            ch3, ch4 = st.columns(2)
            with ch3:
                metric_bar_chart(chunk_frame, x="chunking", y="chunks_per_document", key="matrix_chunks_per_doc", title="Chunkok száma dokumentumonként")
            with ch4:
                percentile_latency_chart(
                    chunk_frame.rename(columns={"mean_chunk_chars": "Mean", "median_chunk_chars": "Median", "p95_chunk_chars": "P95"}),
                    category="chunking",
                    percentiles=["Mean", "Median", "P95"],
                    key="matrix_chunk_size_distribution",
                    title="Chunk méreteloszlás · karakter",
                    value_label="Chunk méret · karakter",
                )
        if not embedding_frame.empty:
            st.markdown("#### Beágyazási benchmark")
            safe_dataframe(embedding_frame, width="stretch", hide_index=True)
            e1, e2 = st.columns(2)
            with e1:
                device_metric_bars(
                    embedding_frame,
                    category="embedding_mode",
                    metric="total_ms",
                    device="actual_device",
                    key="matrix_embedding_batch_latency",
                    title="Beágyazási batch késleltetés · CPU/CUDA",
                )
            with e2:
                device_metric_bars(
                    embedding_frame,
                    category="embedding_mode",
                    metric="texts_per_second",
                    device="actual_device",
                    key="matrix_embedding_throughput",
                    title="Beágyazási áteresztőképesség · szöveg/s",
                )
            metric_bar_chart(
                embedding_frame,
                x="embedding_mode",
                y="model_load_ms",
                color="actual_device",
                key="matrix_embedding_load",
                title="Beágyazási modell hidegindítási késleltetése",
            )
        if not vector_frame.empty:
            st.markdown("#### Vector index + search benchmark")
            safe_dataframe(vector_frame, width="stretch", hide_index=True)
            v1, v2 = st.columns(2)
            with v1:
                device_metric_bars(
                    vector_frame,
                    category="embedding_mode",
                    metric="index_build_ms",
                    device="actual_device",
                    key="matrix_vector_build",
                    title="Vector index build latency",
                )
            with v2:
                device_metric_bars(
                    vector_frame,
                    category="embedding_mode",
                    metric="queries_per_second",
                    device="actual_device",
                    key="matrix_vector_qps",
                    title="Vector search throughput · QPS",
                )
            percentile_latency_chart(
                vector_frame,
                category="variant",
                percentiles=["mean_search_ms", "p50_search_ms", "p95_search_ms", "p99_search_ms"],
                key="matrix_vector_percentiles",
                title="Vector search latency percentilisek",
            )

    with retrieval_tab:
        if retrieval.empty:
            st.info("Nincs retrieval eredmény.")
        else:
            retrieval = retrieval.copy()
            retrieval["overall_score"] = _retrieval_overall_score(retrieval)
            retrieval["efficiency_score"] = (0.75 * retrieval["overall_score"] + 0.25 * _scale_inverse(retrieval["mean_latency_ms"])).round(3)
            with st.expander("Nyers retrieval eredmények", expanded=False):
                safe_dataframe(retrieval.sort_values(["overall_score", "ndcg_at_k"], ascending=False), width="stretch", hide_index=True)
            render_dataframe_exports(retrieval, stem="pipeline_matrix_retrieval", key_prefix="pipeline_retrieval_export")
            kpi_cards(
                [
                    ("Legjobb retrieval konfiguráció", str(retrieval.sort_values(["overall_score", "ndcg_at_k"], ascending=False).iloc[0]["Konfiguráció"]), "A legsikeresebb setup a súlyozott quality score szerint."),
                    ("Legjobb overall score", f"{retrieval['overall_score'].max():.3f}", "Recall, MRR, nDCG, F1, forrásdiverzitás és label coverage alapján."),
                    ("Legjobb hatékonysági pontszám", f"{retrieval['efficiency_score'].max():.3f}", "Minőség és átlagos késleltetés egyensúlyi mutatója."),
                    ("Leggyorsabb mean latency", f"{retrieval['mean_latency_ms'].min():.1f} ms", "A retrieval futások legalacsonyabb átlagideje."),
                ],
                columns=4,
            )
            ql1, ql2 = st.columns(2)
            with ql1:
                quality_latency_scatter(
                    retrieval,
                    x="mean_latency_ms",
                    y="overall_score",
                    color="retriever",
                    hover=["embedding_mode", "requested_vector_device", "chunking", "retriever", "reranker", "recall_at_k", "mrr", "ndcg_at_k"],
                    size="queries_per_second",
                    key="matrix_quality_latency",
                    title="Visszakeresési minőség–latency tér · overall score vs mean latency",
                )
            with ql2:
                metric_bar_chart(
                    retrieval.sort_values(["overall_score", "ndcg_at_k"], ascending=False).head(15),
                    x="ID",
                    y="overall_score",
                    color="retriever",
                    key="matrix_retrieval_top_bar",
                    title="Top retrieval konfigurációk · összesített minőségi pontszám",
                )
            d1, d2 = st.columns(2)
            with d1:
                radar_metrics_chart(
                    retrieval.sort_values(["overall_score", "ndcg_at_k"], ascending=False),
                    label_col="ID",
                    metrics=["recall_at_k", "precision_at_k", "f1_at_k", "mrr", "ndcg_at_k"],
                    key="matrix_retrieval_radar",
                    title="Top retrieval konfigurációk · minőségi profil",
                    max_series=5,
                )
            with d2:
                metric_bar_chart(
                    retrieval.groupby("retriever", as_index=False)[["overall_score", "ndcg_at_k", "mrr"]].mean().sort_values("overall_score", ascending=False),
                    x="retriever",
                    y="overall_score",
                    color="retriever",
                    key="matrix_retrieval_family_bar",
                    title="Visszakeresési családok átlaga · összesített minőségi pontszám",
                )
            percentile_latency_chart(
                retrieval.sort_values(["overall_score", "ndcg_at_k"], ascending=False).head(15),
                category="ID",
                percentiles=["p50_latency_ms", "p95_latency_ms", "p99_latency_ms"],
                key="matrix_retrieval_percentiles",
                title="Visszakeresési késleltetés · P50 / P95 / P99",
            )
            note_box(
                "Összesített visszakeresési pontszám",
                "A retrieval overall score súlyozott diagnosztikai metrika. A részmetrikák közül a Recall@K, MRR és nDCG@K kapnak nagyobb súlyt, de a forrásdiverzitást és a label coverage-et is figyelembe veszi.",
            )

    with rag_tab:
        if rag.empty:
            st.info("Ebben a futásban nem volt end-to-end RAG generálás.")
        else:
            rag = rag.copy()
            rag["overall_score"] = _rag_overall_score(rag)
            rag["efficiency_score"] = (0.75 * rag["overall_score"] + 0.25 * _scale_inverse(rag["mean_total_latency_ms"])).round(3)
            with st.expander("Nyers end-to-end RAG eredmények", expanded=False):
                safe_dataframe(rag.sort_values(["overall_score", "key_fact_coverage"], ascending=False), width="stretch", hide_index=True)
            render_dataframe_exports(rag, stem="pipeline_matrix_rag", key_prefix="pipeline_rag_export")
            kpi_cards(
                [
                    ("Legjobb end-to-end setup", str(rag.sort_values(["overall_score", "key_fact_coverage"], ascending=False).iloc[0]["Konfiguráció"]), "A legsikeresebb RAG konfiguráció a fő minőségi mutatók alapján."),
                    ("Legjobb overall score", f"{rag['overall_score'].max():.3f}", "Hivatkozási pontosság, key-fact coverage és context utilization kombinációja."),
                    ("Legjobb hatékonysági pontszám", f"{rag['efficiency_score'].max():.3f}", "Minőség és teljes késleltetés egyensúlyi mutatója."),
                    ("Legjobb TTFT", f"{rag['mean_ttft_ms'].min():.0f} ms", "A leggyorsabb első token átlag a futott RAG variánsok között."),
                ],
                columns=4,
            )
            r1, r2 = st.columns(2)
            with r1:
                quality_latency_scatter(
                    rag,
                    x="mean_total_latency_ms",
                    y="overall_score",
                    color="rag_strategy",
                    hover=["embedding_mode", "requested_embedding_device", "requested_vector_device", "chunking", "citation_accuracy", "citation_coverage", "mean_ttft_ms", "key_fact_coverage"],
                    size="mean_tokens_per_second",
                    key="matrix_rag_quality_latency",
                    title="RAG quality–latency · overall score vs total latency",
                )
            with r2:
                metric_bar_chart(
                    rag.sort_values(["overall_score", "key_fact_coverage"], ascending=False).head(12),
                    x="ID",
                    y="overall_score",
                    color="rag_strategy",
                    key="matrix_rag_top_bar",
                    title="Top end-to-end konfigurációk · összesített minőségi pontszám",
                )
            r3, r4 = st.columns(2)
            with r3:
                radar_metrics_chart(
                    rag.sort_values(["overall_score", "key_fact_coverage"], ascending=False),
                    label_col="ID",
                    metrics=["citation_accuracy", "citation_coverage", "citation_source_coverage", "key_fact_coverage", "context_utilization"],
                    key="matrix_rag_radar",
                    title="RAG quality profil · top konfigurációk",
                    max_series=5,
                )
            with r4:
                metric_bar_chart(
                    rag.groupby("rag_strategy", as_index=False)[["overall_score", "mean_total_latency_ms", "mean_tokens_per_second"]].mean().sort_values("overall_score", ascending=False),
                    x="rag_strategy",
                    y="overall_score",
                    color="rag_strategy",
                    key="matrix_rag_family_bar",
                    title="RAG stratégiacsaládok átlaga · összesített minőségi pontszám",
                )
            stacked_latency_chart(
                rag.sort_values(["overall_score", "key_fact_coverage"], ascending=False).head(20),
                strategy_col="ID",
                stage_columns=["mean_retrieval_latency_ms", "mean_reranking_latency_ms", "mean_generation_latency_ms"],
                key="matrix_rag_stage_latency",
                title="End-to-end latency bontás · retrieval / reranking / generation",
            )
            percentile_latency_chart(
                rag.sort_values(["overall_score", "key_fact_coverage"], ascending=False).head(20),
                category="ID",
                percentiles=["p50_total_latency_ms", "p95_total_latency_ms", "p99_total_latency_ms"],
                key="matrix_rag_percentiles",
                title="RAG total latency · P50 / P95 / P99",
            )
            note_box(
                "Összesített RAG pontszám",
                "A RAG overall score súlyozott végpont: citation accuracy, citation coverage, citation source coverage, key-fact coverage és context utilization. Összegző nézet, de a részmetrikák továbbra is elsődlegesek maradnak.",
            )

    with device_tab:
        if not embedding_frame.empty:
            st.markdown("#### Beágyazás CPU/CUDA")
            embedding_device_summary = embedding_frame.groupby(["embedding_mode", "actual_device"], as_index=False).agg(
                total_ms=("total_ms", "mean"),
                texts_per_second=("texts_per_second", "mean"),
                model_load_ms=("model_load_ms", "mean"),
                ram_mb=("ram_mb", "mean"),
            )
            safe_dataframe(embedding_device_summary, width="stretch", hide_index=True)
        if not retrieval.empty:
            st.markdown("#### Visszakeresési eszközösszesítés")
            device_summary = retrieval.groupby(["requested_embedding_device", "vector_device"], as_index=False).agg(
                mean_latency_ms=("mean_latency_ms", "mean"),
                p95_latency_ms=("p95_latency_ms", "mean"),
                queries_per_second=("queries_per_second", "mean"),
                recall_at_k=("recall_at_k", "mean"),
                ndcg_at_k=("ndcg_at_k", "mean"),
            )
            safe_dataframe(device_summary, width="stretch", hide_index=True)
            d1, d2 = st.columns(2)
            with d1:
                device_metric_bars(
                    retrieval.groupby(["embedding_mode", "requested_embedding_device"], as_index=False)["mean_latency_ms"].mean(),
                    category="embedding_mode",
                    metric="mean_latency_ms",
                    device="requested_embedding_device",
                    key="matrix_device_latency",
                    title="Beágyazás CPU/CUDA · retrieval mean latency",
                )
            with d2:
                device_metric_bars(
                    retrieval.groupby(["embedding_mode", "requested_embedding_device"], as_index=False)["queries_per_second"].mean(),
                    category="embedding_mode",
                    metric="queries_per_second",
                    device="requested_embedding_device",
                    key="matrix_device_qps",
                    title="Beágyazás CPU/CUDA · retrieval QPS",
                )
        if not rag.empty:
            st.markdown("#### End-to-end RAG device összesítés")
            rag_device = rag.groupby(["requested_embedding_device", "requested_vector_device"], as_index=False).agg(
                total_latency_ms=("mean_total_latency_ms", "mean"),
                ttft_ms=("mean_ttft_ms", "mean"),
                tokens_per_second=("mean_tokens_per_second", "mean"),
                key_fact_coverage=("key_fact_coverage", "mean"),
                citation_accuracy=("citation_accuracy", "mean"),
            )
            safe_dataframe(rag_device, width="stretch", hide_index=True)
        note_box(
            "CPU/CUDA értelmezés",
            "A requested és actual device külön mező. Így a CUDA fallback nem jelenik meg hamis GPU eredményként. Windows alatt a faiss-cpu a stabil baseline; a PyTorch CUDA ettől függetlenül gyorsíthatja az embeddinget és a Cross-Encodert.",
        )

    with errors_tab:
        if errors:
            st.warning(f"{len(errors)} konfiguráció/blokk hibával, fallbackkel vagy runtime-korlát miatt kimaradt.")
            safe_dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
        else:
            st.success("A kiválasztott benchmark tér minden blokkja lefutott hiba nélkül.")
        if not vector_frame.empty and "fallback_used" in vector_frame:
            fallbacks = vector_frame[vector_frame["fallback_used"].astype(bool)]
            if not fallbacks.empty:
                st.info("Vector backend fallbackok: a kért device nem volt ténylegesen elérhető, ezért a rendszer hordozható backendre váltott.")
                safe_dataframe(fallbacks, width="stretch", hide_index=True)
        st.caption(f"Utolsó export: {OUTPUT_PATH}")
