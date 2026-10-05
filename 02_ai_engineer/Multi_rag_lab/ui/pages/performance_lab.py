from __future__ import annotations

import time

import streamlit as st

from rag_engine.evaluation.device_benchmark import (
    benchmark_embedding,
    benchmark_llm,
    benchmark_metadata,
    benchmark_pipeline,
    benchmark_reranking,
    benchmark_retrieval,
)
from rag_engine.platform.config import load_settings
from rag_engine.presets import HUNGARIAN_QUERY_PRESETS
from rag_engine.indexing.hashing import HashingEmbeddingProvider
from rag_engine.indexing.sentence_transformer import SentenceTransformerEmbeddingProvider
from rag_engine.platform.experiments import ExperimentStore
from rag_engine.platform.registry import ExperimentRegistry
from rag_engine.retrieval.rerank_lexical import LexicalReranker
from rag_engine.platform.device import cuda_available
from rag_engine.service import create_rag_pipeline
from ui.components.common import ROOT, cached_cross_encoder, get_lab
from ui.components.education import info_cards, note_box, page_intro
from ui.components.performance import render_performance_dashboard
from ui.components.reranker import build_selected_reranker, reranker_status_text


def _tag_component(record: dict, name: str) -> dict:
    return {**record, "component": name}


def render() -> None:
    page_intro(
        "Teljesítménylabor · teljes RAG folyamat",
        "Nem csak az újrarangsorolót mérjük: ugyanazon lokális környezetben külön benchmarkolható a beágyazás, a Dense/BM25/Hibrid visszakeresés, az újrarangsorolás, az LLM-generálás és a teljes végponttól végpontig RAG folyamat.",
        eyebrow="BEÁGYAZÁS · RETRIEVAL · ÚJRARANGSOROLÁS · LLM · END-TO-END · CPU/CUDA",
    )
    info_cards(
        [
            ("Komponensprofil", "A folyamat minden fontos szakasza külön mérhető, így megkülönböztethető a visszakeresés, az újrarangsorolás és az LLM valódi szűk keresztmetszete."),
            ("CPU / CUDA", "A beágyazás és Cross-Encoder tényleges device-a bekerül az eredménybe; a FAISS CPU backend ettől függetlenül működik."),
            ("Késleltetés + áteresztőképesség", "Átlag, medián, P95, teljes idő és áteresztőképesség együtt kerül kiértékelésre."),
            ("Reprodukálhatóság", "Minden benchmark bekerül az Experiment Registry-be konfigurációval és hardver-metadata mellett."),
        ],
        columns=4,
    )

    workload = st.selectbox("Terhelési profil", ["KICSI", "KÖZEPES", "NAGY"], index=0)
    max_items = {"KICSI": 50, "KÖZEPES": 500, "NAGY": 2000}[workload]
    repeat_queries = st.slider("Visszakeresési lekérdezések száma", 4, 40, 12, step=4)

    with st.expander("Benchmark komponensek", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            do_embedding = st.checkbox("Beágyazás", value=True)
            do_retrieval = st.checkbox("Dense + BM25 + Hibrid visszakeresés", value=True)
        with c2:
            do_reranking = st.checkbox("Újrarangsorolás", value=True)
            do_llm = st.checkbox("LLM-generálás", value=False, help="Ollama esetén valódi lokális generálás, ezért lassabb lehet.")
        with c3:
            do_e2e = st.checkbox("Teljes RAG folyamat", value=False)
            cross_encoder_extra = st.checkbox("Cross-Encoder külön mérés", value=st.session_state.get("reranker_mode") == "cross-encoder")

    global_reranker, reranker_status = build_selected_reranker("reranked", allow_inactive=True)
    cuda_ok = cuda_available()
    st.caption(f"Aktív újrarangsoroló: **{reranker_status_text(reranker_status)}** · {reranker_status.reason}")
    st.caption(f"Benchmark maximum: **{max_items} chunk** · PyTorch CUDA: **{'elérhető' if cuda_ok else 'nem elérhető'}**")
    if cuda_ok and reranker_status.applied == "lexical":
        st.caption("A lexikális újrarangsorolás CPU-algoritmus. CUDA-s rerankinghoz válaszd a Cross-Encoder újrarangsorolót.")

    if st.button("Teljesítménybenchmark futtatása", type="primary"):
        registry = ExperimentRegistry(ROOT / "artifacts" / "experiments" / "experiments.sqlite3")
        run_id = None
        started = time.perf_counter()
        try:
            hashing = st.session_state.get("embedding_mode") == "hashing"
            devices = ["cpu"] + (["cuda"] if cuda_available() and not hashing else [])
            lab = get_lab(embedding_device="cpu")
            texts = [c.text for c in lab.ingestion.chunks if c.metadata.get("role") != "parent"][:max_items]
            if not texts:
                raise RuntimeError("Nincs benchmarkolható szövegrész az aktív korpuszban.")

            settings = load_settings()
            mode = st.session_state.get("embedding_mode")
            model_name = "hashing" if hashing else (settings.multilingual_embedding_model if mode == "multilingual" else settings.e5_embedding_model if mode == "e5-small" else settings.embedding_model)
            config = {
                "benchmark_type": "performance",
                "workload": workload,
                "max_items": max_items,
                "devices": devices,
                "embedding_model": model_name,
                "chunking": st.session_state.get("chunking_strategy", "recursive"),
                "retrieval_mode": st.session_state.get("retrieval_mode", "hybrid"),
                "reranker": reranker_status.applied,
                "reranker_device": reranker_status.device,
                "components": {"embedding": do_embedding, "retrieval": do_retrieval, "reranking": do_reranking, "llm": do_llm, "e2e": do_e2e},
                "repeat_queries": repeat_queries,
                "top_k": int(st.session_state.get("top_k", 5)),
                "candidate_count": int(st.session_state.get("candidate_count", 20)),
                "context_budget": int(st.session_state.get("context_budget", 1800)),
                "vector_device": str(getattr(lab.vector_store, "device", "cpu")),
                "vector_backend": str(getattr(lab.vector_store, "backend_name", "unknown")),
                "llm_provider": st.session_state.get("llm_provider", "dummy"),
            }
            run_id, config_hash = registry.create_run(benchmark_type="performance", config=config, questions=repeat_queries, notes="Streamlit teljes komponensű teljesítménybenchmark")
            records: list[dict] = []

            if do_embedding:
                for device in devices:
                    factory = (lambda _device: HashingEmbeddingProvider()) if hashing else (lambda d, mn=model_name: SentenceTransformerEmbeddingProvider(mn, device=d))
                    records.append(benchmark_embedding(factory, texts, device).to_dict())

            base_queries = [item.query for item in HUNGARIAN_QUERY_PRESETS]
            queries = (base_queries * ((repeat_queries + len(base_queries) - 1) // len(base_queries)))[:repeat_queries]
            if do_retrieval:
                # BM25 nem használ neurális beágyazást, ezért elegendő egyszer CPU-n mérni.
                records.append(
                    _tag_component(
                        benchmark_retrieval(lab.sparse, queries, device="cpu", top_k=config["top_k"]).to_dict(),
                        "retrieval-bm25",
                    )
                )
                # A dense és hibrid visszakeresésnél a query-beágyazás CPU/CUDA hatása külön mérhető.
                for device in devices:
                    retrieval_lab = get_lab(embedding_device=device)
                    actual_device = str(getattr(retrieval_lab.embedder, "device", device))
                    records.append(
                        _tag_component(
                            benchmark_retrieval(retrieval_lab.dense, queries, device=actual_device, top_k=config["top_k"]).to_dict(),
                            "retrieval-dense",
                        )
                    )
                    records.append(
                        _tag_component(
                            benchmark_retrieval(retrieval_lab.hybrid, queries, device=actual_device, top_k=config["top_k"]).to_dict(),
                            "retrieval-hybrid",
                        )
                    )

            candidates = lab.hybrid.retrieve(queries[0], top_k=min(20, len(lab.ingestion.chunks)), candidate_count=min(20, len(lab.ingestion.chunks)))
            if do_reranking:
                lexical = LexicalReranker()
                records.append(_tag_component(benchmark_reranking(lexical, queries[0], candidates, repeats=5).to_dict(), "reranking-lexical"))
                if cross_encoder_extra:
                    for device in devices:
                        try:
                            cross = cached_cross_encoder(settings.reranker_model, device)
                            records.append(
                                _tag_component(
                                    benchmark_reranking(cross, queries[0], candidates, repeats=5).to_dict(),
                                    "reranking-cross-encoder",
                                )
                            )
                        except Exception as exc:
                            st.warning(f"Cross-Encoder {device} mérés kimaradt: {exc}")

            if do_llm:
                prompts = [f"Válaszolj egy rövid mondatban magyarul: {q}" for q in queries[: min(3, len(queries))]]
                records.append(benchmark_llm(lab.llm, prompts, device="Ollama runtime" if st.session_state.get("llm_provider") == "ollama" else "dummy").to_dict())

            if do_e2e:
                reranker = global_reranker or LexicalReranker()
                pipeline = create_rag_pipeline(
                    lab,
                    "reranked",
                    reranker=reranker,
                    execution_device=str(getattr(lab.embedder, "device", "cpu")),
                    top_k=config["top_k"],
                    candidate_count=config["candidate_count"],
                    max_context_tokens=config["context_budget"],
                    context_profile=st.session_state.get("context_profile", "balanced"),
                    prompt_profile=st.session_state.get("prompt_profile", "professional"),
                )
                records.append(benchmark_pipeline(pipeline, queries[: min(4, len(queries))]).to_dict())

            store = ExperimentStore(ROOT / "artifacts" / "experiments" / "performance.jsonl")
            for record in records:
                store.append({"experiment_type": "performance", "run_id": run_id, "workload": workload, **record})
            registry.add_performance_results(
                run_id,
                records,
                chunking=config["chunking"],
                embedding_model=model_name,
                retrieval_mode=config["retrieval_mode"],
                reranker=reranker_status.applied,
                rag_strategy="reranked" if do_e2e else None,
                context_budget=config["context_budget"],
                vector_backend=config["vector_backend"],
                vector_device=config["vector_device"],
                llm_provider=config["llm_provider"],
            )
            registry.complete_run(run_id, duration_ms=(time.perf_counter() - started) * 1000)
            st.session_state["performance_records"] = records
            st.session_state["performance_metadata"] = {device: benchmark_metadata(device) for device in devices}
            st.session_state["last_experiment_run_id"] = run_id
            st.success(f"Teljesítménybenchmark elkészült. run_id: {run_id} · config: {config_hash[:12]}")
        except Exception as exc:
            if run_id is not None:
                registry.fail_run(run_id, exc, duration_ms=(time.perf_counter() - started) * 1000)
            st.error(f"A teljesítménybenchmark sikertelen: {exc}")
            note_box("Hibakeresés", "A részletes traceback helyett a felület a kiváltó hibát mutatja. Ellenőrizd az Infrastructure oldalon a CUDA, FAISS és Ollama állapotát, majd futtasd újra a kisebb profilt.")

    render_performance_dashboard(
        st.session_state.get("performance_records", []),
        st.session_state.get("performance_metadata", {}),
    )
