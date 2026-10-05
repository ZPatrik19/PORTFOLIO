from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine.platform.runtime import cuda_status, faiss_status, ollama_health
from rag_engine.platform.registry import ExperimentRegistry
from ui.components.common import ROOT, active_paths
from ui.components.education import (
    info_cards,
    kpi_cards,
    note_box,
    page_intro,
    section_intro,
    status_cards,
)
from ui.components.runtime_status import runtime_snapshot
from ui.components.tables import safe_dataframe

EVAL_PATH = ROOT / "artifacts" / "evaluations" / "medical_rag_eval.jsonl"
REGISTRY_PATH = ROOT / "artifacts" / "experiments" / "experiments.sqlite3"
INDEX_STATE = ROOT / "artifacts" / "indexes" / "hungarian_medical" / "index_state.json"


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def _index_chunk_count() -> int | None:
    if not INDEX_STATE.exists():
        return None
    try:
        payload = json.loads(INDEX_STATE.read_text(encoding="utf-8"))
        for key in ("chunks", "chunk_count", "vectors", "vector_count"):
            value = payload.get(key)
            if isinstance(value, int):
                return value
    except Exception:
        return None
    return None


def _registry_snapshot() -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        registry = ExperimentRegistry(REGISTRY_PATH)
        return registry.summary(), registry.list_runs(limit=5)
    except Exception:
        return {"runs": 0, "completed": 0, "failed": 0, "results": 0, "unique_configs": 0}, []


def render() -> None:
    snapshot = runtime_snapshot()
    paths = active_paths()
    medical_docs = [path for path in paths if "hungarian_medical" in str(path)]
    eval_count = _line_count(EVAL_PATH)
    indexed_chunks = _index_chunk_count()
    registry_summary, recent_runs = _registry_snapshot()
    cuda = cuda_status()
    faiss = faiss_status()
    ollama = ollama_health()

    page_intro(
        "Multi-RAG Engineering Dashboard",
        "A projekt operatív kezdőképernyője: korpusz, runtime, aktív pipeline és benchmark állapot egy nézetben. A részletes laborok a Build, Run és Evaluate területeken érhetők el.",
        eyebrow="EXECUTIVE OVERVIEW · BUILD → RUN → EVALUATE",
    )

    kpi_cards(
        [
            ("Aktív dokumentum", str(len(paths)), f"Orvosi korpusz: {len(medical_docs)} fájl"),
            ("Indexelt chunk", f"{indexed_chunks:,}" if indexed_chunks is not None else "—", "Perzisztens medical index állapota."),
            ("Evaluation dataset", str(eval_count), "Forrásolt magyar orvosi kérdések."),
            ("Experiment run", str(registry_summary.get("runs", 0)), f"Completed: {registry_summary.get('completed', 0)} · Failed: {registry_summary.get('failed', 0)}"),
        ],
        columns=4,
    )

    status_cards(
        [
            (
                "PyTorch CUDA",
                "Aktív" if cuda.get("torch_cuda_available") else "CPU fallback",
                f"{cuda.get('gpu_name') or 'GPU nem látható'} · CUDA runtime: {cuda.get('torch_cuda_runtime') or '—'}",
                "ok" if cuda.get("torch_cuda_available") else "warn",
            ),
            (
                "FAISS backend",
                f"FAISS {faiss.get('version') or '—'} · CPU" if faiss.get("import_ok") else "Nem elérhető",
                "Windows faiss-cpu esetén a GPU API hiánya nem hiba.",
                "ok" if faiss.get("import_ok") else "warn",
            ),
            (
                "Ollama / Qwen",
                "Elérhető" if ollama.get("ok") else "Nem elérhető",
                f"Ollama {ollama.get('version') or '—'} · {len(ollama.get('models', []))} lokális modell",
                "ok" if ollama.get("ok") else "warn",
            ),
            (
                "Aktív pipeline",
                f"{snapshot['retrieval']} → {snapshot['rag']}",
                f"Chunking: {snapshot['chunking']} · embedding: {snapshot['embedding_device']} · vector: {snapshot['vector_device']}",
                "info",
            ),
        ],
        columns=2,
    )

    left, right = st.columns([1.25, 1])
    with left:
        section_intro("Aktuális konfiguráció", "A napi munka szempontjából releváns beállítások röviden.")
        info_cards(
            [
                ("Chunking", f"{snapshot['chunking']} · size={snapshot['chunk_size']} · overlap={snapshot['overlap']}"),
                ("Beágyazás + FAISS", f"{snapshot['embedding']} · {snapshot['embedding_device']} / {snapshot['vector_device']}"),
                ("Visszakeresés", f"{snapshot['retrieval']} · {snapshot['fusion']} · Top-K={snapshot['top_k']}"),
                ("RAG", f"{snapshot['rag']} · reranker={snapshot['reranker']} · context={snapshot['context_budget']}"),
            ],
            columns=2,
        )

    with right:
        section_intro("Legutóbbi benchmarkok", "Az Experiment Registry legfrissebb futásai.")
        if recent_runs:
            frame = pd.DataFrame(
                [
                    {
                        "Futás": str(run.get("run_id", ""))[-17:],
                        "Típus": run.get("benchmark_type"),
                        "Állapot": run.get("status"),
                        "Kérdés": run.get("questions") or "—",
                        "Idő": f"{float(run.get('duration_ms') or 0) / 1000:.1f} s" if run.get("duration_ms") else "—",
                    }
                    for run in recent_runs
                ]
            )
            safe_dataframe(frame, width="stretch", hide_index=True)
        else:
            note_box("Még nincs experiment run", "Indíts egy retrieval, RAG vagy Full Pipeline benchmarkot az Evaluate területen; az eredmény automatikusan bekerül az Experiment Registry-be.")

    section_intro("Munkafolyamat", "A projektet három fő feladat köré szerveztük; az infrastruktúra külön System nézetben marad.")
    info_cards(
        [
            ("BUILD", "Dokumentumok → chunking → embedding → retrieval. Itt építed és diagnosztizálod az indexelési/retrieval réteget."),
            ("FUTTATÁS", "RAG játszótér és RAG összehasonlítás. Egyedi kérdések, bizonyítékok, kontextus és lokális Qwen-generálás."),
            ("KIÉRTÉKELÉS", "Teljesítmény, forrásolt kiértékelés, teljes pipeline benchmark és Experiment Registry."),
        ],
        columns=3,
    )

    with st.expander("Módszertani megjegyzések", expanded=False):
        st.markdown(
            """
- A retrieval quality és a generálási quality külön réteg: egy gyors vagy magas Recall@K értékű retriever önmagában még nem garantál jó végső választ.
- Az `overall score` csak összegző diagnosztikai mutató; a részmetrikák (Recall, MRR, nDCG, citation, key-fact coverage, latency) továbbra is elsődlegesek.
- A magyar orvosi korpusz oktatási és mérési célú. A generált válasz nem orvosi diagnózis vagy kezelési javaslat.
"""
        )
