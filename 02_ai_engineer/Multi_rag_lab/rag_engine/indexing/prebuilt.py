from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_engine.ingestion.cleaning import CleaningStats
from rag_engine.ingestion.pipeline import IngestionResult
from rag_engine.models import Chunk, ChunkingConfig
from rag_engine.indexing.faiss_cpu import FaissCPUVectorStore
from rag_engine.indexing.faiss_gpu import FaissGPUVectorStore

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "artifacts" / "indexes" / "hungarian_medical" / "index_state.json"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _all_medical_paths(paths: list[Path]) -> bool:
    medical_root = (ROOT / "data" / "raw" / "hungarian_medical").resolve()
    if not paths:
        return False
    for path in paths:
        try:
            path.resolve().relative_to(medical_root)
        except ValueError:
            return False
    return True


def prebuilt_medical_compatible(
    paths: list[Path],
    chunking: ChunkingConfig,
    embedding_model: str,
    *,
    state_path: Path = DEFAULT_STATE,
) -> tuple[bool, dict]:
    if not state_path.exists() or not _all_medical_paths(paths):
        return False, {}
    state = _read_json(state_path)
    config = state.get("config", {})
    if not isinstance(config, dict):
        return False, state
    expected = {
        "strategy": chunking.strategy,
        "chunk_size": int(chunking.chunk_size),
        "overlap": int(chunking.chunk_overlap),
        "embedding_model": embedding_model,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            return False, state
    if int(state.get("documents", -1)) != len(paths):
        return False, state
    source_files = state.get("source_files")
    if isinstance(source_files, list) and source_files:
        expected_files = {
            (path.name, int(path.stat().st_size), int(path.stat().st_mtime_ns))
            for path in paths
        }
        stored_files = {
            (str(item.get("name", "")), int(item.get("size", -1)), int(item.get("mtime_ns", -1)))
            for item in source_files
            if isinstance(item, dict)
        }
        if expected_files != stored_files:
            return False, state
    chunks_path = _resolve_project_path(str(state.get("chunks_path", "")))
    vector_path = _resolve_project_path(str(state.get("vectorstore_path", "")))
    if not chunks_path.exists() or not (vector_path / "index.faiss").exists() or not (vector_path / "metadata.json").exists():
        return False, state
    return True, state


def load_prebuilt_medical(
    paths: list[Path],
    chunking: ChunkingConfig,
    embedding_model: str,
    *,
    vector_device: str = "cpu",
    state_path: Path = DEFAULT_STATE,
) -> tuple[IngestionResult, object] | None:
    compatible, state = prebuilt_medical_compatible(paths, chunking, embedding_model, state_path=state_path)
    if not compatible:
        return None
    chunks_path = _resolve_project_path(str(state["chunks_path"]))
    vector_path = _resolve_project_path(str(state["vectorstore_path"]))
    chunks = [Chunk.model_validate(item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
    normalized_device = str(vector_device).lower()
    if normalized_device in {"cuda", "gpu", "faiss-gpu"}:
        try:
            store = FaissGPUVectorStore.load(vector_path)
        except Exception as exc:
            LOGGER.warning("A perzisztens FAISS index GPU-ra klónozása sikertelen (%s); CPU index betöltése.", exc)
            store = FaissCPUVectorStore.load(vector_path)
    else:
        store = FaissCPUVectorStore.load(vector_path)
    stats = CleaningStats(
        documents_before=int(state.get("documents", len(paths))),
        documents_after=int(state.get("documents", len(paths))),
        characters_before=0,
        characters_after=0,
        duplicates_removed=0,
        empty_sections_removed=0,
        repeated_edge_lines_removed=0,
    )
    return IngestionResult(documents=[], chunks=chunks, cleaning=stats), store
