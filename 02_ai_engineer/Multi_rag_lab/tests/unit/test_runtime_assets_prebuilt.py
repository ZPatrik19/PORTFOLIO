from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.ingestion.cleaning import CleaningStats
from rag_engine.ingestion.pipeline import IngestionResult
from rag_engine.models import Chunk, ChunkingConfig
from rag_engine.service import build_retrieval_resources


def _chunk() -> Chunk:
    return Chunk(chunk_id="c1", document_id="d1", text="magas vérnyomás", metadata={"source": "x"})


def test_build_retrieval_resources_reuses_prebuilt_index_without_document_reembedding(monkeypatch) -> None:
    class FakeEmbedder:
        model_name = "intfloat/multilingual-e5-small"
        device = "cpu"

        def embed_documents(self, texts):  # pragma: no cover - must not run
            raise AssertionError("prebuilt index mellett nem szabad újra embeddingelni a dokumentumokat")

        def embed_query(self, query):
            return [1.0, 0.0]

    class FakeStore:
        backend_name = "faiss"
        device = "cpu"

        def search(self, query, top_k=5):
            return []

    ingestion = IngestionResult(
        documents=[],
        chunks=[_chunk()],
        cleaning=CleaningStats(1, 1, 0, 0, 0, 0, 0),
    )
    monkeypatch.setattr("rag_engine.service.create_embedding_provider", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr("rag_engine.service.load_prebuilt_medical", lambda *a, **k: (ingestion, FakeStore()))

    resources = build_retrieval_resources(
        [Path("dummy.html")],
        chunking=ChunkingConfig(strategy="recursive", chunk_size=500, chunk_overlap=100),
        embedding_model="intfloat/multilingual-e5-small",
    )
    assert resources.build_trace["prebuilt_index_used"] is True
    assert resources.build_trace["document_embedding_ms"] == 0.0
    assert resources.build_trace["index_build_ms"] == 0.0


def test_model_cache_resolves_local_snapshot(monkeypatch, tmp_path: Path) -> None:
    import rag_engine.platform.model_assets as assets

    root = tmp_path
    snapshot = root / ".cache" / "huggingface" / "hub" / "models--demo" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    state = root / "artifacts" / "models" / "runtime_models.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "models": {
                    "embedding_e5": {
                        "repo_id": "demo/model",
                        "snapshot_path": str(snapshot.relative_to(root)),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(assets, "ROOT", root)
    monkeypatch.setattr(assets, "STATE_PATH", state)
    assert assets.resolve_cached_model("demo/model") == str(snapshot)
    assert assets.resolve_cached_model("other/model") == "other/model"


def test_setup_and_infrastructure_prepare_runtime_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    setup = (root / "SETUP.bat").read_text(encoding="utf-8")
    infra = (root / "scripts" / "infrastructure_cli.py").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "prepare_runtime_assets.py --retries 5 --verify" in setup
    assert "ensure_runtime_models" in infra
    assert "ensure_medical_index(force=False)" in infra
    assert "prepare_runtime_assets.py --retries 5 --verify" in dockerfile
    assert "prepare_medical_corpus.py" in dockerfile
