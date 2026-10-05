from pathlib import Path

from rag_engine.evaluation.component_benchmark import run_component_benchmarks
from rag_engine.evaluation.pipeline_matrix import build_matrix_plan
from rag_engine.platform.registry import ExperimentRegistry


def test_matrix_plan_counts_all_dimensions(monkeypatch):
    monkeypatch.setattr("rag_engine.evaluation.pipeline_matrix.cuda_available", lambda: True)
    plan = build_matrix_plan(
        embedding_modes=["multilingual", "hashing"],
        embedding_devices=["cpu", "cuda"],
        vector_devices=["cpu", "cuda"],
        reranker_devices=["cpu", "cuda"],
        chunkings=["recursive", "sentence"],
        retrieval_modes=["dense", "hybrid"],
        rerankers=["none", "cross-encoder"],
        rag_strategies=["baseline", "hybrid"],
        include_rag=True,
        include_components=True,
    )
    assert plan.retrieval_configurations == 96
    assert plan.rag_configurations == 32
    assert plan.component_configurations == 15
    assert plan.total_configurations == 143
    assert plan.cuda_requested is True
    assert plan.cuda_available is True


def test_component_benchmark_hashing_smoke(tmp_path: Path):
    doc = tmp_path / "medical.txt"
    doc.write_text(
        "Magas vérnyomás. A rendszeres vérnyomásmérés fontos.\n\n"
        "A megelőzés és az orvosi kontroll szerepe is lényeges. " * 12,
        encoding="utf-8",
    )
    result = run_component_benchmarks(
        paths=[doc],
        chunkings=["fixed", "recursive"],
        embedding_modes=["hashing"],
        embedding_devices=["cpu"],
        vector_devices=["cpu"],
        chunk_size=180,
        overlap=30,
        semantic_threshold=0.72,
        max_documents=5,
        max_embedding_chunks=32,
        vector_query_count=3,
        top_k=2,
    )
    components = result["components"]
    assert any(row["component"] == "parsing-cleaning" for row in components)
    assert sum(row["component"] == "chunking" for row in components) == 2
    embedding = next(row for row in components if row["component"] == "embedding")
    assert embedding["vector_dimension"] == 256
    assert embedding["texts_per_second"] > 0
    vector = next(row for row in components if row["component"] == "vector-search")
    assert vector["queries_per_second"] > 0
    assert vector["p95_search_ms"] >= 0


def test_registry_matrix_identity_keeps_device_variants(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    run_id, _ = registry.create_run(benchmark_type="pipeline_matrix", config={"x": 1})
    rows = [
        {
            "embedding_mode": "hashing",
            "requested_embedding_device": "cpu",
            "requested_vector_device": "cpu",
            "requested_reranker_device": "cpu",
            "chunking": "recursive",
            "retriever": "dense",
            "reranker": "none",
            "mean_latency_ms": 1.0,
        },
        {
            "embedding_mode": "hashing",
            "requested_embedding_device": "cpu",
            "requested_vector_device": "cuda",
            "requested_reranker_device": "cpu",
            "chunking": "recursive",
            "retriever": "dense",
            "reranker": "none",
            "mean_latency_ms": 2.0,
        },
    ]
    registry.add_matrix_results(run_id, rows, result_type="pipeline_matrix_retrieval")
    stored = registry.get_results(run_id)
    assert len(stored) == 2
    assert {row["requested_vector_device"] for row in stored} == {"cpu", "cuda"}
