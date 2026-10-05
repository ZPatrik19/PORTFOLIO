from __future__ import annotations

from pathlib import Path

from rag_engine.platform.registry import ExperimentRegistry, canonical_config_hash


def test_config_hash_is_order_independent() -> None:
    assert canonical_config_hash({"b": 2, "a": 1}) == canonical_config_hash({"a": 1, "b": 2})


def test_registry_stores_run_and_retrieval_results(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text('{"id":"q1"}\n', encoding="utf-8")
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    run_id, config_hash = registry.create_run(
        benchmark_type="retrieval",
        config={"chunking": ["recursive"], "retrieval": ["hybrid"]},
        dataset_path=dataset,
        questions=1,
    )
    registry.add_retrieval_results(
        run_id,
        [
            {
                "chunking": "recursive",
                "retriever": "hybrid",
                "reranker": "lexical",
                "questions": 1,
                "recall_at_k": 1.0,
                "precision_at_k": 0.2,
                "hit_rate_at_k": 1.0,
                "mrr": 1.0,
                "ndcg_at_k": 1.0,
                "mean_latency_ms": 12.5,
                "p95_latency_ms": 12.5,
                "mean_relevant_chunks": 1.0,
                "labeling_coverage": 1.0,
                "embedding_device": "cpu",
                "vector_backend": "faiss-cpu",
                "vector_device": "cpu",
            }
        ],
        embedding_model="test-model",
    )
    registry.complete_run(run_id, duration_ms=20.0)

    run = registry.get_run(run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["config_hash"] == config_hash
    results = registry.get_results(run_id)
    assert len(results) == 1
    assert results[0]["variant_key"] == "recursive|hybrid|lexical"
    assert results[0]["recall_at_k"] == 1.0
    assert results[0]["embedding_model"] == "test-model"


def test_registry_stores_rag_metrics_and_exports(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    run_id, _ = registry.create_run(
        benchmark_type="rag",
        config={"rag_strategies": ["hybrid"]},
        questions=2,
    )
    registry.add_rag_results(
        run_id,
        [
            {
                "rag_strategy": "hybrid",
                "questions": 2,
                "citation_accuracy": 0.9,
                "key_fact_coverage": 0.7,
                "context_utilization": 0.8,
                "mean_total_latency_ms": 125.0,
                "p95_total_latency_ms": 140.0,
                "mean_context_tokens": 900.0,
                "llm_provider": "dummy",
                "execution_device": "cpu",
            }
        ],
        chunking="recursive",
        embedding_model="test-model",
        context_budget=1800,
        vector_device="cpu",
    )
    registry.complete_run(run_id)

    result = registry.get_results(run_id)[0]
    assert result["rag_strategy"] == "hybrid"
    assert result["key_fact_coverage"] == 0.7
    export = registry.export_json(tmp_path / "export.json")
    assert export.exists()
    assert run_id in export.read_text(encoding="utf-8")


def test_registry_stores_performance_results(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    run_id, _ = registry.create_run(benchmark_type="performance", config={"device": "cpu"})
    registry.add_performance_results(
        run_id,
        [
            {
                "component": "embedding",
                "device": "cpu",
                "total_ms": 100.0,
                "throughput_per_sec": 500.0,
                "mean_ms": 2.0,
                "median_ms": 1.8,
                "p95_ms": 3.0,
                "ram_mb": 256.0,
                "peak_gpu_memory_mb": None,
                "workload_size": 50,
                "cold_start_ms": 25.0,
            }
        ],
        chunking="recursive",
        embedding_model="test-model",
        vector_device="cpu",
    )
    registry.complete_run(run_id)
    result = registry.get_results(run_id)[0]
    assert result["result_type"] == "performance"
    assert result["variant_key"] == "embedding|cpu"
    assert result["throughput_per_sec"] == 500.0
    assert result["cold_start_ms"] == 25.0
