from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, cast
from uuid import uuid4


SCHEMA_VERSION = 5


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_dict(row: object) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(cast(Any, row))
    raise TypeError(f"Nem támogatott benchmark sor típus: {type(row)!r}")


class ExperimentRegistry:
    """SQLite-alapú benchmark registry.

    Egy benchmark invocation egy ``run`` rekord. A run alatt több retrieval vagy RAG
    variáns eredménye tárolható. A config hash determinisztikus és a teljes normalizált
    konfigurációból készül, ezért két azonos konfiguráció könnyen felismerhető.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    benchmark_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    dataset_path TEXT,
                    dataset_hash TEXT,
                    questions INTEGER,
                    duration_ms REAL,
                    notes TEXT,
                    config_json TEXT NOT NULL,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash);
                CREATE INDEX IF NOT EXISTS idx_runs_benchmark_type ON runs(benchmark_type);

                CREATE TABLE IF NOT EXISTS results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    variant_key TEXT NOT NULL,
                    chunking TEXT,
                    embedding_model TEXT,
                    retrieval_mode TEXT,
                    reranker TEXT,
                    rag_strategy TEXT,
                    context_budget INTEGER,
                    embedding_device TEXT,
                    vector_backend TEXT,
                    vector_device TEXT,
                    llm_provider TEXT,
                    questions INTEGER,
                    mean_latency_ms REAL,
                    p95_latency_ms REAL,
                    recall_at_k REAL,
                    precision_at_k REAL,
                    hit_rate_at_k REAL,
                    mrr REAL,
                    ndcg_at_k REAL,
                    citation_accuracy REAL,
                    key_fact_coverage REAL,
                    context_utilization REAL,
                    mean_context_tokens REAL,
                    labeling_coverage REAL,
                    mean_relevant_chunks REAL,
                    total_ms REAL,
                    median_latency_ms REAL,
                    throughput_per_sec REAL,
                    ram_mb REAL,
                    peak_gpu_memory_mb REAL,
                    cold_start_ms REAL,
                    workload_size INTEGER,
                    extra_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, result_type, variant_key)
                );

                CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
                CREATE INDEX IF NOT EXISTS idx_results_type ON results(result_type);
                """
            )
            self._ensure_result_columns(connection)
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _ensure_result_columns(connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(results)").fetchall()}
        migrations = {
            "total_ms": "REAL",
            "median_latency_ms": "REAL",
            "throughput_per_sec": "REAL",
            "ram_mb": "REAL",
            "peak_gpu_memory_mb": "REAL",
            "cold_start_ms": "REAL",
            "workload_size": "INTEGER",
            "p50_latency_ms": "REAL",
            "p99_latency_ms": "REAL",
            "f1_at_k": "REAL",
            "map_at_k": "REAL",
            "mean_first_relevant_rank": "REAL",
            "source_diversity_at_k": "REAL",
            "duplicate_ratio_at_k": "REAL",
            "queries_per_second": "REAL",
            "citation_coverage": "REAL",
            "citation_source_coverage": "REAL",
            "citation_density": "REAL",
            "mean_retrieval_latency_ms": "REAL",
            "mean_reranking_latency_ms": "REAL",
            "mean_generation_latency_ms": "REAL",
            "mean_ttft_ms": "REAL",
            "mean_tokens_per_second": "REAL",
            "mean_answer_tokens": "REAL",
            "answer_redundancy": "REAL",
            "component": "TEXT",
            "variant": "TEXT",
            "requested_embedding_device": "TEXT",
            "requested_vector_device": "TEXT",
            "requested_reranker_device": "TEXT",
            "reciprocal_rank_at_k": "REAL",
            "r_precision": "REAL",
            "context_precision_at_k": "REAL",
            "recall_ci_low": "REAL",
            "recall_ci_high": "REAL",
            "ndcg_ci_low": "REAL",
            "ndcg_ci_high": "REAL",
            "no_hit_rate": "REAL",
            "late_hit_rate": "REAL",
            "latency_cv": "REAL",
            "mean_latency_ci_low_ms": "REAL",
            "mean_latency_ci_high_ms": "REAL",
            "key_fact_ci_low": "REAL",
            "key_fact_ci_high": "REAL",
            "fallback_rate": "REAL",
            "repair_rate": "REAL",
            "mean_total_latency_ci_low_ms": "REAL",
            "mean_total_latency_ci_high_ms": "REAL",
        }
        for column, sql_type in migrations.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE results ADD COLUMN {column} {sql_type}")

    def create_run(
        self,
        *,
        benchmark_type: str,
        config: dict[str, Any],
        dataset_path: Path | None = None,
        questions: int | None = None,
        notes: str | None = None,
    ) -> tuple[str, str]:
        created_at = datetime.now(UTC).isoformat()
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        normalized_config = dict(config)
        if dataset_path is not None:
            normalized_config.setdefault("dataset_path", str(dataset_path))
            normalized_config.setdefault("dataset_hash", file_sha256(dataset_path))
        config_hash = canonical_config_hash(normalized_config)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, created_at, benchmark_type, status, config_hash,
                    dataset_path, dataset_hash, questions, notes, config_json
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    benchmark_type,
                    config_hash,
                    str(dataset_path) if dataset_path else None,
                    file_sha256(dataset_path) if dataset_path else None,
                    questions,
                    notes,
                    json.dumps(normalized_config, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
        return run_id, config_hash

    def complete_run(self, run_id: str, *, duration_ms: float | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status='completed', completed_at=?, duration_ms=? WHERE run_id=?",
                (datetime.now(UTC).isoformat(), duration_ms, run_id),
            )

    def fail_run(self, run_id: str, error: Exception | str, *, duration_ms: float | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status='failed', completed_at=?, duration_ms=?, error_message=? WHERE run_id=?",
                (datetime.now(UTC).isoformat(), duration_ms, str(error), run_id),
            )

    def add_retrieval_results(
        self,
        run_id: str,
        rows: Iterable[object],
        *,
        embedding_model: str,
        context_budget: int | None = None,
    ) -> None:
        with self._connect() as connection:
            for row_obj in rows:
                row = _as_dict(row_obj)
                variant_key = f"{row.get('chunking')}|{row.get('retriever')}|{row.get('reranker')}"
                self._upsert_result(
                    connection,
                    run_id=run_id,
                    result_type="retrieval",
                    variant_key=variant_key,
                    values={
                        "chunking": row.get("chunking"),
                        "embedding_model": embedding_model,
                        "retrieval_mode": row.get("retriever"),
                        "reranker": row.get("reranker"),
                        "context_budget": context_budget,
                        "embedding_device": row.get("embedding_device"),
                        "vector_backend": row.get("vector_backend"),
                        "vector_device": row.get("vector_device"),
                        "questions": row.get("questions"),
                        "mean_latency_ms": row.get("mean_latency_ms"),
                        "p95_latency_ms": row.get("p95_latency_ms"),
                        "recall_at_k": row.get("recall_at_k"),
                        "precision_at_k": row.get("precision_at_k"),
                        "f1_at_k": row.get("f1_at_k"),
                        "hit_rate_at_k": row.get("hit_rate_at_k"),
                        "mrr": row.get("mrr"),
                        "map_at_k": row.get("map_at_k"),
                        "ndcg_at_k": row.get("ndcg_at_k"),
                        "mean_first_relevant_rank": row.get("mean_first_relevant_rank"),
                        "reciprocal_rank_at_k": row.get("reciprocal_rank_at_k"),
                        "r_precision": row.get("r_precision"),
                        "context_precision_at_k": row.get("context_precision_at_k"),
                        "recall_ci_low": row.get("recall_ci_low"),
                        "recall_ci_high": row.get("recall_ci_high"),
                        "ndcg_ci_low": row.get("ndcg_ci_low"),
                        "ndcg_ci_high": row.get("ndcg_ci_high"),
                        "no_hit_rate": row.get("no_hit_rate"),
                        "late_hit_rate": row.get("late_hit_rate"),
                        "source_diversity_at_k": row.get("source_diversity_at_k"),
                        "duplicate_ratio_at_k": row.get("duplicate_ratio_at_k"),
                        "p50_latency_ms": row.get("p50_latency_ms"),
                        "p99_latency_ms": row.get("p99_latency_ms"),
                        "latency_cv": row.get("latency_cv"),
                        "mean_latency_ci_low_ms": row.get("mean_latency_ci_low_ms"),
                        "mean_latency_ci_high_ms": row.get("mean_latency_ci_high_ms"),
                        "queries_per_second": row.get("queries_per_second"),
                        "labeling_coverage": row.get("labeling_coverage"),
                        "mean_relevant_chunks": row.get("mean_relevant_chunks"),
                        "extra_json": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
                    },
                )

    def add_rag_results(
        self,
        run_id: str,
        rows: Iterable[object],
        *,
        chunking: str,
        embedding_model: str,
        context_budget: int,
        vector_device: str,
    ) -> None:
        with self._connect() as connection:
            for row_obj in rows:
                row = _as_dict(row_obj)
                strategy = str(row.get("rag_strategy"))
                effective_chunking = "parent-child" if strategy == "parent-document" else chunking
                variant_key = strategy
                self._upsert_result(
                    connection,
                    run_id=run_id,
                    result_type="rag",
                    variant_key=variant_key,
                    values={
                        "chunking": effective_chunking,
                        "embedding_model": embedding_model,
                        "rag_strategy": strategy,
                        "context_budget": context_budget,
                        "embedding_device": row.get("execution_device"),
                        "vector_device": vector_device,
                        "llm_provider": row.get("llm_provider"),
                        "questions": row.get("questions"),
                        "mean_latency_ms": row.get("mean_total_latency_ms"),
                        "p50_latency_ms": row.get("p50_total_latency_ms"),
                        "p95_latency_ms": row.get("p95_total_latency_ms"),
                        "p99_latency_ms": row.get("p99_total_latency_ms"),
                        "citation_accuracy": row.get("citation_accuracy"),
                        "citation_coverage": row.get("citation_coverage"),
                        "citation_source_coverage": row.get("citation_source_coverage"),
                        "citation_density": row.get("citation_density_per_100_words"),
                        "key_fact_coverage": row.get("key_fact_coverage"),
                        "key_fact_ci_low": row.get("key_fact_ci_low"),
                        "key_fact_ci_high": row.get("key_fact_ci_high"),
                        "context_utilization": row.get("context_utilization"),
                        "answer_redundancy": row.get("answer_redundancy"),
                        "fallback_rate": row.get("fallback_rate"),
                        "repair_rate": row.get("repair_rate"),
                        "mean_answer_tokens": row.get("mean_answer_tokens"),
                        "mean_retrieval_latency_ms": row.get("mean_retrieval_latency_ms"),
                        "mean_reranking_latency_ms": row.get("mean_reranking_latency_ms"),
                        "mean_generation_latency_ms": row.get("mean_generation_latency_ms"),
                        "mean_ttft_ms": row.get("mean_ttft_ms"),
                        "mean_tokens_per_second": row.get("mean_tokens_per_second"),
                        "mean_context_tokens": row.get("mean_context_tokens"),
                        "latency_cv": row.get("latency_cv"),
                        "mean_total_latency_ci_low_ms": row.get("mean_total_latency_ci_low_ms"),
                        "mean_total_latency_ci_high_ms": row.get("mean_total_latency_ci_high_ms"),
                        "extra_json": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
                    },
                )

    def add_matrix_results(self, run_id: str, rows: Iterable[dict[str, Any]], *, result_type: str) -> None:
        """Store heterogeneous pipeline-matrix rows while preserving their full JSON payload."""
        with self._connect() as connection:
            for index, row in enumerate(rows):
                identity_keys = (
                    "component",
                    "variant",
                    "embedding_mode",
                    "requested_embedding_device",
                    "requested_vector_device",
                    "requested_reranker_device",
                    "chunking",
                    "retriever",
                    "reranker",
                    "rag_strategy",
                )
                identity = "|".join(str(row.get(key) or "-") for key in identity_keys)
                variant_key = str(row.get("variant_key") or identity or f"row-{index}")
                self._upsert_result(
                    connection,
                    run_id=run_id,
                    result_type=result_type,
                    variant_key=variant_key,
                    values={
                        "component": row.get("component"),
                        "variant": row.get("variant"),
                        "requested_embedding_device": row.get("requested_embedding_device")
                        or row.get("requested_device"),
                        "requested_vector_device": row.get("requested_vector_device"),
                        "requested_reranker_device": row.get("requested_reranker_device"),
                        "chunking": row.get("chunking"),
                        "embedding_model": row.get("embedding_model"),
                        "retrieval_mode": row.get("retriever"),
                        "reranker": row.get("reranker"),
                        "rag_strategy": row.get("rag_strategy"),
                        "embedding_device": row.get("embedding_device") or row.get("execution_device"),
                        "vector_backend": row.get("vector_backend"),
                        "vector_device": row.get("vector_device"),
                        "llm_provider": row.get("llm_provider"),
                        "questions": row.get("questions"),
                        "mean_latency_ms": row.get("mean_latency_ms") or row.get("mean_total_latency_ms"),
                        "p50_latency_ms": row.get("p50_latency_ms") or row.get("p50_total_latency_ms"),
                        "p95_latency_ms": row.get("p95_latency_ms") or row.get("p95_total_latency_ms"),
                        "p99_latency_ms": row.get("p99_latency_ms") or row.get("p99_total_latency_ms"),
                        "recall_at_k": row.get("recall_at_k"),
                        "precision_at_k": row.get("precision_at_k"),
                        "f1_at_k": row.get("f1_at_k"),
                        "hit_rate_at_k": row.get("hit_rate_at_k"),
                        "mrr": row.get("mrr"),
                        "map_at_k": row.get("map_at_k"),
                        "ndcg_at_k": row.get("ndcg_at_k"),
                        "reciprocal_rank_at_k": row.get("reciprocal_rank_at_k"),
                        "r_precision": row.get("r_precision"),
                        "context_precision_at_k": row.get("context_precision_at_k"),
                        "recall_ci_low": row.get("recall_ci_low"),
                        "recall_ci_high": row.get("recall_ci_high"),
                        "ndcg_ci_low": row.get("ndcg_ci_low"),
                        "ndcg_ci_high": row.get("ndcg_ci_high"),
                        "no_hit_rate": row.get("no_hit_rate"),
                        "late_hit_rate": row.get("late_hit_rate"),
                        "latency_cv": row.get("latency_cv"),
                        "mean_latency_ci_low_ms": row.get("mean_latency_ci_low_ms"),
                        "mean_latency_ci_high_ms": row.get("mean_latency_ci_high_ms"),
                        "citation_accuracy": row.get("citation_accuracy"),
                        "citation_coverage": row.get("citation_coverage"),
                        "citation_source_coverage": row.get("citation_source_coverage"),
                        "key_fact_coverage": row.get("key_fact_coverage"),
                        "key_fact_ci_low": row.get("key_fact_ci_low"),
                        "key_fact_ci_high": row.get("key_fact_ci_high"),
                        "context_utilization": row.get("context_utilization"),
                        "fallback_rate": row.get("fallback_rate"),
                        "repair_rate": row.get("repair_rate"),
                        "mean_total_latency_ci_low_ms": row.get("mean_total_latency_ci_low_ms"),
                        "mean_total_latency_ci_high_ms": row.get("mean_total_latency_ci_high_ms"),
                        "mean_ttft_ms": row.get("mean_ttft_ms"),
                        "mean_tokens_per_second": row.get("mean_tokens_per_second"),
                        "mean_answer_tokens": row.get("mean_answer_tokens"),
                        "extra_json": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
                    },
                )

    def add_performance_results(
        self,
        run_id: str,
        rows: Iterable[object],
        *,
        chunking: str | None = None,
        embedding_model: str | None = None,
        retrieval_mode: str | None = None,
        reranker: str | None = None,
        rag_strategy: str | None = None,
        context_budget: int | None = None,
        vector_backend: str | None = None,
        vector_device: str | None = None,
        llm_provider: str | None = None,
    ) -> None:
        with self._connect() as connection:
            for row_obj in rows:
                row = _as_dict(row_obj)
                component = str(row.get("component") or "unknown")
                device = str(row.get("device") or "unknown")
                variant_key = f"{component}|{device}"
                self._upsert_result(
                    connection,
                    run_id=run_id,
                    result_type="performance",
                    variant_key=variant_key,
                    values={
                        "chunking": chunking,
                        "embedding_model": embedding_model,
                        "retrieval_mode": retrieval_mode,
                        "reranker": reranker,
                        "rag_strategy": rag_strategy,
                        "context_budget": context_budget,
                        "embedding_device": device if component == "embedding" else None,
                        "vector_backend": vector_backend,
                        "vector_device": vector_device,
                        "llm_provider": llm_provider,
                        "questions": row.get("workload_size"),
                        "mean_latency_ms": row.get("mean_ms"),
                        "p95_latency_ms": row.get("p95_ms"),
                        "total_ms": row.get("total_ms"),
                        "median_latency_ms": row.get("median_ms"),
                        "throughput_per_sec": row.get("throughput_per_sec"),
                        "ram_mb": row.get("ram_mb"),
                        "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb"),
                        "cold_start_ms": row.get("cold_start_ms"),
                        "workload_size": row.get("workload_size"),
                        "extra_json": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
                    },
                )

    @staticmethod
    def _upsert_result(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        result_type: str,
        variant_key: str,
        values: dict[str, Any],
    ) -> None:
        columns = ["run_id", "result_type", "variant_key", *values.keys()]
        allowed_columns = {row[1] for row in connection.execute("PRAGMA table_info(results)").fetchall()}
        unknown_columns = set(columns) - allowed_columns
        if unknown_columns:
            raise ValueError(f"Unknown result columns: {sorted(unknown_columns)}")

        placeholders = ",".join("?" for _ in columns)
        update_cols = [column for column in values if column != "variant_key"]
        update_sql = ",".join(f"{column}=excluded.{column}" for column in update_cols)
        query = f"""
            INSERT INTO results({",".join(columns)}) VALUES ({placeholders})
            ON CONFLICT(run_id, result_type, variant_key) DO UPDATE SET {update_sql}
            """  # nosec B608 -- identifiers are validated against the live SQLite schema.
        connection.execute(
            query,
            [run_id, result_type, variant_key, *values.values()],
        )

    def list_runs(self, *, limit: int = 200, benchmark_type: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM runs"
        params: list[Any] = []
        if benchmark_type:
            query += " WHERE benchmark_type = ?"
            params.append(benchmark_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY result_type, variant_key",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            completed = connection.execute("SELECT COUNT(*) FROM runs WHERE status='completed'").fetchone()[0]
            failed = connection.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0]
            result_count = connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            hashes = connection.execute("SELECT COUNT(DISTINCT config_hash) FROM runs").fetchone()[0]
        return {
            "runs": run_count,
            "completed": completed,
            "failed": failed,
            "results": result_count,
            "unique_configs": hashes,
            "database": str(self.path),
        }

    def export_json(self, path: Path) -> Path:
        runs = self.list_runs(limit=100000)
        payload = []
        for run in runs:
            run_copy = dict(run)
            try:
                run_copy["config"] = json.loads(run_copy.pop("config_json"))
            except Exception:
                pass
            run_copy["results"] = self.get_results(str(run["run_id"]))
            payload.append(run_copy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path
