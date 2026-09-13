from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from travel_agent.models import AgentRun

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "06_results" / "usage" / "usage_history.sqlite3"


class UsageStore:
    """Persistent local usage analytics backed by SQLite.

    The database stays inside the repository and is never sent to an external
    analytics service. Every chat run stores the question, answer, runtime
    metadata, and normalized tool-call rows. Aggregate methods intentionally
    compute statistics from persisted history instead of in-memory counters so
    the dashboard remains meaningful after application restarts.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    methodology TEXT NOT NULL,
                    language TEXT NOT NULL,
                    data_mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'custom',
                    preset_id TEXT,
                    total_latency_ms REAL NOT NULL DEFAULT 0,
                    tool_call_count INTEGER NOT NULL DEFAULT 0,
                    successful_tool_calls INTEGER NOT NULL DEFAULT 0,
                    failed_tool_calls INTEGER NOT NULL DEFAULT 0,
                    run_success INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    arguments_json TEXT NOT NULL DEFAULT '{}',
                    output_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    FOREIGN KEY(interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_interactions_created_at
                    ON interactions(created_at_utc);
                CREATE INDEX IF NOT EXISTS idx_interactions_methodology
                    ON interactions(methodology);
                CREATE INDEX IF NOT EXISTS idx_interactions_language
                    ON interactions(language);
                CREATE INDEX IF NOT EXISTS idx_interactions_data_mode
                    ON interactions(data_mode);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_name
                    ON tool_calls(tool_name);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_interaction
                    ON tool_calls(interaction_id);
                """
            )

    def record_run(
        self,
        run: AgentRun,
        *,
        methodology: str,
        language: str,
        data_mode: str,
        source: str = "custom",
        preset_id: str | None = None,
    ) -> int:
        successes = sum(1 for call in run.trace if call.success)
        failures = len(run.trace) - successes
        run_success = bool(run.answer.strip()) and failures == 0
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO interactions (
                    created_at_utc, question, answer, methodology, language,
                    data_mode, model, source, preset_id, total_latency_ms,
                    tool_call_count, successful_tool_calls, failed_tool_calls,
                    run_success, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    run.query,
                    run.answer,
                    methodology,
                    language,
                    data_mode,
                    run.model,
                    source,
                    preset_id,
                    float(run.total_latency_ms),
                    len(run.trace),
                    successes,
                    failures,
                    int(run_success),
                    json.dumps(run.metadata or {}, ensure_ascii=False, default=str),
                ),
            )
            interaction_id = int(cur.lastrowid)
            for call in run.trace:
                con.execute(
                    """
                    INSERT INTO tool_calls (
                        interaction_id, step, tool_name, success, latency_ms,
                        arguments_json, output_json, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        int(call.step),
                        call.name,
                        int(call.success),
                        float(call.latency_ms),
                        json.dumps(call.arguments, ensure_ascii=False, default=str),
                        json.dumps(call.output, ensure_ascii=False, default=str),
                        call.error,
                    ),
                )
        return interaction_id

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(x) for x in values)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    def summary(self) -> dict[str, Any]:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT
                    COUNT(*) AS total_questions,
                    COALESCE(SUM(tool_call_count), 0) AS total_tool_calls,
                    COALESCE(AVG(tool_call_count), 0) AS avg_tool_calls,
                    COALESCE(AVG(total_latency_ms), 0) AS avg_latency_ms,
                    COALESCE(AVG(run_success), 0) AS run_success_rate,
                    COALESCE(SUM(successful_tool_calls), 0) AS successful_tool_calls,
                    COALESCE(SUM(failed_tool_calls), 0) AS failed_tool_calls,
                    SUM(CASE WHEN source='preset' THEN 1 ELSE 0 END) AS preset_questions,
                    SUM(CASE WHEN source='custom' THEN 1 ELSE 0 END) AS custom_questions,
                    SUM(CASE WHEN tool_call_count=0 THEN 1 ELSE 0 END) AS no_tool_questions,
                    SUM(CASE WHEN tool_call_count>1 THEN 1 ELSE 0 END) AS multi_tool_questions,
                    COALESCE(AVG(LENGTH(question)), 0) AS avg_question_chars,
                    COALESCE(AVG(LENGTH(answer)), 0) AS avg_answer_chars,
                    COUNT(DISTINCT question) AS unique_questions
                FROM interactions
                """
            ).fetchone()
            out = dict(row)
            total_calls = int(out["successful_tool_calls"] or 0) + int(out["failed_tool_calls"] or 0)
            total_questions = int(out["total_questions"] or 0)
            out["tool_success_rate"] = (
                float(out["successful_tool_calls"] or 0) / total_calls if total_calls else 0.0
            )
            out["multi_tool_rate"] = (
                float(out["multi_tool_questions"] or 0) / total_questions if total_questions else 0.0
            )
            out["no_tool_rate"] = (
                float(out["no_tool_questions"] or 0) / total_questions if total_questions else 0.0
            )
            out["unique_question_rate"] = (
                float(out["unique_questions"] or 0) / total_questions if total_questions else 0.0
            )

            run_latencies = [float(r[0]) for r in con.execute("SELECT total_latency_ms FROM interactions").fetchall()]
            tool_latencies = [float(r[0]) for r in con.execute("SELECT latency_ms FROM tool_calls").fetchall()]
            out["run_latency_p50_ms"] = self._percentile(run_latencies, 0.50)
            out["run_latency_p95_ms"] = self._percentile(run_latencies, 0.95)
            out["tool_latency_p50_ms"] = self._percentile(tool_latencies, 0.50)
            out["tool_latency_p95_ms"] = self._percentile(tool_latencies, 0.95)
            return out

    def recent_interactions(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT id, created_at_utc, source, methodology, language, data_mode,
                       question, answer, tool_call_count, successful_tool_calls,
                       failed_tool_calls, total_latency_ms, run_success
                FROM interactions
                ORDER BY id DESC
                LIMIT ?
                """,
                con,
                params=(int(limit),),
            )

    def tool_usage(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT tool_name,
                       COUNT(*) AS calls,
                       SUM(success) AS successful_calls,
                       COUNT(*) - SUM(success) AS failed_calls,
                       AVG(latency_ms) AS avg_latency_ms,
                       MIN(latency_ms) AS min_latency_ms,
                       MAX(latency_ms) AS max_latency_ms
                FROM tool_calls
                GROUP BY tool_name
                ORDER BY calls DESC, tool_name ASC
                """,
                con,
            )

    def tool_call_history(self, limit: int | None = None) -> pd.DataFrame:
        """Return normalized tool-call events for interactive latency diagnostics."""
        query = """
                SELECT t.id, t.interaction_id, i.created_at_utc, i.methodology,
                       i.language, i.data_mode, i.source, t.step, t.tool_name,
                       t.success, t.latency_ms, t.error
                FROM tool_calls t
                JOIN interactions i ON i.id = t.interaction_id
                ORDER BY t.id DESC
                """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        with self._connect() as con:
            return pd.read_sql_query(query, con, params=params)

    def methodology_usage(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT methodology,
                       COUNT(*) AS questions,
                       AVG(total_latency_ms) AS avg_latency_ms,
                       AVG(run_success) AS success_rate,
                       AVG(tool_call_count) AS avg_tool_calls,
                       SUM(CASE WHEN tool_call_count > 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS multi_tool_rate
                FROM interactions
                GROUP BY methodology
                ORDER BY questions DESC
                """,
                con,
            )

    def daily_usage(self, days: int = 30) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT substr(created_at_utc, 1, 10) AS date,
                       COUNT(*) AS questions,
                       SUM(tool_call_count) AS tool_calls,
                       AVG(total_latency_ms) AS avg_latency_ms,
                       AVG(run_success) AS success_rate
                FROM interactions
                WHERE created_at_utc >= datetime('now', ?)
                GROUP BY substr(created_at_utc, 1, 10)
                ORDER BY date ASC
                """,
                con,
                params=(f"-{int(days)} days",),
            )

    def dimension_usage(self, column: str) -> pd.DataFrame:
        if column not in {"language", "data_mode", "source", "model"}:
            raise ValueError(f"Unsupported dimension: {column}")
        with self._connect() as con:
            return pd.read_sql_query(
                f"""
                SELECT {column} AS value,
                       COUNT(*) AS questions,
                       AVG(total_latency_ms) AS avg_latency_ms,
                       AVG(run_success) AS success_rate,
                       AVG(tool_call_count) AS avg_tool_calls
                FROM interactions
                GROUP BY {column}
                ORDER BY questions DESC, value ASC
                """,
                con,
            )

    def tool_sequences(self, limit: int = 15) -> pd.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT i.id, t.step, t.tool_name
                FROM interactions i
                LEFT JOIN tool_calls t ON i.id=t.interaction_id
                ORDER BY i.id ASC, t.step ASC
                """
            ).fetchall()
        grouped: dict[int, list[str]] = {}
        for row in rows:
            grouped.setdefault(int(row["id"]), [])
            if row["tool_name"]:
                grouped[int(row["id"])].append(str(row["tool_name"]))
        counts = Counter(" → ".join(seq) if seq else "(no tool)" for seq in grouped.values())
        return pd.DataFrame(counts.most_common(limit), columns=["tool_sequence", "runs"])

    def destination_usage(self, limit: int = 15) -> pd.DataFrame:
        with self._connect() as con:
            rows = con.execute("SELECT arguments_json FROM tool_calls").fetchall()
        cities: Counter[str] = Counter()
        for row in rows:
            try:
                args = json.loads(row["arguments_json"] or "{}")
            except json.JSONDecodeError:
                continue
            city = args.get("city")
            if isinstance(city, str) and city.strip():
                cities[city.strip()] += 1
        return pd.DataFrame(cities.most_common(limit), columns=["city", "tool_calls"])

    def error_breakdown(self, limit: int = 20) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT tool_name,
                       COALESCE(NULLIF(TRIM(error), ''), 'unknown') AS error,
                       COUNT(*) AS occurrences
                FROM tool_calls
                WHERE success=0
                GROUP BY tool_name, COALESCE(NULLIF(TRIM(error), ''), 'unknown')
                ORDER BY occurrences DESC, tool_name ASC
                LIMIT ?
                """,
                con,
                params=(int(limit),),
            )

    def period_comparison(self, days: int = 7) -> dict[str, Any]:
        """Compare the latest N days with the immediately preceding N-day window."""
        with self._connect() as con:
            current = con.execute(
                """
                SELECT COUNT(*) q, COALESCE(SUM(tool_call_count),0) tc,
                       COALESCE(AVG(total_latency_ms),0) lat, COALESCE(AVG(run_success),0) sr
                FROM interactions
                WHERE created_at_utc >= datetime('now', ?)
                """,
                (f"-{int(days)} days",),
            ).fetchone()
            previous = con.execute(
                """
                SELECT COUNT(*) q, COALESCE(SUM(tool_call_count),0) tc,
                       COALESCE(AVG(total_latency_ms),0) lat, COALESCE(AVG(run_success),0) sr
                FROM interactions
                WHERE created_at_utc >= datetime('now', ?)
                  AND created_at_utc < datetime('now', ?)
                """,
                (f"-{int(days*2)} days", f"-{int(days)} days"),
            ).fetchone()
        return {"current": dict(current), "previous": dict(previous), "days": days}

    def export_interactions(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT i.*, GROUP_CONCAT(t.tool_name, ' → ') AS tools
                FROM interactions i
                LEFT JOIN tool_calls t ON i.id = t.interaction_id
                GROUP BY i.id
                ORDER BY i.id DESC
                """,
                con,
            )

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM tool_calls")
            con.execute("DELETE FROM interactions")

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
