"""Structured request telemetry, feedback storage and lightweight drift checks."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from typing import Any

from .config import resolve_path
from .utils import jsonl_append


class Telemetry:
    """Persist request traces to JSONL and SQLite for local observability."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        monitoring = cfg["monitoring"]
        self.json_path = resolve_path(monitoring["json_log_path"])
        self.db_path = resolve_path(monitoring["sqlite_path"])
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS request_log "
                "(request_id TEXT PRIMARY KEY, ts TEXT, payload TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS feedback "
                "(request_id TEXT, ts TEXT, helpful INTEGER, feedback_text TEXT)"
            )
            connection.commit()

    def log(self, record: dict[str, Any]) -> None:
        timestamped = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        jsonl_append(self.json_path, timestamped)
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO request_log VALUES(?,?,?)",
                (
                    timestamped.get("request_id"),
                    timestamped["timestamp"],
                    json.dumps(timestamped, ensure_ascii=False, default=str),
                ),
            )
            connection.commit()

    def feedback(
        self,
        request_id: str,
        helpful: bool,
        feedback_text: str | None = None,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO feedback VALUES(?,?,?,?)",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    int(helpful),
                    feedback_text,
                ),
            )
            connection.commit()

    def recent(self, limit: int = 1000) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload FROM request_log ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def summary(self) -> dict[str, Any]:
        rows = self.recent()
        latencies = [
            float(row["total_latency_ms"])
            for row in rows
            if row.get("total_latency_ms") is not None
        ]
        with closing(self._connect()) as connection:
            feedback_rows = connection.execute("SELECT helpful FROM feedback").fetchall()
        costs = [
            float(row["estimated_cost_usd"])
            for row in rows
            if row.get("estimated_cost_usd") is not None
        ]
        tokens = [
            int(row["total_tokens"])
            for row in rows
            if row.get("total_tokens") is not None
        ]
        request_count = len(rows)
        return {
            "requests": request_count,
            "success_rate": sum(row.get("response_status") == "success" for row in rows)
            / max(1, request_count),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "no_answer_rate": sum(bool(row.get("insufficient_evidence")) for row in rows)
            / max(1, request_count),
            "citation_validation_rate": sum(bool(row.get("citation_valid")) for row in rows)
            / max(1, request_count),
            "total_tokens": sum(tokens),
            "estimated_cost_usd": sum(costs) if costs else None,
            "feedback_score": (
                sum(row[0] for row in feedback_rows) / len(feedback_rows)
                if feedback_rows
                else None
            ),
        }


def drift_report(rows: list[dict[str, Any]], window: int = 50) -> dict[str, Any]:
    """Compare two recent windows for simple retrieval/query-distribution drift."""

    if len(rows) < window * 2:
        return {
            "status": "insufficient_data",
            "required": window * 2,
            "observed": len(rows),
        }

    current = rows[:window]
    previous = rows[window : window * 2]
    metrics: dict[str, Any] = {}
    for key in ("top_retrieval_score", "no_answer_numeric", "retrieved_chunks"):
        current_average = _average_numeric(key, current)
        previous_average = _average_numeric(key, previous)
        relative_change = (
            (current_average - previous_average) / abs(previous_average)
            if previous_average
            else 0.0
        )
        metrics[key] = {
            "current": current_average,
            "previous": previous_average,
            "relative_change": relative_change,
        }

    current_distribution = _distribution(current, "query_type")
    previous_distribution = _distribution(previous, "query_type")
    keys = set(current_distribution) | set(previous_distribution)
    total_variation_distance = 0.5 * sum(
        abs(current_distribution.get(key, 0.0) - previous_distribution.get(key, 0.0))
        for key in keys
    )
    metrics["query_type_distribution"] = {
        "total_variation_distance": total_variation_distance,
        "current": current_distribution,
        "previous": previous_distribution,
    }

    alerts = [
        key
        for key, value in metrics.items()
        if key != "query_type_distribution" and abs(value.get("relative_change", 0.0)) > 0.25
    ]
    if total_variation_distance > 0.30:
        alerts.append("query_type_distribution")
    return {"status": "alert" if alerts else "ok", "alerts": alerts, "metrics": metrics}


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _average_numeric(key: str, items: list[dict[str, Any]]) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return sum(values) / max(1, len(values))


def _distribution(items: list[dict[str, Any]], key: str) -> dict[str, float]:
    counts = Counter(str(item.get(key, "unknown")) for item in items)
    total = sum(counts.values()) or 1
    return {name: count / total for name, count in counts.items()}
