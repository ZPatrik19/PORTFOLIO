"""Utilities for turning production feedback into regression data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import resolve_path


def export_negative_feedback(
    config: dict[str, Any],
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Export negative feedback rows as deterministic regression examples."""

    database_path = resolve_path(config["monitoring"]["sqlite_path"])
    output_path = out_path or (
        resolve_path(config["paths"]["evaluation"]) / "feedback_regression_samples.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        output_path.write_text("[]", encoding="utf-8")
        return []

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT f.request_id, f.feedback_text, r.payload, f.ts
            FROM feedback AS f
            JOIN request_log AS r ON r.request_id = f.request_id
            WHERE f.helpful = 0
            ORDER BY f.ts
            """
        ).fetchall()
    finally:
        connection.close()

    samples: list[dict[str, Any]] = []
    for index, (request_id, feedback_text, payload, timestamp) in enumerate(rows):
        request_payload = json.loads(payload)
        samples.append(
            {
                "question_id": f"feedback_{index:04d}_{request_id[:8]}",
                "source_request_id": request_id,
                "question": request_payload.get("query", ""),
                "category": request_payload.get("query_type", "production_feedback").lower(),
                "difficulty": "production",
                "expected_documents": [],
                "expected_chunks": [],
                "expected_concepts": [],
                "expected_answer_available": not bool(
                    request_payload.get("insufficient_evidence")
                ),
                "expected_tool": None,
                "feedback_text": feedback_text,
                "feedback_timestamp": timestamp,
                "retrieved_chunk_count": request_payload.get("retrieved_chunks"),
            }
        )

    output_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return samples
