from __future__ import annotations

import json
from pathlib import Path

from tkip.feedback import export_negative_feedback
from tkip.monitoring import Telemetry


def test_negative_feedback_becomes_regression_sample(
    isolated_config: dict,
    tmp_path: Path,
) -> None:
    # Arrange
    telemetry = Telemetry(isolated_config)
    telemetry.log(
        {
            "request_id": "req-12345678",
            "query": "Explain RAG",
            "query_type": "CONCEPTUAL",
            "retrieved_chunks": 4,
            "insufficient_evidence": False,
            "response_status": "success",
        }
    )
    telemetry.feedback("req-12345678", helpful=False, feedback_text="Too shallow")
    output_path = tmp_path / "regression.json"

    # Act
    samples = export_negative_feedback(isolated_config, output_path)

    # Assert
    assert len(samples) == 1
    assert samples[0]["source_request_id"] == "req-12345678"
    assert samples[0]["question"] == "Explain RAG"
    assert samples[0]["feedback_text"] == "Too shallow"
    assert json.loads(output_path.read_text(encoding="utf-8")) == samples


def test_feedback_export_returns_empty_list_when_database_is_missing(
    isolated_config: dict,
    tmp_path: Path,
) -> None:
    # Arrange
    isolated_config["monitoring"]["sqlite_path"] = str(tmp_path / "missing.sqlite3")
    output_path = tmp_path / "regression.json"

    # Act
    samples = export_negative_feedback(isolated_config, output_path)

    # Assert
    assert samples == []
    assert output_path.read_text(encoding="utf-8") == "[]"
