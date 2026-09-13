from __future__ import annotations

import json
from pathlib import Path

import travel_agent.training.router_training as rt


def _patch_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data = tmp_path / "intent_router_dataset.csv"
    model = tmp_path / "intent_router.joblib"
    metadata = tmp_path / "intent_router_metadata.json"
    metrics = tmp_path / "intent_router_metrics.json"
    data.write_text("query,tool_labels,split\nhello,weather,train\n", encoding="utf-8")
    # Deliberately not a pickle: the status check must reject it before joblib.load.
    model.write_bytes(b"this-is-not-a-joblib-model")
    monkeypatch.setattr(rt, "DATA_PATH", data)
    monkeypatch.setattr(rt, "MODEL_PATH", model)
    monkeypatch.setattr(rt, "MODEL_METADATA_PATH", metadata)
    monkeypatch.setattr(rt, "METRICS_PATH", metrics)
    return data, model, metadata, metrics


def test_legacy_model_without_metadata_is_not_unpickled(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    status = rt.model_status()
    assert status["exists"]
    assert status["stale"]
    assert not status["compatible"]
    assert status["reason"] == "model_metadata_missing"


def test_runtime_version_mismatch_is_detected_before_unpickle(monkeypatch, tmp_path):
    data, _, metadata, _ = _patch_paths(monkeypatch, tmp_path)
    runtime = rt._runtime_metadata()
    metadata.write_text(
        json.dumps({
            "artifact_format": 2,
            "dataset_sha256": rt.dataset_hash(data),
            "python_major_minor": runtime["python_major_minor"],
            "scikit_learn_version": "0.0.0-test",
        }),
        encoding="utf-8",
    )
    status = rt.model_status()
    assert status["exists"]
    assert status["stale"]
    assert not status["compatible"]
    assert status["reason"] == "runtime_version_mismatch"
    assert "scikit_learn_version" in status["runtime_mismatches"]
