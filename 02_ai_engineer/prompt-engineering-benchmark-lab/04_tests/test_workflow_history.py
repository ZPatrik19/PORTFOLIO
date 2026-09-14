"""EN: Persistent benchmark-run history and Playground preset round trips.

HU: A tartós benchmark history és Playground preset mentés-visszatöltés körét ellenőrzi.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prompt_benchmark.ui import run_history
from prompt_benchmark.ui.playground_store import save_generation_preset, load_generation_preset


def test_run_history_round_trip(tmp_path, monkeypatch):
    """EN: Verifies benchmark run manifests/results can be saved and loaded without losing metadata.

    HU: Ellenőrzi, hogy a benchmark run manifest és eredmények menthetők/visszatölthetők metadata-vesztés nélkül.
    """
    monkeypatch.setattr(run_history, "HISTORY_ROOT", tmp_path / "history")
    rid = run_history.make_run_id("mock", "challenge")
    dataset = pd.DataFrame([{"sample_id": "x", "text": "hello", "true_label": "api"}])
    root = run_history.initialize_run(rid, {"provider": "mock", "model": "m", "suite": "challenge"}, dataset)
    summary = pd.DataFrame([{"strategy": "p0_zero_shot", "macro_f1": 0.5, "total_tokens": 10}])
    run_history.finalize_run(rid, summary, {"elapsed_wall_seconds": 1.2})
    assert (root / "manifest.json").exists()
    assert run_history.load_summary(rid) is not None
    runs = run_history.list_runs(provider="mock")
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == rid


def test_generation_preset_round_trip(tmp_path, monkeypatch):
    """EN: Verifies generative Playground presets can be saved and restored losslessly.

    HU: Ellenőrzi, hogy a generatív Playground preset veszteségmentesen menthető és visszatölthető.
    """
    import prompt_benchmark.ui.playground_store as store
    monkeypatch.setattr(store, "PRESET_DIR", tmp_path / "presets")
    path = store.save_generation_preset("My Prompt", "system", "user")
    loaded = store.load_generation_preset(path)
    assert loaded["name"] == "My Prompt"
    assert loaded["system_prompt"] == "system"
