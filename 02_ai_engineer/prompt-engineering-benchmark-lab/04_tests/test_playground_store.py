"""EN: Robust Playground history loading in the presence of missing/corrupted JSONL rows.

HU: A Playground history robusztus betöltését ellenőrzi hiányzó vagy sérült JSONL sorok esetén.
"""

from __future__ import annotations

import json

import pandas as pd

from prompt_benchmark.ui import playground_store


def test_load_playground_history_skips_malformed_rows(tmp_path, monkeypatch):
    """EN: Ensures one corrupted history row is logged/skipped without breaking the complete Playground history load.

    HU: Biztosítja, hogy egy sérült history sor logolódjon/kimaradjon, de a teljes Playground history betöltés ne álljon le.
    """
    history = tmp_path / "history.jsonl"
    history.write_text('{"mode":"generation","value":1}\nnot-json\n{"mode":"classification","value":2}\n', encoding="utf-8")
    monkeypatch.setattr(playground_store, "HISTORY_PATH", history)

    frame = playground_store.load_playground_history(limit=10)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert frame.iloc[0]["value"] == 2
    assert frame.iloc[1]["value"] == 1


def test_load_playground_history_missing_file_returns_empty(tmp_path, monkeypatch):
    """EN: Ensures a first-run missing Playground history file is treated as an empty history, not an application error.

    HU: Biztosítja, hogy első futáskor a hiányzó Playground history üres historyként kezelődjön.
    """
    monkeypatch.setattr(playground_store, "HISTORY_PATH", tmp_path / "missing.jsonl")
    assert playground_store.load_playground_history().empty
