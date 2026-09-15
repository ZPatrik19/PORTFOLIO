"""Shared pytest fixtures for the Technical Knowledge Intelligence test suite."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
from tkip.config import gemini_api_key, load_config


@pytest.fixture
def isolated_config(tmp_path: Path) -> dict:
    """Return a fully isolated, offline-first configuration.

    No user-library documents, Gemini calls, external vector service, production logs, or
    project indexes are touched by tests using this fixture.
    """
    cfg = copy.deepcopy(load_config())
    project_root = Path(cfg["_project_root"])

    empty_user_library = tmp_path / "user_library"
    empty_user_library.mkdir(parents=True, exist_ok=True)

    cfg["embedding"]["provider"] = "local_hashing"
    cfg["vector_store"]["provider"] = "numpy"
    cfg["paths"]["user_library"] = str(empty_user_library)
    cfg["paths"]["reference_docs"] = str(project_root / cfg["paths"]["reference_docs"])
    cfg["paths"]["processed"] = str(tmp_path / "processed")
    cfg["paths"]["interim"] = str(tmp_path / "interim")
    cfg["paths"]["indexes"] = str(tmp_path / "indexes")
    cfg["monitoring"]["json_log_path"] = str(tmp_path / "telemetry.jsonl")
    cfg["monitoring"]["sqlite_path"] = str(tmp_path / "telemetry.sqlite3")
    return cfg


@pytest.fixture
def live_gemini_key() -> str:
    """Return a live Gemini key only when the user explicitly opts in.

    Live API tests are intentionally skipped by default so the normal test suite
    remains deterministic and does not consume quota or billing.
    """
    if os.getenv("TKI_RUN_LIVE_GEMINI") != "1":
        pytest.skip("Live Gemini tests are disabled. Set TKI_RUN_LIVE_GEMINI=1 to opt in.")

    key = (os.getenv("GEMINI_API_KEY") or gemini_api_key() or "").strip()
    if not key:
        pytest.skip("GEMINI_API_KEY is not configured for live Gemini tests.")
    return key
