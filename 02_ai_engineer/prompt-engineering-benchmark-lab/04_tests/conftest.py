"""EN: Shared pytest configuration and deterministic test-layer classification.

The project package is installed editable during setup (``pip install -e .``),
so no ``sys.path`` mutation is required. Test markers are assigned centrally by
file because the suite predates marker adoption; this keeps the documented
``pytest -m unit|integration|smoke`` commands executable without duplicating
module-level boilerplate across every test file.

HU: Közös pytest konfiguráció és determinisztikus tesztréteg-besorolás.

A projekt package editable módban települ (``pip install -e .``), ezért nincs
szükség ``sys.path`` módosításra. A marker besorolás központilag, fájlszinten
történik, így a dokumentált ``pytest -m unit|integration|smoke`` parancsok
konzisztensen futnak anélkül, hogy minden tesztmodulban boilerplate marker kellene.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Keep tests deterministic and prevent accidental use of a cloud provider.
os.environ.setdefault("LLM_PROVIDER", "mock")

INTEGRATION_TEST_FILES = {
    "test_custom_prompts.py",
    "test_expanded_challenge_set.py",
    "test_provider_factory.py",
    "test_runner_cache.py",
    "test_ui_language_switch.py",
    "test_workflow_history.py",
    "test_cli.py",
}
SMOKE_TEST_FILES = {"test_smoke_pipeline.py"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign exactly one high-level test-layer marker to every collected test."""
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in SMOKE_TEST_FILES:
            item.add_marker(pytest.mark.smoke)
        elif filename in INTEGRATION_TEST_FILES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
