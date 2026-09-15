"""Regression tests for strict HU/EN Streamlit localization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
I18N_PATH = PROJECT_ROOT / "05_ui" / "i18n.py"
APP_PATH = PROJECT_ROOT / "05_ui" / "app.py"


def _load_i18n():
    spec = importlib.util.spec_from_file_location("tki_ui_i18n", I18N_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_translation_key_sets_match_exactly() -> None:
    i18n = _load_i18n()

    assert set(i18n.TEXT["hu"]) == set(i18n.TEXT["en"])


def test_legacy_language_labels_are_normalized_to_stable_codes() -> None:
    i18n = _load_i18n()

    assert i18n.normalize_language("Magyar") == "hu"
    assert i18n.normalize_language("Angol") == "en"
    assert i18n.normalize_language("Hungarian") == "hu"
    assert i18n.normalize_language("English") == "en"
    assert i18n.tr("Angol", "benchmark_title") == i18n.TEXT["en"]["benchmark_title"]


def test_english_ui_copy_contains_no_hungarian_diacritic_leaks() -> None:
    i18n = _load_i18n()
    hungarian_diacritics = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")

    leaked = {
        key: value
        for key, value in i18n.TEXT["en"].items()
        if any(character in hungarian_diacritics for character in value)
    }

    assert leaked == {}


def test_core_labels_are_different_between_hungarian_and_english() -> None:
    i18n = _load_i18n()
    keys = {
        "title",
        "pipeline",
        "temperature",
        "monitoring_tab",
        "retrieval_benchmark_tab",
        "prompt_engineering",
        "tool_calling",
        "prompt_benchmark",
        "tool_calling_benchmark",
        "grounding",
        "system_prompt",
    }

    for key in keys:
        assert i18n.TEXT["hu"][key] != i18n.TEXT["en"][key], key


def test_backend_enum_values_are_localized_for_hungarian_only() -> None:
    i18n = _load_i18n()

    assert i18n.localize_value("hu", "COMPARISON") == "Összehasonlítás"
    assert i18n.localize_value("hu", "balanced grounded evidence") == (
        "kiegyensúlyozott, forrásalapú bizonyíték"
    )
    assert i18n.localize_value("en", "COMPARISON") == "COMPARISON"


def test_language_switch_contract_clears_language_bound_generated_content() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert "LANGUAGE_BOUND_SESSION_KEYS" in app_source
    for key in (
        "workspace_question",
        "last_answer",
        "previous_answer",
        "ab_question",
        "ab_results",
        "prompt_benchmark_df",
    ):
        assert f'"{key}"' in app_source


def test_hungarian_benchmark_catalog_localizes_every_catalog_term() -> None:
    from tkip.benchmark_catalog import BENCHMARK_CATALOG

    i18n = _load_i18n()
    missing: list[str] = []
    for specification in BENCHMARK_CATALOG.values():
        for term in [*specification["methods"], *specification["metrics"]]:
            if term not in i18n.BENCHMARK_TERM_HU:
                missing.append(term)

    assert missing == []
