"""EN: Single-UI launcher policy, shared runtime controls, and removal of obsolete launch paths.

HU: Az egyetlen UI-indító elvét, a közös runtime beállításokat és az elavult launcher-ek eltávolítását ellenőrzi.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_launchers_are_explicit_and_keep_single_ui_entrypoint():
    """EN: Ensures root launchers delegate to one canonical Streamlit entrypoint.

    HU: Biztosítja, hogy a root launcherek egyetlen kanonikus Streamlit entrypointra delegálnak.
    """
    root_bats = sorted(path.name for path in ROOT.glob("*.bat"))
    assert root_bats == ["RUN_UI.bat", "run_project.bat", "run_tests.bat"]
    assert not any(name.startswith("RUN_UI_") for name in root_bats)


def test_unified_ui_contains_language_provider_and_api_key_controls():
    """EN: Ensures language, provider, model, and API-key controls live in the shared UI runtime layer.

    HU: Ellenőrzi, hogy a nyelv/provider/model/API-key kontrollok a közös UI runtime rétegben vannak.
    """
    text = (ROOT / "05_scripts" / "11_ui_app.py").read_text(encoding="utf-8")
    assert "Language / Nyelv" in text
    assert "runtime_provider" in text
    assert "GEMINI_API_KEY" in text
    assert "OPENROUTER_API_KEY" in text
    assert "GROQ_API_KEY" in text
    assert "OPENAI_API_KEY" in text
    assert "Test connection" in text or "Kapcsolat tesztelése" in text


def test_obsolete_language_and_gemini_config_launchers_are_removed():
    """EN: Prevents obsolete HU/EN/Gemini-specific launchers from reappearing and fragmenting UX.

    HU: Megakadályozza az elavult HU/EN/Gemini-specifikus launcherek visszakerülését és az UX széttöredezését.
    """
    obsolete = [
        ROOT / "RUN_UI_HU.bat",
        ROOT / "RUN_UI_EN.bat",
        ROOT / "CONFIGURE_GEMINI.bat",
        ROOT / "00_setup" / "02_run_ui_en.bat",
        ROOT / "00_setup" / "06_configure_gemini.bat",
        ROOT / "00_setup" / "06_configure_gemini.py",
    ]
    assert all(not path.exists() for path in obsolete)
