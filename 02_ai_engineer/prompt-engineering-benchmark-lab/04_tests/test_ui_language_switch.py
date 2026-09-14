"""EN: Bilingual single-UI wiring, shared runtime controls, and feature parity between HU/EN views.

HU: A kétnyelvű egy-UI architektúrát, közös runtime beállításokat és HU/EN funkcióparitást ellenőrzi.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bilingual_ui_launcher_exists_and_loads_both_views() -> None:
    """EN: Ensures the shared launcher can load both Hungarian and English views.

    HU: Biztosítja, hogy a közös launcher mind a magyar, mind az angol nézetet be tudja tölteni.
    """
    launcher = (ROOT / "05_scripts" / "11_ui_app.py").read_text(encoding="utf-8")
    assert "Language / Nyelv" in launcher
    assert "11_ui_view_hu.py" in launcher
    assert "11_ui_view_en.py" in launcher
    assert "ui_language" in launcher


def test_language_views_do_not_reset_page_config() -> None:
    """EN: Prevents language view modules from reinitializing global Streamlit page configuration.

    HU: Megakadályozza, hogy a nyelvi view modulok újrainicializálják a globális Streamlit page configot.
    """
    for name in ["11_ui_view_hu.py", "11_ui_view_en.py"]:
        source = (ROOT / "05_scripts" / name).read_text(encoding="utf-8")
        assert "st.set_page_config(" not in source


def test_ui_bat_delegates_to_single_project_launcher() -> None:
    """EN: Ensures the legacy RUN_UI.bat delegates to the canonical project launcher.

    HU: Ellenőrzi, hogy a legacy RUN_UI.bat a kanonikus project launcherre delegál.
    """
    bat = (ROOT / "00_setup" / "02_run_ui.bat").read_text(encoding="utf-8")
    assert "run_project.bat" in bat
    root_launcher = (ROOT / "run_project.bat").read_text(encoding="utf-8")
    assert "11_ui_app.py" in root_launcher
    assert "11_ui_app_hu.py" not in bat


def test_runtime_controls_live_only_in_shared_launcher() -> None:
    """EN: Prevents provider/model/API-key controls from being duplicated in language-specific view modules.

    HU: Megakadályozza a provider/model/API-key kontrollok duplikálását a nyelvspecifikus view-kban.
    """
    launcher = (ROOT / "05_scripts" / "11_ui_app.py").read_text(encoding="utf-8")
    assert "runtime_provider" in launcher
    assert "GEMINI_API_KEY" in launcher
    assert "Test connection" in launcher or "Kapcsolat tesztelése" in launcher
    for name in ["11_ui_view_hu.py", "11_ui_view_en.py"]:
        source = (ROOT / "05_scripts" / name).read_text(encoding="utf-8")
        assert "runtime_provider" not in source


def test_both_languages_expose_custom_prompt_ui() -> None:
    """EN: Checks HU and EN views both expose custom-prompt functionality.

    HU: Ellenőrzi, hogy a HU és EN felület is elérhetővé teszi a custom prompt funkciót.
    """
    hu = (ROOT / "05_scripts" / "11_ui_view_hu.py").read_text(encoding="utf-8")
    en = (ROOT / "05_scripts" / "11_ui_view_en.py").read_text(encoding="utf-8")
    assert "with tab_custom" in hu
    assert "with tab_custom" in en
