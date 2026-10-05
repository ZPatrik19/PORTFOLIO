from __future__ import annotations

from pathlib import Path

import pytest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - exercised only in minimal CI/runtime images
    AppTest = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "ui" / "app.py"
pytestmark = pytest.mark.skipif(AppTest is None, reason="Streamlit nincs telepítve ebben a tesztkörnyezetben")


def test_streamlit_app_starts_without_uncaught_exception() -> None:
    """Real Streamlit headless smoke test when Streamlit is installed."""
    assert AppTest is not None
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["page"] = "overview"
    at.run(timeout=30)
    assert not at.exception, [str(item.value) for item in at.exception]


def test_streamlit_evaluation_page_starts_without_uncaught_exception() -> None:
    """Open the evaluation route through the real Streamlit runtime."""
    assert AppTest is not None
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["page"] = "evaluation"
    at.run(timeout=30)
    assert not at.exception, [str(item.value) for item in at.exception]


def test_streamlit_runtime_page_profile_actions_do_not_raise_widget_state_errors() -> None:
    """Exercise the page that mutates shared sidebar settings through deferred updates."""
    assert AppTest is not None
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["page"] = "runtime_lab"
    at.run(timeout=30)
    assert not at.exception, [str(item.value) for item in at.exception]

    profile_buttons = [button for button in at.button if button.label == "Profil alkalmazása"]
    if profile_buttons:
        profile_buttons[0].click().run(timeout=30)
        assert not at.exception, [str(item.value) for item in at.exception]
