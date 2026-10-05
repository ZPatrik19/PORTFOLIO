from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_root_batch_launchers_force_utf8() -> None:
    for name in ("SETUP.bat", "INFRASTRUCTURE.bat", "RUN.bat", "EVALUATION.bat"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "chcp 65001 >nul" in text
        assert 'set "PYTHONUTF8=1"' in text
        assert 'set "PYTHONIOENCODING=utf-8"' in text


def test_infrastructure_stream_forces_utf8_for_children() -> None:
    text = (ROOT / "scripts" / "infrastructure_cli.py").read_text(encoding="utf-8")
    assert 'child_env["PYTHONUTF8"] = "1"' in text
    assert 'child_env["PYTHONIOENCODING"] = "utf-8"' in text
    assert "env=child_env" in text


def test_medical_prepare_configures_utf8_console() -> None:
    text = (ROOT / "scripts" / "prepare_medical_corpus.py").read_text(encoding="utf-8")
    assert "def _configure_console_encoding()" in text
    assert 'reconfigure(encoding="utf-8", errors="replace")' in text
    assert "_configure_console_encoding()" in text
