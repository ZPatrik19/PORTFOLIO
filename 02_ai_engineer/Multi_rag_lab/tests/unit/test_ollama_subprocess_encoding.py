from __future__ import annotations

from types import SimpleNamespace

from rag_engine.platform import runtime as runtime_manager


def test_decode_subprocess_output_handles_utf8_progress_bytes() -> None:
    raw = "pulling manifest ⠋ 42% — készül\rkövetkező sor".encode("utf-8")
    decoded = runtime_manager._decode_subprocess_output(raw)
    assert "pulling manifest" in decoded
    assert "42%" in decoded
    assert "készül" in decoded
    assert "következő sor" in decoded


def test_decode_subprocess_output_handles_none() -> None:
    assert runtime_manager._decode_subprocess_output(None) == ""


def test_run_ollama_uses_binary_capture_and_safe_decode(monkeypatch) -> None:
    monkeypatch.setattr(runtime_manager, "ollama_executable", lambda: "ollama")
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout="siker ✓".encode("utf-8"),
            stderr="progress ⠙".encode("utf-8"),
        )

    monkeypatch.setattr(runtime_manager.subprocess, "run", fake_run)
    result = runtime_manager._run_ollama(["pull", "qwen3:4b"])

    assert captured["text"] is False
    assert result["ok"] is True
    assert "siker" in result["stdout"]
    assert "progress" in result["stderr"]


def test_run_ollama_stream_output_inherits_console(monkeypatch) -> None:
    monkeypatch.setattr(runtime_manager, "ollama_executable", lambda: "ollama")
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime_manager.subprocess, "run", fake_run)
    result = runtime_manager._run_ollama(["pull", "qwen3:4b"], stream_output=True)

    assert "capture_output" not in captured
    assert "text" not in captured
    assert result["ok"] is True
