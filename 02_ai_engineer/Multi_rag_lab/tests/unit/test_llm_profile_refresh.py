from __future__ import annotations

import json
from pathlib import Path

import rag_engine.platform.runtime as runtime_manager


def test_profile_refresh_detects_changed_modelfile(tmp_path: Path, monkeypatch) -> None:
    config_root = tmp_path / "config"
    modelfile_dir = config_root / "ollama"
    modelfile_dir.mkdir(parents=True)
    modelfile = modelfile_dir / "Modelfile.balanced"
    modelfile.write_text("SYSTEM magyar-v1", encoding="utf-8")
    state = tmp_path / ".cache" / "ollama_profile_state.json"

    monkeypatch.setattr(runtime_manager, "CONFIG_ROOT", config_root)
    monkeypatch.setattr(runtime_manager, "PROFILE_STATE_PATH", state)

    assert runtime_manager.llm_profile_needs_refresh("balanced")
    runtime_manager._write_profile_state("balanced")
    assert not runtime_manager.llm_profile_needs_refresh("balanced")

    modelfile.write_text("SYSTEM magyar-v2", encoding="utf-8")
    assert runtime_manager.llm_profile_needs_refresh("balanced")


def test_balanced_modelfile_has_hungarian_grounding_system_prompt() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "config" / "ollama" / "Modelfile.balanced").read_text(encoding="utf-8")
    assert "KIZÁRÓLAG magyarul" in text
    assert "forrásokra támaszkodó" in text
    assert "PARAMETER temperature 0.10" in text
