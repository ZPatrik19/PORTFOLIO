from __future__ import annotations

from pathlib import Path

from rag_engine.platform import profiles as profile_module
from rag_engine.platform.profiles import resolve_profile_name


def test_loader_supplies_ui_metadata_for_legacy_profile(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text("profiles:\n  legacy_profile:\n    value: 1\n", encoding="utf-8")
    profiles = profile_module._load_profiles(path)
    assert profiles["legacy_profile"]["display_name"] == "Legacy Profile"
    assert profiles["legacy_profile"]["description"] == ""


def test_resolve_profile_name_handles_stale_session_value() -> None:
    profiles = {"balanced": {"display_name": "Balanced"}, "cpu": {"display_name": "CPU"}}
    assert resolve_profile_name(profiles, "missing", fallback="balanced") == "balanced"
    assert resolve_profile_name(profiles, "cpu", fallback="balanced") == "cpu"
