from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "config"


class ProfileConfigError(RuntimeError):
    pass


def _humanize_profile_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def _load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise ProfileConfigError(f"Profile config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ProfileConfigError(f"No profiles defined in: {path}")

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in profiles.items():
        name = str(key)
        if not isinstance(value, dict):
            raise ProfileConfigError(f"Profile '{name}' must be a mapping in: {path}")
        profile = dict(value)
        # UI metadata is optional in configuration files.  Supplying safe defaults
        # here prevents one malformed/older profile from crashing the whole UI.
        profile.setdefault("display_name", _humanize_profile_name(name))
        profile.setdefault("description", "")
        normalized[name] = profile
    return normalized


def load_llm_profiles() -> dict[str, dict[str, Any]]:
    return _load_profiles(CONFIG_ROOT / "llm_profiles.yaml")


def load_faiss_profiles() -> dict[str, dict[str, Any]]:
    profiles = _load_profiles(CONFIG_ROOT / "faiss_profiles.yaml")
    for name, profile in profiles.items():
        profile.setdefault("output_dir", f"artifacts/indexes/{name}")
    return profiles


def load_cuda_profiles() -> dict[str, dict[str, Any]]:
    return _load_profiles(CONFIG_ROOT / "cuda_profiles.yaml")


def resolve_profile_name(profiles: dict[str, dict[str, Any]], requested: str | None, *, fallback: str) -> str:
    """Return a valid profile name without allowing stale session/config values to crash the UI."""
    if requested and requested in profiles:
        return requested
    if fallback in profiles:
        return fallback
    return next(iter(profiles))


def get_profile(profiles: dict[str, dict[str, Any]], name: str, *, kind: str) -> dict[str, Any]:
    try:
        return profiles[name]
    except KeyError as exc:
        valid = ", ".join(sorted(profiles))
        raise ProfileConfigError(f"Unknown {kind} profile '{name}'. Valid values: {valid}") from exc
