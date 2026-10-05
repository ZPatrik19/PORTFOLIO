from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / ".cache"
HF_HOME = CACHE_ROOT / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
STATE_PATH = ROOT / "artifacts" / "models" / "runtime_models.json"


def configure_hf_environment() -> None:
    """Configure a project-local Hugging Face cache before Hub imports happen."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_ROOT / "sentence-transformers"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # Lower Xet range concurrency is more reliable on consumer/unstable connections.
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")


def runtime_model_ids(settings: Any) -> dict[str, str]:
    return {
        "embedding_default": str(settings.embedding_model),
        "embedding_multilingual": str(settings.multilingual_embedding_model),
        "embedding_e5": str(settings.e5_embedding_model),
        "reranker": str(settings.reranker_model),
    }


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _absolute_cached_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def resolve_cached_model(model_id: str) -> str:
    """Return a verified local snapshot path when setup already cached the model."""
    state = _load_state()
    models = state.get("models", {})
    if not isinstance(models, dict):
        return model_id
    for item in models.values():
        if not isinstance(item, dict) or item.get("repo_id") != model_id:
            continue
        raw_path = str(item.get("snapshot_path", ""))
        if not raw_path:
            continue
        path = _absolute_cached_path(raw_path)
        if path.exists():
            return str(path)
    return model_id


def model_cache_status() -> dict[str, Any]:
    state = _load_state()
    models = state.get("models", {}) if isinstance(state, dict) else {}
    ready = 0
    total = 0
    if isinstance(models, dict):
        for item in models.values():
            if not isinstance(item, dict):
                continue
            total += 1
            raw_path = str(item.get("snapshot_path", ""))
            if raw_path and _absolute_cached_path(raw_path).exists():
                ready += 1
    return {
        "state_path": str(STATE_PATH),
        "ready": ready,
        "total": total,
        "complete": total > 0 and ready == total,
    }
