from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Any

import pandas as pd

from prompt_benchmark.paths import PATHS

LOGGER = logging.getLogger(__name__)

HISTORY_ROOT = PATHS.results / "history"


def make_run_id(provider: str, suite: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_provider = provider.replace("/", "-").replace(" ", "_")
    safe_suite = suite.replace("/", "-").replace(" ", "_")
    return f"{stamp}__{safe_provider}__{safe_suite}"


def run_dir(run_id: str) -> Path:
    return HISTORY_ROOT / run_id


def initialize_run(run_id: str, manifest: dict[str, Any], dataset: pd.DataFrame) -> Path:
    root = run_dir(run_id)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest.setdefault("run_id", run_id)
    manifest.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    manifest.setdefault("status", "running")
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset.to_csv(root / "dataset_used.csv", index=False)
    return root


def finalize_run(run_id: str, summary: pd.DataFrame, updates: dict[str, Any] | None = None) -> Path:
    root = run_dir(run_id)
    summary.to_csv(root / "summary.csv", index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"run_id": run_id}
    manifest.update(updates or {})
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["strategies_completed"] = summary.get("strategy", pd.Series(dtype=str)).astype(str).tolist()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def mark_failed(run_id: str, error: str) -> None:
    root = run_dir(run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": run_id}
    manifest.update({"status": "failed", "error": error, "completed_at": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(run_id: str) -> pd.DataFrame | None:
    path = run_dir(run_id) / "summary.csv"
    return pd.read_csv(path) if path.exists() else None


def load_raw(run_id: str, strategy: str) -> pd.DataFrame | None:
    path = run_dir(run_id) / "raw" / f"{strategy}.csv"
    return pd.read_csv(path) if path.exists() else None


def list_runs(provider: str | None = None, status: str | None = "complete") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not HISTORY_ROOT.exists():
        return pd.DataFrame()
    for manifest_path in HISTORY_ROOT.glob("*/manifest.json"):
        try:
            item = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Skipping unreadable run manifest %s: %s", manifest_path, exc)
            continue
        if provider and item.get("provider") != provider:
            continue
        if status and item.get("status") != status:
            continue
        summary_path = manifest_path.parent / "summary.csv"
        if summary_path.exists():
            try:
                summary = pd.read_csv(summary_path)
                if not summary.empty and "macro_f1" in summary:
                    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
                    item["best_strategy"] = best.get("strategy")
                    item["best_macro_f1"] = float(best.get("macro_f1", 0.0))
                    item["total_tokens"] = float(summary.get("total_tokens", pd.Series([0])).sum())
                    item["strategies"] = int(len(summary))
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                LOGGER.warning("Could not summarize run %s: %s", manifest_path.parent.name, exc)
        rows.append(item)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    sort_col = "completed_at" if "completed_at" in frame else "started_at"
    return frame.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


def latest_run_id(provider: str | None = None) -> str | None:
    runs = list_runs(provider=provider)
    if runs.empty:
        return None
    return str(runs.iloc[0]["run_id"])


def publish_latest(run_id: str, provider: str) -> None:
    """Keep compatibility files used by Dashboard/report code while preserving history."""
    root = run_dir(run_id)
    summary_src = root / "summary.csv"
    provider_root = PATHS.results / provider
    provider_root.mkdir(parents=True, exist_ok=True)
    if summary_src.exists():
        shutil.copy2(summary_src, provider_root / "benchmark_summary.csv")
    raw_dst = PATHS.results / "raw" / provider
    raw_dst.mkdir(parents=True, exist_ok=True)
    for src in (root / "raw").glob("*.csv"):
        shutil.copy2(src, raw_dst / src.name)
    for name in ("case_type_summary.csv", "difficulty_summary.csv"):
        src = root / name
        if src.exists():
            shutil.copy2(src, provider_root / name)
    (provider_root / "latest_run_id.txt").write_text(run_id, encoding="utf-8")
