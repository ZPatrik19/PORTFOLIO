"""Train/evaluate only when the saved router is missing, stale or runtime-incompatible."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "03_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("TRAVEL_DATA_MODE", "local")

from travel_agent.agent import MLRouterTravelAgent
from travel_agent.evaluation.runner import evaluate_agent
from travel_agent.quality import audit_all, dataset_hash
from travel_agent.training import model_status, train_router


def _existing_quality_report_is_current(path: Path) -> bool:
    """Cheaply verify the saved quality report by hashes instead of rerunning the audit."""
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not all(report.get("quality_gates", {}).values()):
        return False
    files = report.get("files", {})
    if not files:
        return False
    raw = ROOT / "01_data" / "raw"
    for name, info in files.items():
        source = raw / name
        expected = (info or {}).get("sha256")
        if not source.exists() or not expected or dataset_hash(source) != expected:
            return False
    return True


def main() -> None:
    status = model_status()
    quality_report = ROOT / "06_results" / "data_quality" / "data_quality_report.json"

    if status.get("exists") and not status.get("stale"):
        print("ML router is already trained for the current dataset and runtime. Skipping retraining.")
        if not _existing_quality_report_is_current(quality_report):
            print("Data-quality report is missing or stale; generating the report only (no retraining).")
            report = audit_all(write_outputs=True)
            if not all(report["quality_gates"].values()):
                raise SystemExit("Data-quality gates failed.")
        else:
            print("Data-quality report is current. Skipping full audit.")
        return

    reason = status.get("reason", "dataset_changed")
    print(f"ML router preparation required: {reason}.")
    if reason == "runtime_version_mismatch":
        mismatches = status.get("runtime_mismatches", {})
        print("Saved model runtime differs from this environment:")
        for name, versions in mismatches.items():
            print(f" - {name}: trained={versions.get('trained')} current={versions.get('current')}")
        print("No package downgrade/upgrade is required. The router will be retrained once locally.")
    elif reason in {"model_metadata_missing", "model_load_failed"}:
        print("A legacy/incompatible model artifact was found. It will be replaced by a locally trained artifact.")

    if _existing_quality_report_is_current(quality_report):
        print("Existing data-quality report matches the current datasets. Skipping expensive full audit.")
        report = json.loads(quality_report.read_text(encoding="utf-8"))
    else:
        print("Data-quality report is missing/stale. Running full audit once before training...")
        report = audit_all(write_outputs=True)
        if not all(report["quality_gates"].values()):
            raise SystemExit("Data-quality gates failed. Training was stopped.")

    print("Training intent router in the current Python/scikit-learn environment...")
    metrics, _ = train_router(write_outputs=True)

    # A short post-training smoke benchmark verifies the freshly written artifact.
    agent = MLRouterTravelAgent(language="hu")
    with tempfile.TemporaryDirectory(prefix="travel-agent-smoke-") as tmpdir:
        _, summary = evaluate_agent(
            agent,
            ROOT / "01_data" / "benchmark" / "agent_tasks.json",
            Path(tmpdir),
            100,
        )
    print(json.dumps({
        "quality_gate_pass_rate": report["quality_gate_pass_rate"],
        "router_training": {
            "test_micro_f1": metrics.get("test_micro_f1"),
            "challenge_micro_f1": metrics.get("challenge_micro_f1"),
            "scikit_learn_version": metrics.get("training_runtime", {}).get("scikit_learn_version"),
            "python_version": metrics.get("training_runtime", {}).get("python_version"),
        },
        "quick_agent_evaluation": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
