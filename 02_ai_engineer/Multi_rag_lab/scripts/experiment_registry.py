from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.platform.registry import ExperimentRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-RAG Experiment Registry utility")
    parser.add_argument("command", choices=["summary", "list", "export"], nargs="?", default="summary")
    parser.add_argument("--db", type=Path, default=ROOT / "artifacts" / "experiments" / "experiments.sqlite3")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experiments" / "experiments_export.json")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    registry = ExperimentRegistry(args.db)
    if args.command == "summary":
        print(json.dumps(registry.summary(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "list":
        runs = registry.list_runs(limit=args.limit)
        for run in runs:
            print(
                f"{run['created_at']} | {run['status']:9} | {run['benchmark_type']:9} | "
                f"{str(run['config_hash'])[:12]} | {run['run_id']}"
            )
        return 0
    path = registry.export_json(args.output)
    print(f"Export elkészült: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
