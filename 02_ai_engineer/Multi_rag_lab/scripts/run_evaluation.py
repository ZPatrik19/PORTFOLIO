from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.evaluation.runner import evaluate_retrieval


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval from a JSONL file with retrieved/relevant chunk IDs.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = []
    for row in rows:
        metrics.append(evaluate_retrieval(row["retrieved"], set(row["relevant"]), args.k).to_dict())
    if not metrics:
        print("No evaluation rows.")
        return 0
    averages = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
    print(json.dumps({"queries": len(metrics), "k": args.k, "metrics": averages}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
