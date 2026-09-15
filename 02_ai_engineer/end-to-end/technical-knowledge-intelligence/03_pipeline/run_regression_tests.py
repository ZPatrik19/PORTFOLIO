from __future__ import annotations

import json

from tkip.evaluation import generate_eval_dataset, run_retrieval_benchmark
from tkip.orchestration import KnowledgePlatform

MIN_RECALL_AT_5 = 0.10


def main() -> int:
    """Run the deterministic retrieval regression gate against the current index."""
    platform = KnowledgePlatform().load_index()
    samples = generate_eval_dataset(platform.chunks, 50, seed=7)
    benchmark = run_retrieval_benchmark(platform.retriever, samples)
    metrics = benchmark.mean(numeric_only=True).to_dict() if not benchmark.empty else {}
    print(json.dumps(metrics, indent=2))

    if float(metrics.get("recall@5", 0.0)) < MIN_RECALL_AT_5:
        raise SystemExit(f"Regression: Recall@5 below floor {MIN_RECALL_AT_5:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
