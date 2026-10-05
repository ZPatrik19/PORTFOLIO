from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_engine.evaluation.device_benchmark import benchmark_metadata
from rag_engine.evaluation.load import benchmark_retrieval_load
from rag_engine.evaluation.robustness import evaluate_retriever_robustness
from rag_engine.models import ChunkingConfig
from rag_engine.presets import HUNGARIAN_QUERY_PRESETS
from rag_engine.service import build_lab


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline-friendly measurement suite: load/latency variability + query robustness."
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=ROOT / "data" / "demo" / "magyar_rag_demo.md",
        help="Input document. Defaults to the versioned demo corpus.",
    )
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluations" / "measurement_suite.json")
    args = parser.parse_args()

    if not args.document.exists():
        raise FileNotFoundError(args.document)

    lab = build_lab(
        [args.document],
        chunking=ChunkingConfig(strategy="recursive", chunk_size=700, chunk_overlap=100),
        embedding_model="__offline_hashing_fallback__",
        embedding_device="cpu",
        vector_device="cpu",
        llm_provider="dummy",
        fallback_embedding=True,
    )
    retriever = {"dense": lab.dense, "bm25": lab.sparse, "hybrid": lab.hybrid}[args.retriever]
    queries = [preset.query for preset in HUNGARIAN_QUERY_PRESETS]

    load_results = [
        benchmark_retrieval_load(
            retriever,
            queries,
            concurrency=level,
            repeats=args.repeats,
            warmup_requests=args.warmup,
            top_k=args.top_k,
            candidate_count=args.candidate_count,
            bootstrap_samples=500,
        ).to_dict()
        for level in args.concurrency
        if level >= 1
    ]
    robustness = [
        {
            "label": preset.label,
            "topic": preset.topic,
            **evaluate_retriever_robustness(
                retriever,
                preset.query,
                top_k=args.top_k,
                candidate_count=args.candidate_count,
            ).to_dict(),
        }
        for preset in HUNGARIAN_QUERY_PRESETS
    ]

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "retriever": args.retriever,
            "embedding": "hashing-fallback",
            "vector_device": str(getattr(lab.vector_store, "device", "cpu")),
            "vector_backend": str(getattr(lab.vector_store, "backend_name", "unknown")),
            "top_k": args.top_k,
            "candidate_count": args.candidate_count,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "concurrency": args.concurrency,
        },
        "environment": benchmark_metadata("cpu"),
        "load": load_results,
        "robustness": robustness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
