from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.platform.config import load_settings
from rag_engine.evaluation.medical_benchmark import (
    corpus_paths_from_manifest,
    rows_to_dicts,
    run_rag_benchmark,
    run_retrieval_benchmark,
)
from rag_engine.evaluation.medical_dataset import load_medical_evaluation_dataset
from rag_engine.platform.registry import ExperimentRegistry


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Magyar orvosi RAG benchmark: retrieval + end-to-end RAG.")
    parser.add_argument("--mode", choices=["retrieval", "rag", "all"], default="all")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts" / "evaluations" / "medical_rag_eval.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "raw" / "hungarian_medical" / "manifest.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "hungarian_medical")
    parser.add_argument(
        "--questions", type=int, default=20, help="Benchmarkhoz használt első N kérdés; 0 = minden kérdés."
    )
    parser.add_argument("--chunking", default="fixed,recursive,sentence,paragraph,structure-aware")
    parser.add_argument("--retrievers", default="dense,bm25,hybrid-rrf,hybrid-weighted")
    parser.add_argument("--rerankers", default="none,lexical")
    parser.add_argument(
        "--rag-strategies", default="baseline,lexical,hybrid,reranked,dense-reranked,compression,corrective"
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vector-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reranker-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--llm-provider", choices=["dummy", "ollama"], default="dummy")
    parser.add_argument("--ollama-profile", choices=["low_memory", "balanced", "extended_context"], default="balanced")
    parser.add_argument("--context-budget", type=int, default=1800)
    parser.add_argument("--context-profile", default="balanced")
    parser.add_argument("--prompt-profile", default="professional")
    parser.add_argument("--hashing", action="store_true", help="Gyors offline benchmark hashing embeddinggel.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluations" / "medical_benchmark.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "experiments" / "experiments.sqlite3")
    parser.add_argument(
        "--no-registry", action="store_true", help="Ne mentse a futást az SQLite Experiment Registry-be."
    )
    parser.add_argument("--notes", default="", help="Opcionális megjegyzés a benchmark futáshoz.")
    args = parser.parse_args()

    items = load_medical_evaluation_dataset(args.dataset)
    if not items:
        raise SystemExit("Az evaluation dataset hiányzik. Futtasd: python scripts/build_medical_eval_dataset.py")
    if args.questions > 0:
        items = items[: args.questions]
    paths = corpus_paths_from_manifest(args.manifest, args.raw_dir)
    if not paths:
        raise SystemExit("A magyar orvosi korpusz hiányzik. Előbb futtasd a korpusz letöltését.")

    settings = load_settings()
    embedding_model = "hashing" if args.hashing else settings.multilingual_embedding_model
    kwargs = dict(
        paths=paths,
        items=items,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=embedding_model,
        embedding_device=args.embedding_device,
        vector_device=args.vector_device,
        top_k=args.top_k,
        candidate_count=args.candidates,
        fallback_embedding=args.hashing,
    )
    config = {
        "mode": args.mode,
        "questions": len(items),
        "chunking": _csv_list(args.chunking),
        "retrievers": _csv_list(args.retrievers),
        "rerankers": _csv_list(args.rerankers),
        "rag_strategies": _csv_list(args.rag_strategies),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "top_k": args.top_k,
        "candidate_count": args.candidates,
        "embedding_model": embedding_model,
        "embedding_device": args.embedding_device,
        "vector_device": args.vector_device,
        "reranker_device": args.reranker_device,
        "fusion": args.fusion,
        "rrf_k": args.rrf_k,
        "dense_weight": args.dense_weight,
        "llm_provider": args.llm_provider,
        "ollama_profile": args.ollama_profile,
        "context_budget": args.context_budget,
        "context_profile": args.context_profile,
        "prompt_profile": args.prompt_profile,
        "hashing": args.hashing,
    }
    output: dict[str, object] = {
        "dataset": str(args.dataset),
        "questions": len(items),
    }

    registry = None if args.no_registry else ExperimentRegistry(args.registry)
    run_id: str | None = None
    started = time.perf_counter()
    if registry is not None:
        run_id, config_hash = registry.create_run(
            benchmark_type=args.mode,
            config=config,
            dataset_path=args.dataset,
            questions=len(items),
            notes=args.notes or None,
        )
        output["run_id"] = run_id
        output["config_hash"] = config_hash

    try:
        if args.mode in {"retrieval", "all"}:
            retrieval_rows = run_retrieval_benchmark(
                **kwargs,
                chunking_strategies=_csv_list(args.chunking),
                retrieval_modes=_csv_list(args.retrievers),
                rerankers=_csv_list(args.rerankers),
                reranker_device=args.reranker_device,
                fusion=args.fusion,
                rrf_k=args.rrf_k,
                dense_weight=args.dense_weight,
            )
            output["retrieval"] = rows_to_dicts(retrieval_rows)
            if registry is not None and run_id is not None:
                registry.add_retrieval_results(
                    run_id,
                    retrieval_rows,
                    embedding_model=config["embedding_model"],
                )

        if args.mode in {"rag", "all"}:
            rag_rows = run_rag_benchmark(
                **kwargs,
                rag_strategies=_csv_list(args.rag_strategies),
                chunking_strategy="recursive",
                llm_provider=args.llm_provider,
                ollama_profile=args.ollama_profile,
                max_context_tokens=args.context_budget,
                context_profile=args.context_profile,
                prompt_profile=args.prompt_profile,
            )
            output["rag"] = rows_to_dicts(rag_rows)
            if registry is not None and run_id is not None:
                registry.add_rag_results(
                    run_id,
                    rag_rows,
                    chunking="recursive",
                    embedding_model=config["embedding_model"],
                    context_budget=args.context_budget,
                    vector_device=args.vector_device,
                )

        duration_ms = (time.perf_counter() - started) * 1000
        if registry is not None and run_id is not None:
            registry.complete_run(run_id, duration_ms=duration_ms)
        output["duration_ms"] = duration_ms
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        if registry is not None and run_id is not None:
            registry.fail_run(run_id, exc, duration_ms=duration_ms)
        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nBenchmark eredmény: {args.output}")
    if run_id:
        print(f"Experiment run_id: {run_id}")
        print(f"Registry: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
