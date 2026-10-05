from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.evaluation.medical_dataset import build_medical_evaluation_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Forrásolt magyar orvosi RAG evaluation dataset építése.")
    parser.add_argument("--questions", type=int, default=80, choices=range(50, 101), metavar="50-100")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "hungarian_medical")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "raw" / "hungarian_medical" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluations" / "medical_rag_eval.jsonl")
    args = parser.parse_args()

    dataset = build_medical_evaluation_dataset(
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        target_questions=args.questions,
    )
    print("=" * 78)
    print("MAGYAR ORVOSI RAG EVALUATION DATASET KÉSZ")
    print("=" * 78)
    print(f"Kérdések      : {len(dataset.items)}")
    print(f"Forráscikkek  : {len({item.source_id for item in dataset.items})}")
    print(f"Kimenet       : {args.output}")
    print(f"Összefoglaló  : {args.output.with_suffix('.summary.json')}")
    print()
    print("A kérdések és expected key factek kizárólag a lokálisan letöltött forráscikkekből készültek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
