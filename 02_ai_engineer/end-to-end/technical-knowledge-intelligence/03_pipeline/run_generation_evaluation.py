from __future__ import annotations

import json

import pandas as pd

from tkip.config import resolve_path
from tkip.evaluation import evaluate_answer_record, generate_eval_dataset
from tkip.models import AskRequest
from tkip.orchestration import KnowledgePlatform

EVALUATION_SAMPLE_COUNT = 50
EVALUATION_SEED = 11


def main() -> int:
    """Run deterministic, quota-free generation/citation evaluation.

    This suite validates structured response behavior, citation resolution and
    abstention. Semantic faithfulness/correctness are deliberately not fabricated
    without a human label or an explicit judge model.
    """
    platform = KnowledgePlatform().load_index()
    samples = generate_eval_dataset(platform.chunks, EVALUATION_SAMPLE_COUNT, seed=EVALUATION_SEED)
    all_chunk_ids = {chunk.chunk_id for chunk in platform.chunks}

    rows: list[dict] = []
    for sample in samples:
        answer = platform.ask(AskRequest(question=sample["question"]))
        rows.append(
            {
                "question_id": sample["question_id"],
                "category": sample["category"],
                **evaluate_answer_record(
                    answer,
                    sample["expected_answer_available"],
                    all_chunk_ids,
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    output_dir = resolve_path(platform.cfg["paths"]["results"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "generation_deterministic_metrics.csv", index=False)

    measured_columns = [
        "citation_correctness",
        "citation_completeness",
        "no_answer_accuracy",
        "structured_output_validity",
    ]
    summary = {
        column: float(metrics[column].dropna().mean())
        for column in measured_columns
    }
    not_measured = "NOT_MEASURED_WITHOUT_LLM_JUDGE_OR_HUMAN_LABELS"
    summary.update(
        {
            "faithfulness": not_measured,
            "answer_correctness": not_measured,
            "hallucination_rate": not_measured,
        }
    )
    (output_dir / "generation_deterministic_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
