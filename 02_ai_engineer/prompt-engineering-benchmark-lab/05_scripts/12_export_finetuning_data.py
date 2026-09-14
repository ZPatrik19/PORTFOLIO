from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging

SYSTEM = "You are a precise SaaS customer-support routing classifier. Return only the correct label."


def to_example(row: pd.Series) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": str(row["text"])},
            {"role": "assistant", "content": str(row["true_label"])},
        ]
    }


def write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for _, row in frame.iterrows():
            fh.write(json.dumps(to_example(row), ensure_ascii=False) + "\n")


def main() -> None:
    configure_logging()
    source = PATHS.processed_data / "development.csv"
    if not source.exists():
        raise FileNotFoundError("Run 05_scripts/02_prepare_data.py first.")
    frame = pd.read_csv(source)
    n_classes = int(frame["true_label"].nunique())
    validation_size = max(n_classes, int(round(len(frame) * 0.2)))
    if len(frame) - validation_size < n_classes:
        raise ValueError("Development split is too small for a stratified fine-tuning train/validation export.")
    train, valid = train_test_split(
        frame,
        test_size=validation_size,
        random_state=42,
        stratify=frame["true_label"],
    )
    out = PATHS.data / "fine_tuning"
    write_jsonl(train, out / "sft_train.jsonl")
    write_jsonl(valid, out / "sft_validation.jsonl")
    print(f"SFT export complete: train={len(train)} validation={len(valid)}")
    print("The final benchmark.csv is NOT used for fine-tuning, preserving holdout integrity.")


if __name__ == "__main__":
    main()
