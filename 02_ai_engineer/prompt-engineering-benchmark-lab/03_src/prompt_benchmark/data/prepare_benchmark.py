from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prompt_benchmark.constants import LABELS

DATASET_NAME = "cngchis/Support-Ticket-Router-12K-Cleaned"
OPTIONAL_METADATA_COLUMNS = (
    "case_type",
    "difficulty",
    "secondary_label",
    "scenario_id",
    "scenario_notes",
)


def load_huggingface_dataframe(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    """Download and combine all Hugging Face splits into one normalized dataframe."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    dataset = load_dataset(dataset_name)
    frames: list[pd.DataFrame] = []
    for split_name in dataset.keys():
        split = dataset[split_name].to_pandas()
        if {"text", "label"}.issubset(split.columns):
            frames.append(split[["text", "label"]])
    if not frames:
        raise ValueError("No dataset split contains both 'text' and 'label' columns.")
    return pd.concat(frames, ignore_index=True)


def load_sample_dataframe(path: str | Path) -> pd.DataFrame:
    """Load any local CSV with text,label columns (sample or synthetic mock data)."""
    return pd.read_csv(path)


def validate_and_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, normalize strings, remove blanks/duplicates and unknown labels.

    Synthetic benchmark metadata is retained when present. Public datasets that
    only expose ``text`` and ``label`` continue to work unchanged.
    """
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected columns {required}; got {set(frame.columns)}")

    keep = ["text", "label", *[c for c in OPTIONAL_METADATA_COLUMNS if c in frame.columns]]
    cleaned = frame[keep].copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned = cleaned[(cleaned["text"] != "") & cleaned["label"].isin(LABELS)]
    cleaned = cleaned.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    for column in OPTIONAL_METADATA_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("").astype(str)
    return cleaned


def prepare_splits(
    frame: pd.DataFrame,
    benchmark_per_class: int,
    development_per_class: int,
    few_shot_per_class: int,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Create disjoint balanced benchmark, development and few-shot subsets."""
    frame = validate_and_clean(frame)
    rng = np.random.default_rng(random_seed)
    benchmark_parts: list[pd.DataFrame] = []
    development_parts: list[pd.DataFrame] = []
    few_shot: list[dict[str, str]] = []
    required = benchmark_per_class + development_per_class + few_shot_per_class

    for label in LABELS:
        label_frame = frame[frame["label"] == label].reset_index(drop=True)
        if len(label_frame) < required:
            raise ValueError(
                f"Label '{label}' has {len(label_frame)} rows, but {required} are required."
            )
        order = rng.permutation(len(label_frame))
        label_frame = label_frame.iloc[order].reset_index(drop=True)
        b = label_frame.iloc[:benchmark_per_class].copy()
        d0 = benchmark_per_class
        d1 = d0 + development_per_class
        d = label_frame.iloc[d0:d1].copy()
        f = label_frame.iloc[d1:d1 + few_shot_per_class]
        benchmark_parts.append(b)
        development_parts.append(d)
        # Few-shot examples only need text+label in the prompt library.
        few_shot.extend(f[["text", "label"]].to_dict(orient="records"))

    benchmark = pd.concat(benchmark_parts, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    development = pd.concat(development_parts, ignore_index=True).sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
    benchmark.insert(0, "sample_id", [f"bench_{i:04d}" for i in range(len(benchmark))])
    development.insert(0, "sample_id", [f"dev_{i:04d}" for i in range(len(development))])
    benchmark = benchmark.rename(columns={"label": "true_label"})
    development = development.rename(columns={"label": "true_label"})

    overlap = set(benchmark["text"]) & set(development["text"])
    if overlap:
        raise AssertionError("Benchmark and development sets overlap.")
    return benchmark, development, few_shot


def save_prepared_data(
    benchmark: pd.DataFrame,
    development: pd.DataFrame,
    few_shot: list[dict[str, str]],
    output_dir: str | Path,
    examples_path: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_dir / "benchmark.csv", index=False)
    development.to_csv(output_dir / "development.csv", index=False)
    examples_path = Path(examples_path)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.write_text(json.dumps(few_shot, indent=2, ensure_ascii=False), encoding="utf-8")
