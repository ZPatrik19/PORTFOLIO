from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prompt_benchmark.constants import LABELS
from prompt_benchmark.paths import PATHS
from prompt_benchmark.data.prepare_benchmark import validate_and_clean


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    rows: int
    labels: int
    duplicate_rate: float
    mean_words: float
    min_class_size: int
    max_class_size: int
    has_case_types: bool
    hard_share: float | None


def normalize_benchmark_frame(frame: pd.DataFrame, *, prefix: str = "custom") -> pd.DataFrame:
    """Normalize ``text/label`` or ``text/true_label`` input into benchmark schema.

    Existing ``sample_id`` values are preserved when they are present and
    unique after cleaning. Otherwise deterministic IDs are generated.
    """
    work = frame.copy()
    if "true_label" in work.columns and "label" not in work.columns:
        work = work.rename(columns={"true_label": "label"})

    existing_ids: pd.Series | None = None
    if "sample_id" in work.columns:
        existing_ids = work["sample_id"].copy()
        work = work.drop(columns=["sample_id"])

    # Keep original row position so preserved IDs can be aligned after filtering.
    work = work.reset_index(names="__source_index")
    cleaned = validate_and_clean(work.drop(columns=["__source_index"]))

    # Reconstruct source-index alignment using text+label. Because cleaning drops
    # duplicate text/label pairs, the first matching source row is authoritative.
    source_lookup = work.drop_duplicates(subset=["text", "label"], keep="first").copy()
    source_lookup["text"] = source_lookup["text"].astype(str).str.strip()
    source_lookup["label"] = source_lookup["label"].astype(str).str.strip().str.lower()
    aligned = cleaned.merge(source_lookup[["__source_index", "text", "label"]], on=["text", "label"], how="left")

    if existing_ids is not None:
        candidate_ids = aligned["__source_index"].map(existing_ids).astype("string")
        if candidate_ids.notna().all() and candidate_ids.is_unique and candidate_ids.str.len().gt(0).all():
            cleaned.insert(0, "sample_id", candidate_ids.astype(str).tolist())

    cleaned = cleaned.rename(columns={"label": "true_label"})
    if "sample_id" not in cleaned.columns:
        cleaned.insert(0, "sample_id", [f"{prefix}_{i:05d}" for i in range(len(cleaned))])
    for col in ("case_type", "difficulty", "secondary_label", "scenario_id", "scenario_notes"):
        if col not in cleaned.columns:
            cleaned[col] = ""
    return cleaned


def profile_dataset(frame: pd.DataFrame, name: str = "dataset") -> DatasetProfile:
    """Return descriptive data-quality statistics without hiding duplicates.

    Duplicate rate is calculated on the raw logical ``text + label`` pairs
    before normalization removes them. This makes the profile useful for audit
    and upload validation rather than always reporting zero.
    """
    raw = frame.copy()
    if "true_label" in raw.columns and "label" not in raw.columns:
        raw = raw.rename(columns={"true_label": "label"})
    if {"text", "label"}.issubset(raw.columns) and len(raw):
        raw_text = raw["text"].astype(str).str.strip()
        raw_label = raw["label"].astype(str).str.strip().str.lower()
        duplicate_rate = float(pd.DataFrame({"text": raw_text, "label": raw_label}).duplicated().mean())
    else:
        duplicate_rate = 0.0

    normalized = normalize_benchmark_frame(frame, prefix="profile")
    counts = normalized["true_label"].value_counts()
    mean_words = float(normalized["text"].astype(str).str.split().str.len().mean()) if len(normalized) else 0.0
    has_case_types = bool("case_type" in normalized and normalized["case_type"].astype(str).str.len().gt(0).any())
    hard_share = None
    if "difficulty" in normalized and normalized["difficulty"].astype(str).str.len().gt(0).any():
        hard_share = float(normalized["difficulty"].astype(str).str.lower().eq("hard").mean())
    return DatasetProfile(
        name=name,
        rows=len(normalized),
        labels=int(counts.size),
        duplicate_rate=duplicate_rate,
        mean_words=mean_words,
        min_class_size=int(counts.min()) if len(counts) else 0,
        max_class_size=int(counts.max()) if len(counts) else 0,
        has_case_types=has_case_types,
        hard_share=hard_share,
    )


def balanced_sample(frame: pd.DataFrame, per_class: int, random_seed: int = 42, prefix: str = "suite") -> pd.DataFrame:
    normalized = normalize_benchmark_frame(frame, prefix=prefix)
    parts: list[pd.DataFrame] = []
    for label in LABELS:
        group = normalized[normalized["true_label"] == label]
        if len(group) < per_class:
            raise ValueError(f"Label '{label}' only has {len(group)} rows; {per_class} are required.")
        parts.append(group.sample(n=per_class, random_state=random_seed + LABELS.index(label)))
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    out["sample_id"] = [f"{prefix}_{i:05d}" for i in range(len(out))]
    return out


def prepare_hf_support_router_suite(
    output_path: str | Path = PATHS.benchmark_suites / "hf_support_router_300.csv",
    examples_path: str | Path = PATHS.benchmark_suites / "hf_support_router_few_shot.json",
    per_class: int = 50,
    few_shot_per_class: int = 2,
    random_seed: int = 42,
) -> tuple[Path, Path]:
    """Download the public Support-Ticket-Router dataset and create an external suite.

    This source is synthetic but independent from the bundled challenge generator.
    Its test split is used for evaluation while few-shot examples come from train.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required. Run RUN_UI.bat to update dependencies.") from exc

    dataset = load_dataset("cngchis/Support-Ticket-Router-12K-Cleaned")
    if "test" not in dataset or "train" not in dataset:
        raise ValueError("Expected train and test splits in the Hugging Face dataset.")

    test = dataset["test"].to_pandas()
    train = dataset["train"].to_pandas()
    test_norm = normalize_benchmark_frame(test[["text", "label"]], prefix="hf")
    train_clean = validate_and_clean(train[["text", "label"]])
    suite = balanced_sample(test_norm, per_class=per_class, random_seed=random_seed, prefix="hf")
    suite["case_type"] = "external_clean"
    suite["difficulty"] = "unknown"
    suite["scenario_notes"] = "Independent Hugging Face Support-Ticket-Router test example."

    few_shot: list[dict[str, str]] = []
    for label in LABELS:
        group = train_clean[train_clean["label"] == label]
        if len(group) < few_shot_per_class:
            raise ValueError(f"Not enough train examples for label {label}.")
        sample = group.sample(n=few_shot_per_class, random_state=random_seed + LABELS.index(label))
        few_shot.extend(sample[["text", "label"]].to_dict(orient="records"))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suite.to_csv(output, index=False)
    import json
    ep = Path(examples_path)
    ep.parent.mkdir(parents=True, exist_ok=True)
    ep.write_text(json.dumps(few_shot, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, ep



def representative_stratified_subset(
    frame: pd.DataFrame,
    n: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select a deterministic representative subset for pilot/API runs.

    The selection cycles through labels while rotating case types per label.
    This avoids the common failure mode where ``head(n)`` accidentally contains
    mostly easy examples and produces misleading 1.00 pilot metrics.
    """
    normalized = normalize_benchmark_frame(frame, prefix="pilot").reset_index(drop=True)
    if n >= len(normalized):
        return normalized.copy()
    if n <= 0:
        raise ValueError("n must be positive")

    rng = __import__("numpy").random.default_rng(random_seed)
    selected_indices: list[int] = []
    used: set[int] = set()

    per_label_cases: dict[str, list[str]] = {}
    per_group_rows: dict[tuple[str, str], list[int]] = {}
    for label in LABELS:
        label_frame = normalized[normalized["true_label"] == label]
        cases = [c for c in label_frame.get("case_type", pd.Series(dtype=str)).astype(str).unique() if c]
        if not cases:
            cases = [""]
        cases = list(cases)
        rng.shuffle(cases)
        per_label_cases[label] = cases
        for case in cases:
            if case:
                rows = label_frame.index[label_frame["case_type"].astype(str) == case].tolist()
            else:
                rows = label_frame.index.tolist()
            rng.shuffle(rows)
            per_group_rows[(label, case)] = rows

    counters = {key: 0 for key in per_group_rows}
    round_idx = 0
    while len(selected_indices) < n:
        progressed = False
        for label_idx, label in enumerate(LABELS):
            if len(selected_indices) >= n:
                break
            cases = per_label_cases[label]
            case = cases[(round_idx + label_idx) % len(cases)]
            key = (label, case)
            rows = per_group_rows[key]
            pos = counters[key]
            while pos < len(rows) and rows[pos] in used:
                pos += 1
            counters[key] = pos + 1
            if pos < len(rows):
                idx = rows[pos]
                selected_indices.append(idx)
                used.add(idx)
                progressed = True
        round_idx += 1
        if not progressed:
            break

    if len(selected_indices) < n:
        remaining = [i for i in normalized.index if i not in used]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[: n - len(selected_indices)])

    subset = normalized.loc[selected_indices[:n]].copy().reset_index(drop=True)
    return subset

def suite_summary(frame: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_benchmark_frame(frame, prefix="summary")
    counts = normalized["true_label"].value_counts().reindex(LABELS, fill_value=0)
    return {
        "rows": int(len(normalized)),
        "class_counts": counts.to_dict(),
        "mean_words": float(normalized["text"].astype(str).str.split().str.len().mean()),
        "hard_share": float(normalized["difficulty"].astype(str).str.lower().eq("hard").mean()) if "difficulty" in normalized else None,
        "case_types": sorted(x for x in normalized.get("case_type", pd.Series(dtype=str)).astype(str).unique() if x),
    }
