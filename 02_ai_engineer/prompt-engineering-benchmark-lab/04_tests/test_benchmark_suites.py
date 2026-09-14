"""EN: Benchmark-suite normalization, profiling, balanced/representative sampling, identifier preservation, and duplicate-rate reporting.

HU: A benchmark suite normalizálását, profilozását, kiegyensúlyozott/reprezentatív mintavételét, azonosító-megőrzését és duplikációs statisztikáit ellenőrzi.
"""

import pandas as pd

from prompt_benchmark.data.benchmark_suites import normalize_benchmark_frame, profile_dataset, balanced_sample, representative_stratified_subset


def test_normalize_custom_csv_schema():
    """EN: Checks that uploaded/custom CSV data is normalized into the canonical benchmark schema.

    HU: Ellenőrzi, hogy a feltöltött/saját CSV adatok a kanonikus benchmark sémára normalizálódnak.
    """
    frame = pd.DataFrame({
        "text": ["cancel please", "charged twice", "app crashes", "api 429", "bad service", "upgrade plan"],
        "label": ["cancellation", "billing", "technical", "api", "complaint", "upgrade"],
    })
    out = normalize_benchmark_frame(frame, prefix="x")
    assert {"sample_id", "text", "true_label", "case_type", "difficulty"}.issubset(out.columns)
    assert len(out) == 6


def test_profile_reports_six_classes():
    """EN: Ensures dataset profiling reports all six supported intent classes.

    HU: Biztosítja, hogy a dataset profil mind a hat támogatott intent osztályt megjeleníti.
    """
    frame = pd.DataFrame({
        "text": [f"sample {i}" for i in range(6)],
        "label": ["api", "billing", "cancellation", "complaint", "technical", "upgrade"],
    })
    profile = profile_dataset(frame)
    assert profile.labels == 6
    assert profile.rows == 6


def test_balanced_sample_is_balanced():
    """EN: Verifies that balanced sampling preserves equal class representation.

    HU: Ellenőrzi, hogy a balanced sampling megtartja az egyenlő class-reprezentációt.
    """
    labels = ["api", "billing", "cancellation", "complaint", "technical", "upgrade"]
    rows = []
    for label in labels:
        for i in range(4):
            rows.append({"text": f"{label} {i}", "label": label})
    out = balanced_sample(pd.DataFrame(rows), per_class=2)
    counts = out["true_label"].value_counts()
    assert set(counts.tolist()) == {2}


def test_representative_subset_covers_all_labels_and_multiple_scenarios():
    """EN: Checks that pilot subsets cover every label and multiple scenario families instead of taking only easy leading rows.

    HU: Ellenőrzi, hogy a pilot subset minden labelt és több scenario családot lefed, nem csak az első könnyű sorokat veszi.
    """
    rows = []
    labels = ["api", "billing", "cancellation", "complaint", "technical", "upgrade"]
    cases = ["easy_clear", "ambiguous_boundary", "prompt_injection", "double_negation"]
    for label in labels:
        for case in cases:
            for i in range(3):
                rows.append({"text": f"{label} {case} {i}", "label": label, "case_type": case, "difficulty": "hard" if case != "easy_clear" else "easy"})
    subset = representative_stratified_subset(pd.DataFrame(rows), n=24, random_seed=42)
    assert len(subset) == 24
    assert subset["true_label"].nunique() == 6
    assert subset["case_type"].nunique() >= 3


def test_normalize_preserves_unique_sample_ids():
    """EN: Ensures valid user-provided sample identifiers survive normalization unchanged.

    HU: Biztosítja, hogy a valid felhasználói sample_id-k normalizáláskor változatlanok maradnak.
    """
    frame = pd.DataFrame({
        "sample_id": ["original-a", "original-b"],
        "text": ["cancel please", "charged twice"],
        "label": ["cancellation", "billing"],
    })
    out = normalize_benchmark_frame(frame, prefix="x")
    assert out["sample_id"].tolist() == ["original-a", "original-b"]


def test_profile_reports_raw_duplicate_rate_before_cleaning():
    """EN: Ensures duplicate-rate telemetry is calculated on raw input before deduplication can hide the problem.

    HU: Biztosítja, hogy a duplikációs arány a nyers adaton számolódjon, még a deduplikáció előtt.
    """
    frame = pd.DataFrame({
        "text": ["cancel please", "cancel please", "charged twice"],
        "label": ["cancellation", "cancellation", "billing"],
    })
    profile = profile_dataset(frame)
    assert profile.rows == 2
    assert profile.duplicate_rate == 1 / 3
