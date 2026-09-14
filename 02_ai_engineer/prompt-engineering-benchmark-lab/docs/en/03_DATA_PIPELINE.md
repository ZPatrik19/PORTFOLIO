# Data Pipeline

## Sources
- `mock`: local synthetic Challenge Set for deterministic offline validation.
- `huggingface`: optional external support-ticket dataset for generalization checks.
- `sample`: tiny local dataset for development.
- UI CSV upload: user-supplied labelled evaluation set.

## Canonical schema
`sample_id` must be unique and non-empty, `text` must be non-empty, and `true_label` must belong to the six-label vocabulary. Challenge data may add `case_type`, `difficulty`, `scenario_id`, and notes.

## Split policy
Default synthetic source: 10,800 rows. Final holdout: 6,000. Development: 3,000. Few-shot library: 24. The fixed random seed is stored in `configs/benchmark.yaml`.

## Leakage controls
Development, holdout, and few-shot selections are disjoint; fine-tuning export uses development data only; duplicate IDs and unknown labels fail before inference; cached predictions are rejected if the underlying sample identity changes.

## Pilot sampling
Small runs use deterministic representative sampling over labels and scenarios rather than `head(n)`, reducing the risk of misleading all-easy pilot results.
