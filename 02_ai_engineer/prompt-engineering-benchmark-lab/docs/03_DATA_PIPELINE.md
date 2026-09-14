# Data Pipeline

## Sources
1. `mock` — local synthetic Challenge Set, designed for offline software validation and prompt-sensitivity demonstrations.
2. `huggingface` — optional external Support-Ticket-Router source for generalization checks.
3. `sample` — tiny repository sample for development.
4. UI CSV upload — user-provided benchmark data.

## Required schema
Before any benchmark starts, the core boundary validates:

```text
sample_id  unique non-empty identifier
text       non-empty ticket text
true_label one of api/billing/cancellation/complaint/technical/upgrade
```

Synthetic challenge metadata may additionally contain `case_type`, `difficulty`, `secondary_label`, `scenario_id`, and `scenario_notes`.

Malformed input fails before API requests are made.

## Split policy
The mock/default configuration creates disjoint sets with a fixed seed:

- final holdout: 1000 samples/class (6000 total);
- development: 500 samples/class (3000 total);
- few-shot library: 4 examples/class;
- raw synthetic source: 1800 samples/class (10800 total).

Few-shot examples never come from the final holdout.

## Leakage controls
- benchmark/development overlap is explicitly checked;
- sample IDs must be unique;
- few-shot rows are selected outside benchmark/development ranges;
- fine-tuning export uses development data only;
- cached predictions are rejected if the same ID maps to changed text/label data.

## Pilot sampling
A small `--limit` run no longer uses `head(n)`. It uses deterministic representative sampling across labels and case types to avoid misleading all-easy pilot scores.

## Reproducibility
`random_seed` is stored in `configs/benchmark.yaml`. pandas/NumPy sampling routines use deterministic seeds. External LLM responses can still be nondeterministic depending on provider/model implementation.
