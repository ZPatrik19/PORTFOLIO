# Data and data quality

This chapter explains what data the system uses, how it is generated, how template leakage is controlled, and which quality gates protect benchmark credibility.

## 1. Role of the data layer

The project uses two broad data classes: travel inventory/lookup data for tools, and labelled language data for intent-router training and evaluation. Row count alone is not treated as quality; explicit gates guard against template repetition and leakage.

## 2. Main datasets

- `cities.csv`: city metadata, currency, language, timezone, coordinates and cost profile.
- `hotels.csv`: 180,000 synthetic hotel records.
- `attractions.csv`: 90,000 synthetic POIs.
- `restaurants.csv`: 90,000 synthetic restaurant records.
- `transport.csv`: transport profiles.
- `weather_fallback.csv`: 72,000 offline weather fallback records.
- `fx_rates_fallback.csv`: offline FX fallback.
- `sample_user_queries.csv`: 90,000 varied sample queries.
- `intent_router_dataset.csv`: 240,000 labelled router examples.
- `intent_router_challenge.csv`: 36,000 indirect/noisy challenge examples.
- `benchmark/agent_tasks.json`: 22,500 end-to-end agent benchmark cases.

## 3. Train/validation/test split

The router corpus is split into 192,000 train, 24,000 validation and 24,000 held-out test rows. Normalized linguistic pattern overlap is currently zero across train/test, train/validation and test/validation. This is a stronger protection against template leakage than a naive random row split.

## 4. Data-quality gates

The audit checks normalized query diversity, split overlap and entity-name skeleton diversity. All 6 current gates pass. JSON/CSV summaries and plots are stored under `06_results/data_quality/`.

## 5. Meaning of synthetic data

Hotel, restaurant and attraction records exist for deterministic filtering/ranking experiments. The logic is real, but the inventory is not live booking availability. Tool outputs explicitly carry data notes where appropriate.

## 6. Reproducibility and hashes

The SHA-256 hash of the router training dataset is persisted with model metadata. If the dataset changes, setup marks the model stale and retrains it once.

## 7. Why row count is not enough

Earlier iterations contained many rows derived from too few language templates. The current audit therefore measures normalized pattern count, largest pattern share, exact duplicates, split leakage and entity-name diversity rather than relying on row count alone.

## Conclusion

The data layer is designed for reproducible tool execution and meaningful routing generalization tests, not merely for maximizing dataset size.
