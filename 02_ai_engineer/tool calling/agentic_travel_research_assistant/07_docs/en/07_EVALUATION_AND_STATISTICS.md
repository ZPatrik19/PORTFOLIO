# Evaluation and statistics

This document explains how routing, arguments and execution are measured separately, and how static versus live statistics are generated.

The repository contains two distinct statistics layers:

1. **static project statistics** covering datasets, model evaluation, benchmark size and repository composition;
2. **live usage analytics** calculated from actual UI interactions stored in local SQLite history.

Regenerate the static report with:

```powershell
python 04_scripts/09_generate_project_statistics.py
```

Outputs are written to `06_results/project_statistics/`.

## 1. Repository statistics

The current project contains:

- 8 registered tools;
- 30 bilingual preset scenarios;
- 60 Hungarian/English preset surface forms;
- 22,500 end-to-end agent benchmark cases;
- 36 Python source files, about 2,800 source-code lines;
- 17 test files containing 56 automated test functions;
- 27 Markdown documentation files;
- 4 notebooks;
- 9 runnable scripts including the project-statistics generator.

## 2. Raw data layer

The ten primary CSV datasets contain **798,150 rows** in total, with 0 exact duplicate rows in the current generated data.

| Dataset | Rows |
|---|---:|
| `intent_router_dataset.csv` | 240,000 |
| `sample_user_queries.csv` | 90,000 |
| `hotels.csv` | 180,000 |
| `intent_router_challenge.csv` | 36,000 |
| `attractions.csv` | 90,000 |
| `restaurants.csv` | 90,000 |
| `weather_fallback.csv` | 72,000 |
| `cities.csv` | 60 |
| `transport.csv` | 60 |
| `fx_rates_fallback.csv` | 30 |

Missing values in optional router argument columns are intentional; a weather-only query does not need hotel or restaurant constraints.

## 3. Intent-router statistics

Dataset split:

- train: 192,000;
- validation: 24,000;
- held-out test: 24,000.

Saved metrics:

- validation micro-F1: **0.7602**;
- validation macro-F1: **0.7539**;
- held-out test micro-F1: **0.7826**;
- held-out test macro-F1: **0.7846**;
- test Hamming loss: **0.1314**.

Feature model:

```text
word TF-IDF 1–3 grams
+
Unicode normalization
+
One-vs-Rest SGD logistic classifier
```

## 4. Data-quality statistics

Current quality gates: **6/6 PASS**.

The gates cover router linguistic diversity, sample-query diversity, normalized split-pattern overlap, and entity-name diversity for hotels, restaurants and attractions.

Current normalized split leakage:

- train ↔ test: 0;
- train ↔ validation: 0;
- validation ↔ test: 0.

## 5. Agent benchmark statistics

On 500 comparable benchmark cases:

| Method | Exact tool selection | Tool F1 | Argument accuracy | Task success | P95 latency |
|---|---:|---:|---:|---:|---:|
| rule-based | 19.6% | 46.7% | 36.6% | 16.6% | 17.66 ms |
| plan-execute | 60.6% | 80.4% | 74.0% | 47.6% | 27.47 ms |
| ML router | **88.2%** | **97.2%** | **95.0%** | **68.0%** | 32.25 ms |

The gap between tool F1 and task success is meaningful: selecting the correct capabilities does not guarantee full task completion if argument extraction or tool execution fails.

## 6. Preset validation

30 scenarios × 2 languages × 2 offline methodologies = 120 validation runs.

Saved results:

- `plan_execute`: 60/60 exact routes;
- `ml_router`: 60/60 exact routes;
- both: 60/60 successful runs.

Preset validation is a controlled regression suite, not a replacement for held-out evaluation.

## 7. Live usage analytics

The `Live Statistics` UI tab reads `06_results/usage/usage_history.sqlite3` and calculates metrics from actual use:

- total questions and tool calls;
- average tools/question;
- tool and run success rates;
- p50/p95 run and tool latency;
- multi-tool and no-tool rates;
- unique-question rate;
- custom/preset mix;
- average question/answer length;
- daily activity;
- latest 7 days versus previous 7 days;
- tool frequency;
- methodology usage;
- common tool sequences;
- top destination cities;
- language/data-mode/source distributions;
- error breakdown;
- recent question/answer history.

The dashboard is operational state, not a static screenshot, and changes after every stored run.

## 8. Generated statistics artifacts

`06_results/project_statistics/` contains both machine-readable sources and reproducible static plots. Important CSV/JSON artifacts include:

- `project_statistics.json` — full project summary;
- `dataset_statistics.csv` — scale, memory, missingness and duplicates;
- `query_statistics.csv` — query diversity and complexity;
- `entity_statistics.csv` — hotel/restaurant/attraction inventory quality;
- `tool_label_distribution.csv` — router label balance;
- `query_complexity_distribution.csv` — intents per query;
- `benchmark_complexity.csv` — expected tool-call depth;
- `benchmark_expected_tool_distribution.csv` — benchmark capability mix;
- `city_inventory_coverage.csv` — per-city inventory;
- `router_per_label_metrics.csv` — precision/recall/F1 by intent;
- `evaluation_statistics.csv` — methodology comparison;
- cuisine/category/weather distribution CSVs.

Saved PNGs are static snapshots of the same statistics: dataset rows/memory, query/entity diversity, intent balance, query and benchmark complexity, benchmark tool mix, city coverage, price/category distributions, router per-label metrics, methodology quality and latency. Static snapshots use a fixed high-contrast light style. The interactive Plotly dashboards use the same CSV/JSON sources and can switch dynamically between Light and Dark themes.

## Evaluation logic

Core metrics include exact tool-selection accuracy, tool precision/recall/F1, argument accuracy, task success, unnecessary tool-call rate, mean latency and P95 latency. Tool selection and task success are intentionally separate because correct routing can still be followed by argument or execution failure.

In the current 500-case methodology snapshot, the ML router reaches about 88.2% exact tool selection, 97.2% tool-F1, 95.0% argument accuracy and 68.0% task success. These are generated benchmark results, not production SLAs.

Live Statistics changes with the SQLite usage history, while Project Statistics can be regenerated from repository contents and saved benchmark artifacts.

## Conclusion

The project has two distinct statistical layers: evaluation measures system quality on reproducible cases, while live analytics describes actual usage patterns and operational behavior.

## Extended project statistics and diagnostic plots

The v1.5 statistics layer is designed to show more than a few aggregate counters. The project separates **scale**, **diversity**, **routing complexity**, **benchmark complexity**, and **end-to-end performance**. Running `python 04_scripts/09_generate_project_statistics.py` produces reproducible artifacts including:

- `dataset_statistics.csv`: rows, columns, missingness, memory footprint, and city coverage per dataset;
- `query_statistics.csv`: unique-query ratio, normalized linguistic diversity, query length, and multi-intent rate;
- `entity_statistics.csv`: hotel/restaurant/attraction name diversity, rating and price quantiles, and per-city inventory;
- `tool_label_distribution.csv`: routing-label balance;
- `query_complexity_distribution.csv`: number of intents per user query;
- `benchmark_complexity.csv`: expected tool calls per benchmark case;
- `benchmark_expected_tool_distribution.csv`: expected tool mix across the benchmark;
- `city_inventory_coverage.csv`: city-level hotel/restaurant/POI coverage.

The generated plots are derived from these files rather than hard-coded illustration values. Separate figures cover dataset scale, memory footprint, language diversity, entity-name diversity, intent balance, query complexity, benchmark workflow complexity, benchmark tool distribution, city inventory coverage, methodology quality, and runtime latency.

### Why these metrics matter

A larger row count is not automatically a better dataset. Two hundred thousand rows generated from twenty underlying templates can still have poor information content. For that reason, this project reports normalized linguistic-pattern diversity, duplicate counts, entity-name diversity, and split leakage in addition to raw scale. Likewise, an agent benchmark should not be judged by one average accuracy value: workflow depth, capability balance, unnecessary calls, argument quality, and latency trade-offs are all relevant.

## Data-scaling decision

The high-cardinality inventory tables are scaled by roughly 10×: 180,000 hotels, 90,000 attractions, 90,000 restaurants, and 72,000 weather fallback records. Reference/dimension tables such as `cities.csv`, the FX fallback table, and city transport profiles are deliberately not duplicated tenfold because that would only create artificial duplicate reference data. Language/routing data is scaled separately by roughly 3×: 240,000 router examples, 36,000 challenge queries, 90,000 sample queries, and 22,500 end-to-end agent benchmark cases.

The purpose of the scaling is therefore not file size for its own sake, but greater search inventory and greater linguistic variation.
