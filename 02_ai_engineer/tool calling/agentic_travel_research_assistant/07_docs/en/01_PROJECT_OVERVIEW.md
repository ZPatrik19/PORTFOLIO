# Project overview — Agentic Travel Research Assistant

This document explains the problem the system solves, the user experience it provides, and how its major components fit together.

## 1. Objective

The **Agentic Travel Research Assistant** is a local-first tool-calling AI system that decomposes natural-language travel requests into structured subtasks, selects the required capabilities, validates arguments, executes tools, and synthesizes the resulting observations into a single response.

The project is not designed as a conventional chatbot. Its central engineering problem is **orchestration**: deciding which capabilities are required for a request, which arguments must be passed to them, and how the resulting tool executions should be coordinated.

A Streamlit browser application is the primary interface, so normal usage does not require manually typing JSON arguments in a terminal.

## 2. Example task

User request:

> I am going to Vienna for 3 days. Check the weather, find a hotel under 150 EUR, recommend restaurants and attractions, and show public transport options.

The system creates a route such as:

```text
get_weather
    ↓
search_hotels
    ↓
search_attractions
    ↓
search_restaurants
    ↓
get_transport_options
    ↓
structured observations
    ↓
final response + trace
```

Each call records its arguments, output, execution latency and success state.

## 3. Main components

### 3.1 Browser UI

`08_ui/app.py` provides seven working areas:

- **Chat** — custom or preset natural-language requests, built-in HU/EN question-writing guidance and complete tool traces;
- **Tool Explorer** — direct tool calls through forms;
- **Data Quality** — duplicate, linguistic-diversity, split-leakage and entity-name checks;
- **Train & Evaluate** — router retraining and benchmark execution;
- **Live Statistics** — persistent KPIs, time series, tool/methodology distributions and Q&A history calculated from real Chat usage;
- **Project Statistics** — reproducible dataset, model, benchmark and repository statistics;
- **Dataset Explorer** — inspection and filtering of the underlying CSV datasets.

The Chat view includes **30 heterogeneous bilingual preset scenarios**. Presets can be filtered by category and difficulty. Every preset also contains a ground-truth expected tool route so the catalogue is regression-testable.

### 3.2 Tool registry

A shared `ToolRegistry` is the single execution boundary for all capabilities. It is responsible for:

1. tool discovery;
2. OpenAI-compatible JSON schema generation;
3. Pydantic argument validation;
4. execution;
5. latency measurement;
6. structured error handling.

### 3.3 Tools

The system exposes eight capabilities:

| Tool | Purpose |
|---|---|
| `get_location_info` | country, currency, language, timezone and local cost profile |
| `get_weather` | multi-day weather forecast |
| `convert_currency` | FX conversion |
| `search_hotels` | hotel filtering and ranking |
| `search_attractions` | attraction filtering and ranking |
| `search_restaurants` | restaurant filtering and ranking |
| `get_transport_options` | local transport passes and estimated trip cost |
| `calculate` | safe dependent arithmetic |

Hotel, restaurant and attraction tools operate on reproducible local synthetic inventories. Weather and FX support `local`, `auto` and `live` provider modes.

## 4. Routing and agent methodologies

### `plan_execute`

An inspectable deterministic planner based on lexical, regex and morphology-aware rules. It creates a structured plan and extracts tool arguments. This is the offline reproducible baseline.

### `ml_router`

A trained multi-label intent router. It combines word TF-IDF 1–3 gram representations with Unicode accent normalization with One-vs-Rest logistic SGD classifiers. Structured argument extraction remains deterministic, allowing routing quality and argument-extraction quality to be measured separately.

### `openai_direct`

An OpenAI tool/function-calling workflow. The LLM selects tools and arguments, while Python validates and executes all functions. Tool outputs are returned to the model until it emits another tool call or a final answer.

## 5. Data

The project separates operational travel inventories from routing/evaluation corpora.

### Travel inventories

- 180,000 hotels;
- 90,000 attractions;
- 90,000 restaurants;
- 60 cities;
- local transport profiles;
- multi-day weather fallback records;
- FX fallback records.

These inventories are **synthetic**. They are intended for reproducible filtering, ranking, routing and evaluation, not to represent live booking availability.

### Routing data

- 240,000 labelled bilingual intent-router examples;
- 36,000 indirect/noisy challenge queries;
- 90,000 sample user queries;
- 22,500 end-to-end agent benchmark cases.

Router split:

```text
192,000 train
 24,000 validation
 24,000 held-out test
```

The data-quality pipeline checks exact duplication, normalized linguistic patterns and train/validation/test pattern overlap.

## 6. ML intent router

Model pipeline:

```text
query
  ↓
word TF-IDF (1–2 grams)
  +
Unicode accent normalization + word TF-IDF 1–3 grams
  ↓
One-vs-Rest SGD logistic classifiers
  ↓
per-label probability thresholds
  ↓
capability set
```

Character n-grams improve robustness to Hungarian suffixes, missing accents and small spelling variations.

The serialized model stores the SHA-256 hash of its training dataset. If training data changes, the UI marks the model as stale and recommends retraining.

## 7. Data-quality methodology

An earlier generated corpus contained a high row count but too few underlying language templates. The current project explicitly detects and prevents that failure mode through quality gates.

Core checks include:

- exact duplicate queries;
- normalized linguistic-pattern count;
- largest template-family share;
- normalized train ↔ validation ↔ test overlap;
- entity-name diversity;
- model/dataset freshness.

Audit artifacts are written to `06_results/data_quality/`.

## 8. Evaluation

The system evaluates orchestration at multiple levels rather than scoring only the final natural-language response:

- Tool Selection Accuracy;
- Tool Precision;
- Tool Recall;
- Tool F1;
- Argument Accuracy;
- Task Success Rate;
- Unnecessary Tool-call Rate;
- Average Tool Calls;
- Mean Latency;
- P95 Latency.

Current saved router results:

- held-out test micro-F1: **0.783**;
- held-out test macro-F1: **0.785**;
- challenge micro-F1: **0.677**.

Current 500-case end-to-end agent benchmark:

- exact tool selection: **88.2%**;
- tool F1: **97.2%**;
- argument accuracy: **95.0%**;
- task success: **68.0%**;
- unnecessary tool-call rate: **3.31%**.

The lower challenge score is intentional and informative: indirect/noisy language remains a genuine generalization problem rather than being hidden behind template leakage.

## 9. Thirty preset scenarios

`travel_agent.presets` defines 30 different use cases, each with Hungarian and English wording. The catalogue includes:

- single-tool weather, hotel, restaurant, attraction and transport requests;
- currency conversion;
- destination metadata;
- multi-tool city planning;
- local trip budgets;
- budget requests without accommodation;
- Hungarian queries without accents;
- negated hotel requests;
- car-free transport planning;
- indirect weather + restaurant requests;
- complex requests requiring five or six tools.

Automated regression tests run every preset in both languages and verify the expected tool route as well as successful execution.

## 10. Reproducibility

Normal Windows setup:

```text
SETUP_AND_START_UI.bat
```

Later runs:

```text
RUN_UI.bat
```

Complete quality → training → evaluation pipeline:

```powershell
python 04_scripts/07_prepare_train_evaluate.py --limit 500
```

Tests:

```powershell
pytest -q
```

## 11. Limitations

- hotel, restaurant and attraction inventories are synthetic;
- `plan_execute` is not a general natural-language parser;
- the ML router predicts capabilities rather than performing end-to-end generative planning;
- local weather/FX fallbacks are reproducible fixtures, not current market data;
- live provider mode depends on network access;
- OpenAI orchestration requires an API key.

These limitations are kept explicit so reproducible offline evaluation is not confused with live travel data quality.

## Preset validation

All 30 bilingual scenarios can be regression-checked with both offline routing methodologies:

```powershell
python 04_scripts/08_validate_presets.py
```

The current saved validation contains 120 runs (30 scenarios × 2 languages × 2 methodologies). Both `plan_execute` and `ml_router` reproduce the expected route in 60/60 language-specific cases and all tool executions succeed. Results are stored under `06_results/preset_validation/`.

## Custom questions, persistent history and live usage statistics

Chat supports unrestricted natural-language questions in addition to presets. The UI provides HU/EN guidance for supported travel capabilities and useful argument details such as city, duration, budget, rating and preferences.

Every executed Chat interaction is written by `travel_agent.usage.UsageStore` to a local SQLite database. Storage is normalized into `interactions` and `tool_calls` tables. The live dashboard calculates KPIs, tool/methodology distribution, daily usage and recent question/answer history directly from these real runs.

Setup is intentionally idempotent: compatible dependencies are not upgraded or reinstalled, and the router is retrained only when the saved model is missing or stale according to the training-dataset hash.

## Conclusion

The project is a measurable tool-calling system with multiple routing strategies. Its core value is converting natural-language requests into validated, observable, structured execution rather than merely generating prose.
