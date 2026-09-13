# Code reference and repository audit

This document answers two practical questions: which file owns each responsibility, and why each repository component remains.

This document maps the working Python components to their responsibilities so the repository can be read layer by layer rather than as a flat file list.

## `03_src/travel_agent/models.py`
Shared runtime dataclasses. `AgentRun` represents one complete agent execution and `ToolCallRecord` represents one traced tool call. UI, evaluation and analytics use the same structures.

## `config.py`
Environment-based configuration for agent mode, model, step limit and default language.

## `data_store.py`
Shared local CSV loading/cache layer used by tools.

## `presets.py`
Canonical source for 30 preset scenarios, including ID, category, difficulty, HU/EN text and expected tools.

## `agent/heuristics.py`
Entity and constraint extraction: canonical cities, duration, budgets, ratings, cuisine/category filters and lexical intent cues.

## `agent/offline_agent.py`
Deliberately limited rule-based baseline retained for comparison.

## `agent/plan_execute_agent.py`
Inspectable deterministic planner/executor based on explicit `PlanStep` objects.

## `agent/ml_router_agent.py`
Supervised intent-router wrapper. Loads the persisted model, aggregates clause-level probabilities, applies high-precision lexical guardrails, then delegates structured argument extraction to deterministic parsers.

## `agent/openai_agent.py`
Real LLM tool-calling loop. Receives schemas from the registry, executes returned function calls, sends tool outputs back to the model, and stops within a bounded `max_steps` loop.

## `agent/prompts.py`
System-level instructions for the OpenAI execution path.

## `agent/service.py`
Methodology factory used by programmatic and CLI execution.

## `tools/base.py`
Common tool contract connecting Pydantic input models to validation and OpenAI-compatible schemas.

## `tools/registry.py`
Central execution boundary. Registers all eight tools, exports schemas, validates inputs, times calls and normalizes errors.

## Individual tools

- `weather.py` — local/auto/live weather providers;
- `currency.py` — local/live FX conversion;
- `hotels.py`, `restaurants.py`, `attractions.py` — inventory filtering/ranking;
- `transport.py` — transit products and trip-cost estimates;
- `location.py` — canonical city metadata;
- `calculator.py` — safe AST arithmetic without unrestricted `eval()`.

## `training/router_training.py`
Corpus loading, TF-IDF/SGD training, per-label threshold selection, metrics, joblib persistence and runtime compatibility metadata.

## `quality/data_quality.py`
Dataset hashes, linguistic normalization, duplicate/pattern diversity, split leakage and entity-name diversity audits.

## `evaluation/metrics.py`
Tool-selection and argument-level metric calculations.

## `evaluation/runner.py`
Ground-truth benchmark execution and persistent case/summary outputs.

## `evaluation/plotting.py`
Reproducible benchmark figures.

## `usage/store.py`
SQLite persistence and operational analytics: p50/p95 latency, tool usage, common sequences, destination frequency, period trends and error breakdown.

## `08_ui/app.py`
Streamlit presentation layer. It intentionally calls shared core modules instead of reimplementing tool/agent behavior. The sidebar also exposes the shared Light/Dark analytics chart theme.

## `08_ui/chart_theme.py`
Shared visualization layer for all three analytics dashboards. It defines the high-contrast Light/Dark palettes, the 460 px standard chart height, axis/legend/hover/annotation styling, and the `render_plotly()` wrapper.

## `08_ui/project_statistics_dashboard.py`
Interactive Plotly dashboard for repository, dataset, routing, model and benchmark statistics.

## `08_ui/data_quality_dashboard.py`
Interactive quality-gate, query/entity-diversity, leakage and router-label diagnostics.

## `08_ui/live_statistics_dashboard.py`
Operational dashboard built from persistent SQLite usage history, including trends, tool reliability, latency and behavior views.

## Setup scripts

- `02_setup_check.py` — structure/runtime smoke checks;
- `04_generate_data.py` — deterministic base data generation;
- `05_upgrade_data_quality.py` — high-diversity corpus/entity stage;
- `06_check_dependencies.py` — no-install dependency compatibility check;
- `07_prepare_if_needed.py` — conditional audit/training/smoke evaluation.

## Runnable scripts

- `01_run_demo.py` — agent demo;
- `02_run_evaluation.py` — benchmark;
- `03_compare_methodologies.py` — baseline comparison;
- `04_run_live_openai_agent.py` — live OpenAI path;
- `05_call_tool.py` — direct tool invocation;
- `06_train_intent_router.py` — router training;
- `07_prepare_train_evaluate.py` — one-command pipeline;
- `08_validate_presets.py` — bilingual preset regression;
- `09_generate_project_statistics.py` — static project statistics.

---

## Audit objective

The final repository was reviewed as both software and a portfolio artifact. Every retained component should have a clear purpose: runtime execution, reproducibility, data, testing, measurement or documentation.

## Removed redundant artifacts

The following generated outputs were removed from `06_results` because they duplicated retained results or were temporary smoke-run artifacts:

- `notebook_eval/`
- `prepare_pipeline_eval/`
- `demo_transcript_en.txt`
- `demo_transcript_hu.txt`
- `evaluation_console.txt`
- `methodology_console.txt`
- `direct_tool_call.json`

The durable results already live under `rule_based/`, `plan_execute/`, `ml_router/`, `models/`, `data_quality/`, `preset_validation/` and `project_statistics/`.

Setup and UI smoke evaluations now use temporary directories, so future one-off validation runs do not pollute `06_results`.

## Why each top-level directory remains

- `00_setup/` — environment checks, dependency gating, data regeneration, quality upgrade and conditional training. Required for reproducibility.
- `01_data/` — tool datasets, training/challenge corpora and end-to-end benchmark. Required.
- `02_notebooks/` — analysis artifacts for data, tools, orchestration and evaluation. Not a runtime dependency, but intentionally retained for inspectability.
- `03_src/` — the actual Python package and core logic. Required.
- `04_scripts/` — thin CLI entry points over shared package modules. Required for automation and reproducibility.
- `05_tests/` — unit/integration/regression coverage. Required.
- `06_results/` — durable, interpretable outputs only. Required.
- `07_docs/` — targeted technical docs plus bilingual master documentation. Required.
- `08_ui/` — the primary user interface. Required.

## Root file decisions

- `README.md` stays as the GitHub entry point.
- `START_HERE_HU.md` and `START_HERE_EN.md` provide fast bilingual onboarding.
- `SETUP_AND_START_UI.bat` stays as an idempotent first-run launcher.
- `RUN_UI.bat` stays as the fast daily launcher with no install/train work.
- `pyproject.toml` is the canonical package/dependency definition.
- `requirements.txt` remains as a human-readable compatibility list; it is not the canonical source, so dependency edits must remain synchronized.
- `.env.example` stays as the configuration contract.
- `.gitignore` stays and explicitly excludes the local SQLite usage history.

## Code-structure findings

Core behavior lives in shared modules rather than being duplicated in Streamlit or CLI code. UI, scripts, tests and evaluations all call the same agent/tool implementation.

`ToolRegistry` remains the execution boundary for schema validation, timing and structured errors.

The weaker `rule_based` implementation remains intentionally as a baseline. `plan_execute` remains as an inspectable deterministic orchestration strategy. `ml_router` remains as the trainable routing experiment. `openai_direct` remains as the real LLM function/tool-calling path.

## Intentional duplication

### PNG + SVG + DOT diagrams
PNG supports convenient previews, SVG provides scalable documentation, and DOT is the reproducible diagram source.

### `pyproject.toml` + `requirements.txt`
Partially duplicate dependency declarations are retained deliberately for packaging plus simple environment inspection. `pyproject.toml` is authoritative.

### Executive summary + master documentation
`PROJECT_DESCRIPTION_*` is short. `PROJECT_DOCUMENTATION_*` is the comprehensive technical reference. They serve different audiences and reading depths.

## Components intentionally not added

Docker, Kubernetes, a standalone API server, a message queue, LangGraph and external observability infrastructure are not required by the current project scope. Adding them now would increase infrastructure complexity without improving the core tool-calling experiment. They become relevant only if the project moves toward production deployment.

## Audit conclusion

The final structure contains no major runtime folder retained purely for decoration. Redundant generated artifacts were removed; retained baselines, datasets, tests and documents each support reproducibility, comparison, debugging or measurable system behavior.

## Conclusion

The numbered top-level structure intentionally follows setup → data → notebooks → source → scripts → tests → results → docs → UI. Persistent artifacts provide reproducible or diagnostic value; one-off smoke outputs are excluded.
