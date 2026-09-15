# Project Refactor Report

## 1. Initial State

The project already contained a substantial working AI/RAG implementation: document parsing, chunking, local/Gemini embeddings, hybrid retrieval, reranking, context engineering, Gemini structured generation, tool calling, Streamlit, FastAPI, evaluation, telemetry and a layered pytest suite.

The main problem was not missing functionality; it was engineering debt accumulated while features were added iteratively. The most important symptoms were:

- runtime import hacks using `PYTHONPATH` / `sys.path.insert(...)`;
- an oversized Streamlit entrypoint and overloaded orchestration service;
- compressed one-line Python that was difficult to review;
- broad exception fallbacks that could hide why optional components failed;
- stale version-specific README material mixed with current behavior;
- Windows and Linux runner drift;
- incomplete separation between software tests, AI evaluation, regression and live-provider validation;
- deployment artifacts that existed but had not been reviewed as one coherent release.

The refactor therefore preserved the working algorithms and focused on architecture, maintainability, reproducibility, observability and release quality.

## 2. Identified Problems

### Architecture

- `05_ui/app.py` had grown into a God File.
- `KnowledgePlatform` mixed request lifecycle and index lifecycle responsibilities.
- UI/API/runtime scripts relied on package-path workarounds.
- architecture diagrams implied a `Streamlit → FastAPI → core` chain although the implementation actually used two parallel adapters over the same core package.

### Code quality

- multiple modules used compressed one-line statements;
- some optional fallbacks used `except Exception: pass`;
- mutable Pydantic defaults required safer factories;
- cache and telemetry resource lifecycle needed deterministic cleanup;
- several imports were unused after previous refactors.

### Testing/evaluation

- the suite was strong but the custom test runner exposed an import-path issue not seen in direct pytest runs;
- live Gemini checks had to be isolated from deterministic tests because quota/provider failures are external;
- semantic generation metrics needed to remain explicitly unmeasured when no valid judge/reference exists.

### Deployment/reproducibility

- package imports were not consistently install-based;
- Windows/Linux shell runners had drifted;
- Docker API/UI health checks needed different endpoints;
- Kubernetes secret handling and probe behavior required explicit documentation.

## 3. Architecture Changes

### Installable package

`tkip` is now an installable package through `pyproject.toml` and an editable development installation:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

Runtime code, notebooks, API and UI no longer depend on `sys.path` mutation.

### Index lifecycle extraction

`IndexService` owns primary index build/load state. `KnowledgePlatform` remains the request-level application service. This reduces orchestration responsibility without changing public behavior.

### UI decomposition

The Streamlit code is separated into:

- `app.py` — application bootstrap/navigation/workspace composition;
- `ui_rendering.py` — answer, evidence, pipeline, ranking and QA rendering;
- `ui_pages.py` — library/monitoring/workflow pages;
- `ui_experiments.py` — benchmarking and A/B experiment views.

### Parallel adapters

Streamlit and FastAPI are explicitly documented as parallel adapters over `tkip`. External users can use FastAPI; local Streamlit use does not incur an unnecessary HTTP hop.

## 4. Clean Code Improvements

The refactor preserved the algorithms while making core modules easier to review and test.

Key improvements include:

- clearer domain-specific names;
- public function/class docstrings;
- type hints on core boundaries;
- constants instead of repeated magic values where useful;
- shorter single-purpose helper functions;
- deterministic resource cleanup for SQLite connections;
- specific or logged exception handling instead of silent `pass` fallbacks;
- repository-relative `pathlib.Path` handling;
- centralized configuration validation and logging setup;
- dedicated domain exception hierarchy for config, index, provider, quota and authentication failures.

Modules receiving substantial cleanup include retrieval, chunking, context, citations, parsing, cache, embeddings, reranking, tools, public-doc download, feedback, figures and CLI logic.

## 5. Testing

The suite is intentionally split by purpose:

```text
unit
integration
evaluation
regression
robustness
performance
smoke
live_gemini
```

Latest deterministic release run:

```text
106 passed
1 live_gemini test deselected
0 failed
```

Branch-aware core coverage:

```text
60.6% measured (61% displayed)
```

Coverage is not presented as universally high. Core deterministic domain layers have substantially stronger coverage than external-provider, optional LangChain/Qdrant, multimodal and alternate-index branches.

Test profiles are available through the single platform runner (`run_project.bat/.sh test <profile>`), and the test infrastructure was itself regression-tested after fixing a repository-root helper import issue.

## 6. Deployment

### Local

- Windows `.bat` runners;
- Linux/macOS `.sh` runners;
- editable package installation;
- first-run and daily-start workflows.

### Docker

The project includes:

- `Dockerfile`;
- `.dockerignore`;
- `docker-compose.yml`;
- non-root runtime user;
- API healthcheck;
- Streamlit-specific Compose healthcheck override;
- bind-mounted local corpus/index/log directories;
- no baked secrets/private books.

### Kubernetes

Reference manifests include:

- namespace;
- ConfigMap;
- PVC;
- API/UI Deployments and Services;
- readiness/liveness probes;
- requests/limits;
- example secret only.

Docker and Kubernetes runtime execution could not be validated in the release environment because Docker and kubectl were not installed. This limitation is explicit rather than reported as a passed runtime test.

## 7. Documentation

The repository now includes dedicated documents for:

- technical audit;
- project overview;
- architecture;
- data pipeline;
- AI pipeline;
- evaluation;
- testing;
- deployment;
- troubleshooting;
- design decisions;
- detailed testing strategy and metrics;
- project refactor report;
- interview-ready project story.

The root README was rewritten as a single current v1.0 document rather than an accumulation of V3/V5/V6/V8 patch notes.

## 8. Validation Results

### Software/test validation

- Python compile: PASS
- Offline pytest suite: 106 passed, 0 failed
- Live Gemini test: excluded by default
- Branch-aware core coverage: 60.6% measured (61% displayed)
- Secret/absolute-path audit: 0 suspicious hits
- YAML parse audit: PASS
- 11 notebook JSON structures: PASS

### Notebook execution

All 11 notebooks were successfully executed during release validation. Ten were executed in the shared nbclient validation pass and the monitoring notebook was additionally verified through `jupyter nbconvert --execute` after an nbclient kernel-cleanup issue in the build environment.

### Packaging

- `tki --help`: PASS
- `python -m tkip.cli list-indexes`: PASS
- wheel build: PASS

### Demo ingestion

First build:

```text
documents: 7
chunks: 10
parsing failures: 0
embeddings computed: 10
```

Second build:

```text
parse cache hits: 7
embedding cache hits: 10
embeddings computed: 0
```

### Demo retrieval benchmark

| Method | Recall@5 | MRR | Hit Rate | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.892 | 0.803 | 1.000 | 0.815 |
| Hybrid + Reranking | 0.875 | 0.800 | 0.950 | 0.811 |
| Hybrid + Reranking + Rewrite | 0.875 | 0.800 | 0.950 | 0.811 |
| Hybrid | 0.875 | 0.726 | 1.000 | 0.752 |
| Dense local fallback | 0.763 | 0.560 | 1.000 | 0.585 |

### Deterministic generation evaluation

```text
citation correctness:        1.00
citation completeness:       1.00
no-answer accuracy:          0.90
structured output validity:  1.00
faithfulness:                not measured without valid judge/reference
answer correctness:          not measured without valid judge/reference
hallucination rate:          not measured without valid judge/reference
```

### API runtime smoke

A real Uvicorn process was started and returned HTTP 200 from:

- `/health`;
- `/ready`;
- `/library/statistics`;
- `/indexes`.

## 9. Known Limitations

- Gemini provider quality and availability depend on external quota/model access.
- The deterministic demo corpus is intentionally tiny and not a domain-quality benchmark.
- Local hashing embeddings are a reproducible fallback, not a production semantic baseline.
- Full Docker build/runtime was not executed in the release environment.
- Kubernetes manifests were parsed/static-tested but not applied to a live cluster.
- Ruff/mypy configuration exists, but the release environment did not contain those binaries and package installation was network-restricted; they are therefore not claimed as locally executed release gates.
- Streamlit source compiles and its smoke contracts are covered by tests, but the release environment did not contain the Streamlit package for a standalone browser-process smoke test.

## 10. Future Improvements

- human-curated domain RAG benchmark;
- semantic embedding benchmark on the full private corpus;
- provider-aware request budgeting and quota forecasting;
- OpenTelemetry/Prometheus export;
- stronger layout/OCR/equation parsing;
- live Docker/Kubernetes CI environment;
- expanded optional-integration coverage;
- controlled live Gemini regression project with stable quota/billing.


## 11. Repository Surface Simplification

The v1.0.4 release keeps the repository surface deliberately small while restoring a reliable first-run experience on Windows and Unix-like systems:

- one dependency source of truth: `pyproject.toml`;
- one convenience runtime file: `requirements.txt`;
- one thin first-run wrapper per platform: `setup.bat` / `setup.sh`;
- one daily command dispatcher per platform: `run_project.bat` / `run_project.sh`;
- the setup wrappers delegate to the shared `setup-run` command instead of duplicating installation logic;
- test and quality dependencies are installed lazily only for the corresponding commands;
- unused runtime packages (`scikit-learn`, `pypdf`) were removed and `httpx` moved to the development extra;
- the Streamlit UI now uses one HU/EN language state and clears language-bound generated content when the language changes;
- redundant tutorial/prose blocks were removed or moved behind collapsed diagnostic sections.
