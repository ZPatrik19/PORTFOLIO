# Technical Audit — Pre-refactor baseline

## Scope

The audit covered repository structure, entrypoints, document ingestion, parsing, chunking, embeddings, indexes, retrieval, reranking, context construction, Gemini integration, tools, API/UI, telemetry, evaluation, tests, configuration, deployment readiness and repository hygiene.

## Initial architecture findings

### Strengths

- The project already had a complete AI-engineering lifecycle instead of a notebook-only demo.
- Private books were excluded from Git and a public demo corpus existed.
- Retrieval was explicitly split into BM25, dense search, RRF hybrid fusion and reranking.
- Pydantic response/request models, citation validation, guardrails, telemetry and offline evaluation were present.
- Tests were separated into unit/integration/evaluation/regression/robustness/performance/smoke/live API categories.

### Problems found

1. **Import architecture** — runtime depended on `PYTHONPATH=03_pipeline` and `sys.path.insert(...)` in UI/API/notebooks. This is fragile in Docker, notebooks, test runners and external tooling.
2. **God files** — `05_ui/app.py` was ~1,600 lines; `orchestration.py` mixed index lifecycle and request orchestration.
3. **Compressed code style** — several core modules used multiple statements per line and terse names that made review difficult.
4. **Configuration validation** — YAML was loaded as an unvalidated dictionary; invalid chunk overlap or malformed values failed later in the pipeline.
5. **Mutable model defaults** — several Pydantic list/dict fields used mutable literal defaults.
6. **Broad exception handling** — recoverable fallbacks existed, but a few broad exception paths hid diagnostics.
7. **External AI error semantics** — Gemini quota/auth/transient errors were not clearly separated at the API boundary.
8. **Telemetry resource lifecycle** — SQLite usage required explicit close/commit semantics for deterministic resource cleanup.
9. **Cross-platform parity** — Windows scripts were stronger than Linux/macOS support.
10. **Deployment gap** — no Docker/Kubernetes release layer matched the API/UI services.
11. **Notebook hygiene** — notebooks contained import path hacks and missing stable cell IDs.
12. **Release hygiene** — no automated secret/absolute-path/YAML/notebook audit existed.

## AI-specific audit

- Retrieval/generation are measured separately: good.
- Offline embeddings provide deterministic tests: good.
- The generated evaluation set is a **silver-label regression aid**, not a replacement for a manually curated golden set.
- Faithfulness/answer correctness/hallucination metrics are intentionally not fabricated when no semantic judge or human ground truth is available.
- Gemini live validation remains quota/credential dependent and must be isolated from deterministic tests.
- Multiple persistent chunking indexes are useful for retrieval experiments, but must never mix vector spaces in a single retrieval run.

## Refactor strategy

The refactor preserves algorithms and public behavior. Changes focus on package architecture, service boundaries, validation, readability, testing, portability, deployment and documentation. No working notebook, benchmark, retrieval strategy or user-visible capability is intentionally removed.
