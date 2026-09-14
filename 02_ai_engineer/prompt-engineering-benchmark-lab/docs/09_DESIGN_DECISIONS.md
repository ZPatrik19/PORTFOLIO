# Design Decisions

## Installable package instead of `sys.path` mutation
**Decision:** core code lives in `03_src/prompt_benchmark` and is installed editable for development.

**Reason:** imports must behave identically in scripts, notebooks, pytest, Docker and Kubernetes.

## Central repository paths
**Decision:** `prompt_benchmark.paths.PATHS` constructs repository-relative locations.

**Reason:** eliminates user-specific Windows paths and current-working-directory coupling in the core package.

## Separate provider adapters
**Decision:** each provider has its own module under `llm/providers/`.

**Reason:** the previous monolithic client module mixed six APIs and a large simulator, making changes and tests risky.

## Bounded retry, not blanket retry
**Decision:** retry only transient network/rate-limit/5xx-like failures with bounded exponential backoff.

**Reason:** authentication and malformed request failures do not improve by retrying and can waste quota.

## Strict validation before API execution
**Decision:** benchmark schema is validated before request execution.

**Reason:** a malformed upload should fail cheaply, before paid or rate-limited requests begin.

## Representative pilot sampling
**Decision:** `--limit` uses deterministic label/scenario-aware sampling instead of `head(n)`.

**Reason:** small pilots otherwise over-sample easy examples and can produce misleading 1.00 metrics.

## Streamlit health endpoint reused
**Decision:** Docker/Kubernetes use `/_stcore/health`.

**Reason:** adding a second API framework solely for a health endpoint would be unnecessary infrastructure.

## Setup does not run a full benchmark
**Decision:** bootstrap runs smoke/tests but not P0–P16 over the full 6000-row holdout.

**Reason:** environment installation must be predictable and inexpensive; benchmark execution is an explicit experiment.

## Mock provider remains clearly labelled
**Decision:** keep the deterministic prompt-sensitive simulator.

**Reason:** it enables offline testing of the entire experiment platform, but documentation and UI distinguish simulated metrics from real provider evidence.
