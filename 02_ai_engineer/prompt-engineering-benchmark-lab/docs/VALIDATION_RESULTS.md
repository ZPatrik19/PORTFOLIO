# Validation Results

Validation date: 2026-09-14

## Executed successfully in this environment
- `python -m pytest -q` → **76 passed, 0 failed**.
- `python -m pytest --cov=prompt_benchmark --cov-report=term-missing -q` → **77% configured core-package coverage**.
- Marker subsets verified separately: `unit` **53 passed**, `integration` **22 passed**, `smoke` **1 passed**.
- Test marker collection supports `unit`, `integration`, and `smoke` subsets.
- Python package imports and typed benchmark configuration load successfully.
- Offline provider boundaries are exercised with mocks rather than consuming cloud quota.

| Module area | Representative coverage |
|---|---:|
| prompt strategies | 94% |
| parsing | 100% |
| mock generator | 95% |
| path management | 81% |
| provider adapters | ~77–85% |
| benchmark runner | 74% |
| config | 85% |
| overall configured core package | **77%** |

## Environment-dependent checks
Live cloud-provider requests require user credentials/network/quota and are not counted as default passing tests. Docker image execution requires a Docker daemon; Kubernetes apply requires kubectl/cluster access. Ruff and mypy are configured and should be executed in CI/dev environments where the binaries are installed.

## Integrity statement
No mock score is presented as real LLM quality. No live provider, Docker runtime, or Kubernetes deployment is marked successful unless actually executed in an environment that supports it.

> Canonical bilingual pair: `docs/en/16_VALIDATION_RESULTS.md` and `docs/hu/16_VALIDATION_RESULTS.md`.
