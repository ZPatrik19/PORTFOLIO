# Testing Strategy

## Goal
The test suite protects experiment correctness, not only Python syntax. The most dangerous failures in this project are silent leakage, stale cache reuse, wrong provider payloads, malformed output parsing, misleading statistics, secret/config mistakes, and UI regressions that invalidate benchmark interpretation.

## Layers
- `unit`: isolated deterministic behavior.
- `integration`: several project components together, still without paid/live external requests.
- `smoke`: smallest complete offline benchmark path.
- `external`: reserved for opt-in live-provider tests requiring credentials/network.

## Commands
```bash
pytest
pytest -m unit
pytest -m integration
pytest -m smoke
pytest --cov=prompt_benchmark --cov-report=term-missing
```

Windows:
```text
run_tests.bat
```
Linux/macOS:
```bash
./run_tests.sh
```

## Current verified result
Validation on 2026-09-14:
- 76 tests passed, 0 failed.
- configured core-package coverage: 77%.
- exact module-by-module coverage is recorded in `16_VALIDATION_RESULTS.md`.

## What is mocked
Provider SDK/network boundaries are mocked in unit/integration tests. This verifies request construction, normalized response handling, credential guards, retry classification, and schema contracts without consuming quota.

## What is not falsely claimed
A live Gemini/Groq/OpenRouter/OpenAI request is not part of the default deterministic suite because it depends on user credentials, quota, model availability, and network access.

## Full catalog
See [13_TEST_CATALOG.md](13_TEST_CATALOG.md) for every test file and every test function with its engineering purpose.

> Canonical bilingual pair: `docs/en/06_TESTING.md` and `docs/hu/06_TESTING.md`.
