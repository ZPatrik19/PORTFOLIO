# Test Suite

This directory contains the automated validation layers for the Technical Knowledge Intelligence Platform.

The most important design rule is simple:

> **The default test suite must work without a Gemini API key, without external services, and without consuming API quota.**

Live Gemini validation is isolated in `live_gemini/` and is opt-in only.

## Recommended commands

On Windows, from the project root:

```bat
run_project.bat test quick
run_project.bat test offline
run_project.bat test evaluation
run_project.bat test regression
run_project.bat test robustness
run_project.bat test performance
run_project.bat test live
```

If you run only:

```bat
run_project.bat test
```

the `offline` profile is used.

## Test layers

| Folder | Question answered | External API? |
|---|---|---:|
| `unit/` | Does a small function/component behave correctly in isolation? | No |
| `integration/` | Do ingestion, indexing, retrieval and orchestration work together? | No |
| `evaluation/` | Are AI/retrieval metric calculations mathematically correct? | No |
| `regression/` | Did a previously stable behavior change unexpectedly? | No |
| `robustness/` | Does the system reject hostile or unusual inputs safely? | No |
| `performance/` | Did local latency regress catastrophically? | No |
| `smoke/` | Can the application/API start and answer basic health checks? | No |
| `live_gemini/` | Can the configured Gemini key/model currently make a real request? | **Yes** |

## Test profiles

### Quick developer check

```bat
run_project.bat test quick
```

Runs unit + smoke tests. Use this after small code changes.

### Recommended offline validation

```bat
run_project.bat test offline
```

Runs unit, integration, evaluation, regression, robustness and smoke tests.
This is the normal pre-commit / pre-push profile.

### Everything local

```bat
run_project.bat test all-offline
```

Also includes performance guardrails.

### Live Gemini validation

```bat
run_project.bat test live
```

This consumes real Gemini API quota and is intentionally separate.
See `09_docs/testing/LIVE_GEMINI_TESTING.md` before using it.

## Clean-code conventions used in tests

Tests follow these conventions:

1. Descriptive behavior-oriented test names.
2. Arrange → Act → Assert structure for non-trivial tests.
3. Small deterministic fixtures.
4. No dependency on user-library documents for normal tests.
5. No dependency on live Gemini for normal tests.
6. One main behavior per test.
7. Hand-calculable examples for metric validation.
8. Generous performance thresholds to avoid flaky hardware-specific failures.
9. Explicit markers (`unit`, `integration`, `evaluation`, etc.).
10. Failure messages should identify the actual broken behavior.

For the complete strategy, read `09_docs/testing/TESTING_STRATEGY.md`.
