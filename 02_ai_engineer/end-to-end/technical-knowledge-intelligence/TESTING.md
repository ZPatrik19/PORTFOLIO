# Testing Quick Start

The test system is offline-first so normal validation remains usable when Gemini quota is exhausted or no API credential is configured.

## Recommended local suite

Windows:

```bat
run_project.bat test offline
```

Linux/macOS:

```bash
./run_project.sh test offline
```

This runs deterministic unit, integration, evaluation, regression, robustness and smoke tests without consuming Gemini quota.

## Fast developer check

```bat
run_project.bat test quick
```

## Full local validation

```bat
run_project.bat test all-offline
```

This also includes local performance guardrails.

## Branch coverage

```bash
python -m pytest 06_tests -m "not live_gemini" \
  --cov=03_pipeline/tkip --cov-branch --cov-report=term-missing
```

Release measurement for v1.0.4:

```text
106 passed
1 live Gemini test deselected by default
0 offline failures
61% branch-aware core coverage
```

## Live Gemini

```bat
run_project.bat test live
```

or:

```bash
./run_project.sh test live
```

The live profile is explicit because it makes a real external API call and may consume quota/billing.

## Test layers

| Layer | Purpose |
|---|---|
| unit | isolated deterministic behavior |
| integration | multiple components working together |
| evaluation | metric/evaluation correctness |
| regression | stable/golden behavior gates |
| robustness | hostile, malformed or unusual input |
| performance | local latency guardrails |
| smoke | lightweight application/API contracts |
| live_gemini | external provider connectivity only |

## Detailed documentation

- `06_tests/README.md`
- `09_docs/testing/TESTING_STRATEGY.md`
- `09_docs/testing/TEST_COMMANDS.md`
- `09_docs/testing/TEST_MATRIX.md`
- `09_docs/testing/AI_EVALUATION_METRICS.md`
- `09_docs/testing/FIXTURES_AND_TEST_DATA.md`
- `09_docs/testing/REGRESSION_AND_GOLDEN_SETS.md`
- `09_docs/testing/LIVE_GEMINI_TESTING.md`
- `09_docs/testing/HOW_TO_WRITE_TESTS.md`
- `09_docs/testing/TROUBLESHOOTING_TESTS.md`
- `09_docs/testing/BENCHMARKING_VS_TESTING.md`
