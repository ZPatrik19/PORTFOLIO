# Test Troubleshooting

## `429 RESOURCE_EXHAUSTED` / quota exceeded

Cause: live Gemini quota or rate limit.

This should only affect:

```bat
run_project.bat test live
```

Use offline validation instead:

```bat
run_project.bat test offline
```

## `GEMINI_API_KEY` missing

Normal tests do not need it.

Only the live profile requires a key.

## Tests are indexing my user-library documents

That should not happen in tests using `isolated_config`.

Integration tests use:
- empty temporary private-book directory;
- public demo corpus;
- temporary processed/index paths.

If a new integration test constructs `KnowledgePlatform(load_config())` directly, review it carefully.

## Test modifies my real index/logs

Use the `isolated_config` fixture instead of project config directly for write-heavy tests.

## Performance test fails on a slow machine

First run it alone:

```bat
run_project.bat test performance
```

The thresholds are intentionally generous. If it still fails, inspect algorithmic complexity before simply increasing the threshold.

## Marker error

The project uses `--strict-markers`. Every marker must be declared in `pyproject.toml`.

Current markers:

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

## Import error in tests

Run through the provided runner from project root:

```bat
run_project.bat test quick
```

The project configuration adds `03_pipeline` to the Python path.

## Need more detail from pytest

```bat
.venv\Scripts\python.exe 06_tests\run_suite.py offline --verbose
```
