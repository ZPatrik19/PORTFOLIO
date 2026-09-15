# Test Commands

## Windows — recommended interface

All commands are run from the repository root.

### Default offline suite

```bat
run_project.bat test
```

Equivalent to:

```bat
run_project.bat test offline
```

### Quick check

```bat
run_project.bat test quick
```

Runs:
- unit
- smoke

Recommended after small edits.

### Individual suites

```bat
run_project.bat test unit
run_project.bat test integration
run_project.bat test evaluation
run_project.bat test regression
run_project.bat test robustness
run_project.bat test performance
run_project.bat test smoke
```

### All local/offline tests

```bat
run_project.bat test all-offline
```

### Live Gemini

```bat
run_project.bat test live
```

Requires:

```env
GEMINI_API_KEY=...
```

and consumes real quota.

## Through the project runner

```bat
run_project.bat test
run_project.bat test-quick
run_project.bat test-evaluation
run_project.bat test-regression
run_project.bat test-robustness
run_project.bat test-performance
run_project.bat test live
```

## Direct pytest usage

```bat
.venv\Scripts\python.exe -m pytest 06_tests\unit -vv
```

By marker:

```bat
.venv\Scripts\python.exe -m pytest 06_tests -m unit
.venv\Scripts\python.exe -m pytest 06_tests -m "not live_gemini"
```

## Recommended workflow

During development:

```text
small code change
→ run_project.bat test quick
```

Before commit:

```text
run_project.bat test offline
```

Before release:

```text
run_project.bat test all-offline
→ retrieval benchmark
→ regression benchmark
→ optionally run_project.bat test live
```
