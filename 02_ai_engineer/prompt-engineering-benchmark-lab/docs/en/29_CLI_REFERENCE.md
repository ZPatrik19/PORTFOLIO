# CLI Reference

## Installed command
```bash
prompt-benchmark --help
prompt-benchmark prepare-data --source mock
prompt-benchmark benchmark --provider mock --strategy p0_zero_shot --limit 12
prompt-benchmark ablation --provider mock
prompt-benchmark parameter-sweep --provider mock
prompt-benchmark report --provider mock
prompt-benchmark smoke
prompt-benchmark ui --port 8501
```

Workflow-specific flags are forwarded to the existing scripts, preserving a single source of truth for benchmark argument parsing.

## Direct scripts
The numbered scripts in `05_scripts/` remain available for education/debugging. Prefer the installed CLI or UI for normal use; use direct scripts when reproducing a specific pipeline step.
