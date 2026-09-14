#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run ./run_project.sh once or create the environment first." >&2
  exit 1
fi
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
exec .venv/bin/python -m pytest 04_tests -v --cov=prompt_benchmark --cov-report=term-missing
