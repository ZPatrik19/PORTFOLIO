#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [[ ! -x .venv/bin/python ]]; then
  echo '[INFO] A .venv nem talalhato. SETUP.sh indul...'
  ./SETUP.sh
fi

if [[ $# -eq 0 ]]; then
  exec .venv/bin/python scripts/infrastructure_cli.py menu balanced
fi
exec .venv/bin/python scripts/infrastructure_cli.py "$@"
