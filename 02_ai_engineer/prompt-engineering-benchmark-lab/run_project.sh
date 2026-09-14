#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" - <<'PYVER'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 15) else 1)
PYVER
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python - <<'PYVER'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 15) else 1)
PYVER

.venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade
.venv/bin/python -c "import packaging" >/dev/null 2>&1 || .venv/bin/python -m pip install "packaging>=24.0"
.venv/bin/python 00_setup/00_dependency_manager.py --requirements requirements-dev.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation

if [[ ! -f .env ]]; then
  echo "[INFO] .env is optional for Mock/Ollama. Cloud keys can be entered in the UI."
fi

exec .venv/bin/python -m streamlit run 05_scripts/11_ui_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=false
