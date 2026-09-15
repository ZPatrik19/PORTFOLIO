#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
CMD="${1:-run}"
ARG="${2:-}"

ensure_python() {
  if [[ -x "$PY" ]]; then
    return 0
  fi
  echo "[INFO] Creating .venv..."
  python3 -m venv .venv
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    "$PY" -m ensurepip --upgrade
  fi
}

install_environment() {
  ensure_python
  echo "[1/2] Project runtime dependencies..."
  "$PY" -m pip install -e . --no-build-isolation

  echo "[2/2] Environment validation..."
  "$PY" 00_setup/01_check_environment.py
  "$PY" -c 'import tkip; print("[OK] tkip", tkip.__version__)'
}

ensure_env() {
  ensure_python
  if ! "$PY" -c 'import tkip, streamlit, fastapi' >/dev/null 2>&1; then
    echo "[INFO] Environment is incomplete; repairing it..."
    install_environment
  fi
}

ensure_dev_environment() {
  ensure_env
  if ! "$PY" -c 'import pytest, pytest_cov, ruff, mypy, httpx' >/dev/null 2>&1; then
    echo "[INFO] Installing developer and test tooling..."
    "$PY" -m pip install -e '.[dev]' --no-build-isolation
  fi
}

case "$CMD" in
  setup)
    install_environment
    "$PY" 00_setup/02_check_gemini_api.py || echo "[INFO] Gemini validation is optional; local retrieval remains available."
    "$PY" 00_setup/03_check_optional_services.py
    ;;
  setup-run)
    "$0" setup
    "$0" run
    ;;
  run)
    ensure_env
    [[ -f 01_data/indexes/numpy/chunks.json ]] || "$0" ingest
    echo "[INFO] Starting Streamlit UI..."
    "$PY" 05_scripts/launch_app.py --ui-only
    ;;
  public)
    ensure_env; "$PY" -m tkip.cli download-public ;;
  ingest|index)
    ensure_env; "$PY" -m tkip.cli ingest ;;
  index-variants)
    ensure_env; "$PY" -m tkip.cli index-variants --strategies fixed recursive semantic ;;
  list-indexes)
    ensure_env; "$PY" -m tkip.cli list-indexes ;;
  benchmark)
    ensure_env; "$PY" -m tkip.cli benchmark ;;
  evaluate)
    ensure_env; "$PY" 03_pipeline/run_generation_evaluation.py ;;
  failure)
    ensure_env; "$PY" 03_pipeline/generate_failure_report.py ;;
  feedback)
    ensure_env; "$PY" 03_pipeline/export_feedback_regression.py ;;
  figures)
    ensure_env; "$PY" -m tkip.cli figures ;;
  test)
    ensure_dev_environment
    PROFILE="${ARG:-offline}"
    [[ "$PROFILE" == "live" ]] && echo "[WARNING] Live tests consume real Gemini quota."
    echo "[INFO] Test profile: $PROFILE"
    "$PY" 06_tests/run_suite.py "$PROFILE"
    ;;
  quality)
    ensure_dev_environment
    "$PY" -m ruff check .
    "$PY" -m ruff format --check .
    "$PY" -m mypy 03_pipeline/tkip
    "$PY" 06_tests/run_suite.py offline
    ;;
  api)
    ensure_env; exec "$PY" -m uvicorn 04_api.main:app --host 0.0.0.0 --port 8000 ;;
  ui)
    ensure_env; exec "$PY" -m streamlit run 05_ui/app.py --server.address 0.0.0.0 --server.port 8501 ;;
  all)
    "$0" setup
    [[ -f 01_data/reference_docs/demo_python.md ]] || "$0" public
    "$0" ingest
    "$0" test
    "$0" benchmark
    "$0" evaluate
    "$0" failure
    "$0" feedback
    "$0" figures
    "$0" run
    ;;
  *)
    echo "[ERROR] Unknown command: $CMD" >&2
    echo "Usage: ./run_project.sh [run|setup|setup-run|ingest|benchmark|test [profile]|quality|api|ui|all]" >&2
    exit 2
    ;;
esac
