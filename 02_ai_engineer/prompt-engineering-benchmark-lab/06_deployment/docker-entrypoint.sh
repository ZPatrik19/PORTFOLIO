#!/usr/bin/env sh
set -eu

if [ ! -f /app/01_data/processed/benchmark.csv ]; then
  echo "[INIT] Preparing local mock benchmark data..."
  python /app/05_scripts/02_prepare_data.py --source mock
fi

exec python -m streamlit run /app/05_scripts/11_ui_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501
