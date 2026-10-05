#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

PROFILE="${1:-balanced}"
if [[ ! -x .venv/bin/python ]]; then
  echo '[INFO] A .venv meg nincs kesz. SETUP.sh indul...'
  ./SETUP.sh
fi

if ! .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)' >/dev/null 2>&1; then
  echo '[INFO] A .venv nem Python 3.14. Ujraepites...'
  ./SETUP.sh force
fi

set +e
.venv/bin/python scripts/setup_state.py check >/dev/null 2>&1
state_rc=$?
set -e
if [[ $state_rc -eq 10 ]]; then
  echo '[INFO] A dependency/config fingerprint megvaltozott. SETUP frissites indul...'
  ./SETUP.sh
fi

[[ -f .env ]] || cp .env.example .env
export CACHE_ROOT="$PWD/.cache"
export TMP="$CACHE_ROOT/temp"
export TEMP="$CACHE_ROOT/temp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export SENTENCE_TRANSFORMERS_HOME="$CACHE_ROOT/sentence-transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export OLLAMA_MODELS="$PWD/infrastructure/llm/models"
mkdir -p "$TMP"

echo '[1/2] Infrastruktur ellenorzese / inditasa...'
./INFRASTRUCTURE.sh run "$PROFILE"

export LLM_PROVIDER=ollama
export OLLAMA_PROFILE="$PROFILE"
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export VECTOR_DEVICE=cpu
export EMBEDDING_DEVICE=auto
export RERANKER_DEVICE=auto
export HF_HUB_OFFLINE=1

echo '[2/2] Streamlit inditasa: http://localhost:8501'
.venv/bin/python scripts/open_when_ready.py http://localhost:8501 >/dev/null 2>&1 &
exec .venv/bin/python -m streamlit run ui/app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true
