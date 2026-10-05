#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CACHE_ROOT="$PWD/.cache"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export SENTENCE_TRANSFORMERS_HOME="$CACHE_ROOT/sentence-transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export OLLAMA_MODELS="$PWD/infrastructure/llm/models"

FORCE_SETUP=0
if [[ "${1:-}" == "force" ]]; then FORCE_SETUP=1; fi

printf '%s\n' '=============================================================================='
printf '%s\n' '  Multi-RAG Engineering Lab - Python 3.14 SETUP'
printf '%s\n' '=============================================================================='

PYTHON=""
for candidate in python3.14 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)' >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo '[ERROR] Python 3.14 nem talalhato.'
  echo 'Telepits Python 3.14-et, majd futtasd ujra: ./SETUP.sh'
  exit 1
fi

mkdir -p .cache/temp .cache/pip .cache/huggingface .cache/sentence-transformers .cache/torch \
  infrastructure/llm/models data/raw data/processed artifacts/indexes artifacts/evaluations \
  artifacts/experiments logs
[[ -f .env ]] || cp .env.example .env

NEW_VENV=0
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)' >/dev/null 2>&1; then
  echo '[INFO] A meglevo .venv nem Python 3.14; ujraepites.'
  rm -rf .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  echo '[1/9] Python 3.14 virtual environment letrehozasa...'
  "$PYTHON" -m venv .venv
  NEW_VENV=1
else
  echo '[1/9] Meglevo Python 3.14 .venv hasznalata.'
fi

NEED_INSTALL=$NEW_VENV
if [[ $FORCE_SETUP -eq 1 ]]; then NEED_INSTALL=1; fi
if [[ $NEED_INSTALL -eq 0 ]]; then
  set +e
  .venv/bin/python scripts/setup_state.py check >/dev/null 2>&1
  state_rc=$?
  set -e
  if [[ $state_rc -eq 10 ]]; then NEED_INSTALL=1; fi
fi

echo '[2/9] Dependency fingerprint ellenorzese...'
if [[ $NEED_INSTALL -eq 1 ]]; then
  echo '[3/9] pip / setuptools / wheel frissitese...'
  .venv/bin/python -m pip install --upgrade pip setuptools wheel
  echo '[4/9] Projekt + dev + FAISS CPU fuggosegek telepitese...'
  .venv/bin/python -m pip install -e '.[dev,cpu]'
else
  echo '[3/9] Build tooling frissites kihagyva.'
  echo '[4/9] Dependency install kihagyva.'
fi

echo '[5/9] Hugging Face embedding es reranker modellek elotoltese...'
.venv/bin/python scripts/prepare_runtime_assets.py --retries 5 --verify

echo '[6/9] Kritikus importok ellenorzese...'
.venv/bin/python -c "import streamlit, plotly, requests, bs4, torch, sentence_transformers, faiss; print('Streamlit / Torch / SentenceTransformers / FAISS: OK')"

echo '[7/9] CPU-kompatibilis regresszios tesztek...'
.venv/bin/python -m pytest -q -m 'not gpu'

echo '[8/9] NVIDIA / CUDA gyorsitas ellenorzese...'
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python scripts/infrastructure_cli.py cuda --install-cuda yes
else
  echo '      NVIDIA GPU/driver nem lathato - CPU runtime marad aktiv.'
fi

echo '[9/9] Setup state mentese...'
.venv/bin/python scripts/setup_state.py write >/dev/null

cat <<'TXT'

===============================================================================
  SETUP KESZ - Python 3.14 virtual environment mukodik
===============================================================================
Elso telepitesnel:
  ./INFRASTRUCTURE.sh setup
Napi hasznalathoz:
  ./RUN.sh
Benchmark / evaluation:
  ./EVALUATION.sh
TXT
