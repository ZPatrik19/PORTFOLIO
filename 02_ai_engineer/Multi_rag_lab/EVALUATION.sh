#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [[ ! -x .venv/bin/python ]]; then ./SETUP.sh; fi
PY=.venv/bin/python

while true; do
  cat <<'MENU'
===============================================================================
  MULTI-RAG EVALUATION / BENCHMARK
===============================================================================
 [1] 80 kerdeses evaluation dataset ujraepitese
 [2] QUICK retrieval benchmark - 20 kerdes
 [3] FULL retrieval benchmark - 80 kerdes
 [4] QUICK RAG benchmark - Dummy LLM
 [5] RAG benchmark - Ollama/Qwen
 [6] QUICK teljes evaluation
 [7] Experiment Registry osszefoglalo
 [8] Measurement suite - latency + robustness
 [0] Vissza
MENU
  read -r -p 'Valassz: ' choice
  case "$choice" in
    1) "$PY" scripts/build_medical_eval_dataset.py --questions 80 ;;
    2) "$PY" scripts/evaluate_medical_rag.py --mode retrieval --questions 20 ;;
    3) "$PY" scripts/evaluate_medical_rag.py --mode retrieval --questions 0 ;;
    4) "$PY" scripts/evaluate_medical_rag.py --mode rag --questions 20 --llm-provider dummy ;;
    5) ./INFRASTRUCTURE.sh run balanced && "$PY" scripts/evaluate_medical_rag.py --mode rag --questions 20 --llm-provider ollama ;;
    6) "$PY" scripts/evaluate_medical_rag.py --mode all --questions 20 --llm-provider dummy ;;
    7) "$PY" scripts/experiment_registry.py summary ;;
    8) "$PY" scripts/run_measurement_suite.py --repeats 5 --warmup 3 --concurrency 1 2 4 8 ;;
    0) exit 0 ;;
    *) echo 'Ismeretlen valasztas.' ;;
  esac
  read -r -p 'ENTER a folytatashoz...' _
done
