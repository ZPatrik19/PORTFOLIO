# Setup és futtatás

## Runtime követelmény

A projekt fő Python runtime-ja **CPython 3.14 x64**. A launcherek kizárólag Python 3.14 környezetet fogadnak el.

## Windows

Első telepítés:

```bat
SETUP.bat
INFRASTRUCTURE.bat setup
RUN.bat
```

A négy felhasználói launcher szerepe:

- `SETUP.bat` – Python 3.14 `.venv`, runtime/dev/FAISS CPU dependency-k, `.env`, cache könyvtárak, import smoke, CPU-kompatibilis regressziós tesztek és setup fingerprint.
- `INFRASTRUCTURE.bat` – CUDA/PyTorch, FAISS, Ollama/Qwen, diagnosztika és opcionális corpus/index műveletek.
- `RUN.bat` – hiányzó/elavult környezet javítása, infrastruktúra runtime check, Streamlit indítás és böngészőnyitás.
- `EVALUATION.bat` – retrieval/RAG benchmark, experiment registry és measurement suite.

A setup idempotens: egy működő Python 3.14 `.venv`-et nem töröl, és változatlan dependency fingerprint esetén nem telepít újra mindent. `SETUP.bat force` kényszeríti a frissítést.

## Linux / WSL2

```bash
chmod +x SETUP.sh INFRASTRUCTURE.sh RUN.sh EVALUATION.sh
./SETUP.sh
./INFRASTRUCTURE.sh setup
./RUN.sh
```

A shell launcherek ugyanazt a felelősségszétválasztást követik, és Python 3.14-et várnak.

## CUDA és FAISS

Natív Windows alatt a stabil alap a FAISS CPU backend. A PyTorch-alapú embedding és Cross-Encoder reranker CUDA-val gyorsítható, ha a driver és a PyTorch CUDA runtime elérhető. A valódi FAISS GPU validáció külön GPU környezetben történik; lásd `docs/faiss-gpu.md`.

Az infrastructure menü:

```text
[1] Setup / javítás      CUDA + FAISS + Ollama/Qwen
[2] Runtime ellenőrzés   napi indítás előtti check
[3] Teljes diagnosztika  státusz + LLM smoke
[4] Orvosi corpus/index  letöltés + újraindexelés
[5] Csak CUDA javítás
[6] Csak FAISS smoke
[7] Csak Ollama/Qwen
```

## Modellek és corpus

A Hugging Face / Sentence Transformers modellek első tényleges használatkor töltődhetnek le. A teljes magyar orvosi corpus nincs a source repositoryban; reprodukálhatóan építhető:

```bash
python scripts/prepare_medical_corpus.py
```

## Docker

```bash
docker compose up --build
```

A Docker image Python 3.14 CPU baseline-t használ. GPU-specifikus validáció külön workflow/környezet.
