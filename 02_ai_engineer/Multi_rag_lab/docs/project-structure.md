# Project structure

## Döntés

A repository domain-alapú felépítést használ. A korábbi `src/rag_lab/...` és sok apró top-level backend package helyett a RAG életciklus hat nagy mérnöki egységre van bontva:

```text
rag_engine/
  ingestion/
  indexing/
  retrieval/
  generation/
  evaluation/
  platform/
```

Ez szándékosan kompromisszum a túl lapos és a túl fragmentált architektúra között.

## Felelősségi határok

| Domain | Felelősség |
|---|---|
| `ingestion` | források, parsing, cleaning, provenance, chunking |
| `indexing` | embedding, hashing fallback, FAISS CPU/GPU, NumPy vector store |
| `retrieval` | dense/BM25/hybrid, fusion, query transformation, reranking, context, RAG orchestration |
| `generation` | LLM providerek, promptok, grounded generation, validation, repair/fallback |
| `evaluation` | retrieval/generation metrics, benchmark, pipeline matrix, latency |
| `platform` | config, device/hardware detection, profiling, experiment registry |

A `chunking/` külön alpackage marad az `ingestion/` alatt, mert nyolc valódi algoritmust tartalmaz. Ennek további összevonása már rontaná az áttekinthetőséget.

## UI

```text
ui/
  pages/
  components/
  charts/
```

A Streamlit csak presentation layer. A `rag_engine` nem importálhat `streamlit` vagy `ui` modult; ezt a CI repository-contract checker automatikusan ellenőrzi.

## Data és artifacts

```text
data/demo/       verziózott kis bemenet
data/raw/        letöltött corpus
data/processed/  feldolgozott corpus

artifacts/indexes/      vector indexek
artifacts/evaluations/  evaluation outputok
artifacts/experiments/  experiment/benchmark outputok
```

A `raw`, `processed` és `artifacts/*` runtime outputok újragenerálhatók és nincsenek verziózva. Ez megakadályozza, hogy egy korábbi gép benchmarkeredménye vagy indexe „projektigazságként” bekerüljön a repositoryba.

## Repository-audit eredménye

A v0.35 audit után nincs strukturális blocker. Kifejezetten megmaradt néhány nagyobb modul/page, ahol a további darabolás nem adna valódi separation-of-concerns előnyt. A projekt preferált iránya: **kevesebb, jelentéssel bíró domain**, nem „egy class = egy mappa”.

A struktúra regresszióját a következő parancs ellenőrzi:

```bash
python scripts/check_repository.py
```

A checker tiltja többek között a legacy `src/rag_lab` visszatérését, cache/build metadata verziózását, generált benchmark/index fájlokat, hard-coded gépútvonalakat és a backend → Streamlit függést.
