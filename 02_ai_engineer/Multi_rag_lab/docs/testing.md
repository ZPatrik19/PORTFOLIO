# Testing és quality gates

A projekt tesztstratégiája réteges. A cél nem az, hogy minden teszt ugyanazt az end-to-end folyamatot ismételje, hanem hogy a hibák gyorsan lokalizálhatók legyenek.

## Unit

```bash
pytest tests/unit -q
```

Fő területek: parsing/cleaning/chunking, embeddings/vector store, retrieval/reranking, evaluation metrikák, runtime/config, grounded generation guardok és UI view-model contractok.

## Integration

```bash
pytest tests/integration -q
```

Több komponensből álló RAG stratégiákat és pipeline-integrációt vizsgál.

## UI smoke

```bash
pytest tests/ui -q
```

Streamlit `AppTest` segítségével ellenőrzi, hogy a fő UI és az evaluation route uncaught exception nélkül elindul-e.

## Performance smoke

```bash
pytest tests/performance -q -m performance
```

Nem hardverbenchmarkot rögzít, hanem a latency/statistics mérési infrastruktúra működését ellenőrzi.

## Repository contract

```bash
python scripts/check_repository.py
```

Ellenőrzi a repository határait és reprodukálhatóságát: nincs generált output, cache, legacy package, machine-specific path vagy Streamlit-függőség a core backendben.

## Syntax / lint

```bash
python -m compileall -q rag_engine ui scripts tests
ruff check rag_engine ui scripts tests
```

A Ruff jelenlegi gate-je a kritikus szintaktikai és névfeloldási hibákra fókuszál. A cél stabil CI, nem automatikus stílus-átírás.

## Package és Docker

CI-ben külön épül:

```bash
python -m build
python -m twine check dist/*
docker build -t multi-rag-engineering-lab:ci .
```

## GPU

A GPU-tesztek külön markerrel és külön self-hosted CI workflow-val futnak. A standard CPU CI nem szimulál CUDA teljesítményt.
