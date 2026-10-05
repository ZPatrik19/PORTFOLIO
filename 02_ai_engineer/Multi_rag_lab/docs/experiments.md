# Experiments és reproducibility

A kísérleti eredmények az `artifacts/experiments/` alatt keletkeznek, de nem kerülnek verziózásra. A cél az, hogy minden futás a saját hardware/runtime környezetben reprodukálható legyen, ne pedig egy másik gép korábbi benchmarkeredményét örökölje.

Javasolt workflow:

```text
fixed dataset
   ↓
experiment config
   ↓
pipeline run
   ↓
metrics + runtime trace
   ↓
artifacts/experiments
```

Összehasonlításkor rögzítsd legalább:

- dataset és query set;
- chunking / embedding / retrieval / reranker konfiguráció;
- CPU vagy CUDA device;
- modellek;
- candidate count és Top-K;
- warm-up mód;
- latency/quality metrikák.

A default CI nem használ külső fizetős LLM API-t és nem állít GPU benchmarkeredményt. A valódi CUDA mérés külön self-hosted workflow-ban vagy lokálisan futtatható.
