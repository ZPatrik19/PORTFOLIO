# Evaluation

A projekt külön méri a retrieval minőségét, a válaszgenerálást és a runtime teljesítményt.

## Retrieval

- Recall@K
- Precision@K
- F1@K
- Hit Rate
- MRR
- MAP@K
- nDCG@K
- source diversity

## Generation

- citation accuracy / coverage
- answer completeness
- context utilization
- question-part coverage
- deterministic fallback behavior

## Performance

- retrieval latency
- reranking latency
- generation latency
- total latency
- TTFT
- token/s
- throughput
- P50/P95/P99

## Pipeline matrix

A matrix ugyanazon dataseten változtatja például:

```text
chunking × retrieval × reranker × device × RAG strategy
```

Ezzel elkülöníthető, hogy egy minőség- vagy latency-változás mely komponenshez köthető.

## Eredmények értelmezése

A repository nem tárol „hivatalos” CPU/CUDA benchmarkszámot. A latency és throughput erősen függ a hardvertől, drivertől, modelltől, batch/candidate mérettől és attól, hogy a modell/index már warm állapotban van-e.

A helyes összehasonlítás ugyanazon gépen, azonos datasettel és konfigurációval történik. CUDA-kompatibilis embedding és Cross-Encoder workloadnál a gyorsulás nagyobb terhelés mellett jellemzően jól látható lehet, de a projekt ezt mérendő eredményként kezeli, nem előre rögzített konstansként.

```bash
python scripts/benchmark_devices.py --workload 500
```


## Bootstrap és robosztusság

A V36-tól a retrieval benchmark Recall és nDCG, a RAG benchmark key-fact coverage, valamint a latency átlag körül bootstrap confidence intervalt is számol. A külön Measurement Lab páros bootstrap A/B összehasonlítást, concurrency benchmarkot és query-perturbációs ranking-stability mérést ad. A részleteket lásd: [measurement.md](measurement.md).
