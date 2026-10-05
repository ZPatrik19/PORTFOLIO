# Measurement methodology

A projekt mérési rétege szándékosan külön kezeli a **minőséget**, a **késleltetést**, a **robosztusságot** és az **erőforrás-használatot**. Egy gyorsabb pipeline nem automatikusan jobb retrieval rendszer, és egy magas Recall@K sem mondja meg, hogy production terhelés alatt stabil-e a latency.

## Retrieval quality

A forrásolt evaluation dataseten a projekt több, egymást kiegészítő metrikát számol:

- `Recall@K`, `Precision@K`, `F1@K`, `Hit Rate@K`;
- `MRR`, `MRR@K`, `MAP@K`, `nDCG@K`;
- `R-Precision`;
- determinisztikus `Context Precision@K`;
- első releváns találat rangja;
- source diversity és duplicate ratio;
- no-hit és late-hit failure rate;
- Recall és nDCG bootstrap konfidenciaintervallum.

A több metrika oka, hogy más hibát látnak. Recall a coverage-et, MRR az első jó találat helyét, nDCG a top ranking minőségét, R-Precision pedig a releváns halmaz méretéhez igazított pontosságot mutatja.

## Grounded answer quality

A generálási evaluation nem igényel külső LLM-judge szolgáltatást. A projekt reprodukálható, forrásalapú jeleket használ:

- citation accuracy és sentence-level citation coverage;
- source coverage;
- key-fact coverage a forrásból épített reference adaton;
- context-utilization lexikális proxy;
- válaszredundancia;
- answer/context token mennyiség;
- repair-rate és deterministic fallback-rate;
- key-fact coverage bootstrap CI.

Ezek nem helyettesítik az emberi szakmai review-t, de alkalmasak regressziók és pipeline-változások összehasonlítására.

## Latency és serving

A benchmarkok warm-up után mért mintákból dolgoznak. A fontosabb mutatók:

- mean, P50, P90, P95 és P99 latency;
- standard deviation és coefficient of variation (CV);
- requests/s és tokens/s;
- TTFT;
- CPU RAM és ahol elérhető, GPU VRAM;
- bootstrap 95% confidence interval az átlagos latency körül.

A CV különösen hasznos: nemcsak azt mutatja, hogy gyors-e a pipeline, hanem azt is, mennyire szóródik a válaszidő.

## Concurrency/load test

A `Measurement Lab` és a `scripts/run_measurement_suite.py` ugyanazon retrievert több concurrency szinten képes mérni. Ez nem production load generator, hanem lokális engineering benchmark, amellyel látható, mikor kezd nőni a tail latency.

Példa:

```bash
python scripts/run_measurement_suite.py \
  --retriever hybrid \
  --repeats 5 \
  --warmup 3 \
  --concurrency 1 2 4 8
```

## Query robustness

A projekt determinisztikus stressztesztet is tartalmaz. Ugyanazt a kérdést casing-, punctuation-, whitespace- és ékezetváltozatokkal futtatja, majd méri:

- Top-K Jaccard overlap;
- prefix ranking stability;
- Top-1 találat megtartási arányát.

Ez nem nyelvi paraphrase benchmark, hanem olcsó regressziós teszt arra, hogy egy apró inputváltozás indokolatlanul átírja-e a retrieval eredményt.

## Statisztikai A/B összehasonlítás

Két konfigurációt ugyanazon kérdésekre érdemes mérni. A projekt páros bootstrap összehasonlítást tartalmaz, amely visszaadja:

- `B - A` átlagos deltát;
- bootstrap confidence intervalt;
- empirikus bootstrap támogatást arra, hogy B kedvezőbb A-nál;
- relatív változást;
- páros standardizált effect-size proxy-t, ha a szórás értelmezhető.

Az empirikus bootstrap támogatás **nem p-érték**. A cél az, hogy ne egyetlen zajos átlag alapján döntsünk.

## Failure analysis

A diagnosztikai réteg géppel értelmezhető failure code-okat ad, például:

- `retrieval.no_hit`;
- `retrieval.late_hit`;
- `retrieval.duplicates`;
- `retrieval.low_source_diversity`;
- `retrieval.latency_budget`;
- `generation.invalid_citation`;
- `generation.citation_gap`;
- `generation.key_fact_gap`;
- `generation.low_context_use`;
- `generation.latency_budget`.

Ez segít megkülönböztetni, hogy a végső válasz azért rossz, mert nem találtunk jó evidence-et, rosszul rangsoroltunk, túl zajos lett a context, vagy a generation réteg nem használta fel megfelelően a bizonyítékokat.

## CPU és CUDA

A repository nem tárol hardverfüggetlen „győztes” benchmarkszámot. CUDA gyorsítás embeddingnél és Cross-Encoder rerankingnál megfelelő workload mellett látványos lehet; FAISS GPU Linux/WSL2 környezetben külön mérhető. A tényleges gyorsulás függ többek között a GPU-tól, VRAM-tól, driver/runtime verziótól, batch size-tól, candidate pooltól és corpus mérettől.

A korrekt összehasonlítás szabálya:

1. azonos dataset és config hash;
2. azonos query-set;
3. warm-up a mérés előtt;
4. több ismétlés;
5. P95/P99 és CV az átlag mellett;
6. CPU és CUDA tényleges device ellenőrzése;
7. saját hardveren újrafuttatott eredmény.
