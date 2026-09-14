# Mock Prompt-Sensitivity Benchmark és mérési rendszer

## Mi változott?

A `mock` provider már nem egy egyszerű keyword classifier, amely szinte minden prompttechnikára ugyanazt az eredményt adja.

A cél az, hogy API-kulcs nélkül is végig lehessen próbálni a **teljes benchmarkrendszert**, és a UI-ban látszódjanak a prompttechnikák közötti tipikus trade-offok.

Fontos: ez továbbra is **szimuláció**, nem valódi LLM benchmark. A README-ben és a UI-ban ezt mindenhol külön jelöljük.

## 8 valószerű ticket-szcenárió

A 10 800 soros mock dataset 6 intentet és 18 különböző case type-ot tartalmaz:

1. `easy_clear` – egyértelmű, explicit kérés;
2. `implicit_request` – az intent nincs szó szerint kimondva;
3. `ambiguous_boundary` – két kategória erős jelzése is megjelenik;
4. `multi_intent_primary` – több kérés, de csak az egyik a primary intent;
5. `noisy_typo` – elgépelés, rossz írásjelek, informális szöveg;
6. `long_context` – hosszú, irreleváns vagy korábbi kontextus;
7. `prompt_injection` – a ticketen belül szerepel egy utasításnak tűnő, de nem megbízható mondat;
8. `resolved_history` – régi, lezárt probléma és új aktuális kérés egyszerre.

Minden sorhoz tartozik:

- `case_type`;
- `difficulty`;
- `secondary_label`;
- `scenario_id`;
- `scenario_notes`.

Így a benchmark nem csak egyetlen összesített F1 értéket ad, hanem megmutatja, hogy **melyik prompttechnika milyen problématípuson segít**.

## Hogyan viselkedik a prompt-sensitive mock provider?

A simulator determinisztikusan modellezi a prompttechnikák tipikus hatását.

Példák:

- label definitions segítenek implicit és határeset kérdéseknél;
- few-shot javítja az implicit/noisy eseteket, de sok tokent fogyaszt;
- decision policy segíti a multi-intent és historical-context döntéseket;
- delimited input különösen erős a prompt-injection eseteken;
- contrastive few-shot a boundary eseteken segít;
- reasoning mode javítja a komplex eseteket, de latency/token overheadet ad;
- branch-and-vote nagyon robusztus, viszont három modellhívásnyi prompt/token/latency költsége lehet;
- schema/grammar constrained output főleg a kimeneti megbízhatóságot javítja;
- full advanced template általában erős, de jelentős input-token overheadet okoz.

A helyes/hibás döntést stabil SHA-256 alapú determinisztikus draw választja ki, ezért ugyanazzal a datasettel, prompttal és sampling beállítással a benchmark reprodukálható.

## Melyik szám valódi és melyik szimulált mock módban?

### Szimulált / becsült

- classification quality;
- token usage;
- latency.

A raw CSV egyértelműen jelzi:

- `token_source = estimated_mock`;
- `latency_source = simulated_mock`.

### Valódi API/Ollama futásnál

- prediction: valódi modellkimenet;
- token usage: provider/runtime usage mezője, ahol elérhető;
- latency: wall-clock API request idő;
- API error: valódi SDK/API hiba;
- output validity: valódi response parsing eredménye.

## A benchmark által számolt metrikák

### Quality

- Accuracy;
- Macro Precision;
- Macro Recall;
- Macro F1;
- Weighted F1;
- per-class Precision/Recall/F1;
- confusion matrix;
- bootstrap 95% confidence interval.

### Output reliability

- Invalid Output Rate;
- Output Contract Valid Rate;
- Invalid JSON Rate;
- JSON Grammar Valid Rate;
- API Error Rate.

### Token efficiency

- Mean Input Tokens;
- Mean Output Tokens;
- Mean Total Tokens;
- Total Input Tokens;
- Total Output Tokens;
- Total Tokens;
- Tokens per Correct Prediction.

### Latency

- Mean;
- Median;
- P50;
- P95;
- P99;
- Total sequential latency;
- sequential throughput (requests/sec).

### Cost

- mean estimated cost/request;
- estimated cost / 1000 requests;
- full benchmark estimated cost;
- cost / correct prediction.

A cost mindig a `configs/pricing.yaml` fájl alapján számolódik. Free-tier esetben ez **referencia/list-price equivalent** is lehet, nem feltétlen a tényleges számlázott összeg.

### Prompt complexity

- prompt characters;
- prompt words;
- approximate template tokens;
- output mode;
- structured output flag;
- reasoning effort;
- branch count.

### Baseline deltas

P0-hoz képest:

- absolute Macro F1 improvement;
- relative Macro F1 improvement;
- token overhead %;
- latency overhead %;
- invalid-output reduction percentage pointban.

## Miért fontos a case-type bontás?

Egy prompt lehet összességében jó, de productionben egy bizonyos típusú kérdésen veszélyesen gyenge.

Például:

- delimiter prompt javíthat prompt-injectionön;
- contrastive examples javíthatják az ambiguous boundary eseteket;
- reasoning/branching javíthat hard multi-intent eseteken;
- persona lehet, hogy szinte semmit nem ad, csak token overheadet.

Ezért a UI külön `strategy × case_type` heatmapet és easy/medium/hard bontást is mutat.
