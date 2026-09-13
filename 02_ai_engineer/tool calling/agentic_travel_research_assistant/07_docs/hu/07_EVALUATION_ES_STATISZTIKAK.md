# Evaluation és statisztikák

Ez a dokumentum azt mutatja be, hogyan mérjük külön a routingot, az argumentumokat, a végrehajtást, valamint hogyan keletkeznek a statikus és élő statisztikák.

A projekt két külön statisztikai réteget tartalmaz:

1. **statikus projektstatisztika** — dataset-, modell-, benchmark- és repository-méretek;
2. **élő használati statisztika** — a Streamlit UI-ban ténylegesen lefuttatott kérdésekből számolt SQLite-alapú metrikák.

A statikus projektstatisztika újragenerálható:

```powershell
python 04_scripts/09_generate_project_statistics.py
```

Az output a `06_results/project_statistics/` alatt jelenik meg.

## 1. Repository statisztika

A jelenlegi projektben:

- 8 regisztrált tool;
- 30 kétnyelvű preset scenario;
- 60 HU/EN preset surface form;
- 22 500 end-to-end agent benchmark eset;
- 36 Python source file, kb. 2 800 source code sor;
- 17 tesztfájl, 56 automatikus test function;
- 27 Markdown dokumentációs fájl;
- 4 notebook;
- 9 újrafuttatható script a statisztika-generátorral együtt.

## 2. Nyers adatréteg

A 10 fő CSV összesen **798 150 sort** tartalmaz. Az aktuális audit szerint a CSV-kben összesen 0 exact duplicate row található.

Főbb datasetméretek:

| Dataset | Sor |
|---|---:|
| `intent_router_dataset.csv` | 240 000 |
| `sample_user_queries.csv` | 90 000 |
| `hotels.csv` | 180 000 |
| `intent_router_challenge.csv` | 36 000 |
| `attractions.csv` | 90 000 |
| `restaurants.csv` | 90 000 |
| `weather_fallback.csv` | 72 000 |
| `cities.csv` | 60 |
| `transport.csv` | 60 |
| `fx_rates_fallback.csv` | 30 |

A router dataset opcionális argumentum-oszlopjaiban található üres értékek szándékosak: például weather queryhez nem kell hotelár vagy restaurant budget.

## 3. Intent router statisztika

Train/validation/test:

- train: 192 000;
- validation: 24 000;
- held-out test: 24 000.

Mentett modellmetrikák:

- validation micro-F1: **0.7602**;
- validation macro-F1: **0.7539**;
- held-out test micro-F1: **0.7826**;
- held-out test macro-F1: **0.7846**;
- test hamming loss: **0.1314**.

Feature pipeline:

```text
word TF-IDF 1–3 gram
+
Unicode normalizálás
+
One-vs-Rest SGD logistic classifier
```

## 4. Data-quality statisztika

A quality gate-ek aktuális állapota: **6/6 PASS**.

A fontosabb gate-ek:

- router corpus linguistic diversity;
- sample query linguistic diversity;
- train/validation/test normalizált pattern overlap;
- hotelnév-diverzitás;
- restaurantnév-diverzitás;
- attractionnév-diverzitás.

A normalizált split leakage az aktuális reportban:

- train ↔ test: 0;
- train ↔ validation: 0;
- validation ↔ test: 0.

## 5. Agent benchmark statisztika

500 összehasonlítható benchmark eseten:

| Módszer | Exact tool selection | Tool F1 | Argument accuracy | Task success | P95 latency |
|---|---:|---:|---:|---:|---:|
| rule-based | 19.6% | 46.7% | 36.6% | 16.6% | 17.66 ms |
| plan-execute | 60.6% | 80.4% | 74.0% | 47.6% | 27.47 ms |
| ML router | **88.2%** | **97.2%** | **95.0%** | **68.0%** | 32.25 ms |

Ez a táblázat két fontos dolgot mutat. Egyrészt a tanítható router lényegesen jobb routingot ad. Másrészt a task success alacsonyabb, mint a tool F1, tehát a jó tool selection önmagában nem garantál teljes feladatsikert; argument extraction és tool execution is számít.

## 6. Preset validáció

30 scenario × 2 nyelv × 2 offline módszertan = 120 validációs futás.

A mentett eredményben:

- `plan_execute`: 60/60 exact route;
- `ml_router`: 60/60 exact route;
- mindkettőnél 60/60 successful run.

A preset validáció kontrollált regressziós suite, nem helyettesíti a held-out benchmarkot.

## 7. Élő használati statisztika

A UI `Live Statistics` lapja a `06_results/usage/usage_history.sqlite3` adatbázisból számol.

Mért KPI-k:

- total questions;
- total tool calls;
- avg tools/question;
- tool success rate;
- run success rate;
- p50 run latency;
- p95 run latency;
- p50/p95 tool latency;
- multi-tool rate;
- no-tool rate;
- unique-question rate;
- custom vs preset arány;
- átlagos kérdés- és válaszhossz;
- napi question/tool-call trend;
- utolsó 7 nap vs előző 7 nap;
- tool-frequency;
- methodology usage;
- top tool sequences;
- top destination cities;
- language/data-mode/source megoszlás;
- error breakdown;
- recent Q&A history.

Az élő usage statisztika ezért nem fix screenshot, hanem minden új kérdés után változó operational dashboard.

## 8. Generált statisztikai artifactok

A `06_results/project_statistics/` könyvtár egyszerre tartalmaz géppel olvasható adatforrásokat és reprodukálható statikus plotokat. A fontosabb CSV/JSON fájlok:

- `project_statistics.json` — teljes projektösszefoglaló;
- `dataset_statistics.csv` — datasetméret, memória, missing és duplicate statisztika;
- `query_statistics.csv` — query-diverzitás és komplexitás;
- `entity_statistics.csv` — hotel/étterem/látnivaló inventory minőség;
- `tool_label_distribution.csv` — router label balance;
- `query_complexity_distribution.csv` — intents/query eloszlás;
- `benchmark_complexity.csv` — elvárt tool-hívásszám benchmark esetenként;
- `benchmark_expected_tool_distribution.csv` — benchmark capability mix;
- `city_inventory_coverage.csv` — városonkénti inventory;
- `router_per_label_metrics.csv` — intentenkénti precision/recall/F1;
- `evaluation_statistics.csv` — methodology összehasonlítás;
- cuisine/category/weather distribution CSV-k.

A mentett PNG-k ugyanennek az adatnak statikus snapshotjai: dataset rows/memory, query- és entity-diverzitás, intent balance, query/benchmark komplexitás, benchmark tool-mix, city coverage, ár- és kategóriaeloszlások, router per-label metrikák, methodology quality és latency. Ezek fix magas kontrasztú világos stílust használnak. Az interaktív Plotly dashboard ugyanazon CSV/JSON artifactokból épül, de Light/Dark témára dinamikusan váltható.

## Evaluation logika

A fő metrikák: exact tool-selection accuracy, tool precision/recall/F1, argument accuracy, task success, unnecessary tool-call rate, átlagos és P95 latency. A tool-selection és task success külön metrika, mert a helyes tool kiválasztása után is hibázhat az argumentum vagy execution.

A jelenlegi 500-as methodology snapshotban az ML router ~88.2% exact tool-selection, ~97.2% tool-F1, ~95.0% argument accuracy és ~68.0% task success értéket ad. Ezek generált benchmark eredmények, nem production SLA-k.

A Live Statistics SQLite historyból változik, a Project Statistics pedig a repository és mentett benchmark artifactok aktuális állapotából újragenerálható.

## Összegzés

A projekt statisztikái két külön célt szolgálnak: az evaluation a rendszer minőségét méri reprodukálható teszteseteken, a live analytics pedig a tényleges használati mintát és üzemi viselkedést mutatja.

## Kibővített projektstatisztikák és diagnosztikai plotok

A v1.5 statisztikai réteg célja nem pusztán néhány összesített számláló megjelenítése. A projekt külön választja a **méretet**, a **diverzitást**, a **routing-komplexitást**, a **benchmark-komplexitást** és az **end-to-end teljesítményt**. A `python 04_scripts/09_generate_project_statistics.py` parancs a következő reprodukálható artifactokat készíti:

- `dataset_statistics.csv`: sor-/oszlopszám, hiányzó értékek, memóriaigény és városlefedettség datasetenként;
- `query_statistics.csv`: unique query arány, normalizált nyelvi diverzitás, query-hossz, multi-intent arány;
- `entity_statistics.csv`: hotel/étterem/látnivaló névdiverzitás, rating- és árkvantilisek, városonkénti inventory;
- `tool_label_distribution.csv`: routing label balance;
- `query_complexity_distribution.csv`: hány intent található egy kérdésben;
- `benchmark_complexity.csv`: hány tool-hívást vár el egy benchmark eset;
- `benchmark_expected_tool_distribution.csv`: benchmark tool-eloszlás;
- `city_inventory_coverage.csv`: városonkénti hotel/restaurant/POI lefedettség.

A plotok ezekből az adatokból készülnek, ezért nem kézzel megadott illusztrációk. Külön ábra mutatja a datasetméreteket, memóriaigényt, nyelvi diverzitást, entity-név diverzitást, intent balance-ot, query-komplexitást, benchmark-komplexitást, várható tool-eloszlást, városonkénti inventoryt, methodology qualityt és runtime latencyt.

### Miért relevánsak ezek a statisztikák?

A nagyobb adatmennyiség önmagában nem jelent jobb datasetet. Például 200 000 sor kevés értéket ad, ha ugyanannak a 20 sablonnak a variációja. Ezért a projekt a rekordszám mellett külön méri a normalizált nyelvi minták arányát, a duplikációt, a névdiverzitást és a split leakage-et. Hasonlóan, egy agent benchmarknál nem elég az átlagos accuracy: fontos látni, hogy a tesztesetek hány toolt igényelnek, mennyire kiegyensúlyozottak a capabilityk és mekkora latency trade-off tartozik a jobb routinghoz.

## Adatskálázási döntés

A nagy inventory táblák körülbelül tízszeresére nőttek: 180 000 hotel, 90 000 látnivaló, 90 000 étterem és 72 000 időjárási fallback rekord. A referencia/dimenzió táblákat (`cities.csv`, `fx_rates_fallback.csv`, városi transport profilok) nem másoljuk tízszer, mert az csak mesterséges duplikációt hozna létre. A nyelvi/routing adatok külön, körülbelül háromszoros skálázást kaptak: 240 000 router példa, 36 000 challenge query, 90 000 sample query és 22 500 agent benchmark eset.

Ez a különbségtétel fontos: a cél nem a fájlméret növelése, hanem több tényleges keresési inventory és több nyelvi variáció létrehozása.
