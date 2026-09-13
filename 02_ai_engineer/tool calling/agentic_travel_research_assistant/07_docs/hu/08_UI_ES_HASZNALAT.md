# UI és használat

Ez a fejezet a projekt gyakorlati használatát, a Streamlit felületet, a saját kérdésbevitelt, a preseteket és a lokális history működését írja le.

## 1. Indítás

Első használatkor Windows alatt `SETUP_AND_START_UI.bat`. A script csak hiányzó/inkompatibilis dependency esetén telepít, és csak stale/hiányzó modell esetén tanít újra. Napi használatra `RUN_UI.bat`.

## 2. UI lapok

- **Chat**: saját kérdés vagy preset.
- **Tool Explorer**: egyedi tool kézi kipróbálása form mezőkkel.
- **Data Quality**: quality gate-ek és dataset-diverzitás.
- **Train & Evaluate**: tréning és benchmark.
- **Live Statistics**: perzisztens használati analytics SQLite-ból.
- **Project Statistics**: interaktív Plotly dashboard repository-, adat-, routing-, modell- és benchmark statisztikákkal; külön statikus plot-galériával.
- **Dataset Explorer**: CSV-k vizsgálata.

## 3. Saját kérdés megadása

A Chat mező szabad szöveg. Jó kérdésben érdemes megadni a várost, napok számát, budgetet és preferenciákat. Például: „4 napra megyek Rómába. Nézd meg az időjárást, keress 180 euró alatti hotelt és ajánlj olasz éttermet 40 euró/fő alatt.”

## 4. Presetek

30 scenario × 2 nyelv áll rendelkezésre. Ezek nem csak demo gombok: regressziós katalógusként is futtathatók a `08_validate_presets.py` scripttel.

## 5. Usage history

A Chat futások lokálisan a `06_results/usage/usage_history.sqlite3` adatbázisba kerülnek. A DB gitignored. A felhasználó a UI-ból törölheti vagy CSV-ként exportálhatja a historyt.

## 6. Adatvédelem

A lokális usage analytics önmagában nem küldi ki a kérdéseket külső analytics szolgáltatásba. `openai_direct` vagy live provider használatakor viszont az adott külső szolgáltatásnak szükséges request adatok természetesen elküldésre kerülnek.

## 7. Interaktív Project Statistics dashboard

A Project Statistics oldal a v1.6-tól Plotly-alapú, ezért a diagramok hover információval, zoommal, legendakapcsolással és dinamikus szűrőkkel vizsgálhatók. A dashboard hat nézetre bontja az információt:

1. **Áttekintés** — fő KPI-k, dataset skála, methodology radar és quality/latency trade-off.
2. **Adatok & diverzitás** — nyelvi és entity-diverzitás, interaktív árhistogramok, kategóriaeloszlások.
3. **Routing & benchmark** — intent-egyensúly, multi-intent komplexitás, benchmark workflow komplexitás és tool-eloszlás.
4. **Modell & módszerek** — intentenkénti precision/recall/F1, választható methodology metrika, radar és latency trade-off.
5. **Mentett plotok** — a `06_results/project_statistics/*.png` statikus, reprodukálható snapshotjainak galériája.
6. **Adattáblák** — a grafikonok mögötti CSV-k megtekintése és letöltése.

Az interaktív nézetek ugyanazokból a CSV/JSON artifactokból épülnek, amelyeket a `04_scripts/09_generate_project_statistics.py` generál, ezért a dashboard nem rejtett vagy kézzel megadott számokra támaszkodik. Az árhistogramoknál dataset és bin-szám választható; a category chartoknál Top-N szűrés, a methodology nézetben pedig külön teljesítmény- vagy latency-metrika választható.

## Összegzés

A UI a projekt elsődleges emberi interfésze; a CLI megmarad reprodukálhatósághoz, automatizáláshoz és teszteléshez.


## Data Quality dashboard (v1.7)

A Data Quality fül a v1.7-től ugyanazt az interaktív Plotly dashboard-szemléletet követi, mint a Project Statistics. A nézetek külön kezelik a quality gate-eket, a query-corpus diverzitást, az entity-név diverzitást, a train/validation/test leakage-et, a router label balance-ot és a mentett audit snapshotokat.

Fő interaktív elemek:

- quality-gate gauge és gate status chart;
- query scale-vs-diversity bubble chart;
- query repetition-risk chart;
- entity unique-name és skeleton-diversity chart;
- split leakage heatmap;
- router intent-label balance;
- választható Top-N normalizált query pattern explorer;
- mentett PNG audit gallery és letölthető JSON/CSV report.

A teljes audit továbbra sem fut minden UI-indításkor. Csak kézi újrafuttatás vagy adatváltozás esetén szükséges.

## Live Statistics dashboard (v1.7)

A Live Statistics fül a lokális SQLite usage historyból épít interaktív dashboardot. Az Overview, Trends, Tools, Behavior és History & export nézetek különválasztják az operatív használat különböző aspektusait.

A dashboard többek között megjeleníti:

- kérdésszám és tool-call trend;
- run success és latency trend;
- P50/P95 latency;
- tool használat és tool-sikerarány;
- tool latency boxplot;
- leggyakoribb tool-szekvenciák;
- methodology quality-vs-latency;
- top célvárosok treemap;
- hét napja × óra usage heatmap;
- kérdéshossz vs end-to-end latency scatter;
- nyelv / data mode / custom-vs-preset megoszlás;
- tool error breakdown;
- exportálható és törölhető lokális history.

Minden Plotly grafikon egyedi Streamlit key-t használ, és a jelenlegi `width="stretch"` API-ra épül.


## Közös Light/Dark chart theme és egységes magasság (v1.8)

A sidebar **Analytics chart theme** vezérlője két magas kontrasztú módot kínál: `Light` és `Dark`. A beállítás egyszerre vonatkozik a Project Statistics, Data Quality és Live Statistics interaktív Plotly-ábráira. Light módban a plot háttér fehér és a fő/tengely/legend szöveg fekete; Dark módban a háttér sötét és a szöveg fehér. A három dashboard minden elsődleges Plotly chartja közös 460 px magasságot kap, így a kétoszlopos kártyák vizuálisan egy vonalban maradnak.

A részletes vizualizációs szabályokat a [12_VIZUALIZACIOS_RENDSZER.md](12_VIZUALIZACIOS_RENDSZER.md) tartalmazza.
