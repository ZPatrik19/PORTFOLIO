# Streamlit + Plotly UI használata

## 1. Egykattintásos indítás

A projekt gyökerében futtasd:

```text
RUN_UI.bat
```

Ha még nincs virtuális környezet, automatikusan lefut a smart setup. Ha már minden telepítve van, közvetlenül elindul a UI.

Alternatíva:

```text
00_setup\01_setup_windows.bat
00_setup\02_run_ui.bat
```

Csak dependency ellenőrzés/javítás:

```text
00_setup\05_update_environment.bat
```

## 2. Mit kezel automatikusan a setup?

1. Python 3.10–3.14 keresése.
2. Ha nincs megfelelő Python és van `winget`, Python 3.12 automatikus telepítési kísérlete.
3. `.venv` létrehozása, vagy a meglévő kompatibilis környezet megtartása.
4. `requirements.txt` elemzése.
5. A hiányzó vagy túl régi csomagok célzott telepítése/frissítése.
6. `pip check` futtatása.
7. Streamlit + Plotly import ellenőrzése.
8. Mock benchmark adatok előkészítése, ha hiányoznak.
9. Koncepcionális ábrák generálása.
10. Pytest suite futtatása.

A cél az idempotens setup: egy második futtatás nem építi újra feleslegesen az egész környezetet.

## 3. UI oldalak

### Dashboard

A mentett provider eredményeiből készít interaktív Plotly dashboardot:

- Macro F1 leaderboard;
- 95% bootstrap confidence interval;
- Quality vs Cost;
- Quality vs P95 Latency;
- Token overhead vs Macro F1;
- Invalid output rate;
- per-class F1 heatmap;
- interaktív eredménytábla.

A pontokra/oszlopokra rámutatva további provider-, model-, token-, latency- és cost-információ jelenik meg.

### Techniques

P0–P16 technikák, kategória, hipotézis és tényleges prompt preview. A multi-branch P14 külön áganként megtekinthető.

### Playground

Egy saját support ticket futtatása bármely stratégiával. Megjelenik a prediction, latency, tokenhasználat, output-validity és raw response.

### Benchmark Runner

A UI-ból kiválasztható:

- development vagy final benchmark dataset;
- egy vagy több P0–P16 stratégia;
- row limit;
- cache használata vagy `force` rerun;
- automatikus report generálás.

A cloud provider esetén a UI figyelmeztet a kvóta/költség kockázatára.

### Parameter Lab

A prompt fix marad, miközben külön történik a `temperature`, `top_p` és – ahol a provider támogatja – `top_k` sweep. A kiválasztott metric Plotly line charton jelenik meg.

### Output Validation

Egy adott stratégia raw eredményeire:

- interaktív confusion matrix;
- per-class precision/recall/F1;
- hibás/invalid példák táblázata;
- latency boxplot;
- token boxplot.

### Fine-tuning Prep

Leakage-safe SFT train/validation JSONL export. A UI nem indít automatikusan fizetős fine-tuning jobot.

### System

Megmutatja a projektkomponensek állapotát, dependency checket tud futtatni, és teszteli a kiválasztott LLM providert.

## 4. API key kezelés

A Groq/Gemini/OpenRouter/OpenAI kulcs beírható a sidebaron. A UI csak az aktuális Streamlit process környezeti változójában tartja, és nem írja `.env` vagy más fájlba.

Ollama és mock módhoz nem kell cloud API key.

## 5. Miért Plotly + Matplotlib együtt?

A Plotly az interaktív elemzéshez jobb: hover, zoom, vizuális összehasonlítás és dinamikus provider-váltás. A Matplotlib pipeline viszont stabil PNG artifactokat készít, amelyek közvetlenül használhatók GitHub README-ben, dokumentációban és portfólió-prezentációban. A projekt ezért mindkettőt megtartja.

## Prompt Gallery

A `🧠 Techniques` tabon a `Show all P0–P16 templates` nézet az összes promptot a **valódi strategy kódból rendereli**. Minden promptnál látszik:

- category;
- hypothesis;
- system prompt;
- user prompt template;
- branch count;
- karakter- és szószám;
- approximate token count;
- output mode;
- structured output;
- reasoning effort;
- aktuális temperature/top_p/top_k.

## Benchmark case filtering

A `🏁 Benchmark Runner` a synthetic dataset esetén külön szűrhető:

- case type;
- difficulty.

Így például külön összevethető P0 vs P11 prompt-injection eseteken vagy P0 vs P12 ambiguous boundary eseteken.

A részleges/limitált UI futások alapból `07_outputs/results/ui_runs/...` alá mennek, ezért nem írják felül a canonical 300-row benchmarkot.

## API Integration

A `🔌 API Integration` tab élő provider requestet futtat és kiírja a teljes telemetryt:

- prediction;
- raw response;
- input/output/total tokens;
- token source;
- latency + latency source;
- estimated cost;
- output/JSON validity;
- decoding settings;
- branch count.

## Egyetlen kétnyelvű UI

A projekt egyetlen launchert használ:

```text
RUN_UI.bat
```


A `Magyar / English` nyelvváltó, a provider-, modell-, API-key és sampling beállítások mind a közös sidebarban találhatók. Nincs külön magyar/angol launcher.

A magyar nézet fő fülei:

1. Dashboard
2. Prompttechnikák
3. Saját prompt
4. Playground
5. Benchmark
6. Paraméterlabor
7. Output validáció
8. Fine-tuning előkészítés
9. API-integráció
10. Rendszer

### Saját prompt

A `{ticket}` helyőrző kötelező. Példa:

```text
Te egy routing osztályozó vagy.

<ticket>
{ticket}
</ticket>

Csak egy címkét adj vissza.
```

A preset mentése után a prompt megjelenik a Benchmark és API-integráció nézetben is.

### Hibakezelés

A UI-tabok a provider/dependency hibákat lokálisan jelenítik meg, így egy rossz API key vagy hiányzó provider nem állítja le a teljes Streamlit alkalmazást.
