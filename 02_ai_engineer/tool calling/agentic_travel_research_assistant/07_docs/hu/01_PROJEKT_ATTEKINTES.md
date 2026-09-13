# Projekt áttekintés — Agentic Travel Research Assistant

Ez a dokumentum azt foglalja össze, hogy milyen problémát old meg a rendszer, milyen felhasználói élményt ad, és hogyan kapcsolódnak egymáshoz a fő komponensek.

## 1. Projektcél

Az **Agentic Travel Research Assistant** egy lokálisan is futtatható, tool-calling alapú AI-rendszer, amely természetes nyelvű utazási kérdéseket több strukturált részfeladatra bont, kiválasztja a megfelelő toolokat, validálja az argumentumokat, végrehajtja a toolokat, majd a rész-eredményeket egységes válasszá állítja össze.

A projekt célja nem egy hagyományos chatbot létrehozása. A rendszer központi problémája az **orchestration**: eldönteni, hogy egy felhasználói kéréshez milyen capabilityk szükségesek, milyen argumentumokkal kell őket meghívni, és milyen sorrendben kell a részfeladatokat végrehajtani.

A projekt böngészős Streamlit UI-val használható, ezért a normál kipróbáláshoz nem szükséges parancssori JSON argumentumokat írni.

## 2. Példa feladat

Felhasználói kérés:

> 3 napra megyek Bécsbe. Nézd meg az időjárást, keress 150 euró alatti hotelt, ajánlj éttermet és látnivalókat, valamint mondd meg a tömegközlekedési lehetőségeket.

A rendszer ebből a következő tool-láncot állítja elő:

```text
get_weather
    ↓
search_hotels
    ↓
search_attractions
    ↓
search_restaurants
    ↓
get_transport_options
    ↓
structured results
    ↓
final response + trace
```

Minden tool-hívás külön rögzíti az argumentumokat, a kimenetet, a futási időt és a sikerességet.

## 3. Fő komponensek

### 3.1. Böngészős UI

A `08_ui/app.py` Streamlit alkalmazás hét fő területet biztosít:

- **Chat** — saját vagy preset természetes nyelvű kérdések, beépített HU/EN kérdésírási instrukciók és teljes tool trace;
- **Tool Explorer** — az egyedi toolok közvetlen kipróbálása űrlapokkal;
- **Data Quality** — duplikáció, nyelvi diverzitás, split-leakage és entity-név minőség;
- **Train & Evaluate** — intent-router újratanítás és benchmark;
- **Live Statistics** — a tényleges Chat használatból számolt perzisztens KPI-k, idősorok, tool/módszertan megoszlás és kérdés-válasz előzmények;
- **Project Statistics** — újragenerálható dataset-, modell-, benchmark- és repository-statisztikák;
- **Dataset Explorer** — a CSV-adatkészletek böngészése és szűrése.

A Chat felület **30 különböző, kétnyelvű preset scenario-t** tartalmaz. A kérdések kategória és nehézség szerint szűrhetők. Mindegyik scenario rendelkezik ground-truth tool-route-tal, ezért regressziós teszttel ellenőrizhető.

### 3.2. Tool registry

A toolokat egy közös `ToolRegistry` kezeli. Ez a réteg felel:

1. a toolok felderítéséért;
2. az OpenAI-kompatibilis JSON schema előállításáért;
3. a Pydantic argumentum-validációért;
4. a tool végrehajtásáért;
5. a latency méréséért;
6. a strukturált hibakezelésért.

### 3.3. Toolok

A rendszer nyolc capabilityt tartalmaz:

| Tool | Feladat |
|---|---|
| `get_location_info` | ország, pénznem, nyelv, időzóna, lokális költségprofil |
| `get_weather` | többnapos időjárás-előrejelzés |
| `convert_currency` | devizaátváltás |
| `search_hotels` | hotelkeresés és rangsorolás |
| `search_attractions` | látnivaló-keresés és rangsorolás |
| `search_restaurants` | étteremkeresés és rangsorolás |
| `get_transport_options` | helyi közlekedési jegyek és becsült költség |
| `calculate` | biztonságos függő számítások |

A hotel-, étterem- és attraction toolok lokális, reprodukálható szintetikus inventorykon dolgoznak. A weather és FX tool `local`, `auto` és `live` provider módban használható.

## 4. Routing és agent módszerek

### `plan_execute`

Determinista, inspectable planner. Regex-, lexikai- és morfológiai szabályokból állítja elő a tool tervet és a strukturált argumentumokat. Offline és reprodukálható baseline.

### `ml_router`

Tanított multi-label intent router. A modell 1–3 szavas word TF-IDF feature-öket és Unicode ékezet-normalizálást használ, majd One-vs-Rest logisztikus SGD classifier választja ki a capabilityket. A strukturált argumentumok kinyerését továbbra is determinisztikus parser végzi, így a routing és az argument extraction külön mérhető.

### `openai_direct`

Az OpenAI tool/function-calling megközelítése. Az LLM választ toolt és argumentumokat, a Python alkalmazás pedig validálja és végrehajtja a függvényeket. A tool output visszakerül a modellhez, amely új toolt kérhet vagy végső választ adhat.

## 5. Adatok

A projekt több, külön célra használt adatcsoportot tartalmaz.

### Utazási inventoryk

- 1240 000 hotel;
- 90 000 látnivaló;
- 90 000 étterem;
- 60 város;
- lokális transport profilok;
- többnapos weather fallback adatok;
- FX fallback adatok.

Ezek az adatok **szintetikusak**. A cél a reprodukálható tool-filtering, ranking, routing és evaluation, nem valós idejű foglalhatóság imitálása.

### Routing adatok

- 240 000 címkézett kétnyelvű intent-router példa;
- 36 000 indirekt/noisy challenge query;
- 90 000 sample user query;
- 22 500 end-to-end agent benchmark eset.

A 240 000-es router dataset splitje:

```text
192 000 train
 24 000 validation
 24 000 held-out test
```

A data-quality pipeline ellenőrzi az exact duplikációkat, normalizált nyelvi mintákat és a train/validation/test mintázatátfedést.

## 6. ML intent router

A modell pipeline:

```text
query
  ↓
word TF-IDF (1–2 gram)
  +
Unicode ékezet-normalizálás + word TF-IDF 1–3 gram
  ↓
One-vs-Rest SGD logistic classifiers
  ↓
labelenként optimalizált probability threshold
  ↓
capability set
```

A Unicode ékezet-normalizálás és a determinisztikus magyar parser guardrail együtt kezeli az ékezet nélküli és toldalékos edge case-ek jelentős részét.

A mentett modell tartalmazza a training dataset SHA-256 hashét. Ha a training adat megváltozik, a UI `stale` modellállapotot jelez.

## 7. Data-quality megközelítés

A korábbi generált corpus problémája az volt, hogy a magas sorszám mögött kevés valós nyelvi minta állt. A jelenlegi pipeline explicit quality gate-ekkel védekezik ez ellen.

A fő ellenőrzések:

- exact duplicate query;
- normalizált nyelvi minták száma;
- legnagyobb template-család aránya;
- train ↔ validation ↔ test normalizált overlap;
- entity-név diverzitás;
- model/dataset freshness.

Az audit eredményei a `06_results/data_quality/` könyvtárba kerülnek.

## 8. Evaluation

A projekt nem csak végső válasz alapján értékel. Külön méri az orchestration különböző szintjeit:

- Tool Selection Accuracy;
- Tool Precision;
- Tool Recall;
- Tool F1;
- Argument Accuracy;
- Task Success Rate;
- Unnecessary Tool-call Rate;
- Average Tool Calls;
- Mean Latency;
- P95 Latency.

A mentett router aktuális eredményei:

- held-out test micro-F1: **0.783**;
- held-out test macro-F1: **0.785**;
- challenge micro-F1: **0.677**.

Az aktuális 500-case end-to-end agent benchmark eredménye:

- exact tool selection: **88.2%**;
- tool F1: **97.2%**;
- argument accuracy: **95.0%**;
- task success: **68.0%**;
- unnecessary tool-call rate: **3.31%**.

A challenge eredmény szándékosan lényegesen alacsonyabb a tiszta tesztnél: ez mutatja, hogy a nehezebb, indirekt és zajos nyelvezet még valódi generalizációs problémát jelent.

## 9. 30 preset scenario

A `travel_agent.presets` modul 30 eltérő use case-t definiál magyar és angol változatban. A scenario-k között található:

- egytoolos időjárás, hotel, étterem, attraction és transport kérdés;
- devizaátváltás;
- destination metadata;
- többtoollos várostervezés;
- teljes helyi költségterv;
- szállás nélküli budget scenario;
- ékezet nélküli magyar kérdés;
- negációs hotel scenario;
- közlekedés autó nélkül;
- időjárás + étterem indirekt megfogalmazás;
- 5–6 toolt igénylő komplex kérés.

A presetek automatikus regressziós tesztje mindkét nyelven ellenőrzi az elvárt tool-route-ot és a tool végrehajtás sikerességét.

## 10. Reprodukálhatóság

A normál Windows indítás:

```text
SETUP_AND_START_UI.bat
```

Későbbi indítás:

```text
RUN_UI.bat
```

Teljes quality → training → evaluation pipeline:

```powershell
python 04_scripts/07_prepare_train_evaluate.py --limit 500
```

Tesztek:

```powershell
pytest -q
```

## 11. Korlátok

- a hotel-, restaurant- és attraction inventory szintetikus;
- a `plan_execute` parser nem általános természetesnyelv-értelmező;
- az ML router routingot végez, nem teljes end-to-end generatív planninget;
- a local weather/FX fallback reprodukálható tesztadat, nem aktuális piaci adat;
- a live provider mód hálózatfüggő;
- az OpenAI orchestration használatához API-kulcs szükséges.

Ezek a korlátok tudatosan külön vannak választva a mérhető, reprodukálható offline komponensektől.

## 12. Preset-validáció

A 30 kétnyelvű scenario mindkét offline routing módszerrel külön regressziósan ellenőrizhető:

```powershell
python 04_scripts/08_validate_presets.py
```

A jelenlegi mentett validáció összesen 120 futást tartalmaz (30 scenario × 2 nyelv × 2 módszertan). A `plan_execute` és az `ml_router` is 60/60 nyelvspecifikus esetben pontosan az elvárt tool-route-ot állítja elő, és minden tool-hívás sikeresen lefut. Az eredmények a `06_results/preset_validation/` könyvtárban találhatók.

## Saját kérdések, perzisztens használati előzmény és élő statisztika

A Chat nem csak presetekkel működik: a felhasználó tetszőleges természetes nyelvű kérdést adhat meg. A UI magyar/angol instrukciós blokkban jelzi a támogatott travel capabilityket és a jó argumentumkinyeréshez hasznos mezőket (város, időtartam, keret, értékelés, preferenciák).

Minden lefuttatott Chat interakciót a `travel_agent.usage.UsageStore` helyi SQLite adatbázisba ment. A tárolás normalizált: külön `interactions` és `tool_calls` táblák vannak. Az élő dashboard innen számolja a KPI-kat, a tool/módszertan megoszlást, a napi használatot és a kérdés-válasz előzményeket.

A setup idempotens jellegű: kompatibilis dependencyket nem frissít újra, és a router modellt csak hiányzó vagy dataset-hash alapján elavult állapotban tanítja újra.

## Összegzés

A projekt egy mérhető, több routing módszert támogató tool-calling rendszer. A lényeg nem pusztán a válaszgenerálás, hanem a természetes nyelvű kérdés strukturált, validált és megfigyelhető végrehajtássá alakítása.
