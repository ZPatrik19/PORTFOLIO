# Kódreferencia és repository audit

Ez a dokumentum két kérdésre válaszol: melyik fájl felel egy adott funkcióért, és miért marad meg az adott repository elem.

Ez a dokumentum fájlról fájlra összefoglalja a projekt működő Python komponenseit. A cél az, hogy a repository olvasásakor azonnal látható legyen, melyik modul miért létezik, és melyik réteghez tartozik.

## `03_src/travel_agent/models.py`
Közös runtime dataclassok. Az `AgentRun` egy teljes agent futást, a `ToolCallRecord` egy tool-hívás trace rekordját reprezentálja. Ezeket a UI, evaluation és usage analytics is ugyanúgy használja.

## `03_src/travel_agent/config.py`
Environment-alapú konfiguráció: agent mód, modell, maximális lépésszám és default nyelv.

## `03_src/travel_agent/data_store.py`
A lokális CSV-adatok közös betöltési/cache rétege. A toolok nem saját adathozzáférési implementációt tartanak fenn minden hívásnál.

## `03_src/travel_agent/presets.py`
A 30 preset scenario canonical forrása. Egy scenario tartalmaz azonosítót, kategóriát, nehézséget, HU/EN szöveget és expected tool-listát.

## `agent/heuristics.py`
Entity- és constraint extraction: városnormalizálás, days, price, rating, cuisine, attraction és intent cue-k. A magyar ragozás regresszióinak jelentős része itt van kezelve.

## `agent/offline_agent.py`
Korlátozott rule-based baseline. Szándékosan nem a legerősebb agent, hanem összehasonlítási referencia.

## `agent/plan_execute_agent.py`
Inspectable deterministic planner/executor. `PlanStep` objektumokból explicit execution tervet épít, majd registry-n keresztül végrehajtja.

## `agent/ml_router_agent.py`
A supervised intent-router wrapper. Betölti a mentett pipeline-t, clause-level probability aggregációt használ, explicit lexical guardrailokat alkalmaz, majd a deterministic parserrel készít structured argumentsot.

## `agent/openai_agent.py`
Valódi LLM tool-calling loop. A tool schemákat a registryből kapja, function callokat végrehajtja, outputot visszaadja a modellnek, és `max_steps` korláttal fejezi be a loopot.

## `agent/prompts.py`
Az OpenAI agent system instructionje. Toolhasználati és válaszkészítési contract.

## `agent/service.py`
Factory/router a methodology kiválasztásához. CLI és programmatic használatban egységes agent constructiont ad.

## `tools/base.py`
A tool contract. Pydantic input modelből OpenAI-compatible schema készül és egységes validation interface-et biztosít.

## `tools/registry.py`
A rendszer execution boundaryja. Regisztrálja a nyolc toolt, exportálja a schemákat, validál, időt mér és structured errort ad vissza.

## `tools/weather.py`
Local/auto/live időjárás provider logika.

## `tools/currency.py`
Local/live FX conversion logika.

## `tools/hotels.py`, `restaurants.py`, `attractions.py`
Strukturált inventory search, constraint filtering és ranking.

## `tools/transport.py`
Városi transport-pass és többnapos becsült költség.

## `tools/location.py`
Canonical city metaadat lookup.

## `tools/calculator.py`
Safe AST arithmetic. Nem használ unrestricted Python `eval()` végrehajtást.

## `training/router_training.py`
Router corpus beolvasása, TF-IDF/SGD model training, per-label threshold selection, metrics, joblib persistence és runtime metadata compatibility.

## `quality/data_quality.py`
Dataset hash, linguistic normalization, duplicate/pattern diversity, split leakage és entity diversity audit.

## `evaluation/metrics.py`
Tool-selection és argument-level metrikák.

## `evaluation/runner.py`
Benchmark runner: ground-truth case-ek futtatása, case-level eredmények, summary metrics és output írás.

## `evaluation/plotting.py`
Reprodukálható benchmark vizualizációk.

## `usage/store.py`
SQLite persistence és operational analytics. Ment minden Chat runhoz interaction/tool-call rekordot, majd p50/p95 latencyt, tool usage-ot, sequence-eket, destinations-t, trendet és error breakdownot számol.

## `08_ui/app.py`
Streamlit presentation layer. Nem tartalmaz külön agent/tool üzleti logikát; shared core modulokat hív. A fő tabok: Chat, Tool Explorer, Data Quality, Train & Evaluate, Live Statistics, Project Statistics és Dataset Explorer. A sidebarban a routing/data-mode beállítások mellett közös Light/Dark analitikai chart theme is választható.

## `08_ui/chart_theme.py`
A három analytics dashboard közös vizualizációs rétege. Itt található a Light/Dark magas kontrasztú paletta, a 460 px közös chartmagasság, az axis/legend/hover/annotation styling és a `render_plotly()` wrapper. Ez akadályozza meg, hogy az egyes dashboard-modulok egymástól eltérő Plotly stílust használjanak.

## `08_ui/project_statistics_dashboard.py`
Repository-, dataset-, routing-, model- és benchmark-statisztikák interaktív Plotly dashboardja. A chart-builder függvények a közös theme-rétegen keresztül renderelődnek.

## `08_ui/data_quality_dashboard.py`
Quality gate, query/entity-diverzitás, leakage és router-label diagnosztika interaktív felülete.

## `08_ui/live_statistics_dashboard.py`
SQLite usage historyból számolt trends/tool reliability/latency/behavior dashboard.

## Setup scriptek

- `00_setup/02_setup_check.py` — repository/runtime smoke check.
- `00_setup/04_generate_data.py` — determinisztikus base data generation.
- `00_setup/05_upgrade_data_quality.py` — high-diversity corpus/entity generation stage.
- `00_setup/06_check_dependencies.py` — install nélküli version compatibility check.
- `00_setup/07_prepare_if_needed.py` — csak szükség esetén audit/training/smoke evaluation.

## Runnable scriptek

- `01_run_demo.py` — agent demo;
- `02_run_evaluation.py` — benchmark;
- `03_compare_methodologies.py` — baseline comparison;
- `04_run_live_openai_agent.py` — live OpenAI path;
- `05_call_tool.py` — direct tool call;
- `06_train_intent_router.py` — router training;
- `07_prepare_train_evaluate.py` — one-command pipeline;
- `08_validate_presets.py` — bilingual preset regression;
- `09_generate_project_statistics.py` — statikus project statistics.

---

## Audit célja

A projekt végső struktúráját nem csak funkcionális szempontból, hanem portfólió-repositoryként is átnéztük. A cél az volt, hogy minden megmaradó elemnek legyen egyértelmű szerepe: futtatás, reprodukálhatóság, tesztelés, adat, mérés vagy dokumentáció.

## Eltávolított redundáns artifactok

A `06_results` mappából eltávolításra kerültek azok a generált fájlok, amelyek más, megtartott eredményeket duplikáltak vagy csak egyszeri smoke run melléktermékei voltak:

- `notebook_eval/`
- `prepare_pipeline_eval/`
- `demo_transcript_en.txt`
- `demo_transcript_hu.txt`
- `evaluation_console.txt`
- `methodology_console.txt`
- `direct_tool_call.json`

Ezek nem adtak új információt a megtartott `rule_based/`, `plan_execute/`, `ml_router/`, `models/`, `data_quality/`, `preset_validation/` és `project_statistics/` eredményekhez képest.

A setup és a UI smoke evaluation most temporary directoryban fut, ezért a jövőben sem szemeteli tele a `06_results` könyvtárat.

## Miért maradnak meg a fő mappák?

### `00_setup/`
Szükséges. Környezetellenőrzés, dependency gate, adatregenerálás, quality upgrade és conditional train logic található itt. Ezek nélkül a projekt nem reprodukálható.

### `01_data/`
Szükséges. A toolok működési adatait, a router training/challenge corpusokat és az agent benchmarkot tartalmazza.

### `02_notebooks/`
Szükséges, de nem runtime dependency. A négy notebook külön szerepet tölt be: data exploration, tool layer, agent workflow és evaluation. Portfólió és reprodukálható elemzés szempontból indokoltak.

### `03_src/`
Szükséges. Ez a tényleges Python package. Az agent, tool, training, quality, evaluation és usage modulok külön vannak választva.

### `04_scripts/`
Szükséges. A scriptek nem a core logikát duplikálják, hanem CLI entry pointok a shared package modulokhoz. Ide került a projekt-statisztika generátor is.

### `05_tests/`
Szükséges. Parser, tool, router, evaluation, persistence, preset és model compatibility regressziókat véd.

### `06_results/`
Szükséges, de csak tartós, értelmezhető outputokkal. A redundáns console/smoke artifactokat eltávolítottuk.

### `07_docs/`
Szükséges. Rövid célzott dokumentumok + magyar/angol master dokumentáció + tool-calling tervezési checklist található itt.

### `08_ui/`
Szükséges. Ez a normál felhasználói interface.

## Root fájlok auditja

### `README.md`
Marad. GitHub belépési pont, rövid angol áttekintés.

### `START_HERE_HU.md` és `START_HERE_EN.md`
Marad. Gyors indulási útmutató két nyelven.

### `SETUP_AND_START_UI.bat`
Marad. Első indításkor idempotens setup: csak hiányzó dependencyt telepít és csak szükség esetén tanít.

### `RUN_UI.bat`
Marad. Napi használat gyors indítója; nem telepít és nem tanít.

### `pyproject.toml`
Marad és ez a canonical package metadata/dependency definíció.

### `requirements.txt`
Marad kompatibilitási/olvashatósági segédfájlként. A setup nem erre épít, ezért a canonical forrás továbbra is a `pyproject.toml`. Dokumentációban jelezve van, hogy módosításnál a két dependency-listát szinkronban kell tartani.

### `.env.example`
Marad. OpenAI és provider konfiguráció dokumentált mintája.

### `.gitignore`
Marad. Különösen fontos, mert a helyi SQLite usage history nem kerülhet Gitbe.

## Kódstruktúra audit

A core logika shared module-okban van, nem a Streamlit UI-ban vagy a CLI scriptekben. Ez fontos döntés, mert ugyanazt a tool/agent implementationt használja:

- UI;
- CLI;
- teszt;
- evaluation;
- preset validation.

A `ToolRegistry` marad az egyetlen execution boundary. A Pydantic validáció és trace ugyanazon az úton történik.

A `rule_based` agent marad, bár a minősége gyengébb, mert baseline-ként mérési értéke van. Nem production default.

A `plan_execute` agent marad, mert inspectable deterministic orchestration baseline.

Az `ml_router` marad, mert ez a tanítható routing komponens és a data-centric kísérlet fő része.

Az `openai_direct` marad, mert valódi LLM tool-calling execution pathot demonstrál.

## Tudatosan megtartott duplikációk

### PNG + SVG + DOT diagramok
Mindhárom formátumnak más szerepe van: PNG GitHub-kompatibilis preview, SVG skálázható dokumentáció, DOT regenerálható forrás.

### `pyproject.toml` + `requirements.txt`
Technikailag részben duplikált dependency-lista. A `pyproject.toml` canonical; a `requirements.txt` egyszerű környezetolvasás/kompatibilitás miatt marad.

### Rövid és részletes projektleírás
A `PROJECT_DESCRIPTION_*` executive summary. A `PROJECT_DOCUMENTATION_*` a teljes technikai master dokumentáció. Nem ugyanazt a célt szolgálják.

## További, jelenleg nem szükséges elemek

Nem került bele Docker, Kubernetes, adatbázis-szerver, message queue, LangGraph vagy külön backend API, mert a jelenlegi projekt céljához ezek csak infrastruktúra-komplexitást adnának. Ha a projekt production deployment irányba fejlődik, ezek újraértékelhetők.

## Audit eredmény

A végső struktúrában nincs olyan nagyobb mappa vagy runtime komponens, amely pusztán dekorációs céllal maradt volna. A generált result-artifactok közül a redundáns elemeket eltávolítottuk; a megmaradó baseline-ok, datasetek és dokumentációk mind konkrét reprodukálhatósági vagy mérési célt szolgálnak.

## Összegzés

A repository számozott top-level struktúrája tudatosan követi a setup → data → notebooks → source → scripts → tests → results → docs → UI folyamatot. A tartós artifactok reprodukálható vagy diagnosztikai értéket hordoznak; az egyszeri smoke-outputok nem maradnak a repositoryban.
