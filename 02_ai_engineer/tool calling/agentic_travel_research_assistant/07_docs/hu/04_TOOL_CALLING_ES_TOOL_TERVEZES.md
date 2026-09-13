# Tool Calling és tool-tervezés

Ez a dokumentum a tool-contractok technikai szerepét, a trust boundaryt és a legfontosabb tool-calling rendszertervezési kérdéseket foglalja össze.

## Tool-katalógus

A rendszer nyolc regisztrált toolt használ: `get_location_info`, `get_weather`, `convert_currency`, `search_hotels`, `search_attractions`, `search_restaurants`, `get_transport_options`, `calculate`.

Minden tool Pydantic input modellt használ `extra="forbid"` beállítással. Ez azt jelenti, hogy az agent által generált extra mezők sem némán kerülnek át a végrehajtásba. A `ToolRegistry` egységesen kezeli a schema exportot, validációt, végrehajtást, latencyt és trace-et.

A `calculate` tool nem használ Python `eval()`-t; AST whitelist alapján kizárólag numerikus aritmetikát enged. Weather és FX esetén külön local/auto/live provider mód létezik.

## Tool Calling rendszertervezési kérdések

Az alábbi kérdések nem interjúanyagok, hanem a rendszer tervezése és review-ja során használható engineering checklist.

Ez a dokumentum nem interjúkérdés-gyűjtemény. A célja, hogy egy tool-calling / agentic rendszer tervezésekor végigjárható legyen az a mérnöki checklist, amely meghatározza a toolok határait, a sémákat, az orchestrationt, a hibakezelést, az observabilityt és az evaluationt.

## 1. Capability és tool-határok

### 1.1 Mi legyen külön tool, és mi maradjon egy toolon belüli paraméter?
Egy tool egy koherens, jól tesztelhető üzleti capabilityt képviseljen. Ha két műveletnek eltérő adatforrása, hibamódja, jogosultsága vagy latencyprofilja van, általában indokolt külön toolként kezelni.

A projektben ezért külön tool a `search_hotels`, `search_restaurants` és `search_attractions`, miközben a hotel ár-, rating- és top-k szűrése ugyanannak a toolnak paramétere.

### 1.2 Mikor túl finom a tool-granularitás?
Ha az agent minden feladathoz 10–20 mikrolépést kénytelen végrehajtani, nő a latency, a hibafelület és a routing komplexitás. A tool ne legyen egyetlen adatbázis-oszlop lekérdezésére redukálva.

### 1.3 Mikor túl nagy egy tool?
Ha egyetlen tool egyszerre keres hotelt, időjárást, éttermet és számol budgetet, az agent elveszíti a composabilityt és nehezen mérhető, mely capability hibázott.

## 2. Tool schema és argumentumtervezés

### 2.1 Mely mezők legyenek kötelezők?
Csak olyan mező legyen required, amely nélkül a művelet értelmetlen. A túl sok kötelező mező felesleges argument-hallucinációt okoz.

### 2.2 Milyen típus- és tartománykorlátok kellenek?
Például `nights > 0`, `top_k` korlátozott, ár nem lehet negatív. A modell által generált JSON nem tekinthető megbízható belső adatnak.

### 2.3 Enum vagy szabad szöveg?
Szűk, stabil domain esetén enum jobb. Nyitott keresési térnél szabad szöveg szükséges. Cuisine vagy category esetén a projekt normalizálással kezeli a szabadabb inputot.

### 2.4 Hogyan kezeljük az opcionális feltételeket?
Az `None` jelentse azt, hogy nincs szűrés. Ne találjunk ki default üzleti feltételt csak azért, mert az agent nem adott meg valamit.

### 2.5 Hogyan verziózzuk a tool schema-t?
Schema-változásnál a teszteknek, promptnak, benchmark expected argumentumoknak és OpenAI tool definícióknak együtt kell változniuk. Production rendszerben érdemes explicit schema-versiont vezetni.

## 3. Tool selection és routing

### 3.1 LLM, szabályrendszer vagy külön classifier válasszon toolt?
A projekt mindhárom logikát összehasonlíthatóvá teszi: rule-based baseline, deterministic plan-execute és supervised ML router, plusz opcionális OpenAI direct tool calling.

### 3.2 Mikor használjunk high-precision lexical guardrailt?
Ha egy explicit szókapcsolat nagyon nagy pontossággal jelzi a capabilityt — például „ne keress hotelt” — érdemes deterministic override-ot alkalmazni a classifier false positive-ja ellen.

### 3.3 Mi történjen bizonytalan routingnál?
Lehetséges stratégiák: ne hívjon toolt, kérjen pontosítást, válassza a legvalószínűbb toolt, vagy használjon confidence thresholdot. A választás üzleti kockázatfüggő.

### 3.4 Hogyan mérjük a routing minőségét?
Nem elég az accuracy. Fontos a precision, recall, F1, exact tool-set accuracy és unnecessary tool-call rate is.

## 4. Argument extraction

### 4.1 Ki extrahálja a strukturált argumentumokat?
Lehet ugyanaz az LLM, külön parser, regex/heuristic réteg vagy külön extraction modell. A projekt ML routere szándékosan különválasztja a tool selectiont és a deterministic argument extractiont.

### 4.2 Mit tegyünk implicit vagy hiányzó argumentum esetén?
Ne értékeljük hibának azt, amit a felhasználói szövegből nem lehet kikövetkeztetni. A benchmark expected argumentumai csak inferálható mezőket kérjenek számon.

### 4.3 Hogyan kezeljük a magyar ragozást és entitásváltozatokat?
A `Bécs`, `Bécsbe`, `Bécsben`, `Bécsből` alakokat ugyanahhoz a canonical city értékhez kell kötni. Ez normalizációs probléma, nem tool-probléma.

## 5. Orchestration és dependency

### 5.1 Mely tool-hívások futtathatók párhuzamosan?
Független weather/hotel/restaurant keresések párhuzamosíthatók. Ha a második tool az első outputját használja, dependency graph szükséges.

### 5.2 Hogyan reprezentáljuk a dependency-t?
Egyszerű flow esetén explicit `PlanStep` és dependency lista elég. Bonyolultabb workflow-nál DAG/state-machine/agent graph lehet indokolt.

### 5.3 Kell-e maximális lépésszám?
Igen. Az agent loopnak legyen `max_steps`, hogy tool-loop vagy hibás újrapróbálkozás ne fusson végtelenül.

### 5.4 Mikor használjunk plan-then-execute módszert?
Ha fontos az inspectability és előre látható a feladat szerkezete. Dinamikus, tool-output alapján változó feladatnál iterative agent loop jobb lehet.

## 6. Validation és trust boundary

### 6.1 Megbízhatunk-e a model-generated JSON-ban?
Nem. A function schema segít, de application-side validation továbbra is szükséges.

### 6.2 Hol legyen a validáció?
A ToolRegistry előtt/ben. A projekt Pydantic input modellel validál minden végrehajtást.

### 6.3 Mi történjen invalid argumentumnál?
Strukturált, trace-elhető error térjen vissza. Ne crash-eljen az egész process és ne fusson részben invalid adattal.

## 7. Hibakezelés, timeout és retry

### 7.1 Mely hibák retry-olhatók?
Transient network/HTTP timeout igen; schema validation vagy negatív ár nem. Retry-policy legyen hibatípus-függő.

### 7.2 Mennyi retry engedélyezett?
Kevés, bounded retry. Exponential backoff használható live provider esetén.

### 7.3 Mi a fallback stratégia?
A projekt `local / auto / live` módja ezt demonstrálja. `auto` élő providert próbál, majd kontrollált local fallbackre vált.

### 7.4 Hogyan jelezzük a degraded módot?
A tool outputban/provider metadata-ban látszódjon, hogy live vagy fallback adatból született az eredmény.

## 8. Idempotency és mellékhatások

### 8.1 Read-only vagy write tool?
A jelenlegi toolok alapvetően read-only jellegűek. Foglalás, fizetés vagy email-küldés esetén sokkal szigorúbb approval és idempotency kontroll szükséges.

### 8.2 Hogyan kerüljük el a dupla végrehajtást?
Write toolnál idempotency key, transaction ID vagy deduplikáció szükséges. Tool retry nem jelentheti automatikusan a művelet megismétlését.

## 9. Jogosultság és biztonság

### 9.1 Mely toolokhoz milyen permission kell?
Tool-level allowlist és capability-based permission modell célszerű.

### 9.2 Hogyan kezeljük a secretet?
API kulcs `.env`/secret managerből jön, nem tool argumentumból és nem kerül trace-be.

### 9.3 Tool output tartalmazhat prompt injectiont?
Igen, külső web/data output untrusted content. A tool outputot adatként, nem új system instructionként kell kezelni.

### 9.4 Milyen inputot logolhatunk?
A usage DB jelenleg helyi. Production környezetben PII-redaction, retention policy és consent kérdés is szükséges.

## 10. Observability

### 10.1 Mit logoljunk minden tool-callnál?
Toolnév, step, argumentumok, success/error, latency, output metadata, provider és correlation/run ID.

### 10.2 Milyen aggregált metrikák fontosak?
Tool success rate, p50/p95 latency, tool frequency, multi-tool rate, unnecessary calls, error breakdown, top tool sequences és methodology success.

### 10.3 Miért kell trace és aggregate statisztika is?
A trace egyetlen hibát magyaráz. Az aggregate dashboard rendszerszintű trendet mutat.

## 11. Evaluation

### 11.1 Mi a ground truth?
Expected tool set + inferálható expected arguments + optional task-level success criteria.

### 11.2 Milyen dataset kell?
Single-tool, multi-tool, negation, indirect wording, noisy text, bilingual input, hard negatives és edge case-ek.

### 11.3 Hogyan kerüljük a leakage-et?
Train/validation/test között ne ugyanannak a sablonnak minimális variációi legyenek. A projekt normalizált pattern-overlap quality gate-et használ.

### 11.4 Miért kell challenge set?
A clean held-out teszt nem méri jól az indirekt és zajos queryket. Külön challenge corpus szükséges.

## 12. Latency és költség

### 12.1 Mennyit ér a jobb accuracy extra latencyért?
Ezt nem elméleti kérdésként, hanem Pareto trade-offként kell mérni. A methodology benchmark ezért accuracy és latency metrikákat is tárol.

### 12.2 Mikor cache-elhető egy tool?
Stabil read-only lookup igen; időjárás/FX csak megfelelő TTL-lel.

### 12.3 Hogyan csökkenthető a tool-call szám?
Jobb routing, kompozit tool csak indokolt esetben, parallel calls, korai stop és dependency-aware planning.

## 13. Data freshness

### 13.1 Mely adat élő, melyik fixture?
A rendszernek ezt expliciten dokumentálnia kell. A szintetikus hotel/restaurant inventory nem mutatható valódi foglalási adatként.

### 13.2 Hogyan mérjük az adat frissességét?
Provider timestamp, dataset generation metadata és hash használható.

## 14. Tesztelés

### 14.1 Mit unit teszteljünk?
Parser, schema validation, tool logic, routing, metric calculation és persistence.

### 14.2 Mit integration teszteljünk?
Agent → registry → tool → output láncot, multi-tool workflow-t, fallbacket és model artifact compatibilityt.

### 14.3 Mi legyen regressziós teszt?
Minden egyszer már előfordult érdemi hiba: például `Bécsbe`, magyar budget-szórend, sklearn model persistence mismatch és negált hotel intent.

## 15. Production továbbfejlesztési kérdések

- Kell-e authentication és per-user history?
- Kell-e async tool execution?
- Kell-e queue a hosszú toolokhoz?
- Kell-e distributed tracing?
- Kell-e explicit human approval write action előtt?
- Milyen SLA/SLO vonatkozik a toolokra?
- Mekkora retention engedhető a prompt/answer historyra?
- Mi történik provider outage alatt?
- Milyen model/tool schema version van egy adott trace mögött?
- Hogyan roll backeljük a router modellt?

A lényeg: tool callingnál nem az a fő tervezési kérdés, hogy „az LLM képes-e függvényt hívni”, hanem hogy a capability contracts, validation, dependency, failure semantics, evaluation és observability együtt mennyire kontrollált rendszert alkotnak.

## Összegzés

Egy jó agentrendszerben a tool nem puszta Python-függvény: explicit capability-határ, validált contract, megfigyelhető végrehajtási egység és biztonsági boundary.
