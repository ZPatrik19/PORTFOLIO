# Architektúra és adatfolyam

Ez a fejezet a rendszer technikai felépítését és a request teljes életútját mutatja be.

## 1. Rendszerhatár

A rendszer bemenete egy természetes nyelvű utazási kérdés. A kimenet egy ember számára olvasható válasz, amely mögött strukturált tool-hívások, validáció és trace áll.

## 2. Runtime folyamat

```text
Felhasználó / Streamlit UI
        ↓
AgentService
        ↓
Routing / Planner
        ↓
ToolRegistry
        ↓
Pydantic validáció
        ↓
Tool execution
        ↓
Local CSV / Live API
        ↓
Structured result + trace
        ↓
Válasz + usage analytics
```

A fontos architekturális döntés, hogy minden routing módszer ugyanazt a `ToolRegistry` réteget használja. Emiatt a rule-based, plan-execute, ML-router és OpenAI-direct útvonal ugyanazokat a tool-contractokat hajtja végre.

## 3. Fő komponensek

- `08_ui/app.py`: böngészős felület.
- `agent/service.py`: egységes agent belépési pont.
- `agent/*`: routing és orchestration stratégiák.
- `tools/*`: tool-contractok és végrehajtás.
- `data_store.py`: lokális adatréteg és gyorsítótárazott city lookup.
- `training/*`: intent router tanítás, modellmetaadat, freshness check.
- `evaluation/*`: benchmark és metrikák.
- `usage/*`: SQLite-alapú élő használati statisztika.

## 4. Provider módok

`local`: csak reprodukálható lokális adatok.

`auto`: megpróbál live API-t, hiba esetén lokális fallback.

`live`: kizárólag live API, hiba esetén nincs automatikus lokális eredmény.

Az időjárásnál Open-Meteo, devizánál Frankfurter lehet a live provider. A hotel/restaurant/attraction inventory szintetikus, ezért booking vagy availability rendszerként nem értelmezhető.

## 5. Dependency-k és párhuzamosság

A független tool-hívások elméletileg párhuzamosíthatók. A kalkulátor viszont lehet downstream dependency: például a hotel nightly price és nights eredményéből teljes költséget számol. A projekt jelenlegi implementációja egyszerű, jól trace-elhető orchestrationt preferál a túl korai async komplexitás helyett.

## 6. Megfigyelhetőség

Minden tool-hívás trace rekordot készít: tool név, argumentumok, output, latency, success/error. A Chat futások összesített metaadata SQLite-ba kerül, így a UI Live Statistics lapja tényleges használati adatból számol.

## 7. Diagramok

- `../shared/visuals/01_runtime.png` — runtime komponensek.
- `../shared/visuals/02_data_flow.png` — adatfolyam.
- `../shared/visuals/03_agent_loop.png` — agent loop.
- `../shared/visuals/04_provider_modes.png` — provider módok.
- `../shared/visuals/06_ui_flow.png` — UI folyamat.

## Összegzés

Az architektúra tudatosan moduláris: routing, tool-contract, adatforrás, evaluation és UI külön réteg. Ez teszi lehetővé, hogy ugyanazt a tool-réteget több agent-stratégiával lehessen mérni.
