# Dokumentációs térkép

Ez az oldal a magyar technikai dokumentáció belépési pontja.

A dokumentáció célja, hogy a projektet elejétől a végéig, logikus sorrendben lehessen megérteni. Ha először látod a repositoryt, haladj a számozás szerint.

| # | Dokumentum | Mit magyaráz? |
|---|---|---|
| 01 | [Projekt áttekintés](01_PROJEKT_ATTEKINTES.md) | cél, use case, fő komponensek |
| 02 | [Architektúra és adatfolyam](02_ARCHITEKTURA_ES_ADATFOLYAM.md) | runtime komponensek, request flow, provider módok |
| 03 | [Adatok és adatminőség](03_ADATOK_ES_ADATMINOSEG.md) | datasetek, generálás, leakage, quality gate-ek |
| 04 | [Tool Calling és tool-tervezés](04_TOOL_CALLING_ES_TOOL_TERVEZES.md) | tool contract, schema, trust boundary, tervezési kérdések |
| 05 | [Agent routing és orchestration](05_AGENT_ROUTING_ES_ORCHESTRATION.md) | rule-based, plan-execute, ML router, OpenAI direct |
| 06 | [Modell, tanítás és kompatibilitás](06_MODELL_TANITAS_ES_KOMPATIBILITAS.md) | TF-IDF router, split, thresholdok, model persistence |
| 07 | [Evaluation és statisztikák](07_EVALUATION_ES_STATISZTIKAK.md) | metrikák, benchmark, live/project statistics |
| 08 | [UI és használat](08_UI_ES_HASZNALAT.md) | setup, Streamlit lapok, saját kérdések, presetek |
| 09 | [Kódreferencia és repository audit](09_KODREFERENCIA_ES_REPOSITORY_AUDIT.md) | fájlok szerepe, mi marad és miért |
| 10 | [Korlátok és további irányok](10_KORLATOK_ES_TOVABBI_IRANYOK.md) | jelenlegi scope és production irány |
| 11 | [Preset kérdések](11_PRESET_KERDESEK.md) | 30 HU/EN scenario és regressziós szerepük |
| 12 | [Vizualizációs rendszer](12_VIZUALIZACIOS_RENDSZER.md) | Light/Dark téma, kontraszt, egységes chartmagasság és Plotly render pipeline |

A diagramok közösek, és a `../shared/visuals/` mappában találhatók.

## Ajánlott olvasási sorrend

Első olvasáskor haladj 01-től 12-ig. Hibakereséskor közvetlenül a releváns témadokumentumhoz ugorhatsz.
