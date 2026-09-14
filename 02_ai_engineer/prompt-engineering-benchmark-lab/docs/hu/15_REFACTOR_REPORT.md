# Refaktor riport

## Kiinduló állapot
A projekt már értékes benchmark, UI, provider, vizualizációs és notebook funkciókkal rendelkezett, de portability/maintainability adósság halmozódott fel: nem standard package/import megoldások, nagy provider/UI modulok, szétszórt path/config feltételezések, túl sok feladatot végző setup és hiányos production/deployment dokumentáció.

## Azonosított problémák
- package/import működés fejlesztői layouttól függött;
- provider logika túl centralizált volt;
- több config/path referencia duplikált volt;
- a cache identity erősebb védelemre szorult;
- már előfordult UI regresszió token reshape körül;
- a setup véletlenül túl drága munkát indíthatott;
- a tesztek léteztek, de a rétegek és célok dokumentációja hiányos volt;
- deployment/security/reproducibility elvárások nem voltak konzisztensen HU/EN dokumentálva.

## Architektúraváltozások
- telepíthető `prompt_benchmark` package a `03_src` alatt;
- centralizált `ProjectPaths` és típusos config;
- providerenként szétválasztott adapterek közös base/retry contracttal;
- normalizált `07_outputs` artifact struktúra;
- CLI és cross-platform launcherek;
- Docker/Kubernetes assetek Streamlit health probe-bal.

## Clean Code javítások
A reusable logika ahol ésszerű volt kikerült notebook/UI rétegből, javult a naming/type/docstring minőség, specifikusabb lett az exception handling, megszűntek secret/path feltételezések, szigorúbb lett a cache- és adatvalidáció.

## Tesztelés
A jelenlegi suite 76 sikeres unit/integration/smoke tesztet tartalmaz, 77% konfigurált core-package coverage mellett. A pontos felelősségeket a kétnyelvű tesztkatalógus írja le.

## Ismert korlátok
Az élő provider viselkedés, Docker runtime és Kubernetes apply olyan toolt/credentialt/hálózatot igényel, amely a validációs környezetben nem garantált. A kétnyelvű Streamlit view fájlok továbbra is viszonylag nagyok a refaktorációs regressziókockázat csökkentése érdekében.

## Következő fejlesztések
CI quality gate, human-reviewed adversarial dataset, opt-in live-provider contract test, perzisztens multi-user storage és további UI componentization.
