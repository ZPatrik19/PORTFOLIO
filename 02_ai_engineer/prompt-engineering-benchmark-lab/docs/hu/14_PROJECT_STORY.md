# Projekt történet — interjúnarratíva

## Probléma
A célom az volt, hogy a prompt engineeringet mérhető AI engineeringként mutassam be, ne promptgyűjteményként. A feladatnak quality, reliability, efficiency és robustness trade-offokat kellett láthatóvá tennie.

## Kihívás
A nehéz rész az experiment integritása volt: leakage-mentes split, fair promptösszehasonlítás, structured-output hibák, külső provider instabilitás, félrevezető kis pilotok, stale cache, token/cost accounting és history megőrzés UI sessionök között.

## Megközelítés
Szétválasztottam az adatkezelést, prompt constructiont, provider adaptereket, benchmark futtatást, parsing/validációt, evaluationt, perzisztenciát és presentation réteget. Ugyanazt a core runnert használja a CLI, notebook és UI.

## Fontos engineering döntések
- A Macro F1 elsődleges, de soha nem jelenik meg önmagában.
- Kis tökéletes pilotnál confidence interval jelenik meg.
- Az output validity külön metrika a szemantikai helyességtől.
- A prompt- és decoding-kísérletek el vannak választva.
- A mock eredmény egyértelműen simulation.
- A branch-and-vote csak végső branch döntéseket tárol, privát chain-of-thoughtot nem.
- Minden befejezett run auditálható raw prediction és manifest alapján.

## Kiértékelés
A rendszer task qualityt, statisztikai bizonytalanságot, scenario robustnessot, formatting reliabilityt, tokent, latencyt, throughputot és costot mér. A párosított összehasonlítás javításokat és regressziókat is mutat.

## Productionization
A repository telepíthető package lett típusos configgal, hordozható pathokkal, provider izolációval, retry-val, tesztekkel, cross-platform launcherrel, Docker/Kubernetes assetekkel, strukturált dokumentációval és security szabályokkal.

## Tanulság
Egy F1-ben nyerő prompt productionben veszíthet token overhead, latency, malformed output, provider support vagy adversarial inputokra való törékenység miatt. A benchmark ezeket a trade-offokat teszi láthatóvá.
