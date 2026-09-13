# Adatok és adatminőség

Ez a fejezet azt írja le, milyen adatokból dolgozik a rendszer, hogyan készülnek, hogyan kerülhető el a template leakage, és milyen quality gate-ek védik a benchmark hitelességét.

## 1. Adatréteg szerepe

A projekt kétféle adatot használ: travel inventory/lookup adatokat a toolokhoz, valamint címkézett nyelvi adatot az intent router tanításához és értékeléséhez. A nagy rekordszám önmagában nem cél; a projekt explicit minőségi gate-eket használ a sablonosság és leakage ellen.

## 2. Fő adatkészletek

- `cities.csv`: városmetaadat, pénznem, nyelv, timezone, koordináták, költségprofil.
- `hotels.csv`: 180 000 szintetikus hotelrekord.
- `attractions.csv`: 90 000 szintetikus POI.
- `restaurants.csv`: 90 000 szintetikus étteremrekord.
- `transport.csv`: közlekedési profilok.
- `weather_fallback.csv`: 72 000 offline időjárási fallback rekord.
- `fx_rates_fallback.csv`: offline deviza fallback.
- `sample_user_queries.csv`: 90 000 változatos példakérdés.
- `intent_router_dataset.csv`: 240 000 címkézett router példa.
- `intent_router_challenge.csv`: 36 000 indirekt/noisy challenge példa.
- `benchmark/agent_tasks.json`: 22 500 end-to-end agent benchmark eset.

## 3. Train/validation/test split

Az intent-router corpus 192 000 train, 24 000 validation és 24 000 held-out test sorra oszlik. A normalizált nyelvi minták között a train/test, train/validation és test/validation átfedés jelenleg 0. Ez fontosabb, mint egy véletlen rowsplit, mert csökkenti a template leakage kockázatát.

## 4. Data-quality gate-ek

A quality audit többek között ellenőrzi a normalizált query-diverzitást, split-overlapot és az entity-nevek skeleton diverzitását. Jelenleg 6/6 gate teljesül. Az audit JSON/CSV és grafikus artifactokat is generál a `06_results/data_quality/` alatt.

## 5. Szintetikus adatok jelentése

A hotel/restaurant/attraction rekordok tesztelésre készültek. A filtering és ranking valós programlogika, de az inventory nem live booking adat. Ezt a tool outputok `data_note` mezői is jelzik.

## 6. Reprodukálhatóság és hash-ek

A router tréning dataset SHA-256 hash-e elmentésre kerül a model metadata mellett. Ha az adat megváltozik, a setup stale modellként kezeli az artifactot, és egyszer újratanítja.

## 7. Miért nem elég a nagy adatmennyiség?

Korábbi iterációban sok rekord kevés alaptemplaten alapult. A jelenlegi audit ezért nem csak rowszámot néz, hanem normalizált minták számát, largest pattern share-t, duplicate-okat, split leakage-et és entity-name diverzitást.

## Összegzés

Az adatréteg célja nem a rekordszám maximalizálása, hanem a reprodukálható tool-végrehajtás és a generalizációt mérő routing benchmark támogatása.
