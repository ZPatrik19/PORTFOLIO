# Engineering döntések – magyar változat

## 1. Miért klasszifikáció?
A support-ticket routing egyszerűen értelmezhető, objektív ground truth címkékkel mérhető, ezért jól izolálja a prompt engineering hatását.

## 2. Miért fix benchmark?
Ugyanazokat a mintákat kell minden promptnak látnia. Máskülönben a prompt és a dataset közti különbség összekeveredne.

## 3. Miért development + holdout?
A promptokat development adaton lehet iterálni. A holdout benchmarkot csak végső mérésre használjuk, hogy csökkentsük a prompt overfittinget.

## 4. Miért balanced benchmark?
Minden kategória azonos súlyt kap. Így az Accuracy sem tud pusztán egy domináns osztály miatt magas lenni.

## 5. Miért Macro F1 az elsődleges minőségi metrika?
Minden osztály F1 értéke azonos súllyal számít. Ez jól mutatja, ha egy prompt csak bizonyos intenteken erős.

## 6. Miért nem elég az Accuracy?
Az Accuracy nem mutatja meg, melyik osztály romlik. Ezért per-class precision/recall/F1 és confusion matrix is készül.

## 7. Miért mérünk invalid outputot?
Production környezetben a kimenetet programkód dolgozza fel. A magas minőségű, de parse-olhatatlan output üzletileg hibás válasz.

## 8. Miért külön P6 és P7?
P6 csak prompttal kér JSON-t; P7 provider-szintű schema constraintet használ. Így mérhető a prompt-only és constrained generation különbsége.

## 9. Miért külön token, latency és cost?
A magasabb F1 nem feltétlenül jobb production döntés, ha 4× több tokent vagy 3× nagyobb P95 latencyt igényel.

## 10. Miért P95/P99 latency?
Az átlag elfedi a lassú tail requesteket. Felhasználói SLA szempontból a P95/P99 gyakran fontosabb.

## 11. Miért külön decoding sweep?
A prompt és a sampling paraméter egyidejű változtatása confoundingot okozna. Ezért prompt benchmarknál fix sampling, parameter sweepnél fix prompt van.

## 12. Miért top_p/top_k capability matrix?
Nem minden provider támogatja ugyanazt a decoding paramétert. Nem küldünk nem támogatott mezőt csak azért, hogy a UI egységesnek tűnjön.

## 13. Miért nincs hidden Chain-of-Thought mentés?
A projekt döntési policy-t, provider reasoning effortot és külső branch-and-vote aggregációt használ. Nem kér vagy tárol privát gondolatmenetet.

## 14. Miért Tree-of-Thought-inspired branch + vote?
Több független szakértői ág külön végső címkét ad, majd majority vote dönt. Így a multi-path reasoning hatása mérhető anélkül, hogy hidden reasoningot tárolnánk.

## 15. Miért case_type bontás?
Egy globális F1 nem mondja meg, hogy a technika miért jobb. A `prompt_injection`, `ambiguous_boundary`, `multi_intent_primary`, `long_context` stb. bontás mutatja a mechanizmust.

## 16. Miért prompt-sensitive Mock?
API-key nélkül is demonstrálható az egész benchmark pipeline. Az eredmények szimuláltak és egyértelműen jelölve vannak; valódi LLM evidence-nek nem használjuk őket.

## 17. Miért külön token_source / latency_source?
Mocknál `estimated_mock` / `simulated_mock`; valódi provider esetén provider-reported token és wall-clock latency. Ez megakadályozza a félreértelmezést.

## 18. Miért menthetők a raw predictionök?
Aggregált metrikából nem lehet hibaanalízist végezni. A raw outputból visszanézhető minden rossz, invalid vagy regressziós eset.

## 19. Miért resumable/cache-elt a benchmark?
Cloud API quota vagy hálózati hiba miatt megszakadhat a futás. A már elkészült requesteket nem érdemes újrafizetni/újrafuttatni.

## 20. Miért dataset fingerprint/cache védelem?
Azonos sample ID más adathalmazban nem jelent azonos ticketet. A cache nem használható csendben eltérő datasetre.

## 21. Miért külön custom prompt?
A portfolio ne csak előre definiált P0–P16 recepteket tudjon. A felhasználó saját promptot írhat, menthet és ugyanazzal a benchmarkkal értékelhet.

## 22. Miért `{ticket}` template contract?
A custom prompt reprodukálhatóvá válik, és garantáltan ugyanoda kerül a benchmark input minden mintánál.

## 23. Miért magyar és angol UI?
A tanulási/dokumentációs élmény magyarul kényelmesebb, miközben a GitHub/engineering kód és az angol benchmark is megmarad nemzetközi portfólióhoz.

## 24. Miért OpenRouter Free integráció?
API-költség nélkül több valódi cloud modellt lehet kipróbálni. Szigorú benchmarkhoz viszont fix model ID ajánlott, mert a free router háttérmodellje változhat.

## 25. Miért fine-tuning export, nem automatikus training?
A training provider/hardver/költségfüggő. A projekt leakage-safe JSONL-t készít, de nem indít véletlenül fizetős jobot.

## 26. Production prompt kiválasztás
A döntés többdimenziós: quality + reliability + tokens + latency + cost + complexity. A legmagasabb F1 önmagában nem automatikusan a legjobb production választás.
