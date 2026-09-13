# Modell, tanítás és kompatibilitás

Ez a fejezet a tanítható intent router pipeline-ját, splitstratégiáját, threshold-választását és biztonságos modellbetöltési logikáját dokumentálja.

## 1. Feladat

Az intent router multi-label klasszifikációt végez. Egy kérdés egyszerre több labelt kaphat, például `weather + restaurant + transport`.

## 2. Feature pipeline

```text
word TF-IDF (1–3 gram)
+ Unicode ékezet-normalizálás
      ↓
One-vs-Rest SGD logistic classifiers
      ↓
per-label probability
      ↓
validationon választott threshold
```

A Unicode ékezet-normalizálás csökkenti az ékezetes/ékezet nélküli alakok közötti eltérést; a magyar toldalékos és negációs edge case-eket a determinisztikus parsing guardrail-ek is támogatják.

## 3. Split és threshold

192k train / 24k validation / 24k held-out test. A thresholdok labelenként külön vannak kiválasztva precision-orientált célfüggvénnyel, mert az unnecessary tool call valós agent-hiba. A jelenlegi thresholdok a mentett metrics JSON-ban találhatók.

## 4. Aktuális modellmetrikák

- test micro-F1: ~0.783
- test macro-F1: ~0.785
- test hamming loss: ~0.131
- 36k challenge micro-F1: ~0.677

A challenge eredmény szándékosan alacsonyabb: indirekt/noisy nyelvezetet mér, ezért jobb generalizációs stresszteszt.

## 5. Modell persistence és runtime kompatibilitás

A joblib/scikit-learn artifact nem tekinthető verziófüggetlen formátumnak. Ezért külön `intent_router_metadata.json` rögzíti a Python/scikit-learn verziót és dataset hash-t. A setup betöltés előtt ellenőrzi a metaadatot. Eltérő runtime esetén nem unpickle-öl, hanem egyszer helyben újratanít.

## 6. Mikor történik retraining?

Csak ha nincs modell, runtime-incompatible, hiányzik/hibás a metadata, vagy megváltozott a training dataset hash. Normál startupkor nincs felesleges retraining.

## Összegzés

A router célja nem egy univerzális NLP-modell, hanem egy reprodukálható, mérhető tool-selection komponens. A data split és a runtime-compatibility ellenőrzés ugyanannyira fontos, mint maga a classifier.
