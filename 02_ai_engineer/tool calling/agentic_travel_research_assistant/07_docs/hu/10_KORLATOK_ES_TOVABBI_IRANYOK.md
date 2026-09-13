# Korlátok és további irányok

Ez a fejezet egyértelműen elválasztja, mit tud jelenleg a rendszer, és milyen fejlesztések lennének valóban indokoltak egy következő szinten.

## 1. Jelenlegi korlátok

- A hotel/restaurant/attraction inventory szintetikus, nem live availability.
- A travel domain 60 városra és előre definiált capabilitykre korlátozott.
- Az ML router klasszikus TF-IDF modell, nem szemantikus encoder.
- A deterministic argument parser továbbra is heurisztikus.
- A workflow nem használ production queue-t vagy distributed tracinget.
- A live API integráció időjárásra és devizára korlátozott.
- A Chat válaszminőség a routing módszertől függ; offline módban nincs általános generatív reasoning.

## 2. Értelmes következő fejlesztések

1. Valódi hotel/flight/POI provider integráció megfelelő API szerződésekkel.
2. Async parallel execution független toolokra.
3. DAG/dependency executor komplex workflow-khoz.
4. Semantic router vagy compact transformer összehasonlítása TF-IDF baseline-nal.
5. OpenTelemetry-kompatibilis trace és production monitoring.
6. Tool-level timeout/retry/circuit-breaker policy.
7. Human approval side-effecting toolok előtt.
8. Secrets manager és per-tool permission modell.
9. Evaluation bővítése adversarial/prompt-injection és schema-failure esetekkel.
10. Backend/API réteg Streamlittől különválasztva.

## 3. Mit nem érdemes csak a komplexitás kedvéért hozzáadni?

Multi-agent framework, Kubernetes vagy vector database önmagában nem teszi jobbá ezt a projektet. Csak akkor indokolt, ha konkrét új követelmény — skálázás, állapotmegosztás, szemantikus retrieval vagy side-effect workflow — támasztja alá.

## Összegzés

A jelenlegi projekt koherens portfolio-scope. A következő lépcsőnek valódi új rendszerkövetelményt kell megoldania, nem pusztán új technológiát hozzáadnia.
