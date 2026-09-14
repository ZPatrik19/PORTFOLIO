# Kiértékelés

## Elsődleges minőségi metrikák
- Accuracy Wilson 95% konfidenciaintervallummal.
- Macro Precision, Macro Recall, Macro F1.
- Weighted F1 és balanced accuracy.
- Matthews Correlation Coefficient és Cohen's kappa.
- Class-wise precision/recall/F1 és confusion matrix.

## Statisztikai értelmezés
Egy kis pilot tökéletes eredménye nem bizonyít tökéletes modellminőséget. A Wilson interval és bootstrap F1 interval megmutatja a bizonytalanságot. Párosított összehasonlításnál fixed/regressed sample-ek és indokolt esetben McNemar teszt is használható.

## Robusztussági szeletek
Az eredmény difficulty és scenario family szerint is bontható, például ambiguous, multi-intent, noisy, long-context, prompt-injection, quoted-thread, multilingual, negation és log-noise esetekre.

## Output megbízhatóság
A benchmark a task accuracytól külön méri az output-contract validityt, invalid-output rate-et, JSON-validityt, parser hibákat és provider hibákat.

## Hatékonyság
Input/output/total token, token/correct prediction, P50/P95/P99 latency, teljes futási idő, throughput és becsült költség elsőrangú metrikák.

## Döntési elv
A production winner nem automatikusan a legmagasabb F1-et elérő stratégia. A reliability, latency, token overhead, cost, komplexitás és üzemeltetési támogatás ugyanúgy számít.
