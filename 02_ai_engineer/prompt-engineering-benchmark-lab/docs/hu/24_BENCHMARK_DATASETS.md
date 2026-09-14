# Benchmark datasetek

## Challenge Set
A beépített szintetikus Challenge Set kontrollált stressztesztre készült, nem production-distribution realizmust állít. 10 800 raw sor, 6 000 holdout, 3 000 development minta, hat label és 18 scenario family tartozik hozzá.

## Scenario design
Az esetek között van clear intent, implicit request, ambiguity, multi-intent priority, typo/noise, long context, prompt injection, resolved history, negation/correction, quoted thread, multilingual mix, telegraphic text, primary-first/last, distractor, code/log noise, label-word attack és double negation.

## Külső adat
A Hugging Face support-ticket forrás hasznos kiegészítő generalization setként, de maga is szintetikus; nem szabad production ground truth-ként leírni.

## Erős portfólióbizonyíték
A legerősebb evidence: locked evaluation protocol challenge suite + független külső set + lehetőleg human-reviewed/private holdout kombináción, valódi provider/model futással.
