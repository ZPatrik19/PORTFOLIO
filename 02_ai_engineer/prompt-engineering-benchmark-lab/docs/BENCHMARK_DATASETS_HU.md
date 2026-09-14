# Benchmark dataset stratégia

A Prompt Engineering Benchmarkban **nem egyetlen datasetből** érdemes következtetést levonni.

## 1. Offline Prompt Challenge Set — fő prompt-robosztussági benchmark

Útvonal: `01_data/processed/benchmark.csv`

A projekt saját, determinisztikusan generált challenge setje. Nem valódi ügyféladat, hanem kontrollált kísérleti adat. Ennek előnye, hogy ismert a ground truth és célzottan tartalmaz olyan eseteket, amelyek megmutatják a prompttechnikák közötti különbségeket:

- `easy_clear`
- `implicit_request`
- `ambiguous_boundary`
- `multi_intent_primary`
- `noisy_typo`
- `long_context`
- `prompt_injection`
- `resolved_history`

Ez a suite a legjobb a prompt-design komponensek **robosztussági összehasonlítására**, de önmagában nem bizonyít valós production teljesítményt.

## 2. Hugging Face Support-Ticket-Router külső suite — generalizációs kontroll

A UI a Benchmark oldalon egy gombbal letölti és elkészíti:

`01_data/benchmark_suites/hf_support_router_300.csv`

A few-shot példák külön a train splitből készülnek:

`01_data/benchmark_suites/hf_support_router_few_shot.json`

A dataset független a projekt saját generátorától, ezért hasznos második kontroll. Fontos azonban, hogy a Hugging Face dataset dokumentációja szerint ez is szintetikus GPT-generált adat, és a készítők az ambiguous/noisy példák jelentős részét kiszűrték. Emiatt nem helyettesíti a Challenge Set nehéz eseteit.

## 3. Saját CSV — legjobb domain-specifikus benchmark

A UI-ból közvetlenül feltölthető.

Kötelező oszlopok:

```text
text,label
```

Engedélyezett label-ek:

```text
api
billing
cancellation
complaint
technical
upgrade
```

Ha van saját, ember által címkézett vagy üzleti rendszerből anonimizált adat, ez adja a legerősebb portfolio bizonyítékot.

## Ajánlott értékelési protokoll

1. Prompt fejlesztés: `development.csv`
2. Prompt robustness: Offline Challenge Set
3. Generalizáció: Hugging Face external suite
4. Ha van: saját holdout CSV
5. Ugyanaz a provider/modell/sampling beállítás minden összehasonlított promptnál
6. Ne csak Macro F1-et nézz: token, P95 latency, output validity és esettípusonkénti pontosság is fontos

## Miért jobb ez egyetlen datasetnél?

Egy tiszta, könnyű intent dataset könnyen azt a hamis következtetést adhatja, hogy minden prompttechnika ugyanannyira jó. Egy kizárólag adversarial dataset pedig túl pesszimista lehet. A háromszintű benchmark külön választja:

- alap task quality,
- prompt robustness,
- external generalization.
