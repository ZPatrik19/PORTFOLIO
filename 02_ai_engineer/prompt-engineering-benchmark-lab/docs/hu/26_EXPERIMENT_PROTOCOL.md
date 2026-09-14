# Kísérleti protokoll

## Kontrollált prompt benchmark
Fix dataset/sorrend, provider/model, decoding beállítás, max output token, parser, metric kód, retry/cache szabály. Csak a promptstratégia változzon.

## Experiment családok
1. Prompt strategy benchmark.
2. Decoding sweep fix prompttal.
3. Prompt ablation.
4. Provider/model összehasonlítás fix prompttal.
5. Base vs fine-tuned ugyanazon holdouton.

## Leakage policy
Few-shot példa és fine-tuning train/validation sor nem származhat a final holdoutból.

## Run identity
Rögzíteni kell providert, modellt, stratégiát, sampling settingset, dataset/sample identitást, prompt verziót, pricing configot, timestampet és raw predikciókat. Cache csak kompatibilis identitás mellett használható újra.

## Riportolás
Aggregate és sliced metrika, uncertainty, invalid output, token, latency és cost együtt jelenjen meg. Egyértelműen jelezni kell, hogy mock simulation vagy valódi provider evidence.
