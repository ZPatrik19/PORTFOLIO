# Kísérleti protokoll – magyar változat

## Kontrollált változók
Prompt-stratégia benchmarknál rögzített:
- dataset és mintasorrend;
- modell/provider;
- decoding beállítások;
- max output token;
- parsing és evaluation logic;
- retry/cache szabályok.

Csak a promptstratégia változik.

## Külön experiment family-k
1. **Prompt strategy benchmark** – P0–P16 vagy custom promptok.
2. **Decoding sweep** – fix prompt mellett temperature/top_p/top_k.
3. **Ablation** – advanced promptból egy komponens eltávolítása.
4. **Provider/model comparison** – ugyanaz a prompt több modellen.
5. **Fine-tuned comparison** – base vs fine-tuned ugyanazon holdouton.

## Fő metrikák
- Accuracy;
- Macro Precision/Recall/F1;
- Weighted F1;
- per-class F1;
- invalid output / invalid JSON;
- input/output/total token;
- tokens per correct prediction;
- P50/P95/P99 latency;
- estimated cost;
- 95% bootstrap CI.

## Leakage szabály
Few-shot példa és fine-tuning train/validation nem származhat a final benchmark holdoutból.

## Mock értelmezése
A Mock provider software/pipeline demonstráció. Szimulált minőség- és latencykülönbségei nem valódi modell-teljesítményszámok.
