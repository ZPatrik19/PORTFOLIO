# Fine-tuning readiness

A projekt elsősorban prompt-engineering benchmark, de leakage-mentes SFT adatot exportál a development splitből. A final holdout soha nem kerül training adatba.

## Implementálva
A `05_scripts/12_export_finetuning_data.py` train/validation JSONL-t készít az `01_data/fine_tuning/` mappába.

## Miért nem automatikus a training?
A fine-tuning provider/model/hardver/költségfüggő. Automatikus training job rejtett költséget és rosszabb reprodukálhatóságot okozna. A repository ezért validált adatot készít, majd a létrejött model ID-t ugyanazon locked benchmarkkal értékeli.

## Fair összehasonlítás
Ugyanazon holdouton hasonlítsd össze: base model + baseline prompt, base model + optimized prompt, fine-tuned model + minimal prompt, fine-tuned model + optimized prompt. A kérdés az, hogy a quality gain indokolja-e a training és maintenance költséget.
