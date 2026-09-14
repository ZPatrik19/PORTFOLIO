# Prompt Engineering vs Fine-tuning

A projekt fő fókusza prompt engineering. A fine-tuning külön baseline lehet, de csak leakage-safe módon.

## Mi van implementálva?

A `05_scripts/12_export_finetuning_data.py` a DEVELOPMENT splitből SFT JSONL fájlokat készít:

- `01_data/fine_tuning/sft_train.jsonl`
- `01_data/fine_tuning/sft_validation.jsonl`

A final `benchmark.csv` NEM kerül training adatba.

## Miért nincs automatikus cloud fine-tuning?

Mert provider- és model-specifikus, költséges lehet, és hardver/API függő. A repository ettől még fine-tuning-ready: ha létrejön egy fine-tuned model ID, ugyanazzal a benchmark runnerrel kell értékelni.

Példa:

```bash
python 05_scripts/04_run_benchmark.py \
  --provider openai \
  --model YOUR_FINE_TUNED_MODEL_ID \
  --strategy p16_full_advanced_template
```

Így fair összehasonlítás készíthető:

- base model + zero-shot
- base model + optimized prompt
- fine-tuned model + minimal prompt
- fine-tuned model + optimized prompt

A döntési kérdés nem az, hogy "fine-tuning jobb-e", hanem hogy a quality gain indokolja-e a training/maintenance költséget.
