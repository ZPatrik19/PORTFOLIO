# Fine-tuning Readiness

The project is prompt-engineering first, but exports leakage-safe SFT data from the development split. The final holdout never becomes training data.

## Implemented
`05_scripts/12_export_finetuning_data.py` creates training/validation JSONL under `01_data/fine_tuning/`.

## Why training is not automatic
Fine-tuning is provider/model/hardware/cost specific. Automatically starting a training job would create hidden cost and reduce reproducibility. Instead, the repository prepares validated data and evaluates any resulting model ID with the same locked benchmark.

## Fair comparison
Compare base model + baseline prompt, base model + optimized prompt, fine-tuned model + minimal prompt, and fine-tuned model + optimized prompt on the same holdout. The question is whether quality gain justifies training and maintenance cost.
