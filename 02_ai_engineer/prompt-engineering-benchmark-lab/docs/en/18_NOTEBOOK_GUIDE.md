# Notebook Guide

1. `01_data_understanding.ipynb` — business problem, dataset profile, split rationale.
2. `02_prompt_design.ipynb` — P0–P16 hypotheses and prompt design.
3. `03_benchmark.ipynb` — benchmark execution and leaderboard interpretation.
4. `04_error_analysis.ipynb` — confusion, fixed/regressed examples, hard cases.
5. `05_prompt_ablation.ipynb` — component-level prompt ablation.
6. `06_final_analysis.ipynb` — executive engineering decision summary.
7. `07_advanced_prompt_engineering.ipynb` — persona/context/constraints/reasoning/branching concepts.
8. `08_decoding_parameters.ipynb` — temperature/top_p/top_k sweep design.
9. `09_prompt_template_and_output_validation.ipynb` — reusable prompt template and output contracts.
10. `10_finetuning_readiness.ipynb` — leakage-safe SFT export and fine-tuning decision criteria.

Notebook principle: **Theory → Code → Result → Interpretation → Engineering decision**. Reusable business logic belongs in the package, not duplicated in notebooks.
