# Figure Guide — What Every Chart Means and What Decision It Supports

This guide explains how to read every generated figure. It is designed for portfolio presentation and interview preparation.

## 00 — Experiment pipeline

**File:** `00_experiment_pipeline.png`

**Question answered:** What is the full experimental workflow?

**Interpretation:** The final benchmark changes the prompt strategy while holding the model, holdout and evaluation logic constant.

**Decision supported:** Whether the experiment is controlled enough to attribute changes to prompt design.

## 00b — Leakage-safe split

**File:** `00b_leakage_safe_split.png`

**Question answered:** Where do prompt examples, development rows and final benchmark rows come from?

**Interpretation:** The three subsets are disjoint.

**Decision supported:** Whether the benchmark avoids obvious few-shot leakage and prompt tuning on the final test set.

## 00c — Prompt strategy ladder

**File:** `00c_prompt_strategy_ladder.png`

**Question answered:** What engineering intervention does each P0–P16 version introduce, and which strategies belong to the core ladder versus the advanced layer?

**Interpretation:** The progression is incremental rather than a collection of unrelated prompts.

**Decision supported:** Whether observed changes can be connected to a meaningful prompt component.

## 00d — Production trade-off

**File:** `00d_production_tradeoff.png`

**Question answered:** Why is the highest F1 not automatically the production winner?

**Interpretation:** Quality, reliability and efficiency must be considered together.

**Decision supported:** Final prompt selection.


## 00e — Free-first provider decision map

**File:** `00e_free_provider_options.png`

**Question answered:** Which execution mode should I choose if I want to avoid API cost?

**Interpretation:** `mock` validates the software pipeline, while Ollama, Groq and Gemini can produce real LLM benchmark evidence through different local/cloud trade-offs.

**Decision supported:** Choosing the cheapest credible execution path without confusing mock outputs with model evidence.

## 01 — Class distribution

**File:** `01_class_distribution.png`

**Question answered:** Is each intent equally represented in the benchmark?

**Good sign:** Equal bars for all six classes.

**Decision supported:** Whether Macro F1 and per-class comparisons are based on a controlled balanced holdout.

## 02 — Text length distribution

**File:** `02_text_length_distribution.png`

**Question answered:** How long are support messages?

**Interpretation:** Long-tail message length can affect token usage and potentially difficulty.

**Decision supported:** Whether token/cost results need to be interpreted in light of unusually long inputs.

## 03 — Macro F1 comparison

**File:** `03_macro_f1_comparison.png`

**Question answered:** Which strategy gives the strongest class-balanced classification quality?

**Primary use:** Main quality leaderboard visualization.

**Do not conclude:** A slightly higher bar is automatically statistically or operationally meaningful.

## 04 — Accuracy comparison

**File:** `04_accuracy_comparison.png`

**Question answered:** What fraction of tickets is classified correctly overall?

**Why secondary to Macro F1:** Accuracy can hide class-specific weakness.

## 05 — Invalid output rate

**File:** `05_invalid_output_rate.png`

**Question answered:** How often does a strategy violate its expected output contract?

**Decision supported:** Downstream reliability and parser complexity.

**Especially useful for:** P6 prompt-only JSON vs P7 schema-enforced output.

## 06 — Mean token usage

**File:** `06_token_usage.png`

**Question answered:** How much context/output does each strategy consume per request?

**Interpretation:** Few-shot and long decision policies usually create token overhead.

**Decision supported:** Whether quality improvement justifies prompt size.

## 07 — P95 latency comparison

**File:** `07_latency_comparison.png`

**Question answered:** What is the tail latency of each prompt strategy?

**Why P95:** Production users care about slow requests, not only the average.

## 08 — Cost comparison

**File:** `08_cost_comparison.png`

**Question answered:** How much would 1,000 requests cost under the configured token prices?

**Decision supported:** Production economics.

**Important:** Update `configs/pricing.yaml` before interpreting this plot.

## 09 — Quality vs cost

**File:** `09_quality_vs_cost.png`

**Question answered:** How much Macro F1 do we obtain for the API cost?

**Ideal direction:** High F1 and low cost.

**Decision supported:** Identifying cost-quality dominated strategies.

## 10 — Quality vs latency

**File:** `10_quality_vs_latency.png`

**Question answered:** Does higher classification quality require slower inference?

**Ideal direction:** High F1 and low P95 latency.

## 11 — Macro F1 bootstrap confidence interval

**File:** `11_macro_f1_bootstrap_ci.png`

**Question answered:** How uncertain is each measured Macro F1 on this finite benchmark?

**Interpretation:** Very small score differences should be treated cautiously when uncertainty is large.

**Do not conclude:** CI overlap alone is a complete significance test.

## 12 — Per-class F1 heatmap

**File:** `12_per_class_f1_heatmap.png`

**Question answered:** Which prompt helps or hurts which intent?

**High value:** This can reveal that a prompt improves `api` but regresses `cancellation`, even when overall F1 changes only slightly.

**Decision supported:** Whether a global improvement is acceptable for business-critical classes.

## 13 — Error rate by class

**File:** `13_error_rate_by_class.png`

**Question answered:** Where are systematic errors concentrated?

**Interpretation:** Dark/high-error cells identify class-strategy combinations needing qualitative investigation.

## 14 — Latency distribution boxplot

**File:** `14_latency_distribution_boxplot.png`

**Question answered:** Is latency stable or driven by outliers?

**Why stronger than one mean:** Shows median, spread and outliers across individual requests.

## 15 — Token distribution boxplot

**File:** `15_token_distribution_boxplot.png`

**Question answered:** How variable is token consumption across requests?

**Interpretation:** Variation can come from ticket length even when prompt template is fixed.

## 16 — Baseline confusion matrix

**File:** `16_confusion_matrix_baseline.png`

**Question answered:** What are the baseline's systematic confusion patterns?

**Use:** Establish the error pattern before prompt optimization.

## 17 — Best-quality confusion matrix

**File:** `17_confusion_matrix_best.png`

**Question answered:** Which baseline confusions disappeared or remained after optimization?

**Use:** Compare error structure, not only headline F1.

## 18 — Relative F1 improvement

**File:** `18_relative_f1_improvement.png`

**Question answered:** How many absolute Macro F1 points does each strategy gain or lose versus P0?

**Why useful:** Makes the baseline-relative effect immediately visible.

## 19 — Fixed vs regressed samples

**File:** `19_fixed_vs_regressed_samples.png`

**Question answered:** How many individual baseline mistakes were corrected and how many new mistakes were introduced?

**Decision supported:** Whether the optimization is a true improvement rather than a redistribution of errors.

**Note:** This chart is created only when the best-quality strategy differs from P0.

## 20 — Prompt length vs F1

**File:** `20_prompt_length_vs_f1.png`

**Question answered:** Does prompt complexity correlate with quality?

**Important:** Prompt length here is an approximate template-size estimate. Real token usage remains the authoritative cost measurement.

**Decision supported:** Removing unnecessary prompt components.

## 21 — Token usage vs cost

**File:** `21_token_usage_vs_cost.png`

**Question answered:** Does the configured pricing behave as expected relative to token usage?

**Use:** Sanity-check the economic story of the benchmark.

## Ablation — Macro F1

**File:** `ablation_macro_f1.png`

**Question answered:** Which component of the full prompt actually contributes quality on the development set?

**Decision supported:** Prompt simplification before freezing the final version.

# Recommended README figures after a real benchmark

For a recruiter-facing GitHub README, the most valuable result figures are usually:

1. `03_macro_f1_comparison.png`
2. `09_quality_vs_cost.png`
3. `12_per_class_f1_heatmap.png`
4. `16_confusion_matrix_baseline.png`
5. `17_confusion_matrix_best.png`
6. `19_fixed_vs_regressed_samples.png` when available
7. `ablation_macro_f1.png`

These seven visuals tell a coherent story: **quality → economics → class-level behaviour → error movement → component value**.
