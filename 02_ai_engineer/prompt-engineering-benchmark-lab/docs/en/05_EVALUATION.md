# Evaluation

## Primary quality metrics
- Accuracy with Wilson 95% confidence interval.
- Macro Precision, Macro Recall, Macro F1.
- Weighted F1 and balanced accuracy.
- Matthews Correlation Coefficient and Cohen's kappa.
- Per-class precision/recall/F1 and confusion matrix.

## Statistical interpretation
A perfect score on a tiny pilot is not treated as proof of perfect model quality. Wilson intervals and bootstrap F1 intervals expose uncertainty. Paired comparisons can report fixed/regressed samples and McNemar testing when appropriate.

## Robustness slices
Performance can be grouped by difficulty and scenario family, including ambiguous, multi-intent, noisy, long-context, prompt-injection, quoted-thread, multilingual, negation, and log-noise cases.

## Output reliability
The benchmark records output-contract validity, invalid-output rate, JSON validity, parser errors, and provider errors independently from task accuracy.

## Efficiency
Input/output/total tokens, tokens per correct prediction, P50/P95/P99 latency, total elapsed time, throughput, and estimated cost are first-class metrics.

## Decision principle
The production winner is not automatically the strategy with the highest F1. Reliability, latency, token overhead, cost, complexity, and operational support matter as well.
