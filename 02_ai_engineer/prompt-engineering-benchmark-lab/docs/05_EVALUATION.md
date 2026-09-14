# Evaluation

## Primary quality metric
Macro F1 is the primary classification quality metric because every intent class receives equal weight. Accuracy remains useful but is presented with a Wilson confidence interval, especially for small pilot runs.

## Metric groups

### Quality
- Accuracy + Wilson 95% CI
- Macro Precision / Recall / F1
- Weighted F1
- Balanced Accuracy
- Matthews correlation coefficient
- Cohen's kappa

### Reliability
- Output-contract valid rate
- Invalid output rate
- JSON validity / invalid JSON rate
- API error rate

### Efficiency
- input/output/total tokens
- total benchmark tokens
- tokens per correct prediction
- P50/P95/P99 latency
- sequential throughput
- estimated request / 1000-request / benchmark cost

### Diagnostics
- class-wise precision/recall/F1
- confusion matrix including `__invalid__`
- difficulty breakdown
- scenario heatmap
- hard examples
- baseline-fixed vs regressed examples
- paired McNemar comparison in the UI/report workflow

## Statistical interpretation
A 1.00 score on a very small pilot is not presented as proof of perfect model quality. The UI exposes sample count and confidence intervals and recommends larger/harder suites when strategies are indistinguishable.

## Mock result policy
Mock scores validate experiment mechanics only. They must not be cited as real model benchmark evidence.
