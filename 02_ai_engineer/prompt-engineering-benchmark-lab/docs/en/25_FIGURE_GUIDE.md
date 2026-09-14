# Figure Guide

## Quality with uncertainty
Use Accuracy with Wilson CI and Macro F1 with sample size. Never interpret a small 1.0 score without uncertainty.

## Quality vs tokens / latency / cost
These charts answer whether quality gain justifies operational overhead. A Pareto-efficient prompt dominates one that is slower/more expensive with no quality gain.

## Scenario heatmap
Shows *where* a strategy helps: ambiguity, prompt injection, long context, multi-intent, etc. This is often more informative than one aggregate F1.

## Confusion matrix
Shows which intent pairs are confused. `__invalid__` represents parse/contract failures rather than a semantic class.

## Fixed vs regressed
Paired comparison against baseline distinguishes examples repaired by a strategy from examples it breaks.

## Parameter sweep
Interpret only with a fixed prompt; otherwise decoding and prompt effects are confounded.
