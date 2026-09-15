# Regression Testing and Golden Sets

## Purpose

Regression testing answers:

> Did a new code/prompt/index version make previously stable behavior worse?

It is different from a one-time benchmark.

## Current golden regression examples

The test suite currently locks deterministic intent behavior for examples such as:

```text
"Compare three books on transformers" -> COMPARISON
"Find PyTorch code example"          -> CODE_SEARCH
"Tanítsd meg a PCA-t"                -> LEARNING
```

## Recommended retrieval golden set

Maintain a small manually reviewed set, for example 50 questions:

```json
{
  "question_id": "gold_001",
  "question": "How does Docker bridge networking work?",
  "expected_documents": ["docker-doc"],
  "expected_chunks": ["chunk-..."],
  "must_find_by_k": 5
}
```

Every significant retrieval change should compare:
- Recall@5
- MRR
- nDCG@5
- P95 latency

## Threshold-based regression policy

Example policy:

```text
Recall@5 may not drop > 3 percentage points
MRR may not drop > 0.03
Citation correctness may not drop > 2 percentage points
P95 latency may not increase > 30% without documented reason
```

Thresholds should be project-specific and updated only with an explicit design decision.

## Prompt regression

For prompt changes, preserve:
- prompt profile name/version;
- model;
- generation parameters;
- retrieval/index version;
- expected answer characteristics;
- quality scores;
- token/cost footprint.

Do not compare two prompts if retrieval/model configuration changed silently at the same time.
