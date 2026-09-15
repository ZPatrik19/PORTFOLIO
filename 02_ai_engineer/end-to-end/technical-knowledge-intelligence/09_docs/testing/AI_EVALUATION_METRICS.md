# AI Evaluation Metrics

## Why this document exists

AI quality must not be reduced to a single "accuracy" number. A RAG system has multiple independently failing stages:

```text
query
→ retrieval
→ reranking
→ context selection
→ generation
→ citations
→ tool calls
```

The metric must identify **which stage failed**.

## Retrieval metrics

### Recall@K

```text
relevant retrieved in top K / total expected relevant
```

Use it when the main question is:

> Did retrieval find the evidence needed by generation?

### Precision@K

```text
relevant retrieved in top K / K
```

Use it to measure context noise.

### MRR

Mean Reciprocal Rank rewards finding the first relevant result early.

Example:

```text
rank of first relevant = 2
RR = 1/2 = 0.5
```

### nDCG@K

nDCG rewards relevant results being ranked near the top while supporting multiple relevant items.

The unit suite contains hand-calculable nDCG cases to prove the implementation behaves monotonically.

### Hit Rate

Did the result list contain at least one expected relevant item?

## Reranking evaluation

Compare the same candidate pool before and after reranking:

```text
Hybrid retrieval
→ top N
→ reranker
→ top K
```

Track:
- MRR
- Recall@K
- nDCG@K
- latency uplift

## Generation metrics

Deterministic metrics:
- structured-output validity
- citation validity
- correct abstention / no-answer accuracy

Semantic/judge metrics:
- answer correctness
- answer relevance
- faithfulness
- groundedness
- completeness
- conciseness

Semantic metrics must identify who/what produced the judgment: human, Gemini judge, another model, or deterministic rule.

## Citation metrics

Recommended:
- citation correctness
- citation completeness
- citation precision
- citation recall
- false citation rate

A citation should never be scored as correct merely because it looks plausible. It must resolve to a real selected context chunk/document/page.

## Tool calling metrics

- tool selection accuracy
- argument accuracy
- schema validity
- execution success rate
- task completion rate
- unnecessary tool-call rate
- tool hallucination rate
- retry rate
- average tool calls
- tool latency

## No-answer / hallucination metrics

Create cases where:
- answer exists;
- no answer exists;
- evidence is partial;
- evidence conflicts;
- evidence is outdated.

Track:
- correct-answer rate
- correct-abstention rate
- hallucination rate
- unsupported-claim rate
- false-citation rate

## System metrics

Always pair quality with engineering cost:
- P50/P95/P99 latency
- input/output tokens
- estimated cost
- retries
- timeouts
- failures
- throughput

The best configuration is rarely the one with the highest quality metric alone.
