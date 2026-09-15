# Benchmarking vs Automated Testing

These concepts are related but not interchangeable.

## Automated test

A test normally checks a deterministic contract:

```text
input
→ component
→ expected behavior
```

Example:

```text
first relevant result is rank 2
→ MRR implementation
→ expected MRR = 0.5
```

If this fails, the implementation is wrong or its contract changed.

## Benchmark

A benchmark measures quality/performance on a dataset:

```text
50 / 300 / 500 questions
→ retrieval configuration
→ Recall@5, MRR, nDCG@5, latency
```

A benchmark result is not usually pass/fail by itself. It becomes regression testing only when explicit thresholds are defined.

## Evaluation

Evaluation is broader than both. It can include:
- deterministic metrics;
- human review;
- LLM-as-judge;
- groundedness checks;
- answer correctness;
- cost/latency trade-offs.

## Example: RAG

### Unit test

```text
Does RRF combine both ranked lists?
```

### Metric test

```text
Does nDCG decrease when the relevant result moves from rank 1 to rank 3?
```

### Retrieval benchmark

```text
BM25 vs Dense vs Hybrid vs Reranker on 300 questions
```

### Regression gate

```text
Fail release if Recall@5 decreases by > 3 percentage points.
```

### Live evaluation

```text
Run Gemini grounded generation and score faithfulness/citations.
```

Keeping these categories separate prevents a green pytest result from being misinterpreted as "the AI is good".
