# Evaluation

## Separate the layers

A RAG system can fail in retrieval, context selection, generation, citation or tool execution. The project therefore avoids one opaque "quality" number.

### Retrieval

- Recall@1/3/5/10
- Precision@K
- MRR
- Hit Rate
- nDCG@K
- P50/P95 latency

### Grounded generation

Deterministic checks:
- citation correctness/completeness;
- no-answer accuracy;
- structured-output validity.

Semantic metrics such as faithfulness, answer correctness and hallucination rate require a human/labeled reference or explicit judge evaluation. Offline deterministic code leaves these as `None` rather than fabricating values.

### Tool calling

- selection accuracy;
- argument validity;
- execution success;
- unnecessary call rate;
- average calls;
- latency/cost from telemetry.

## Evaluation dataset warning

`generate_eval_dataset()` creates reproducible silver-label samples from the corpus and is useful for regression comparisons. Portfolio/production claims should additionally use a manually curated golden evaluation set with independently verified relevant chunks/answers.
