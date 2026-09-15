# Testing Strategy

## Purpose

The platform contains two very different kinds of software:

1. **Deterministic engineering components** — parsers, chunkers, retrieval algorithms, citation validation, schemas, APIs, guardrails.
2. **Probabilistic AI behavior** — retrieval quality, prompt behavior, tool selection, grounded generation and semantic answer quality.

Treating both as the same kind of "test" creates misleading results. This project deliberately separates them.

## Testing pyramid

```text
                         Live Gemini tests
                       /                  \
              AI evaluation / benchmarks
             /                            \
        Integration + regression + robustness
       /                                      \
                Unit tests + smoke tests
```

The lower layers should be fast, deterministic, cheap and run frequently.
The upper layers are slower, may require real models/data, and should run intentionally.

## Layer definitions

### Unit tests

Goal: validate one deterministic behavior in isolation.

Examples:
- BM25 ranks exact technical terms correctly.
- RRF keeps candidates from both rank lists.
- structure-aware chunking preserves code blocks.
- citation validator rejects invented chunk IDs.
- tool registry rejects non-allowlisted tools.

A unit test should not need Gemini, Qdrant server, user-library documents or network access.

### Integration tests

Goal: prove that multiple internal components work together.

Current integration path:

```text
public demo corpus
→ parsing
→ chunking
→ local hashing embeddings
→ NumPy index
→ hybrid retrieval
→ orchestration diagnostics
```

The integration suite intentionally uses the public demo corpus and local embeddings so it remains reproducible.

### AI evaluation tests

Goal: validate the **evaluation logic itself** with deterministic examples.

Examples:
- expected MRR = 0.5 when the first relevant item is rank 2.
- Recall@3 = 1.0 when the only relevant item occurs inside the top 3.
- nDCG must decrease when the same relevant item is moved lower in the ranking.

These tests answer:

> "Is our metric implementation mathematically correct?"

They do **not** answer:

> "Is the current RAG system good enough?"

The latter requires benchmark datasets and evaluation runs.

### Regression tests

Goal: prevent known-good behavior from silently changing.

Examples:
- a known code-search query should remain `CODE_SEARCH`.
- a known learning query should remain `LEARNING`.
- future golden retrieval cases can assert minimum Recall@K / MRR thresholds.

Regression tests should use stable golden cases and explicit tolerances.

### Robustness tests

Goal: test hostile, malformed and unusual inputs.

Examples:
- prompt injection.
- source exfiltration request.
- multilingual input.
- empty/very long input.
- malformed tool arguments.

### Performance tests

Goal: catch large performance regressions, not benchmark developer hardware.

A threshold such as `BM25 < 1500 ms for 200 tiny chunks` is intentionally generous. It catches accidental O(N²)-style regressions without failing because one laptop is slower than another.

### Smoke tests

Goal: verify application startup contracts and key endpoints.

Example:
- `/health` returns HTTP 200.
- request/trace IDs survive middleware.

### Live Gemini tests

Goal: verify the external Gemini integration **at the current moment**.

These tests can fail because of:
- quota exhaustion,
- rate limits,
- billing,
- model availability,
- API changes,
- network problems.

For that reason they are never part of the default offline suite.

## What "all tests passed" means

If `run_project.bat test offline` passes, it means:

- deterministic internal logic passed its contracts;
- the public demo ingestion/index/retrieval integration works;
- evaluation metric implementations passed known examples;
- golden deterministic behavior did not regress;
- robustness guards passed the covered cases;
- API smoke checks passed.

It does **not** prove that:

- Gemini currently has quota;
- the live model gives a high-quality answer;
- retrieval quality is good on every private book;
- all hallucinations are prevented;
- the benchmark dataset is representative of production.

Those are evaluated separately.
