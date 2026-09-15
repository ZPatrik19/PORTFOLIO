# Testing

Default deterministic run:

```bat
run_project.bat test offline
```

Equivalent pytest selection:

```bash
pytest -m "not live_gemini"
```

## Layers

- unit — isolated deterministic behavior;
- integration — ingestion/index/retrieval/orchestration boundaries;
- evaluation — metric correctness;
- regression — golden behavior gates;
- robustness — prompt injection and unusual inputs;
- performance — local latency guardrails;
- smoke — API/deployment/application contracts;
- live_gemini — explicit opt-in external calls.

The default suite never requires personal library files, Gemini quota or an external Qdrant service.

Detailed test design, fixtures, golden sets and metric notes are under `09_docs/testing/`.
