# Documentation

Core documents:

1. `01_PROJECT_OVERVIEW.md` — scope and engineering objective.
2. `02_ARCHITECTURE.md` — runtime/core boundaries.
3. `03_DATA_PIPELINE.md` — collections, uploads, parsing, chunking and indexes.
4. `04_AI_PIPELINE.md` — retrieval, context, prompts, tools, Gemini and validation.
5. `05_EVALUATION.md` — evaluation methodology and metrics.
6. `06_TESTING.md` — automated test layers and commands.
7. `07_DEPLOYMENT.md` — local, Docker and Kubernetes execution.
8. `08_TROUBLESHOOTING.md` — operational failure recovery.
9. `09_DESIGN_DECISIONS.md` — important architectural trade-offs.
10. `PROJECT_STORY.md` — portfolio/interview narrative.

Specialized documents are kept only when they contain distinct implementation detail:

- `INDEX_BACKEND_POLICY.md`
- `MULTI_INDEX_CHUNKING.md`
- `LANGCHAIN_FASTAPI_INTEGRATION.md`
- `testing/` for detailed QA/evaluation methodology
- `diagrams/` for source/rendered architecture assets

Historical patch notes are consolidated into the root `CHANGELOG.md` rather than separate Markdown files.
