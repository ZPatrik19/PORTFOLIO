# Final Validation Report — v1.1.0

Validated: 2026-09-15

## What changed

- Fixed legacy Streamlit language values (`Angol`, `Magyar`) by normalizing them to stable `en` / `hu` codes before question-bank and chart access.
- Changed the daily launcher so `run_project.bat run` starts only the Streamlit UI. The FastAPI adapter remains available separately with `run_project.bat api`.
- Added Library UI controls for uploading supported documents and rebuilding the search index.
- Renamed the corpus folders to `01_data/user_library/` and `01_data/reference_docs/`, while keeping backward-compatible config aliases for older projects.
- Reduced the Markdown set and consolidated overlapping technical notes into the core documentation.
- Prevented user-library files and non-demo reference documents from being copied into Docker images by default.

## Validation

- Python compile validation: PASS
- Focused UI / language / Plotly / packaging tests: 37 passed
- Full offline regression suite: 119 passed
- Release audit: PASS
- English Plotly smoke test: PASS for both `en` and legacy `Angol` language values

Docker and Kubernetes runtime validation was not executed in this environment because Docker/Kubectl are unavailable. Live Gemini calls were not required for this hotfix.
