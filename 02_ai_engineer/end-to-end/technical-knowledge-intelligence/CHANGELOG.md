# Changelog

## v1.1.1

- Fixed clean-CI editable installs with a dependency-free local PEP 517/660 backend.
- Fixed runtime project/config discovery for installed Docker/wheel deployments.
- Cleaned Ruff violations without weakening lint rules or CI checks.

## v1.1.0

- Fixed stale-language sessions (`Angol`/`Magyar`) so English A/B and benchmark plots no longer crash.
- `run_project.bat` now starts one Streamlit UI process; FastAPI remains available separately with `run_project.bat api`.
- Added document upload + index rebuild controls to the Library UI.
- Renamed corpus folders to `user_library` and `reference_docs` and clarified their purpose.
- Simplified project documentation and removed redundant Markdown placeholders/history files.

## Earlier releases

- **v1.0.9:** unique Plotly keys and workflow visual fixes.
- **v1.0.8:** interactive Plotly workflow and benchmark dashboard.
- **v1.0.7:** workflow/benchmark redesign and expanded demo corpus.
- **v1.0.6:** reliable Python-based Windows launcher.
- **v1.0.0–1.0.5:** packaging, localization, setup and engineering refactor foundations.
