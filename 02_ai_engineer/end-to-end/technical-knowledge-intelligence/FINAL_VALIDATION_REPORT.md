# Final Validation Report — v1.1.1

Validated: 2026-09-15

## What changed

- Replaced the hidden setuptools CI dependency with a self-contained PEP 517/660 build backend.
- Fixed runtime project/config discovery for installed wheel and Docker execution.
- Updated the Docker build context to include the local build backend before package installation.
- Fixed the Ruff violations reported by GitHub Actions without weakening lint rules.
- Hardened `.gitignore` for private documents, indexes, parse caches, processed runtime state and logs.

## Validation

- Python compile validation: PASS
- Full offline pytest suite: **121 passed, 1 live-Gemini test deselected**
- Release audit: PASS
- Editable install in a clean environment with **no setuptools installed**: PASS
- Normal wheel build/install: PASS
- Installed-wheel config discovery from a Docker-like working directory: PASS
- FastAPI startup + `/health`: PASS

A real Docker daemon and `kubectl` are not available in this execution environment, so container/Kubernetes runtime execution was not claimed as tested.
