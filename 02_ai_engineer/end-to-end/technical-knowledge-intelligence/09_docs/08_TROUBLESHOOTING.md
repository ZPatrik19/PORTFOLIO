# Troubleshooting

## Editable install / build-backend errors

### Symptom

A clean Python 3.12+ environment may fail before dependency installation if a
project relies on a build backend that is not already installed. Earlier TKI
releases could surface:

```text
BackendUnavailable: Cannot import 'setuptools.build_meta'
```

### Current behavior

Since v1.1.1 the repository ships a small dependency-free PEP 517/660 backend
(`build_backend.py`). Therefore this works in a clean environment without
preinstalling setuptools or wheel:

```bash
python -m pip install -e . --no-build-isolation
```

The build backend reads project metadata from `pyproject.toml`, creates normal
wheels for deployment, and creates an editable `.pth` wheel for development.
This keeps `pyproject.toml` as the dependency contract while removing hidden
CI assumptions about preinstalled packaging tools.

If an extracted environment is corrupted, recreate only the virtual environment:

```bat
rmdir /s /q .venv
run_project.bat setup-run
```

---

## `ModuleNotFoundError: tkip`
Run setup so the package is installed editable:
```bash
python -m pip install -e . --no-deps --no-build-isolation
```

## Missing Gemini key
Set `GEMINI_API_KEY` in local `.env` or enter it in the Streamlit session field. Never commit it.

## Gemini `429 RESOURCE_EXHAUSTED`
The key may be valid but quota is exhausted. Normal offline tests and retrieval remain available. Wait for quota reset, use a billing-enabled project/model, or reduce optional model calls.

## Port 8000/8501 already in use
Stop the previous API/UI process or change the local launch ports.

## Graphviz unavailable
The UI has a fallback renderer. For full workflow export install both the Python package and system `dot` executable.

## Docker build failed
Check Docker Desktop/daemon availability and network access for Python dependency installation.

## Kubernetes `CrashLoopBackOff`
Inspect:
```bash
kubectl -n tki get pods
kubectl -n tki describe pod <pod>
kubectl -n tki logs <pod>
```
Most common causes are missing image, write permissions on the data volume or invalid/missing runtime dependencies.

## `The system cannot find the batch label specified - ensure_python`

This is a Windows batch-file line-ending problem, not a missing function. The `:ensure_python` label exists in `run_project.bat`, but Windows `cmd.exe` can fail to resolve `CALL :label` when the file was packaged with Unix-only LF line endings.

Release v1.0.4 stores every `.bat` file with CRLF and adds `.gitattributes` plus automated regression checks so Git/ZIP packaging cannot silently reintroduce the issue.

If you are repairing an older extracted release manually, convert `run_project.bat` to **CRLF** in your editor, save it, and run `setup.bat` again.
