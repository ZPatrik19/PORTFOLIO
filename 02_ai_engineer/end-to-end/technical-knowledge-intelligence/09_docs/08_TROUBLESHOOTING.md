# Troubleshooting

## `BackendUnavailable: Cannot import 'setuptools.build_meta'`

### Symptom

A first Windows setup can fail during:

```text
pip install -e .
BackendUnavailable: Cannot import 'setuptools.build_meta'
```

### Cause

Python 3.12+ virtual environments are not guaranteed to contain `setuptools`.
The project uses the `setuptools.build_meta` PEP 517/660 backend, and a
`--no-build-isolation` editable install therefore requires the backend to
already be installed inside `.venv`.

### Fixed setup path

v1.0.3 bootstraps the build backend before installing the local package:

```bat
run_project.bat setup-run
```

The relevant order is:

```text
create/reuse .venv
    ↓
verify pip
    ↓
install `setuptools` + `wheel` from the `build-system` contract
    ↓
verify setuptools.build_meta + wheel
    ↓
install runtime/dev dependencies
    ↓
pip install -e . --no-deps --no-build-isolation
```

### Repairing an existing `.venv`

You do **not** need to delete the environment first. From the repository root:

```bat
.venv\Scripts\python.exe -m pip install "setuptools>=80" "wheel>=0.45"
.venv\Scripts\python.exe -c "import setuptools.build_meta, wheel; print('build backend OK')"
.venv\Scripts\python.exe -m pip install -e ".[dev]" --no-build-isolation
```

If the environment is corrupted rather than merely missing the backend:

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
