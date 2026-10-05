from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_setup_is_python_314_and_idempotent() -> None:
    batch = (ROOT / "SETUP.bat").read_text(encoding="utf-8")
    assert "py -3.14" in batch
    assert "3.11" not in batch
    assert "3.12" not in batch
    assert '.[dev,cpu]' in batch
    assert "setup_state.py check" in batch
    assert "setup_state.py write" in batch
    assert 'pytest -q -m "not gpu"' in batch
    assert "infrastructure_cli.py cuda --install-cuda yes" in batch
    assert "where nvidia-smi" in batch
    assert "EVALUATION.bat" in batch


def test_linux_setup_is_python_314_and_idempotent() -> None:
    shell = (ROOT / "SETUP.sh").read_text(encoding="utf-8")
    assert "python3.14" in shell
    assert "(3,14)" in shell
    assert ".[dev,cpu]" in shell
    assert "setup_state.py check" in shell
    assert "setup_state.py write" in shell
    assert "pytest -q -m 'not gpu'" in shell
    assert 'infrastructure_cli.py cuda --install-cuda yes' in shell


def test_run_launchers_setup_then_infrastructure_then_streamlit() -> None:
    batch = (ROOT / "RUN.bat").read_text(encoding="utf-8")
    shell = (ROOT / "RUN.sh").read_text(encoding="utf-8")
    for text in (batch, shell):
        assert "SETUP" in text
        assert "INFRASTRUCTURE" in text
        assert "streamlit" in text
        assert "balanced" in text
    assert "py -3.14" not in batch  # RUN uses the project venv, SETUP owns interpreter discovery.
    assert "sys.version_info[:2] == (3,14)" in batch


def test_infrastructure_launchers_wrap_shared_cli() -> None:
    batch = (ROOT / "INFRASTRUCTURE.bat").read_text(encoding="utf-8")
    shell = (ROOT / "INFRASTRUCTURE.sh").read_text(encoding="utf-8")
    assert "SETUP.bat" in batch
    assert "infrastructure_cli.py" in batch
    assert "./SETUP.sh" in shell
    assert "infrastructure_cli.py" in shell


def test_evaluation_launchers_exist_and_cover_main_workflows() -> None:
    batch = (ROOT / "EVALUATION.bat").read_text(encoding="utf-8")
    shell = (ROOT / "EVALUATION.sh").read_text(encoding="utf-8")
    for text in (batch, shell):
        assert "build_medical_eval_dataset.py" in text
        assert "evaluate_medical_rag.py" in text
        assert "experiment_registry.py" in text
        assert "run_measurement_suite.py" in text
        assert "INFRASTRUCTURE" in text


def test_no_legacy_cuda_helper_reference() -> None:
    text = (ROOT / "scripts" / "infrastructure_cli.py").read_text(encoding="utf-8")
    assert "infrastructure/cuda" not in text
    assert "CUDA_SETUP.bat" not in text


def test_runtime_preflight_works_with_dummy_provider() -> None:
    env = os.environ.copy()
    env["LLM_PROVIDER"] = "dummy"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "runtime_preflight.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LLM_PROVIDER=dummy" in completed.stdout


def test_project_declares_python_314_runtime() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    repo_root = ROOT.parents[1]
    ci_candidates = [
        repo_root / ".github" / "workflows" / "multi-rag-ci.yml",
        ROOT / ".github" / "workflows" / "ci.yml",
    ]
    ci_path = next((path for path in ci_candidates if path.exists()), None)
    assert ci_path is not None, "Multi-RAG CI workflow is missing"
    ci = ci_path.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.14,<3.15"' in pyproject
    assert 'target-version = "py314"' in pyproject
    assert "FROM python:3.14-slim" in dockerfile
    assert 'PYTHON_VERSION: "3.14"' in ci or 'python-version: "3.14"' in ci
