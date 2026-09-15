"""FastAPI smoke tests that avoid loading the full private corpus."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.smoke


def _load_api_module():
    project_root = Path(__file__).resolve().parents[2]
    api_path = project_root / "04_api" / "main.py"
    spec = importlib.util.spec_from_file_location("tkip_api_main", api_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_health_endpoint_returns_ok_and_preserves_trace_headers() -> None:
    module = _load_api_module()
    client = TestClient(module.app)

    response = client.get(
        "/health",
        headers={"X-Request-ID": "request-test", "X-Trace-ID": "trace-test"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-test"
    assert response.headers["x-trace-id"] == "trace-test"
