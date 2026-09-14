"""EN: Gemini health-check response parsing, usage accounting, and missing-key behavior.

HU: A Gemini health-check válaszfeldolgozását, tokenhasználat-kezelését és hiányzó kulcs esetét ellenőrzi.
"""

from __future__ import annotations

import json
from io import BytesIO

from prompt_benchmark.llm.gemini_health import test_gemini_connection as run_gemini_health_check


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_gemini_health_parses_response_and_usage(monkeypatch):
    """EN: Checks that Gemini health responses are parsed into model output and token-usage telemetry.

    HU: Ellenőrzi, hogy a Gemini health response modellválasszá és token-telemetriává alakul.
    """
    payload = {
        "candidates": [{"content": {"parts": [{"text": "billing"}]}}],
        "usageMetadata": {
            "promptTokenCount": 17,
            "candidatesTokenCount": 2,
            "totalTokenCount": 19,
        },
    }

    def fake_urlopen(request, timeout):
        assert request.headers["X-goog-api-key"] == "secret-test-key"
        assert "secret-test-key" not in request.full_url
        assert timeout == 30.0
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = run_gemini_health_check(api_key="secret-test-key", model="gemini-3.5-flash-lite")
    assert result.ok is True
    assert result.text == "billing"
    assert result.input_tokens == 17
    assert result.output_tokens == 2
    assert result.total_tokens == 19
    assert result.error is None


def test_gemini_health_requires_key(monkeypatch):
    """EN: Ensures Gemini health checks fail clearly when no API key is configured.

    HU: Biztosítja, hogy API kulcs nélkül a Gemini health check érthető hibával álljon le.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = run_gemini_health_check(api_key="")
    assert result.ok is False
    assert "GEMINI_API_KEY" in (result.error or "")
