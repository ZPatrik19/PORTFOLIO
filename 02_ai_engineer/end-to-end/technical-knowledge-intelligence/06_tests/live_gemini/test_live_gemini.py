"""Explicit opt-in tests that make real Gemini API calls.

These tests are NEVER required for the normal offline test suite. They consume
Gemini quota and can fail for quota/billing/network reasons unrelated to local
application correctness.
"""
from __future__ import annotations

import pytest

from tkip.config import load_config
from tkip.gemini_service import GeminiService

pytestmark = pytest.mark.live_gemini


def test_live_gemini_connection(live_gemini_key: str) -> None:
    """One minimal API request proving that the configured key/model is usable."""
    service = GeminiService(load_config(), api_key=live_gemini_key)

    ok, message = service.test_connection()

    assert ok, f"Gemini live connection failed: {message}"
