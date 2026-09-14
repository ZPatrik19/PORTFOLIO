"""EN: Gemini adapter request construction and interaction payload compatibility.

HU: A Gemini adapter request-összeállítását és interaction payload kompatibilitását ellenőrzi.
"""

from __future__ import annotations

import sys
import types

from prompt_benchmark.llm.client import GeminiClient
from prompt_benchmark.prompts.base import PromptPayload


class _Usage:
    total_input_tokens = 21
    total_output_tokens = 4
    total_tokens = 25


class _Interaction:
    output_text = '{"label":"billing"}'
    usage = _Usage()


class _Interactions:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return _Interaction()


class _FakeSDKClient:
    def __init__(self, api_key: str) -> None:
        assert api_key == "test-gemini-key"
        self.interactions = _Interactions()


def test_gemini_adapter_builds_interactions_request(monkeypatch):
    """EN: Checks that the Gemini adapter builds the expected interactions request payload for the SDK boundary.

    HU: Ellenőrzi, hogy a Gemini adapter az SDK boundary számára megfelelő interactions request payloadot állít össze.
    """
    fake_genai = types.SimpleNamespace(Client=_FakeSDKClient)
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    client = GeminiClient(
        model="gemini-3.5-flash-lite",
        temperature=None,
        top_p=None,
        top_k=None,
        thinking_level="minimal",
    )
    payload = PromptPayload(
        strategy_name="test",
        instructions="System instruction",
        input_text="Classify this ticket",
        output_mode="json",
        structured_output=True,
        reasoning_effort="medium",
    )
    response = client.classify(payload)

    request = client._client.interactions.last_request
    assert request["model"] == "gemini-3.5-flash-lite"
    assert request["input"] == "Classify this ticket"
    assert request["system_instruction"] == "System instruction"
    assert request["generation_config"]["max_output_tokens"] == 64
    assert request["generation_config"]["thinking_level"] == "medium"
    assert "temperature" not in request["generation_config"]
    assert request["response_format"]["mime_type"] == "application/json"
    assert response.raw_output == '{"label":"billing"}'
    assert response.input_tokens == 21
    assert response.output_tokens == 4
    assert response.total_tokens == 25
