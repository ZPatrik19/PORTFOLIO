"""EN: Provider adapter success paths and credential guards without making live paid calls.

HU: A provider adapterek sikeres útvonalait és credential-ellenőrzését vizsgálja élő/fizetős hívások nélkül.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from prompt_benchmark.prompts.base import PromptPayload


def payload(*, structured: bool = True) -> PromptPayload:
    return PromptPayload(
        strategy_name="test",
        instructions="Classify carefully.",
        input_text="Please cancel my subscription.",
        output_mode="json" if structured else "label",
        structured_output=structured,
        reasoning_effort="low",
    )


class _FakeOpenAIResponse:
    output_text = '{"label":"cancellation"}'
    usage = SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14)


class _FakeChatResponse:
    choices = [SimpleNamespace(message=SimpleNamespace(content='{"label":"cancellation"}'))]
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14)


class _FakeOpenAI:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.responses = SimpleNamespace(create=lambda **request: _FakeOpenAIResponse())
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: _FakeChatResponse()))


class _FakeGroq:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: _FakeChatResponse()))


class _FakeOllamaClient:
    def __init__(self, host: str):
        self.host = host

    def chat(self, **request):
        return SimpleNamespace(
            message=SimpleNamespace(content='{"label":"cancellation"}'),
            prompt_eval_count=12,
            eval_count=4,
        )


def test_openai_adapter_success(monkeypatch):
    """EN: Exercises the OpenAI adapter success path with a mocked SDK response and verifies normalized telemetry.

    HU: Mockolt SDK válasszal teszteli az OpenAI adapter sikeres útvonalát és a normalizált telemetriát.
    """
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from prompt_benchmark.llm.providers.openai import OpenAIResponsesClient

    client = OpenAIResponsesClient("test-model", temperature=0.1, top_p=0.9)
    result = client.classify(payload())
    assert result.raw_output == '{"label":"cancellation"}'
    assert result.total_tokens == 14
    assert result.provider == "openai"


def test_openrouter_adapter_success(monkeypatch):
    """EN: Exercises the OpenRouter adapter success path without a live network call.

    HU: Élő hálózati hívás nélkül teszteli az OpenRouter adapter sikeres útvonalát.
    """
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from prompt_benchmark.llm.providers.openrouter import OpenRouterClient

    client = OpenRouterClient("openrouter/free", temperature=0.0, top_p=0.8)
    result = client.classify(payload())
    assert result.total_tokens == 14
    assert result.provider == "openrouter"
    assert _FakeOpenAI.last_kwargs["base_url"] == "https://openrouter.ai/api/v1"


def test_groq_adapter_success(monkeypatch):
    """EN: Exercises the Groq adapter success path without a live network call.

    HU: Élő hálózati hívás nélkül teszteli a Groq adapter sikeres útvonalát.
    """
    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=_FakeGroq))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from prompt_benchmark.llm.providers.groq import GroqClient

    client = GroqClient("test-model", top_p=0.9)
    result = client.classify(payload())
    assert result.total_tokens == 14
    assert result.provider == "groq"


def test_ollama_adapter_success(monkeypatch):
    """EN: Exercises the Ollama adapter success path without requiring a locally running model.

    HU: Lokálisan futó modell nélkül teszteli az Ollama adapter sikeres útvonalát.
    """
    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=_FakeOllamaClient))
    from prompt_benchmark.llm.providers.ollama import OllamaClient

    client = OllamaClient("local-model", top_p=0.9, top_k=20)
    result = client.classify(payload())
    assert result.total_tokens == 16
    assert result.provider == "ollama"


def test_cloud_adapters_require_keys(monkeypatch):
    """EN: Ensures cloud adapters reject missing credentials before attempting network I/O.

    HU: Biztosítja, hogy a cloud adapterek hiányzó credential esetén még network I/O előtt hibázzanak.
    """
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from prompt_benchmark.llm.providers.openai import OpenAIResponsesClient

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIResponsesClient("test")
