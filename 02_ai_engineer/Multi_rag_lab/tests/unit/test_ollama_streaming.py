from __future__ import annotations

import json

import requests

from rag_engine.generation.provider_ollama import OllamaProvider


class FakeResponse:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        assert decode_unicode is False
        for event in self.events:
            yield json.dumps(event).encode("utf-8")


def test_ollama_provider_streams_and_captures_metrics(monkeypatch):
    events = [
        {"response": "Szia ", "done": False},
        {
            "response": "[S1]",
            "done": True,
            "eval_count": 10,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 20,
        },
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(events))
    provider = OllamaProvider(read_timeout=600, num_predict=256)
    answer = provider.generate("teszt")
    assert answer == "Szia [S1]"
    assert provider.last_metrics["output_tokens"] == 10
    assert provider.last_metrics["tokens_per_second"] == 5


def test_ollama_provider_uses_streaming_and_long_read_timeout(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse([{"response": "ok", "done": True, "eval_count": 1, "eval_duration": 1_000_000_000}])

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaProvider(connect_timeout=7, read_timeout=600)
    provider.generate("teszt")
    assert captured["stream"] is True
    assert captured["timeout"] == (7.0, 600.0)


def test_ollama_timeout_retries_with_smaller_generation_budget(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"]["options"]["num_predict"])
        if len(calls) == 1:
            raise requests.ReadTimeout("slow first attempt")
        return FakeResponse([{"response": "ok", "done": True, "eval_count": 1, "eval_duration": 1_000_000_000}])

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaProvider(read_timeout=600, num_predict=320, max_retries=1)
    assert provider.generate("teszt") == "ok"
    assert calls == [320, 192]
    assert provider.last_metrics["attempt"] == 2


def test_ollama_chat_uses_system_and_user_roles(monkeypatch):
    captured = {}
    events = [
        {"message": {"role": "assistant", "content": "Magyar "}, "done": False},
        {
            "message": {"role": "assistant", "content": "válasz [S1]"},
            "done": True,
            "eval_count": 4,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 12,
        },
    ]

    def fake_post(url, *args, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return FakeResponse(events)

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaProvider(read_timeout=600, num_predict=256)
    answer = provider.generate_chat("Csak magyarul.", "Kérdés és bizonyíték.")
    assert answer == "Magyar válasz [S1]"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "Csak magyarul."},
        {"role": "user", "content": "Kérdés és bizonyíték."},
    ]
    assert captured["json"]["think"] is False
    assert provider.last_metrics["endpoint"] == "/api/chat"


def test_runtime_ollama_smoke_uses_streaming_and_long_timeout(monkeypatch):
    from rag_engine.platform.runtime import test_ollama_model

    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse(
            [
                {"response": "RAG runtime ", "done": False},
                {"response": "OK", "done": True, "eval_count": 3, "eval_duration": 1_000_000_000},
            ]
        )

    monkeypatch.setattr(requests, "post", fake_post)
    result = test_ollama_model("rag-qwen-balanced", connect_timeout=9, read_timeout=600, keep_alive="15m")
    assert result["ok"] is True
    assert result["response"] == "RAG runtime OK"
    assert captured["stream"] is True
    assert captured["timeout"] == (9.0, 600.0)
    assert captured["json"]["options"]["num_predict"] == 16
    assert captured["json"]["keep_alive"] == "15m"
