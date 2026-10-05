from __future__ import annotations

from rag_engine.models import ExecutionDevice
import rag_engine.platform.device as device_module


def test_auto_resolves_cpu_when_cuda_missing(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    result = device_module.resolve_device("auto")
    assert result.resolved == ExecutionDevice.CPU
    assert not result.fallback_used


def test_explicit_cuda_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    result = device_module.resolve_device("cuda", allow_cpu_fallback=True)
    assert result.resolved == ExecutionDevice.CPU
    assert result.fallback_used


def test_resolve_device_none_defaults_to_auto(monkeypatch):
    import rag_engine.platform.device as device_module

    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    result = device_module.resolve_device(None)
    assert result.requested.value == "auto"
    assert result.resolved.value == "cpu"
