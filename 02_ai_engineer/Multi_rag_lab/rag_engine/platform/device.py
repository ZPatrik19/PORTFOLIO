from __future__ import annotations

from dataclasses import dataclass

from rag_engine.models import ExecutionDevice


class DeviceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceResolution:
    requested: ExecutionDevice
    resolved: ExecutionDevice
    fallback_used: bool = False
    reason: str = ""


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_device(
    requested: ExecutionDevice | str | None,
    *,
    allow_cpu_fallback: bool = True,
) -> DeviceResolution:
    requested = ExecutionDevice(requested or ExecutionDevice.AUTO.value)
    available = cuda_available()
    if requested == ExecutionDevice.AUTO:
        resolved = ExecutionDevice.CUDA if available else ExecutionDevice.CPU
        return DeviceResolution(requested, resolved, False, "auto-detected")
    if requested == ExecutionDevice.CUDA and not available:
        if allow_cpu_fallback:
            return DeviceResolution(requested, ExecutionDevice.CPU, True, "CUDA unavailable; using CPU fallback")
        raise DeviceUnavailableError(
            "CUDA was explicitly requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build/driver or choose CPU/AUTO."
        )
    return DeviceResolution(requested, requested)
