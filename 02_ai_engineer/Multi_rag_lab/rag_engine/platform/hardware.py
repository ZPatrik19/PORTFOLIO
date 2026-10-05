from __future__ import annotations

import os
import platform

import psutil
from pydantic import BaseModel

from rag_engine.models import ExecutionDevice
from rag_engine.platform.device import resolve_device


class HardwareProfile(BaseModel):
    requested_device: str
    resolved_device: str
    cuda_available: bool
    gpu_name: str | None = None
    gpu_count: int = 0
    gpu_total_memory_mb: int | None = None
    cpu_count: int
    system_memory_mb: int
    platform: str


def get_hardware_profile(requested: ExecutionDevice | str = ExecutionDevice.AUTO) -> HardwareProfile:
    resolution = resolve_device(requested)
    gpu_name = None
    gpu_count = 0
    gpu_mem = None
    cuda_ok = False
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_mem = int(props.total_memory / (1024**2))
    except ImportError:
        pass
    return HardwareProfile(
        requested_device=str(resolution.requested),
        resolved_device=str(resolution.resolved),
        cuda_available=cuda_ok,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_total_memory_mb=gpu_mem,
        cpu_count=os.cpu_count() or 1,
        system_memory_mb=int(psutil.virtual_memory().total / (1024**2)),
        platform=f"{platform.system()} {platform.release()}",
    )
