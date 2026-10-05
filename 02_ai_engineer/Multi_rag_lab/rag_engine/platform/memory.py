from __future__ import annotations

import gc


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_memory_mb() -> dict[str, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    except ImportError:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
