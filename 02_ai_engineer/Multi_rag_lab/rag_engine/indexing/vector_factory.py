from __future__ import annotations

import logging

from rag_engine.indexing.faiss_cpu import FaissCPUVectorStore
from rag_engine.indexing.faiss_gpu import FaissGPUVectorStore
from rag_engine.indexing.numpy_store import NumpyVectorStore

LOGGER = logging.getLogger(__name__)


def create_vector_store(
    *,
    device: str = "cpu",
    fallback_to_numpy: bool = True,
    strict_device: bool = False,
    gpu_id: int = 0,
):
    normalized_device = device.strip().lower()

    if normalized_device in {"cuda", "gpu", "faiss-gpu"}:
        try:
            return FaissGPUVectorStore(gpu_id=gpu_id)
        except Exception as exc:
            if strict_device:
                raise RuntimeError(f"FAISS GPU initialization failed: {exc}") from exc
            LOGGER.warning("FAISS GPU unavailable (%s); falling back to CPU", exc)

    try:
        return FaissCPUVectorStore()
    except Exception as exc:
        if not fallback_to_numpy:
            raise
        LOGGER.warning("FAISS CPU unavailable (%s); falling back to NumPy exact search", exc)
        return NumpyVectorStore()
