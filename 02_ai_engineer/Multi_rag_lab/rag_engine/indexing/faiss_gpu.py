from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_engine.models import Chunk
from rag_engine.indexing.faiss_cpu import FaissCPUVectorStore


class FaissGPUVectorStore(FaissCPUVectorStore):
    """FAISS vector store backed by a single NVIDIA GPU.

    The Python ``faiss`` module must be compiled with GPU support. Official
    pre-built ``faiss-gpu`` packages are Linux-only, so Windows users should
    run this backend from WSL2 or a Linux GPU container.
    """

    backend_name = "faiss-gpu"
    device = "cuda"

    def __init__(
        self,
        dimension: int | None = None,
        gpu_id: int = 0,
        *,
        use_float16: bool = False,
        temp_memory_mb: int | None = 256,
    ) -> None:
        super().__init__(dimension)
        self.gpu_id = gpu_id
        self.use_float16 = use_float16
        self.temp_memory_mb = temp_memory_mb

        self._validate_gpu_runtime()
        self.resources = self.faiss.StandardGpuResources()
        self._configure_resources()

        if self.index is not None:
            self.index = self._cpu_to_gpu(self.index)

    def _validate_gpu_runtime(self) -> None:
        if not hasattr(self.faiss, "StandardGpuResources"):
            raise RuntimeError(
                "The imported FAISS module is CPU-only: StandardGpuResources is missing. "
                "On Windows use WSL2/Linux and install the official faiss-gpu package; "
                "do not install faiss-cpu into the same environment."
            )

        if not hasattr(self.faiss, "index_cpu_to_gpu"):
            raise RuntimeError("The imported FAISS module does not expose index_cpu_to_gpu().")

        get_num_gpus = getattr(self.faiss, "get_num_gpus", None)
        if get_num_gpus is not None:
            gpu_count = int(get_num_gpus())
            if gpu_count <= 0:
                raise RuntimeError(
                    "FAISS has GPU APIs, but no CUDA device is visible. Check nvidia-smi, "
                    "WSL2 GPU passthrough, the NVIDIA driver, and container GPU access."
                )
            if self.gpu_id >= gpu_count:
                raise RuntimeError(f"Requested FAISS GPU id {self.gpu_id}, but only {gpu_count} GPU(s) are visible.")

    def _configure_resources(self) -> None:
        if self.temp_memory_mb is None:
            return
        set_temp_memory = getattr(self.resources, "setTempMemory", None)
        if set_temp_memory is not None:
            set_temp_memory(int(self.temp_memory_mb) * 1024 * 1024)

    def _cpu_to_gpu(self, cpu_index: Any):
        if self.use_float16 and hasattr(self.faiss, "GpuClonerOptions"):
            options = self.faiss.GpuClonerOptions()
            options.useFloat16 = True
            return self.faiss.index_cpu_to_gpu(self.resources, self.gpu_id, cpu_index, options)
        return self.faiss.index_cpu_to_gpu(self.resources, self.gpu_id, cpu_index)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if self.index is None and len(vectors):
            cpu_index = self.faiss.IndexFlatIP(vectors.shape[1])
            self.index = self._cpu_to_gpu(cpu_index)
            self.dimension = vectors.shape[1]
        super().add(vectors, chunks)

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save empty index")
        gpu_index = self.index
        self.index = self.faiss.index_gpu_to_cpu(gpu_index)
        try:
            super().save(path)
        finally:
            self.index = gpu_index

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        gpu_id: int = 0,
        use_float16: bool = False,
        temp_memory_mb: int | None = 256,
    ) -> "FaissGPUVectorStore":
        """Load a persisted CPU FAISS index and clone it onto the selected GPU."""
        import faiss

        cpu_index = cls._read_index_portable(faiss, path / "index.faiss")
        store = cls(
            dimension=None,
            gpu_id=gpu_id,
            use_float16=use_float16,
            temp_memory_mb=temp_memory_mb,
        )
        store.dimension = cpu_index.d
        store.index = store._cpu_to_gpu(cpu_index)
        store.chunks = [
            Chunk.model_validate(item) for item in json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        ]
        return store

    def runtime_info(self) -> dict[str, Any]:
        get_num_gpus = getattr(self.faiss, "get_num_gpus", None)
        return {
            "backend": self.backend_name,
            "device": self.device,
            "gpu_id": self.gpu_id,
            "visible_gpus": int(get_num_gpus()) if get_num_gpus is not None else None,
            "use_float16": self.use_float16,
            "temp_memory_mb": self.temp_memory_mb,
            "faiss_version": getattr(self.faiss, "__version__", "unknown"),
        }
