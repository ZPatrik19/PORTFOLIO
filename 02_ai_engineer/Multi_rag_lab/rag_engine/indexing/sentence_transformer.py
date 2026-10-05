from __future__ import annotations

import logging

import numpy as np

from rag_engine.models import ExecutionDevice
from rag_engine.platform.device import resolve_device
from rag_engine.platform.memory import clear_cuda_cache
from rag_engine.platform.model_assets import configure_hf_environment, resolve_cached_model

LOGGER = logging.getLogger(__name__)


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        *,
        device: ExecutionDevice | str = ExecutionDevice.AUTO,
        batch_size_cpu: int = 16,
        batch_size_cuda: int = 64,
        normalize: bool = True,
        allow_cpu_fallback: bool = True,
        max_oom_retries: int = 2,
    ) -> None:
        configure_hf_environment()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed. Run SETUP first.") from exc

        resolution = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.device = str(resolution.resolved)
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size_cpu = batch_size_cpu
        self.batch_size_cuda = batch_size_cuda
        self.allow_cpu_fallback = allow_cpu_fallback
        self.max_oom_retries = max_oom_retries
        self.model_source = resolve_cached_model(model_name)
        self.model = SentenceTransformer(self.model_source, device=self.device)
        if resolution.fallback_used:
            LOGGER.warning(resolution.reason)

    @property
    def batch_size(self) -> int:
        return self.batch_size_cuda if self.device == "cuda" else self.batch_size_cpu

    def _run_encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                device=self.device,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        batch = self.batch_size
        last_oom: RuntimeError | None = None
        for attempt in range(self.max_oom_retries + 1):
            try:
                return self._run_encode(texts, batch)
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower() and self.device == "cuda"
                if not is_oom:
                    raise
                last_oom = exc
                if attempt < self.max_oom_retries:
                    LOGGER.warning("CUDA OOM during embedding; reducing batch size from %s", batch)
                    clear_cuda_cache()
                    batch = max(1, batch // 2)

        if self.device == "cuda" and self.allow_cpu_fallback:
            LOGGER.warning("CUDA embedding retries exhausted; moving embedding model to CPU")
            clear_cuda_cache()
            self.model.to("cpu")
            self.device = "cpu"
            return self._run_encode(texts, self.batch_size_cpu)
        raise last_oom or RuntimeError("Embedding failed after retries")

    @property
    def _uses_e5_prefixes(self) -> bool:
        return "e5" in self.model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self.model.get_sentence_embedding_dimension() or 0
            return np.empty((0, dimension), dtype=np.float32)
        prepared = [f"passage: {text}" for text in texts] if self._uses_e5_prefixes else texts
        return self._encode(prepared)

    def embed_query(self, query: str) -> np.ndarray:
        prepared = f"query: {query}" if self._uses_e5_prefixes else query
        return self._encode([prepared])[0]
