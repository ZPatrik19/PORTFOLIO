from __future__ import annotations

import logging

from rag_engine.models import ExecutionDevice
from rag_engine.platform.device import resolve_device
from rag_engine.platform.memory import clear_cuda_cache
from rag_engine.platform.model_assets import configure_hf_environment, resolve_cached_model

LOGGER = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: ExecutionDevice | str = "auto",
        batch_size_cpu: int = 4,
        batch_size_cuda: int = 16,
        max_oom_retries: int = 2,
        allow_cpu_fallback: bool = True,
    ) -> None:
        configure_hf_environment()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc

        resolution = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self._cross_encoder_cls = CrossEncoder
        self.device = str(resolution.resolved)
        self.model_name = model_name
        self.batch_size_cpu = batch_size_cpu
        self.batch_size_cuda = batch_size_cuda
        self.max_oom_retries = max_oom_retries
        self.allow_cpu_fallback = allow_cpu_fallback
        self.model_source = resolve_cached_model(model_name)
        self.model = CrossEncoder(self.model_source, device=self.device)
        if resolution.fallback_used:
            LOGGER.warning(resolution.reason)

    def _predict(self, pairs, batch_size: int):
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    def rerank(self, query: str, chunks, top_k: int = 5):
        if not chunks:
            return []
        batch = self.batch_size_cuda if self.device == "cuda" else self.batch_size_cpu
        pairs = []
        for c in chunks:
            metadata = getattr(c, "metadata", {}) or {}
            candidate = "\n".join(
                part
                for part in (
                    str(metadata.get("title", "")),
                    str(metadata.get("section", "")),
                    str(c.text),
                )
                if part
            )
            pairs.append((query, candidate))
        last_oom: RuntimeError | None = None
        for attempt in range(self.max_oom_retries + 1):
            try:
                scores = self._predict(pairs, batch)
                ranked = sorted(
                    zip(scores, chunks, strict=False),
                    key=lambda item: float(item[0]),
                    reverse=True,
                )[:top_k]
                return [
                    chunk.model_copy(update={"score": float(score), "rank": rank})
                    for rank, (score, chunk) in enumerate(ranked, start=1)
                ]
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower() and self.device == "cuda"
                if not is_oom:
                    raise
                last_oom = exc
                if attempt < self.max_oom_retries:
                    LOGGER.warning("CUDA OOM during reranking; reducing batch size from %s", batch)
                    clear_cuda_cache()
                    batch = max(1, batch // 2)

        if self.device == "cuda" and self.allow_cpu_fallback:
            LOGGER.warning("CUDA reranking retries exhausted; reloading reranker on CPU")
            clear_cuda_cache()
            self.device = "cpu"
            self.model = self._cross_encoder_cls(self.model_source, device="cpu")
            scores = self._predict(pairs, self.batch_size_cpu)
            ranked = sorted(
                zip(scores, chunks, strict=False),
                key=lambda item: float(item[0]),
                reverse=True,
            )[:top_k]
            return [
                chunk.model_copy(update={"score": float(score), "rank": rank})
                for rank, (score, chunk) in enumerate(ranked, start=1)
            ]
        raise last_oom or RuntimeError("Reranking failed after retries")
