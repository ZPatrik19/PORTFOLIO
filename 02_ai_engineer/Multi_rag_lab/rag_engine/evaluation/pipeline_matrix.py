from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rag_engine.platform.config import load_settings
from rag_engine.evaluation.component_benchmark import run_component_benchmarks
from rag_engine.evaluation.medical_benchmark import rows_to_dicts, run_rag_benchmark, run_retrieval_benchmark
from rag_engine.evaluation.medical_dataset import MedicalEvaluationItem
from rag_engine.platform.device import cuda_available
from rag_engine.retrieval.rerank_cross_encoder import CrossEncoderReranker
from rag_engine.retrieval.rerank_lexical import LexicalReranker

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class EmbeddingSpec:
    label: str
    model_name: str
    fallback_embedding: bool = False


@dataclass(frozen=True)
class MatrixPlan:
    component_configurations: int
    retrieval_configurations: int
    rag_configurations: int
    total_configurations: int
    cuda_requested: bool
    cuda_available: bool


def embedding_specs_from_modes(modes: list[str]) -> list[EmbeddingSpec]:
    settings = load_settings()
    mapping = {
        "multilingual": EmbeddingSpec("multilingual", settings.multilingual_embedding_model, False),
        "english": EmbeddingSpec("english", settings.embedding_model, False),
        "e5-small": EmbeddingSpec("e5-small", settings.e5_embedding_model, False),
        "hashing": EmbeddingSpec("hashing", "__offline_hashing_fallback__", True),
    }
    return [mapping[mode] for mode in modes if mode in mapping]


def _reranker_variants(rerankers: list[str], reranker_devices: list[str]) -> list[tuple[str, str]]:
    """Return only semantically distinct reranker/device variants.

    ``none`` and the lexical reranker are CPU algorithms, therefore duplicating
    them for every requested CUDA device only repeats the same benchmark. The
    Cross-Encoder is the only reranker that expands across CPU/CUDA here.
    """
    variants: list[tuple[str, str]] = []
    devices = reranker_devices or ["cpu"]
    for kind in rerankers:
        if kind == "cross-encoder":
            variants.extend((kind, device) for device in devices)
        else:
            variants.append((kind, "cpu"))
    return variants


def build_matrix_plan(
    *,
    embedding_modes: list[str],
    embedding_devices: list[str],
    vector_devices: list[str],
    reranker_devices: list[str],
    chunkings: list[str],
    retrieval_modes: list[str],
    rerankers: list[str],
    rag_strategies: list[str],
    include_rag: bool,
    include_components: bool = True,
) -> MatrixPlan:
    n_embeddings = max(1, len(embedding_modes))
    n_embedding_devices = max(1, len(embedding_devices))
    n_vector_devices = max(1, len(vector_devices))
    n_chunkings = max(1, len(chunkings))
    n_retrieval_modes = max(1, len(retrieval_modes))

    retrieval_reranker_variants = _reranker_variants(rerankers or ["none"], reranker_devices)
    retrieval = (
        n_embeddings
        * n_embedding_devices
        * n_vector_devices
        * n_chunkings
        * n_retrieval_modes
        * max(1, len(retrieval_reranker_variants))
    )

    rag = 0
    if include_rag:
        reranked_keys = {"reranked", "dense-reranked"}
        reranked_count = sum(1 for strategy in rag_strategies if strategy in reranked_keys)
        plain_count = max(0, len(rag_strategies) - reranked_count)
        usable_rerankers = [name for name in rerankers if name != "none"] or ["lexical"]
        rag_reranker_variants = _reranker_variants(usable_rerankers, reranker_devices)
        strategy_variants = max(1, plain_count + reranked_count * max(1, len(rag_reranker_variants)))
        rag = n_embeddings * n_embedding_devices * n_vector_devices * n_chunkings * strategy_variants

    components = 0
    if include_components:
        components = (
            1
            + n_chunkings
            + n_embeddings * n_embedding_devices
            + n_embeddings * n_embedding_devices * n_vector_devices
        )

    reranker_cuda_requested = "cross-encoder" in rerankers and "cuda" in reranker_devices
    return MatrixPlan(
        component_configurations=components,
        retrieval_configurations=retrieval,
        rag_configurations=rag,
        total_configurations=components + retrieval + rag,
        cuda_requested=("cuda" in embedding_devices) or ("cuda" in vector_devices) or reranker_cuda_requested,
        cuda_available=cuda_available(),
    )


def run_pipeline_matrix(
    *,
    paths: list[Path],
    items: list[MedicalEvaluationItem],
    embedding_modes: list[str],
    embedding_devices: list[str],
    vector_devices: list[str],
    reranker_devices: list[str],
    chunkings: list[str],
    retrieval_modes: list[str],
    rerankers: list[str],
    rag_strategies: list[str],
    llm_provider: str,
    ollama_profile: str,
    questions: int,
    chunk_size: int,
    overlap: int,
    semantic_threshold: float,
    top_k: int,
    candidate_count: int,
    context_budget: int,
    fusion: str,
    rrf_k: int,
    dense_weight: float,
    context_profile: str,
    prompt_profile: str,
    include_rag: bool = True,
    include_components: bool = True,
    component_documents: int = 20,
    component_chunks: int = 256,
    progress: ProgressCallback | None = None,
) -> dict[str, list[dict[str, object]]]:
    subset = items[:questions]
    specs = embedding_specs_from_modes(embedding_modes)
    embedding_devices_actual = [device for device in embedding_devices if device != "cuda" or cuda_available()]
    if not embedding_devices_actual:
        embedding_devices_actual = ["cpu"]
    reranker_devices_actual = [device for device in reranker_devices if device != "cuda" or cuda_available()]
    if not reranker_devices_actual:
        reranker_devices_actual = ["cpu"]

    retrieval_variants = _reranker_variants(rerankers or ["none"], reranker_devices_actual)
    reranked_keys = {"reranked", "dense-reranked"}
    plain_rag_strategies = [strategy for strategy in rag_strategies if strategy not in reranked_keys]
    reranked_rag_strategies = [strategy for strategy in rag_strategies if strategy in reranked_keys]
    usable_rerankers = [name for name in rerankers if name != "none"] or ["lexical"]
    rag_reranker_variants = _reranker_variants(usable_rerankers, reranker_devices_actual)

    retrieval_rows: list[dict[str, object]] = []
    rag_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    component_steps = 1 if include_components else 0
    retrieval_steps = len(specs) * len(embedding_devices_actual) * len(vector_devices) * max(1, len(retrieval_variants))
    rag_groups_per_chunk = (1 if plain_rag_strategies else 0) + (len(rag_reranker_variants) if reranked_rag_strategies else 0)
    rag_steps = (
        len(specs) * len(embedding_devices_actual) * len(vector_devices) * len(chunkings) * rag_groups_per_chunk
        if include_rag
        else 0
    )
    total_steps = max(1, component_steps + retrieval_steps + rag_steps)
    current = 0

    if include_components:
        current += 1
        if progress:
            progress(current, total_steps, "Komponens microbenchmark · parsing / chunking / embedding / vector search")
        try:
            component_result = run_component_benchmarks(
                paths=paths,
                chunkings=chunkings,
                embedding_modes=embedding_modes,
                embedding_devices=embedding_devices_actual,
                vector_devices=vector_devices,
                chunk_size=chunk_size,
                overlap=overlap,
                semantic_threshold=semantic_threshold,
                max_documents=component_documents,
                max_embedding_chunks=component_chunks,
                top_k=top_k,
            )
            component_rows.extend(component_result["components"])
            errors.extend(component_result["errors"])
        except Exception as exc:
            errors.append({"phase": "components", "error": str(exc)})

    for spec in specs:
        for embedding_device in embedding_devices_actual:
            for vector_device in vector_devices:
                # Retrieval: lexical/none are measured once because their device is CPU.
                # Only the Cross-Encoder expands into separate CPU/CUDA runs.
                for reranker_name, reranker_device in retrieval_variants:
                    current += 1
                    if progress:
                        progress(
                            current,
                            total_steps,
                            f"Retrieval · {spec.label} · embed={embedding_device} · vector={vector_device} · rerank={reranker_name}/{reranker_device}",
                        )
                    try:
                        rows = run_retrieval_benchmark(
                            paths=paths,
                            items=subset,
                            chunking_strategies=chunkings,
                            retrieval_modes=retrieval_modes,
                            rerankers=[reranker_name],
                            chunk_size=chunk_size,
                            overlap=overlap,
                            semantic_threshold=semantic_threshold,
                            embedding_model=spec.model_name,
                            embedding_device=embedding_device,
                            vector_device=vector_device,
                            top_k=top_k,
                            candidate_count=candidate_count,
                            fusion=fusion,
                            rrf_k=rrf_k,
                            dense_weight=dense_weight,
                            fallback_embedding=spec.fallback_embedding,
                            reranker_device=reranker_device,
                        )
                        for row in rows_to_dicts(rows):
                            row["embedding_mode"] = spec.label
                            row["embedding_model"] = spec.model_name
                            row["requested_embedding_device"] = embedding_device
                            row["requested_vector_device"] = vector_device
                            row["requested_reranker_device"] = reranker_device
                            retrieval_rows.append(row)
                    except Exception as exc:
                        errors.append(
                            {
                                "phase": "retrieval",
                                "embedding_mode": spec.label,
                                "embedding_device": embedding_device,
                                "vector_device": vector_device,
                                "reranker": reranker_name,
                                "reranker_device": reranker_device,
                                "error": str(exc),
                            }
                        )

                if not include_rag:
                    continue

                for chunking in chunkings:
                    if plain_rag_strategies:
                        current += 1
                        if progress:
                            progress(
                                current,
                                total_steps,
                                f"RAG · {spec.label} · embed={embedding_device} · vector={vector_device} · {chunking} · {len(plain_rag_strategies)} stratégia (közös index)",
                            )
                        try:
                            rows = run_rag_benchmark(
                                paths=paths,
                                items=subset,
                                rag_strategies=plain_rag_strategies,
                                chunking_strategy=chunking,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                semantic_threshold=semantic_threshold,
                                embedding_model=spec.model_name,
                                embedding_device=embedding_device,
                                vector_device=vector_device,
                                llm_provider=llm_provider,
                                ollama_profile=ollama_profile,
                                top_k=top_k,
                                candidate_count=candidate_count,
                                max_context_tokens=context_budget,
                                fallback_embedding=spec.fallback_embedding,
                                context_profile=context_profile,
                                prompt_profile=prompt_profile,
                                reranker=None,
                            )
                            for row in rows_to_dicts(rows):
                                row["embedding_mode"] = spec.label
                                row["embedding_model"] = spec.model_name
                                row["requested_embedding_device"] = embedding_device
                                row["requested_vector_device"] = vector_device
                                row["chunking"] = "parent-child" if row["rag_strategy"] == "parent-document" else chunking
                                row["reranker"] = "none"
                                row["requested_reranker_device"] = "cpu"
                                row["actual_reranker_device"] = "—"
                                rag_rows.append(row)
                        except Exception as exc:
                            errors.append(
                                {
                                    "phase": "rag",
                                    "embedding_mode": spec.label,
                                    "embedding_device": embedding_device,
                                    "vector_device": vector_device,
                                    "chunking": chunking,
                                    "rag_strategies": plain_rag_strategies,
                                    "reranker": "none",
                                    "error": str(exc),
                                }
                            )

                    if reranked_rag_strategies:
                        for reranker_name, reranker_device in rag_reranker_variants:
                            current += 1
                            if progress:
                                progress(
                                    current,
                                    total_steps,
                                    f"RAG · {spec.label} · embed={embedding_device} · vector={vector_device} · {chunking} · reranked · {reranker_name}/{reranker_device}",
                                )
                            try:
                                if reranker_name == "cross-encoder":
                                    settings = load_settings()
                                    reranker_obj = CrossEncoderReranker(settings.reranker_model, device=reranker_device)
                                else:
                                    reranker_obj = LexicalReranker()
                                rows = run_rag_benchmark(
                                    paths=paths,
                                    items=subset,
                                    rag_strategies=reranked_rag_strategies,
                                    chunking_strategy=chunking,
                                    chunk_size=chunk_size,
                                    overlap=overlap,
                                    semantic_threshold=semantic_threshold,
                                    embedding_model=spec.model_name,
                                    embedding_device=embedding_device,
                                    vector_device=vector_device,
                                    llm_provider=llm_provider,
                                    ollama_profile=ollama_profile,
                                    top_k=top_k,
                                    candidate_count=candidate_count,
                                    max_context_tokens=context_budget,
                                    fallback_embedding=spec.fallback_embedding,
                                    context_profile=context_profile,
                                    prompt_profile=prompt_profile,
                                    reranker=reranker_obj,
                                )
                                for row in rows_to_dicts(rows):
                                    row["embedding_mode"] = spec.label
                                    row["embedding_model"] = spec.model_name
                                    row["requested_embedding_device"] = embedding_device
                                    row["requested_vector_device"] = vector_device
                                    row["chunking"] = "parent-child" if row["rag_strategy"] == "parent-document" else chunking
                                    row["reranker"] = reranker_name
                                    row["requested_reranker_device"] = reranker_device
                                    row["actual_reranker_device"] = str(getattr(reranker_obj, "device", "cpu"))
                                    rag_rows.append(row)
                            except Exception as exc:
                                errors.append(
                                    {
                                        "phase": "rag",
                                        "embedding_mode": spec.label,
                                        "embedding_device": embedding_device,
                                        "vector_device": vector_device,
                                        "chunking": chunking,
                                        "rag_strategies": reranked_rag_strategies,
                                        "reranker": reranker_name,
                                        "reranker_device": reranker_device,
                                        "error": str(exc),
                                    }
                                )

    return {"components": component_rows, "retrieval": retrieval_rows, "rag": rag_rows, "errors": errors}

