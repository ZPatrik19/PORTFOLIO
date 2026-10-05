from __future__ import annotations

from dataclasses import dataclass

from rag_engine.evaluation.pipeline_matrix import MatrixPlan


@dataclass(frozen=True)
class BenchmarkRuntimeEstimate:
    low_seconds: float
    expected_seconds: float
    high_seconds: float
    retrieval_executions: int
    rag_generations: int
    component_runs: int
    basis: str


def estimate_matrix_runtime(
    *,
    plan: MatrixPlan,
    question_count: int,
    component_documents: int,
    component_chunks: int,
    include_components: bool,
    include_rag: bool,
    llm_provider: str,
    ollama_profile: str,
    cuda_selected: bool,
    cuda_available: bool,
    rerankers: list[str],
    gpu_vram_gb: float | None = None,
) -> BenchmarkRuntimeEstimate:
    """Estimate wall-clock time from the selected benchmark workload.

    This is deliberately a range, not a fake precise runtime prediction. The model is
    based on the actual number of component variants, query-level retrieval executions
    and end-to-end generations. CUDA availability, Cross-Encoder use and Ollama profile
    adjust the per-operation cost. The UI should replace this baseline with historical
    calibration when enough registry runs are available.
    """

    questions = max(1, int(question_count))
    docs = max(1, int(component_documents))
    chunks = max(1, int(component_chunks))
    effective_cuda = bool(cuda_selected and cuda_available)

    component_seconds = 0.0
    if include_components:
        # Parsing/chunking are CPU-bound; embedding/vector work benefits from CUDA only
        # for the embedding portion. The multiplier intentionally includes cold-load
        # overhead because every benchmark variant may instantiate model/index objects.
        per_variant = (0.035 * docs) + (0.006 * chunks)
        if effective_cuda:
            per_variant *= 0.72
        component_seconds = plan.component_configurations * per_variant

    retrieval_per_query = 0.026 if effective_cuda else 0.034
    if "cross-encoder" in rerankers:
        retrieval_per_query += 0.020 if effective_cuda else 0.060
    retrieval_executions = plan.retrieval_configurations * questions
    retrieval_seconds = retrieval_executions * retrieval_per_query

    rag_generations = plan.rag_configurations * questions if include_rag else 0
    rag_seconds = 0.0
    if rag_generations:
        if llm_provider == "dummy":
            rag_per_query = 0.018
        else:
            # Local Qwen generation dominates wall-clock time. The 4 GB class receives
            # a conservative penalty because model load/offload and memory pressure can
            # make TTFT and decode speed less stable.
            rag_per_query = {
                "low_memory": 1.6,
                "balanced": 2.7,
                "extended_context": 4.4,
            }.get(ollama_profile, 2.7)
            if gpu_vram_gb is not None and gpu_vram_gb <= 4.5:
                rag_per_query *= 1.25
        rag_seconds = rag_generations * rag_per_query

    # Model/index cold starts and orchestration overhead matter on a local workstation.
    orchestration_seconds = 4.0 + 0.20 * plan.total_configurations
    expected = component_seconds + retrieval_seconds + rag_seconds + orchestration_seconds

    uncertainty = 0.24 if effective_cuda else 0.32
    if llm_provider == "ollama" and rag_generations:
        uncertainty += 0.10

    low = max(1.0, expected * (1.0 - uncertainty))
    high = max(low, expected * (1.0 + uncertainty))
    basis = (
        "workload-count + CUDA-aware component cost + reranker cost + local LLM profile"
        if llm_provider == "ollama"
        else "workload-count + CUDA-aware component/retrieval cost"
    )
    return BenchmarkRuntimeEstimate(
        low_seconds=low,
        expected_seconds=expected,
        high_seconds=high,
        retrieval_executions=retrieval_executions,
        rag_generations=rag_generations,
        component_runs=plan.component_configurations if include_components else 0,
        basis=basis,
    )
