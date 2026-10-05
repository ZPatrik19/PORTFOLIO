from rag_engine.evaluation.pipeline_matrix import MatrixPlan
from rag_engine.evaluation.runtime_estimator import estimate_matrix_runtime


def _plan() -> MatrixPlan:
    return MatrixPlan(
        component_configurations=10,
        retrieval_configurations=20,
        rag_configurations=5,
        total_configurations=35,
        cuda_requested=True,
        cuda_available=True,
    )


def test_runtime_estimator_scales_with_questions() -> None:
    base = estimate_matrix_runtime(
        plan=_plan(),
        question_count=5,
        component_documents=10,
        component_chunks=64,
        include_components=True,
        include_rag=True,
        llm_provider="dummy",
        ollama_profile="balanced",
        cuda_selected=True,
        cuda_available=True,
        rerankers=["none", "lexical"],
        gpu_vram_gb=4.0,
    )
    larger = estimate_matrix_runtime(
        plan=_plan(),
        question_count=20,
        component_documents=10,
        component_chunks=64,
        include_components=True,
        include_rag=True,
        llm_provider="dummy",
        ollama_profile="balanced",
        cuda_selected=True,
        cuda_available=True,
        rerankers=["none", "lexical"],
        gpu_vram_gb=4.0,
    )
    assert larger.expected_seconds > base.expected_seconds
    assert larger.retrieval_executions == 400
    assert larger.rag_generations == 100
    assert larger.low_seconds < larger.expected_seconds < larger.high_seconds


def test_local_ollama_estimate_is_slower_than_dummy() -> None:
    common = dict(
        plan=_plan(),
        question_count=10,
        component_documents=10,
        component_chunks=64,
        include_components=True,
        include_rag=True,
        ollama_profile="balanced",
        cuda_selected=True,
        cuda_available=True,
        rerankers=["cross-encoder"],
        gpu_vram_gb=4.0,
    )
    dummy = estimate_matrix_runtime(llm_provider="dummy", **common)
    ollama = estimate_matrix_runtime(llm_provider="ollama", **common)
    assert ollama.expected_seconds > dummy.expected_seconds
