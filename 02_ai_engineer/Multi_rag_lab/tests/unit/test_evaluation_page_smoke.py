from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class _Item:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _Block:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            if name == "progress":
                return self
            return None

        return _noop


class _FakeStreamlit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = {
            "medical_retrieval_eval": [
                {
                    "chunking": "recursive",
                    "retriever": "hybrid-rrf",
                    "reranker": "lexical",
                    "recall_at_k": 0.8,
                    "precision_at_k": 0.6,
                    "f1_at_k": 0.69,
                    "hit_rate_at_k": 0.95,
                    "mrr": 0.77,
                    "map_at_k": 0.72,
                    "ndcg_at_k": 0.79,
                    "mean_first_relevant_rank": 1.4,
                    "source_diversity_at_k": 0.75,
                    "duplicate_ratio_at_k": 0.05,
                    "mean_latency_ms": 17.5,
                    "p50_latency_ms": 15.0,
                    "p95_latency_ms": 27.0,
                    "p99_latency_ms": 31.0,
                    "queries_per_second": 57.0,
                    "labeling_coverage": 1.0,
                }
            ],
            "medical_rag_eval": [
                {
                    "rag_strategy": "hybrid",
                    "citation_accuracy": 1.0,
                    "citation_coverage": 0.8,
                    "citation_source_coverage": 0.7,
                    "citation_density_per_100_words": 2.4,
                    "key_fact_coverage": 0.75,
                    "context_utilization": 0.68,
                    "answer_redundancy": 0.12,
                    "mean_answer_tokens": 180,
                    "mean_retrieval_latency_ms": 18.0,
                    "mean_reranking_latency_ms": 4.0,
                    "mean_generation_latency_ms": 2800.0,
                    "mean_ttft_ms": 520.0,
                    "mean_tokens_per_second": 18.2,
                    "mean_total_latency_ms": 2822.0,
                    "p50_total_latency_ms": 2700.0,
                    "p95_total_latency_ms": 3400.0,
                    "p99_total_latency_ms": 3900.0,
                    "mean_context_tokens": 1320,
                }
            ],
            "eval_candidates": [],
            "top_k": 5,
            "candidate_count": 20,
            "chunking_strategy": "recursive",
            "chunk_size": 700,
            "chunk_overlap": 100,
            "semantic_threshold": 0.72,
            "embedding_device": "cpu",
            "vector_device": "cpu",
            "reranker_mode": "lexical",
            "reranker_device": "cpu",
            "llm_provider": "dummy",
            "context_budget": 1800,
            "context_profile": "balanced",
            "prompt_profile": "professional",
        }

    @staticmethod
    def _decorator(*args, **kwargs):
        def wrap(func):
            return func

        return wrap

    cache_resource = _decorator
    cache_data = _decorator

    def tabs(self, labels):
        return [_Block() for _ in labels]

    def columns(self, spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Block() for _ in range(count)]

    def expander(self, *args, **kwargs):
        return _Block()

    def spinner(self, *args, **kwargs):
        return _Block()

    def progress(self, *args, **kwargs):
        return _Block()

    def empty(self):
        return _Block()

    def slider(self, *args, **kwargs):
        if "value" in kwargs:
            return kwargs["value"]
        if len(args) >= 4:
            return args[3]
        if len(args) >= 3:
            return args[2]
        return 0

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index]

    def multiselect(self, label, options, default=None, **kwargs):
        return list(default or [])

    def checkbox(self, label, value=False, **kwargs):
        return value

    def text_area(self, label, value="", **kwargs):
        return value

    def text_input(self, label, value="", **kwargs):
        return value

    def button(self, *args, **kwargs):
        return False

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def test_evaluation_page_renders_prefilled_results_without_runtime_keyerrors(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    # Import fresh so decorators and module-level Streamlit imports use the fake module.
    for name in list(sys.modules):
        if name.startswith("ui."):
            sys.modules.pop(name)

    evaluation = importlib.import_module("ui.pages.evaluation")
    sample_item = _Item(
        id="q1",
        question_type="overview",
        query="Mi a stroke?",
        article_title="Stroke",
        section="Áttekintés",
        expected_key_facts=["teszt"],
        source_url="https://example.invalid/stroke",
        source_id="stroke",
    )
    monkeypatch.setattr(evaluation, "load_medical_evaluation_dataset", lambda _path: [sample_item])

    evaluation.render()
