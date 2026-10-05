from rag_engine.retrieval.context import ContextBuilder
from rag_engine.generation.grounded import GroundedGenerator
from rag_engine.generation.provider_dummy import DummyLLMProvider
from rag_engine.models import RetrievedChunk


def test_context_budget_and_citation():
    chunks = [RetrievedChunk(chunk_id="c1", text="RAG uses retrieved evidence.", source="x", score=1.0, rank=1, metadata={"title": "Doc"})]
    context = ContextBuilder(max_tokens=100).build(chunks)
    answer, citations = GroundedGenerator(DummyLLMProvider()).generate("What is RAG?", context.text)
    assert "[S1]" in answer
    assert citations == ["[S1]"]
