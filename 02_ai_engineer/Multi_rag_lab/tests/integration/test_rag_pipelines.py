from rag_engine.retrieval.context import ContextBuilder
from rag_engine.indexing.hashing import HashingEmbeddingProvider
from rag_engine.generation.grounded import GroundedGenerator
from rag_engine.generation.provider_dummy import DummyLLMProvider
from rag_engine.models import Chunk
from rag_engine.retrieval.query_transform import MultiQueryGenerator
from rag_engine.retrieval.query_transform import QueryRewriter
from rag_engine.retrieval.strategies import BaselineRAG
from rag_engine.retrieval.strategies import CompressionRAG
from rag_engine.retrieval.strategies import CorrectiveRAG
from rag_engine.retrieval.strategies import HybridRAG
from rag_engine.retrieval.strategies import MultiQueryRAG
from rag_engine.retrieval.strategies import QueryRewriteRAG
from rag_engine.retrieval.strategies import RerankedRAG
from rag_engine.retrieval.rerank_lexical import LexicalReranker
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.sparse import BM25Retriever
from rag_engine.indexing.numpy_store import NumpyVectorStore


def build_components():
    chunks = [
        Chunk(chunk_id="c1", document_id="d", text="RAG uses retrieval to add external evidence before grounded generation.", metadata={"source": "doc"}),
        Chunk(chunk_id="c2", document_id="d", text="BM25 is sparse retrieval while embeddings enable dense retrieval.", metadata={"source": "doc"}),
    ]
    embedder = HashingEmbeddingProvider()
    store = NumpyVectorStore(); store.add(embedder.embed_documents([c.text for c in chunks]), chunks)
    dense = DenseRetriever(embedder, store)
    hybrid = HybridRetriever(dense, BM25Retriever(chunks))
    llm = DummyLLMProvider()
    common = dict(context_builder=ContextBuilder(500), generator=GroundedGenerator(llm))
    return dense, hybrid, llm, common


def test_core_rag_strategies_run_without_external_llm():
    dense, hybrid, llm, common = build_components()
    pipelines = [
        BaselineRAG(retriever=dense, **common),
        HybridRAG(retriever=hybrid, **common),
        RerankedRAG(retriever=hybrid, reranker=LexicalReranker(), **common),
        MultiQueryRAG(retriever=hybrid, query_generator=MultiQueryGenerator(llm), **common),
        QueryRewriteRAG(retriever=hybrid, rewriter=QueryRewriter(llm), **common),
        CompressionRAG(retriever=dense, **common),
        CorrectiveRAG(retriever=hybrid, rewriter=QueryRewriter(llm), **common),
    ]
    for pipeline in pipelines:
        result = pipeline.answer("How does RAG use retrieval?")
        assert result.answer
        assert result.total_latency_ms >= 0


def test_parent_document_rag_maps_children_to_parent():
    from rag_engine.models import RetrievedChunk
    from rag_engine.retrieval.strategies import ParentDocumentRAG

    _, _, _, common = build_components()
    parent = Chunk(chunk_id="p1", document_id="d", text="Parent context contains the complete RAG explanation.", metadata={"source": "doc", "role": "parent"})
    child = RetrievedChunk(chunk_id="ch1", text="RAG explanation", source="doc", score=0.9, rank=1, metadata={"parent_id": "p1"})

    class ChildRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return [child]

    result = ParentDocumentRAG(retriever=ChildRetriever(), parent_chunks={"p1": parent}, **common).answer("Explain RAG")
    assert result.retrieved_chunks[0].chunk_id == "p1"
    assert "[S1]" in result.answer
