from rag_engine.indexing.hashing import HashingEmbeddingProvider
from rag_engine.models import Chunk
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.fusion import reciprocal_rank_fusion
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.sparse import BM25Retriever
from rag_engine.indexing.numpy_store import NumpyVectorStore


def corpus():
    return [
        Chunk(chunk_id="rag", document_id="d", text="RAG uses retrieval and grounded generation", metadata={}),
        Chunk(chunk_id="docker", document_id="d", text="Docker packages applications into containers", metadata={}),
        Chunk(chunk_id="bm25", document_id="d", text="BM25 is a sparse lexical retrieval method", metadata={}),
    ]


def test_bm25_retrieves_lexical_match():
    results = BM25Retriever(corpus()).retrieve("sparse BM25 retrieval", 2)
    assert results[0].chunk_id == "bm25"


def test_hybrid_and_rrf_return_unique_results():
    chunks = corpus()
    embedder = HashingEmbeddingProvider()
    store = NumpyVectorStore()
    store.add(embedder.embed_documents([c.text for c in chunks]), chunks)
    dense = DenseRetriever(embedder, store)
    sparse = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, sparse)
    results = hybrid.retrieve("RAG retrieval", top_k=3, candidate_count=3)
    assert len({r.chunk_id for r in results}) == len(results)
    fused = reciprocal_rank_fusion([results, results], top_k=3)
    assert fused
