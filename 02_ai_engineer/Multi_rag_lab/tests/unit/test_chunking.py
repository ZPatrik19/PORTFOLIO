from rag_engine.ingestion.chunking.fixed import FixedChunker, FixedTokenChunker
from rag_engine.ingestion.chunking.parent_child import ParentChildChunker
from rag_engine.ingestion.chunking.recursive import RecursiveChunker
from rag_engine.ingestion.chunking.semantic import SemanticChunker
from rag_engine.indexing.hashing import HashingEmbeddingProvider
from rag_engine.models import Document


def sample_doc():
    return Document(
        document_id="d1",
        text=("Retrieval augmented generation uses external evidence. " * 30) + ("Semantic boundaries can improve chunk quality. " * 20),
        metadata={"source": "sample"},
    )


def test_fixed_chunker_respects_max_size():
    chunks = FixedChunker(120, 20).chunk([sample_doc()])
    assert len(chunks) > 2
    assert all(len(c.text) <= 120 for c in chunks)


def test_recursive_chunker_returns_metadata():
    chunks = RecursiveChunker(180, 30).chunk([sample_doc()])
    assert chunks
    assert all(c.metadata["chunking_strategy"] == "recursive" for c in chunks)


def test_semantic_chunker_uses_embedding_interface():
    chunks = SemanticChunker(HashingEmbeddingProvider(), threshold=0.2, max_chars=350).chunk([sample_doc()])
    assert chunks
    assert all(c.metadata["chunking_strategy"] == "semantic" for c in chunks)


def test_parent_child_has_parent_links():
    chunks = ParentChildChunker(parent_size=600, child_size=180).chunk([sample_doc()])
    parents = [c for c in chunks if c.metadata.get("role") == "parent"]
    children = [c for c in chunks if c.metadata.get("role") == "child"]
    assert parents and children
    assert all(c.parent_id for c in children)


def test_fixed_token_chunker_respects_token_count():
    chunks = FixedTokenChunker(20, 5).chunk([sample_doc()])
    assert chunks
    assert all(len(c.text.split()) <= 20 for c in chunks)


def test_sentence_chunker_uses_sentence_boundaries():
    from rag_engine.ingestion.chunking.sentence import SentenceChunker

    document = Document(
        document_id="sentences",
        text="First sentence. Second sentence is longer. Third sentence ends here.",
        metadata={"source": "sample"},
    )
    chunks = SentenceChunker(chunk_size=35).chunk([document])
    assert chunks
    assert all(c.metadata["chunking_strategy"] == "sentence" for c in chunks)
    assert any("First sentence." in c.text for c in chunks)


def test_structure_aware_fallback_keeps_strategy_metadata():
    from rag_engine.ingestion.chunking.structural import StructuralChunker

    chunks = StructuralChunker(chunk_size=120).chunk([sample_doc()])
    assert chunks
    assert all(c.metadata["chunking_strategy"] == "structure-aware" for c in chunks)
