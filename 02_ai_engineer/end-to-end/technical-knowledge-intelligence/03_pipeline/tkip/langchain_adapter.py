from __future__ import annotations

"""Optional LangChain adapter around the project's native retrieval engine.

LangChain is intentionally not the source of truth for retrieval. The project's
HybridRetriever remains directly testable and benchmarkable; this module simply
exposes the same engine through LangChain Core interfaces when desired.
"""


def langchain_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("langchain_core") is not None
    except ImportError:
        return False


def create_langchain_retriever(native_retriever, *, k: int = 8):
    if not langchain_available():
        raise RuntimeError(
            "langchain-core is not installed. Run setup or install langchain-core>=1.6,<2."
        )

    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict, Field

    class TKIHybridRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        engine: object = Field(exclude=True)
        result_k: int = 8

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            hits = self.engine.search(query)[: self.result_k]
            return [
                Document(
                    page_content=h.chunk.text,
                    metadata={
                        "document_id": h.chunk.document_id,
                        "chunk_id": h.chunk.chunk_id,
                        "title": h.chunk.title,
                        "page": h.chunk.page_start,
                        "chapter": h.chunk.chapter,
                        "section": h.chunk.section,
                        "source": h.chunk.source,
                        "bm25_score": h.bm25_score,
                        "dense_score": h.dense_score,
                        "hybrid_score": h.hybrid_score,
                        "reranker_score": h.reranker_score,
                    },
                )
                for h in hits
            ]

    return TKIHybridRetriever(engine=native_retriever, result_k=k)


def create_context_runnable(native_retriever, *, k: int = 8):
    if not langchain_available():
        raise RuntimeError(
            "langchain-core is not installed. Run setup or install langchain-core>=1.6,<2."
        )
    from langchain_core.runnables import RunnableLambda

    retriever = create_langchain_retriever(native_retriever, k=k)
    return RunnableLambda(lambda q: retriever.invoke(q)) | RunnableLambda(
        lambda docs: {
            "documents": docs,
            "context": "\n\n".join(d.page_content for d in docs),
            "citations": [d.metadata for d in docs],
        }
    )
