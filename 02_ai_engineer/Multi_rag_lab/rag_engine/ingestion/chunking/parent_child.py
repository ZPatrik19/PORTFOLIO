from __future__ import annotations

from rag_engine.ingestion.chunking.recursive import RecursiveChunker
from rag_engine.models import Chunk, Document


class ParentChildChunker:
    name = "parent-child"

    def __init__(self, parent_size: int = 1500, child_size: int = 300) -> None:
        if child_size >= parent_size:
            raise ValueError("child_size must be smaller than parent_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent = RecursiveChunker(parent_size, 100)
        self.child = RecursiveChunker(child_size, 50)

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        parents = self.parent.chunk(documents)
        for p_index, parent in enumerate(parents):
            parent_id = f"{parent.document_id}:parent:{p_index}"
            parent_metadata = {
                **parent.metadata,
                "chunk_id": parent_id,
                "chunking_strategy": self.name,
                "role": "parent",
                "parent_chunk_size": self.parent_size,
                "child_chunk_size": self.child_size,
            }
            parent_chunk = parent.model_copy(update={"chunk_id": parent_id, "metadata": parent_metadata})
            output.append(parent_chunk)

            child_doc = Document(
                document_id=parent.document_id,
                text=parent.text,
                metadata={**parent.metadata, "parent_id": parent_id},
            )
            children = self.child.chunk([child_doc])
            for c_index, child in enumerate(children):
                child_id = f"{parent.document_id}:child:{p_index}:{c_index}"
                child_metadata = {
                    **child.metadata,
                    "chunk_id": child_id,
                    "chunking_strategy": self.name,
                    "role": "child",
                    "parent_id": parent_id,
                    "parent_chunk_size": self.parent_size,
                    "child_chunk_size": self.child_size,
                }
                output.append(
                    child.model_copy(
                        update={
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "metadata": child_metadata,
                        }
                    )
                )
        return output
