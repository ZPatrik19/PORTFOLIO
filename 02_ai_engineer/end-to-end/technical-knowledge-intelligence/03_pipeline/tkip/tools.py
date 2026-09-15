"""Allowlisted backend tools exposed to Gemini function calling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models import Chunk

ToolFunction = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    """Metadata and implementation for one allowlisted backend function."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFunction


class ToolRegistry:
    """Validate and execute only explicitly registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ValueError(f"Tool not allowlisted: {name}")
        self._validate_arguments(spec, arguments)
        return spec.fn(**arguments)

    @staticmethod
    def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
        properties = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        unknown = set(arguments) - set(properties)
        missing = required - set(arguments)
        if unknown:
            raise ValueError(f"Unknown tool arguments: {sorted(unknown)}")
        if missing:
            raise ValueError(f"Missing required tool arguments: {sorted(missing)}")


class LibraryTools:
    """Create the RAG-specific tool registry over the active retriever/corpus."""

    MAX_RESULTS = 10
    MAX_COMPARE_DOCUMENTS = 5

    def __init__(self, retriever: Any, chunks: list[Chunk]) -> None:
        self.retriever = retriever
        self.chunks = chunks

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for spec in self._specifications():
            registry.register(spec)
        return registry

    def _specifications(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "search_library",
                "Search the indexed private and public technical library.",
                self._query_schema(),
                self.search_library,
            ),
            ToolSpec(
                "search_public_docs",
                "Search only public technical documentation.",
                self._query_schema(),
                self.search_public_docs,
            ),
            ToolSpec(
                "get_document_metadata",
                "Get metadata for an indexed document.",
                {
                    "type": "object",
                    "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"],
                },
                self.get_document_metadata,
            ),
            ToolSpec(
                "get_page_content",
                "Get indexed chunks from a specific page.",
                {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                    "required": ["document_id", "page"],
                },
                self.get_page_content,
            ),
            ToolSpec(
                "search_code_examples",
                "Search code-like chunks.",
                self._query_schema(),
                self.search_code_examples,
            ),
            ToolSpec(
                "compare_documents",
                "Retrieve evidence from selected documents for comparison.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "document_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "document_ids"],
                },
                self.compare_documents,
            ),
            ToolSpec(
                "create_learning_path",
                "Create the learning-mode section plan for a topic.",
                {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
                self.create_learning_path,
            ),
            ToolSpec(
                "list_library_topics",
                "List dominant indexed topics.",
                {"type": "object", "properties": {}},
                self.list_library_topics,
            ),
            ToolSpec(
                "get_library_statistics",
                "Get document/chunk/language statistics.",
                {"type": "object", "properties": {}},
                self.get_library_statistics,
            ),
        ]

    @staticmethod
    def _query_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }

    def search_library(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return [hit.model_dump() for hit in self.retriever.search(query)[: self._limit(limit)]]

    def search_public_docs(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        hits = self.retriever.search(query, {"source_type": "public"})
        return [hit.model_dump() for hit in hits[: self._limit(limit)]]

    def get_document_metadata(self, document_id: str) -> dict[str, Any]:
        matching = [chunk for chunk in self.chunks if chunk.document_id == document_id]
        return {
            "document_id": document_id,
            "title": matching[0].title if matching else None,
            "chunks": len(matching),
            "pages": sorted({chunk.page_start for chunk in matching}),
        }

    def get_page_content(self, document_id: str, page: int) -> list[str]:
        return [
            chunk.text
            for chunk in self.chunks
            if chunk.document_id == document_id and chunk.page_start <= page <= chunk.page_end
        ][: self.MAX_RESULTS]

    def search_code_examples(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        hits = self.retriever.search(query, {"chunk_type": "code"})
        return [hit.model_dump() for hit in hits[: self._limit(limit)]]

    def compare_documents(self, query: str, document_ids: list[str]) -> dict[str, Any]:
        search_results = self.retriever.search(query)
        output: dict[str, Any] = {}
        for document_id in document_ids[: self.MAX_COMPARE_DOCUMENTS]:
            output[document_id] = [
                hit.model_dump()
                for hit in search_results
                if hit.chunk.document_id == document_id
            ][:3]
        return output

    @staticmethod
    def create_learning_path(topic: str) -> dict[str, Any]:
        return {
            "topic": topic,
            "stages": [
                "intuition",
                "core concepts",
                "mathematics",
                "simple example",
                "Python example",
                "common mistakes",
                "quiz",
                "recommended reading",
            ],
        }

    def list_library_topics(self) -> list[tuple[str, int]]:
        return Counter(keyword for chunk in self.chunks for keyword in chunk.keywords).most_common(40)

    def get_library_statistics(self) -> dict[str, Any]:
        return {
            "documents": len({chunk.document_id for chunk in self.chunks}),
            "chunks": len(self.chunks),
            "languages": sorted({chunk.language for chunk in self.chunks}),
        }

    def _limit(self, requested: int) -> int:
        return max(1, min(int(requested), self.MAX_RESULTS))
