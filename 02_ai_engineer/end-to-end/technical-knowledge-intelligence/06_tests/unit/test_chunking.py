"""Unit tests for document chunking behavior."""
from __future__ import annotations

from pathlib import Path

import pytest

from tkip.chunking import chunk_document
from tkip.models import ParsedBlock
from test_support.factories import make_document_record

pytestmark = pytest.mark.unit


def test_structure_aware_chunking_preserves_code_block_as_code(tmp_path: Path) -> None:
    # Arrange
    document = make_document_record(tmp_path / "example.md", title="Chunking Demo")
    blocks = [
        ParsedBlock(document_id=document.document_id, page=1, block_type="heading", text="Section"),
        ParsedBlock(
            document_id=document.document_id,
            page=1,
            block_type="paragraph",
            text="A paragraph about retrieval. " * 20,
        ),
        ParsedBlock(
            document_id=document.document_id,
            page=1,
            block_type="code",
            text="def answer():\n    return 42",
        ),
    ]

    # Act
    chunks = chunk_document(document, blocks, "structure_aware", 300, 50)

    # Assert
    assert any(chunk.chunk_type == "code" and "def answer" in chunk.text for chunk in chunks)
