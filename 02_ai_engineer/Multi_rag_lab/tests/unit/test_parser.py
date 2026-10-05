from __future__ import annotations

import json
from pathlib import Path

from rag_engine.ingestion.parser import parse_file


def test_text_parser_preserves_source_sidecar(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\nUseful content for a retrieval system.", encoding="utf-8")
    path.with_suffix(".md.source.json").write_text(
        json.dumps({"url": "https://example.test/doc", "sha256": "abc"}),
        encoding="utf-8",
    )
    document = parse_file(path)[0]
    assert document.metadata["source"] == "https://example.test/doc"
    assert document.metadata["local_path"] == str(path)


def test_html_parser_preserves_heading_boundaries(tmp_path: Path):
    path = tmp_path / "doc.html"
    path.write_text(
        "<html><head><title>RAG Guide</title></head><body>"
        "<h1>Retrieval</h1><p>Dense and sparse retrieval.</p>"
        "<h2>Reranking</h2><p>Cross encoders reorder candidates.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    document = parse_file(path)[0]
    assert "# Retrieval" in document.text
    assert "## Reranking" in document.text
    assert document.metadata["title"] == "RAG Guide"


def test_html_parser_removes_navigation_and_footer_noise(tmp_path: Path):
    path = tmp_path / "medical.html"
    path.write_text(
        "<html><head><title>Asztma</title></head><body>"
        "<nav>Ismétlődő navigáció</nav><main><h1>Asztma</h1><p>Orvosi tartalom.</p></main>"
        "<footer>Gyorslinkek és lábléc</footer></body></html>",
        encoding="utf-8",
    )
    document = parse_file(path)[0]
    assert "Orvosi tartalom" in document.text
    assert "Ismétlődő navigáció" not in document.text
    assert "Gyorslinkek és lábléc" not in document.text
