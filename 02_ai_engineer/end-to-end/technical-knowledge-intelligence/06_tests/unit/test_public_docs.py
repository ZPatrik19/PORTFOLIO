from __future__ import annotations

import json
from pathlib import Path

import requests

from tkip import public_docs


class FakeResponse:
    def __init__(self, html: str, error: Exception | None = None) -> None:
        self.text = html
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


def test_public_document_download_extracts_readable_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Arrange
    html = "<html><nav>menu</nav><body><h1>RAG</h1><p>Grounded retrieval.</p></body></html>"
    monkeypatch.setattr(
        public_docs.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(html),
    )

    # Act
    report = public_docs.download_public_docs(
        tmp_path,
        sources={"rag": "https://example.test/rag"},
        timeout=1,
    )

    # Assert
    snapshot = (tmp_path / "rag_docs.md").read_text(encoding="utf-8")
    assert report[0]["status"] == "ok"
    assert "Grounded retrieval." in snapshot
    assert "menu" not in snapshot


def test_public_document_download_records_request_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Arrange
    error = requests.HTTPError("503 unavailable")
    monkeypatch.setattr(
        public_docs.requests,
        "get",
        lambda *args, **kwargs: FakeResponse("", error),
    )

    # Act
    report = public_docs.download_public_docs(
        tmp_path,
        sources={"broken": "https://example.test/broken"},
        timeout=1,
    )

    # Assert
    assert report[0]["status"] == "error"
    persisted = json.loads((tmp_path / "download_report.json").read_text(encoding="utf-8"))
    assert persisted[0]["status"] == "error"
