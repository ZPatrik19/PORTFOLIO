from __future__ import annotations

import json
from pathlib import Path

import requests

from rag_engine.ingestion.catalog import load_source_catalog
from rag_engine.ingestion.downloader import DownloadError, download_url


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "application/pdf") -> None:
        self.body = body
        self.headers = {"content-type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body[:4]
        yield self.body[4:]


def test_hungarian_source_catalog_contains_substantial_default_corpus() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = load_source_catalog(root / "config" / "data_sources.yaml")
    defaults = [source for source in sources if source.default]
    assert len(defaults) >= 10
    assert sum(source.pages_estimate or 0 for source in defaults) >= len(defaults)
    assert all(source.language == "hu" for source in defaults)
    assert all(source.url.startswith("https://") for source in defaults)


def test_download_url_streams_valid_pdf_and_writes_provenance(monkeypatch, tmp_path: Path) -> None:
    response = FakeResponse(b"%PDF-1.7\nmagyar teszt")

    def fake_get(*args, **kwargs):
        assert kwargs["stream"] is True
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    target = download_url(
        "https://example.test/document.pdf",
        tmp_path,
        filename="magyar.pdf",
        metadata={"title": "Magyar tesztdokumentum", "language": "hu"},
    )
    assert target.read_bytes().startswith(b"%PDF")
    assert target.with_suffix(".pdf.sha256").exists()
    sidecar = json.loads(target.with_suffix(".pdf.source.json").read_text(encoding="utf-8"))
    assert sidecar["title"] == "Magyar tesztdokumentum"
    assert sidecar["language"] == "hu"
    assert sidecar["size_bytes"] == len(response.body)


def test_download_rejects_html_disguised_as_pdf(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(b"<html>hiba</html>", "text/html"),
    )
    try:
        download_url(
            "https://example.test/document.pdf",
            tmp_path,
            retries=0,
            filename="dokumentum.pdf",
        )
    except DownloadError:
        pass
    else:
        raise AssertionError("A PDF-validációnak el kellett volna utasítania a HTML választ.")
    assert not (tmp_path / "dokumentum.pdf").exists()
    assert not (tmp_path / "dokumentum.pdf.part").exists()
