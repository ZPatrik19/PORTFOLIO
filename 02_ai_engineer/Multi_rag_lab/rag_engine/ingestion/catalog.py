from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag_engine.ingestion.downloader import download_url


@dataclass(frozen=True)
class SourceSpec:
    id: str
    title: str
    category: str
    organization: str
    language: str
    format: str
    pages_estimate: int | None
    default: bool
    filename: str
    url: str
    source_page: str | None
    description: str

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "SourceSpec":
        return cls(
            id=str(item["id"]),
            title=str(item["title"]),
            category=str(item.get("category", "Egyéb")),
            organization=str(item.get("organization", "")),
            language=str(item.get("language", "hu")),
            format=str(item.get("format", "pdf")),
            pages_estimate=(int(item["pages_estimate"]) if item.get("pages_estimate") else None),
            default=bool(item.get("default", False)),
            filename=str(item.get("filename") or Path(str(item["url"])).name),
            url=str(item["url"]),
            source_page=(str(item["source_page"]) if item.get("source_page") else None),
            description=str(item.get("description", "")),
        )

    def sidecar_metadata(self) -> dict[str, Any]:
        return {
            "source_id": self.id,
            "title": self.title,
            "category": self.category,
            "organization": self.organization,
            "language": self.language,
            "format": self.format,
            "source_page": self.source_page,
            "description": self.description,
        }


def load_source_catalog(path: Path) -> list[SourceSpec]:
    if not path.exists():
        return []
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [SourceSpec.from_mapping(item) for item in config.get("sources", [])]


def default_sources(path: Path) -> list[SourceSpec]:
    return [source for source in load_source_catalog(path) if source.default]


def download_source(source: SourceSpec, destination_dir: Path) -> Path:
    return download_url(
        source.url,
        destination_dir,
        filename=source.filename,
        metadata=source.sidecar_metadata(),
    )


def downloaded_source_path(source: SourceSpec, destination_dir: Path) -> Path:
    return destination_dir / source.filename
