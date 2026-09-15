"""Download small public documentation snapshots for reproducible demo ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import requests
from bs4 import BeautifulSoup

DEFAULT_SOURCES = {
    "python": "https://docs.python.org/3/tutorial/",
    "pytorch": "https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html",
    "scikit-learn": "https://scikit-learn.org/stable/user_guide.html",
    "fastapi": "https://fastapi.tiangolo.com/tutorial/",
    "docker": "https://docs.docker.com/get-started/docker-overview/",
    "kubernetes": "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "gemini": "https://ai.google.dev/gemini-api/docs",
}
USER_AGENT = "TechnicalKnowledgeIntelligencePortfolio/1.0 (+educational demo)"


def _extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )


def download_public_docs(
    out_dir: Path,
    sources: Mapping[str, str] | None = None,
    timeout: int = 30,
) -> list[dict[str, object]]:
    """Download public documentation snapshots without failing the whole batch on one URL."""

    out_dir.mkdir(parents=True, exist_ok=True)
    source_map = dict(sources or DEFAULT_SOURCES)
    report: list[dict[str, object]] = []
    headers = {"User-Agent": USER_AGENT}

    for name, url in source_map.items():
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            text = _extract_readable_text(response.text)
            header = f"# {name.title()} public documentation snapshot\n\nSource: {url}\n\n"
            (out_dir / f"{name}_docs.md").write_text(header + text, encoding="utf-8")
            report.append({"name": name, "url": url, "status": "ok", "chars": len(text)})
        except requests.RequestException as exc:
            report.append(
                {"name": name, "url": url, "status": "error", "error": str(exc)}
            )

    (out_dir / "download_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
