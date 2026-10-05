from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import yaml
from bs4 import BeautifulSoup

from rag_engine.ingestion.downloader import download_url


CATEGORY_SLUGS = (
    "1-9",
    "a-a",
    "b",
    "c-cs",
    "d",
    "e-e",
    "f",
    "g-gy",
    "h",
    "i-j",
    "k",
    "l-ly",
    "m",
    "n-ny",
    "o-o",
    "oo-oo",
    "p-q",
    "r",
    "s-sz",
    "t-ty",
    "u-u",
    "v-w",
    "x-y",
    "z-zs",
)


@dataclass(frozen=True)
class MedicalCorpusConfig:
    corpus_id: str
    display_name: str
    language: str
    organization: str
    target_documents: int
    directory_url: str
    allowed_host: str
    article_path_prefix: str
    output_dir: str
    manifest_path: str
    processed_chunks: str
    vectorstore_dir: str
    state_path: str
    indexing: dict[str, Any]
    evaluation: dict[str, Any]
    license_note: str
    disclaimer: str
    priority_title_contains: tuple[str, ...]
    exclude_title_contains: tuple[str, ...]


@dataclass(frozen=True)
class MedicalArticle:
    title: str
    url: str
    group: str

    @property
    def source_id(self) -> str:
        return "medical-" + hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:14]

    @property
    def filename(self) -> str:
        path = Path(urlparse(self.url).path)
        group = _slugify(self.group)[:24] or "az"
        slug = _slugify(path.stem)[:90] or self.source_id
        return f"{group}__{slug}.html"


def load_medical_corpus_config(path: Path) -> MedicalCorpusConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    item = raw.get("corpus", {})
    return MedicalCorpusConfig(
        corpus_id=str(item["id"]),
        display_name=str(item["display_name"]),
        language=str(item.get("language", "hu")),
        organization=str(item.get("organization", "")),
        target_documents=int(item.get("target_documents", 100)),
        directory_url=str(item["directory_url"]),
        allowed_host=str(item["allowed_host"]),
        article_path_prefix=str(item.get("article_path_prefix", "/egeszseg-a-z/")),
        output_dir=str(item["output_dir"]),
        manifest_path=str(item["manifest_path"]),
        processed_chunks=str(item["processed_chunks"]),
        vectorstore_dir=str(item["vectorstore_dir"]),
        state_path=str(item["state_path"]),
        indexing=dict(item.get("indexing", {})),
        evaluation=dict(item.get("evaluation", {})),
        license_note=str(item.get("license_note", "")),
        disclaimer=str(item.get("disclaimer", "")),
        priority_title_contains=tuple(str(x).lower() for x in item.get("priority_title_contains", [])),
        exclude_title_contains=tuple(str(x).lower() for x in item.get("exclude_title_contains", [])),
    )


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return cleaned or "document"


def _normalized_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    return urlunparse((parsed.scheme or "https", parsed.netloc, parsed.path, "", "", ""))


def _article_group(path: str, prefix: str) -> str | None:
    if not path.startswith(prefix):
        return None
    remainder = path[len(prefix) :].strip("/")
    parts = remainder.split("/")
    # Category pages are /egeszseg-a-z/a-a.html; real articles are
    # /egeszseg-a-z/a-a/asztma.html.
    if len(parts) < 2 or not parts[-1].endswith(".html"):
        return None
    return parts[0]


def _dedupe(items: Iterable[MedicalArticle]) -> list[MedicalArticle]:
    seen: set[str] = set()
    result: list[MedicalArticle] = []
    for item in items:
        if item.url in seen:
            continue
        seen.add(item.url)
        result.append(item)
    return result


def _select_diverse(
    articles: list[MedicalArticle],
    *,
    target: int,
    priority_terms: tuple[str, ...],
) -> list[MedicalArticle]:
    if target <= 0:
        return []

    selected: list[MedicalArticle] = []
    selected_urls: set[str] = set()

    # Put clinically useful/common topics into the corpus first when they are available.
    for term in priority_terms:
        match = next(
            (article for article in articles if term in article.title.lower() and article.url not in selected_urls),
            None,
        )
        if match:
            selected.append(match)
            selected_urls.add(match.url)
            if len(selected) >= target:
                return selected

    groups: dict[str, deque[MedicalArticle]] = defaultdict(deque)
    for article in sorted(articles, key=lambda x: (x.group, x.title.casefold())):
        if article.url not in selected_urls:
            groups[article.group].append(article)

    group_order = sorted(groups, key=str.casefold)
    while len(selected) < target and group_order:
        next_groups: list[str] = []
        for group in group_order:
            queue = groups[group]
            if queue:
                article = queue.popleft()
                selected.append(article)
                selected_urls.add(article.url)
                if len(selected) >= target:
                    break
            if queue:
                next_groups.append(group)
        group_order = next_groups
    return selected[:target]


def _extract_articles_from_html(config: MedicalCorpusConfig, html_text: str, base_url: str) -> list[MedicalArticle]:
    soup = BeautifulSoup(html_text, "html.parser")
    discovered: list[MedicalArticle] = []
    for anchor in soup.find_all("a", href=True):
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not title:
            continue
        lower = title.lower()
        if any(term in lower for term in config.exclude_title_contains):
            continue
        url = _normalized_url(base_url, str(anchor.get("href")))
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        if host != config.allowed_host.lower().removeprefix("www."):
            continue
        group = _article_group(parsed.path, config.article_path_prefix)
        if group is None:
            continue
        discovered.append(MedicalArticle(title=title, url=url, group=group))
    return discovered


def _request_html(url: str, timeout: float) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 Multi-RAG-Engineering-Lab/1.0 educational-corpus-builder",
            "Accept-Language": "hu-HU,hu;q=0.9,en;q=0.5",
        },
    )
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def discover_medical_articles(
    config: MedicalCorpusConfig,
    *,
    target: int | None = None,
    timeout: float = 45.0,
    html_text: str | None = None,
) -> list[MedicalArticle]:
    desired = int(target or config.target_documents)
    discovered: list[MedicalArticle] = []

    if html_text is not None:
        discovered.extend(_extract_articles_from_html(config, html_text, config.directory_url))
    else:
        # Primary source: the public HTML sitemap. It currently exposes the Egészség A-Z article tree.
        try:
            directory_html = _request_html(config.directory_url, timeout)
            discovered.extend(_extract_articles_from_html(config, directory_html, config.directory_url))
        except requests.RequestException:
            pass

        # Robust fallback: crawl the alphabetical Egészség A-Z category pages. This also protects
        # the setup from future sitemap layout changes.
        if len(_dedupe(discovered)) < desired:
            site_root = f"https://{config.allowed_host}"
            for slug in CATEGORY_SLUGS:
                category_url = f"{site_root}{config.article_path_prefix}{slug}.html"
                try:
                    category_html = _request_html(category_url, timeout)
                except requests.RequestException:
                    continue
                discovered.extend(_extract_articles_from_html(config, category_html, category_url))
                if len(_dedupe(discovered)) >= max(desired * 2, desired + 25):
                    break

    discovered = _dedupe(discovered)
    return _select_diverse(
        discovered,
        target=desired,
        priority_terms=config.priority_title_contains,
    )


def download_medical_corpus(
    config: MedicalCorpusConfig,
    destination_dir: Path,
    *,
    target: int | None = None,
    refresh: bool = False,
    timeout: float = 45.0,
    progress_callback=None,
) -> dict[str, Any]:
    desired = int(target or config.target_documents)
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_dir / "manifest.json"

    # Discover a reserve pool as well. One transient HTTP failure should not make a
    # 100-document corpus build fail if other valid articles are available.
    reserve = max(20, desired // 4)
    articles = discover_medical_articles(config, target=desired + reserve, timeout=timeout)
    if len(articles) < desired:
        raise RuntimeError(
            f"Csak {len(articles)} megfelelő Egészségvonal-cikket sikerült felfedezni a kért {desired} helyett."
        )

    downloaded = 0
    reused = 0
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []

    for idx, article in enumerate(articles, start=1):
        if len([record for record in records if record["exists"]]) >= desired:
            break
        target_path = destination_dir / article.filename
        if target_path.exists() and not refresh:
            reused += 1
        else:
            try:
                download_url(
                    article.url,
                    destination_dir,
                    filename=article.filename,
                    timeout=timeout,
                    retries=5,
                    metadata={
                        "source_id": article.source_id,
                        "title": article.title,
                        "category": "Orvosi / egészségügyi ismeretterjesztés",
                        "organization": config.organization,
                        "language": config.language,
                        "format": "html",
                        "description": "Egészségvonal Egészség A-Z magyar nyelvű egészségügyi tájékoztató cikk.",
                        "corpus_id": config.corpus_id,
                    },
                )
                downloaded += 1
            except Exception as exc:  # keep partial corpus inspectable and continue with reserve articles
                failures.append({"title": article.title, "url": article.url, "error": str(exc)})
        records.append(
            {
                "source_id": article.source_id,
                "title": article.title,
                "url": article.url,
                "group": article.group,
                "filename": article.filename,
                "exists": target_path.exists(),
            }
        )
        if progress_callback:
            progress_callback(min(idx, desired), desired, article.title)

    valid_records = [record for record in records if record["exists"]][:desired]
    manifest = {
        "corpus_id": config.corpus_id,
        "display_name": config.display_name,
        "organization": config.organization,
        "requested_documents": desired,
        "available_documents": len(valid_records),
        "downloaded_now": downloaded,
        "reused_existing": reused,
        "failures": failures,
        "license_note": config.license_note,
        "disclaimer": config.disclaimer,
        "documents": valid_records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
