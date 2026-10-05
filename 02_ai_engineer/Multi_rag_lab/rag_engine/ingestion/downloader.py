from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    pass


def _filename_from_response(url: str, response: requests.Response) -> str:
    disposition = response.headers.get("content-disposition", "")
    marker = "filename="
    if marker in disposition.lower():
        raw = disposition.split(marker, 1)[1].strip().strip('"\'')
        if raw:
            return Path(raw).name
    name = Path(urlparse(url).path).name
    if name and Path(name).suffix:
        return name
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    extension = mimetypes.guess_extension(content_type) or ".bin"
    if content_type == "application/pdf":
        extension = ".pdf"
    elif content_type in {"text/html", "application/xhtml+xml"}:
        extension = ".html"
    return f"downloaded_document{extension}"


def _validate_download(target_name: str, content_type: str, first_bytes: bytes, size_bytes: int) -> None:
    if size_bytes <= 0:
        raise DownloadError("A szerver üres dokumentumot adott vissza.")
    suffix = Path(target_name).suffix.lower()
    if suffix == ".pdf":
        # Some official servers send application/octet-stream, therefore validate the file signature too.
        if not first_bytes.startswith(b"%PDF"):
            raise DownloadError(
                f"A letöltött tartalom nem PDF-nek tűnik (Content-Type: {content_type or 'ismeretlen'})."
            )
    elif content_type.startswith("text/html") and suffix not in {".html", ".htm"}:
        raise DownloadError("A szerver dokumentum helyett HTML-oldalt adott vissza.")


def download_url(
    url: str,
    destination_dir: Path,
    *,
    timeout: float = 45.0,
    retries: int = 5,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Download a source atomically and persist checksum + provenance metadata.

    The body is streamed to a temporary file so larger official PDF corpora do not need to be
    held entirely in memory. A failed or invalid response never overwrites an existing source.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        temp_path: Path | None = None
        try:
            with requests.get(
                url,
                timeout=(10.0, timeout),
                headers={"User-Agent": "Multi-RAG-Engineering-Lab/0.3 (+educational project)"},
                stream=True,
            ) as response:
                response.raise_for_status()
                name = Path(filename or _filename_from_response(url, response)).name
                target = destination_dir / name
                temp_path = target.with_suffix(target.suffix + ".part")
                checksum_path = target.with_suffix(target.suffix + ".sha256")
                digest = hashlib.sha256()
                size_bytes = 0
                first_bytes = b""
                with temp_path.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        if not first_bytes:
                            first_bytes = block[:16]
                        handle.write(block)
                        digest.update(block)
                        size_bytes += len(block)
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                _validate_download(name, content_type, first_bytes, size_bytes)
                sha256 = digest.hexdigest()

            if target.exists() and checksum_path.exists() and checksum_path.read_text(encoding="utf-8").strip() == sha256:
                temp_path.unlink(missing_ok=True)
                LOGGER.info("Duplicate download skipped: %s", url)
                return target

            temp_path.replace(target)
            checksum_path.write_text(sha256, encoding="utf-8")
            source_info = {
                "url": url,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "content_type": content_type,
                **(metadata or {}),
            }
            target.with_suffix(target.suffix + ".source.json").write_text(
                json.dumps(source_info, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return target
        except (requests.RequestException, OSError, DownloadError) as exc:
            last_error = exc
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            LOGGER.warning("Download failed (%s/%s): %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(min(2 ** attempt * 2, 20))
    raise DownloadError(f"A dokumentum nem tölthető le: {url}. Utolsó hiba: {last_error}")
