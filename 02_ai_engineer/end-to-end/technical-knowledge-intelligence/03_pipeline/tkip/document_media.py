from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config import PROJECT_ROOT
from .logging_config import get_logger
from .models import Chunk, SourceCitation, SourceVisual

LOGGER = get_logger(__name__)


def _safe_page(page: int | None) -> int:
    return max(1, int(page or 1))


def _chunk_by_id(chunks: Iterable[Chunk], chunk_id: str) -> Chunk | None:
    for c in chunks:
        if c.chunk_id == chunk_id:
            return c
    return None


def extract_source_visuals(citations: list[SourceCitation], chunks: list[Chunk], max_visuals: int = 1) -> list[SourceVisual]:
    """Create query-time source visuals without re-indexing the library.

    For PDFs, prefer the largest embedded image on the cited page. If the page has
    no useful embedded image, render the cited page itself as a readable preview.
    Artifacts are cached under 01_data/interim/source_previews and are excluded
    from source-control by the project hygiene rules.
    """
    out: list[SourceVisual] = []
    cache_root = PROJECT_ROOT / "01_data" / "interim" / "source_previews"
    cache_root.mkdir(parents=True, exist_ok=True)

    for cit in citations:
        if len(out) >= max_visuals:
            break
        chunk = _chunk_by_id(chunks, cit.chunk_id)
        if chunk is None:
            continue
        source = Path(chunk.source)
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        if source.suffix.lower() != ".pdf" or not source.exists():
            continue
        try:
            import pymupdf as fitz

            page_no = _safe_page(cit.page or chunk.page_start)
            pdf = fitz.open(source)
            idx = min(max(0, page_no - 1), max(0, len(pdf) - 1))
            page = pdf[idx]
            doc_dir = cache_root / chunk.document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Prefer a substantial embedded figure instead of logos/icons.
            best = None
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    info = pdf.extract_image(xref)
                    w = int(info.get("width", 0) or 0)
                    h = int(info.get("height", 0) or 0)
                    area = w * h
                    if w >= 240 and h >= 160 and (best is None or area > best[0]):
                        best = (area, info)
                except (OSError, RuntimeError, ValueError) as exc:
                    LOGGER.debug("Embedded PDF image extraction failed for xref %s: %s", xref, exc)
                    continue

            if best is not None:
                info = best[1]
                ext = str(info.get("ext") or "png").lower()
                asset = doc_dir / f"page_{idx+1:04d}_figure.{ext}"
                if not asset.exists():
                    asset.write_bytes(info["image"])
                kind = "embedded_figure"
                caption = f"Extracted figure from {chunk.title}, page {idx+1}."
            else:
                asset = doc_dir / f"page_{idx+1:04d}_preview.png"
                if not asset.exists():
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45), alpha=False)
                    pix.save(str(asset))
                kind = "page_preview"
                caption = f"Source page preview: {chunk.title}, page {idx+1}."
            pdf.close()
            out.append(SourceVisual(
                document_id=chunk.document_id,
                document_title=chunk.title,
                page=idx + 1,
                kind=kind,
                asset_path=str(asset.relative_to(PROJECT_ROOT)),
                caption=caption,
            ))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Source visual extraction failed for %s: %s", source, exc)
            continue
    return out
