from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.platform.config import load_settings
from rag_engine.indexing.embedding_factory import create_embedding_provider
from rag_engine.models import ChunkingConfig
from rag_engine.ingestion.pipeline import ingest_paths

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".docx"}


def _expand_paths(items: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        if item.is_dir():
            paths.extend(
                path
                for path in sorted(item.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED
            )
        elif item.is_file():
            paths.append(item)
    # Stable deduplication matters for reproducible chunk IDs/order.
    return list(dict.fromkeys(path.resolve() for path in paths))


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokális dokumentumok szövegkinyerése, tisztítása és darabolása.")
    parser.add_argument("paths", nargs="+", type=Path, help="Fájlok vagy könyvtárak; könyvtár esetén rekurzív bejárás történik.")
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--semantic-threshold", type=float, default=0.72)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--hashing",
        action="store_true",
        help="Deterministic offline hashing provider használata szemantikus daraboláshoz.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "chunks.json")
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    if not paths:
        raise SystemExit("Nem található támogatott dokumentum a megadott útvonalakon.")

    config = ChunkingConfig(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        semantic_threshold=args.semantic_threshold,
    )
    embedder = None
    if args.strategy == "semantic":
        settings = load_settings()
        model_name = "hashing" if args.hashing else (args.embedding_model or settings.multilingual_embedding_model)
        embedder = create_embedding_provider(
            model_name,
            device=args.embedding_device,
            fallback_to_hashing=args.hashing,
        )

    result = ingest_paths(paths, config, embedder=embedder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([c.model_dump() for c in result.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {"input_files": len(paths), **result.summary(), "output": str(args.out)}
    if embedder is not None:
        summary["semantic_embedding_model"] = getattr(embedder, "model_name", model_name)
        summary["semantic_embedding_device"] = getattr(embedder, "device", "cpu")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
