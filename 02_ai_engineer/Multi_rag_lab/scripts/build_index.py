from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.platform.config import load_settings
from rag_engine.indexing.embedding_factory import create_embedding_provider
from rag_engine.models import Chunk
from rag_engine.indexing.vector_factory import create_vector_store


def main() -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Feldolgozott szövegrészek beágyazása és tartós vektorindex építése.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "processed" / "chunks.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "indexes" / "default")
    parser.add_argument("--model", default=settings.multilingual_embedding_model)
    parser.add_argument("--embedding-device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--vector-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--fallback-to-hashing",
        action="store_true",
        help="Ha a neurális embedding modell nem tölthető be, determinisztikus hashing fallback engedélyezése.",
    )
    args = parser.parse_args()

    chunks = [
        Chunk.model_validate(item)
        for item in json.loads(args.chunks.read_text(encoding="utf-8"))
    ]
    index_chunks = [chunk for chunk in chunks if chunk.metadata.get("role") != "parent"]
    embedder = create_embedding_provider(
        args.model,
        device=args.embedding_device,
        fallback_to_hashing=args.fallback_to_hashing,
    )
    vectors = embedder.embed_documents([chunk.text for chunk in index_chunks])
    store = create_vector_store(device=args.vector_device, fallback_to_numpy=True)
    store.add(vectors, index_chunks)
    store.save(args.out)
    print(
        json.dumps(
            {
                "vectors": len(index_chunks),
                "embedding_model": getattr(embedder, "model_name", args.model),
                "embedding_device": getattr(embedder, "device", args.embedding_device),
                "vector_backend": getattr(store, "backend_name", "unknown"),
                "vector_device": getattr(store, "device", "unknown"),
                "path": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
