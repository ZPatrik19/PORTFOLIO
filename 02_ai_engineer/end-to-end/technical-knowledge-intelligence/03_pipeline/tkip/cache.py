"""Small, deterministic on-disk caches used by ingestion and embedding stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .logging_config import get_logger
from .models import ParsedBlock

LOGGER = get_logger(__name__)


class ParseCache:
    """Cache parsed document blocks by stable document id and checksum."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, document_id: str, checksum: str) -> Path:
        """Return the cache path for a specific document version."""

        return self.root / f"{document_id}_{checksum[:16]}.json"

    def get(self, document_id: str, checksum: str) -> list[ParsedBlock] | None:
        """Load parsed blocks, returning ``None`` when cache data is absent or corrupt."""

        cache_path = self.path(document_id, checksum)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [ParsedBlock(**row) for row in payload]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning("Ignoring corrupt parse cache %s: %s", cache_path, exc)
            return None

    def put(self, document_id: str, checksum: str, blocks: list[ParsedBlock]) -> None:
        """Persist parsed blocks for later deterministic reuse."""

        payload = [block.model_dump() for block in blocks]
        self.path(document_id, checksum).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )


class EmbeddingCache:
    """Persist embedding vectors keyed by chunk id for one provider signature."""

    def __init__(self, root: Path, signature: str) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        safe_signature = signature.replace("/", "_").replace(":", "_")
        self.metadata_path = self.root / f"embeddings_{safe_signature}.json"
        self.array_path = self.root / f"embeddings_{safe_signature}.npy"

    def load(self) -> dict[str, np.ndarray]:
        """Load cached embeddings, falling back to an empty mapping on corruption."""

        if not self.metadata_path.exists() or not self.array_path.exists():
            return {}

        try:
            chunk_ids = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            vectors = np.load(self.array_path, allow_pickle=False)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning(
                "Ignoring corrupt embedding cache %s / %s: %s",
                self.metadata_path,
                self.array_path,
                exc,
            )
            return {}

        return {
            chunk_id: vectors[index]
            for index, chunk_id in enumerate(chunk_ids)
            if index < len(vectors)
        }

    def save(self, mapping: Mapping[str, np.ndarray]) -> None:
        """Persist embeddings in deterministic id order and NumPy binary form."""

        chunk_ids = list(mapping)
        vectors = (
            np.vstack([mapping[chunk_id] for chunk_id in chunk_ids])
            if chunk_ids
            else np.empty((0, 0), dtype=np.float32)
        )
        self.metadata_path.write_text(json.dumps(chunk_ids), encoding="utf-8")
        np.save(self.array_path, vectors)
