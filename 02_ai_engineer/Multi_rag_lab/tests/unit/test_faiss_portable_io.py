from __future__ import annotations

from pathlib import Path

import numpy as np

from rag_engine.indexing.faiss_cpu import FaissCPUVectorStore


class FakeFaiss:
    def __init__(self) -> None:
        self.serialized = None
        self.deserialized = None

    def serialize_index(self, index):
        self.serialized = index
        return np.array([1, 2, 3, 4], dtype=np.uint8)

    def deserialize_index(self, payload):
        self.deserialized = payload.copy()
        return {"restored": payload.tolist()}


def test_faiss_portable_io_handles_unicode_path(tmp_path: Path):
    target = tmp_path / "még1" / "árvíztűrő" / "index.faiss"
    fake = FakeFaiss()

    FaissCPUVectorStore._write_index_portable(fake, "INDEX", target)

    assert target.read_bytes() == bytes([1, 2, 3, 4])
    assert fake.serialized == "INDEX"

    restored = FaissCPUVectorStore._read_index_portable(fake, target)

    assert restored == {"restored": [1, 2, 3, 4]}
    assert fake.deserialized.tolist() == [1, 2, 3, 4]
