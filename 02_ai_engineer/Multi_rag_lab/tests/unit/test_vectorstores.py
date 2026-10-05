from pathlib import Path

import numpy as np
import pytest

from rag_engine.models import Chunk
from rag_engine.indexing.numpy_store import NumpyVectorStore


def chunks():
    return [
        Chunk(chunk_id="a", document_id="d", text="alpha", metadata={"source": "x"}),
        Chunk(chunk_id="b", document_id="d", text="beta", metadata={"source": "x"}),
    ]


def test_numpy_vector_store_search_and_persistence(tmp_path: Path):
    store = NumpyVectorStore()
    store.add(np.array([[1, 0], [0, 1]], dtype=np.float32), chunks())
    assert store.search(np.array([1, 0], dtype=np.float32), 1)[0].chunk_id == "a"
    store.save(tmp_path)
    loaded = NumpyVectorStore.load(tmp_path)
    assert loaded.search(np.array([0, 1], dtype=np.float32), 1)[0].chunk_id == "b"


def test_faiss_cpu_if_available(tmp_path: Path):
    pytest.importorskip("faiss")
    from rag_engine.indexing.faiss_cpu import FaissCPUVectorStore

    store = FaissCPUVectorStore()
    store.add(np.array([[1, 0], [0, 1]], dtype=np.float32), chunks())
    assert store.search(np.array([1, 0], dtype=np.float32), 1)[0].chunk_id == "a"
    store.save(tmp_path)
    loaded = FaissCPUVectorStore.load(tmp_path)
    assert loaded.search(np.array([0, 1], dtype=np.float32), 1)[0].chunk_id == "b"
