import numpy as np

from rag_engine.indexing.sentence_transformer import SentenceTransformerEmbeddingProvider


def test_e5_provider_adds_query_and_passage_prefixes(monkeypatch):
    provider = object.__new__(SentenceTransformerEmbeddingProvider)
    provider.model_name = "intfloat/multilingual-e5-small"
    captured = []

    def fake_encode(texts):
        captured.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(provider, "_encode", fake_encode)
    provider.embed_documents(["egy", "ketto"])
    provider.embed_query("kerdes")
    assert captured[0] == ["passage: egy", "passage: ketto"]
    assert captured[1] == ["query: kerdes"]
