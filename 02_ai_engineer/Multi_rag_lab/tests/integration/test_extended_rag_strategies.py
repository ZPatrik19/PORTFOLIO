from rag_engine.models import ChunkingConfig
from rag_engine.service import build_lab, create_rag_pipeline


def test_extended_rag_strategies_run_with_dummy_llm(tmp_path):
    path = tmp_path / "medical.md"
    path.write_text(
        "# Asztma\nAz asztma légúti betegség. Tünete lehet a köhögés és a zihálás. "
        "A kivizsgálás és a kezelés orvosi feladat.",
        encoding="utf-8",
    )
    bundle = build_lab(
        [path],
        chunking=ChunkingConfig(strategy="recursive", chunk_size=300, chunk_overlap=30),
        embedding_model="hashing",
        embedding_device="cpu",
        vector_device="cpu",
        llm_provider="dummy",
        fallback_embedding=True,
    )
    for strategy in ["lexical", "dense-reranked", "hyde", "multi-hop"]:
        pipeline = create_rag_pipeline(bundle, strategy, top_k=2, candidate_count=3)
        result = pipeline.answer("Mik az asztma tünetei?")
        assert result.answer
        assert result.total_latency_ms >= 0
