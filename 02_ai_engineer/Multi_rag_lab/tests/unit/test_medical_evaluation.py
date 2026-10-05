from __future__ import annotations

import json
from pathlib import Path

from rag_engine.evaluation.medical_dataset import (
    MedicalEvaluationItem,
    build_medical_evaluation_dataset,
    key_fact_coverage,
    load_medical_evaluation_dataset,
    resolve_relevant_chunk_ids,
)
from rag_engine.models import Chunk


def _article_html(title: str) -> str:
    return f"""
    <html><head><title>{title}</title></head><body><main>
      <h1>{title}</h1>
      <p>A {title} egy olyan egészségügyi állapot, amelyről a betegnek érdemes hiteles forrásból tájékozódnia. A pontos megítéléshez orvosi vizsgálat is szükséges lehet.</p>
      <h2>Tünetek</h2>
      <p>A jellemző tünetek közé tartozhat a tartós panasz és az általános rossz közérzet. A tünetek súlyossága személyenként eltérhet.</p>
      <h2>Kezelés</h2>
      <p>A kezelés az állapot jellegétől és súlyosságától függ. A terápiát egészségügyi szakember határozza meg a vizsgálati eredmények alapján.</p>
    </main></body></html>
    """


def test_build_medical_eval_dataset_from_local_sources(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    documents = []
    for idx in range(1, 4):
        filename = f"article-{idx}.html"
        (raw / filename).write_text(_article_html(f"Tesztállapot {idx}"), encoding="utf-8")
        documents.append(
            {
                "source_id": f"source-{idx}",
                "title": f"Tesztállapot {idx}",
                "url": f"https://example.test/article-{idx}",
                "filename": filename,
                "exists": True,
            }
        )
    manifest = {
        "corpus_id": "test-medical",
        "display_name": "Teszt orvosi korpusz",
        "available_documents": 3,
        "documents": documents,
    }
    manifest_path = raw / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "eval.jsonl"

    dataset = build_medical_evaluation_dataset(
        raw_dir=raw,
        manifest_path=manifest_path,
        output_path=output,
        target_questions=3,
    )

    assert len(dataset.items) == 3
    assert output.exists()
    loaded = load_medical_evaluation_dataset(output)
    assert len(loaded) == 3
    assert all(item.source_url.startswith("https://example.test/") for item in loaded)
    assert all(item.expected_key_facts for item in loaded)


def test_resolve_relevant_chunks_uses_source_and_evidence_overlap() -> None:
    item = MedicalEvaluationItem(
        id="q1",
        query="Melyek a tünetek?",
        question_type="symptoms",
        article_title="Teszt",
        source_id="source-a",
        source_url="https://example.test/a",
        section="Tünetek",
        evidence_text="A gyakori tünet a fejfájás és a szédülés.",
        expected_key_facts=["A gyakori tünet a fejfájás és a szédülés."],
    )
    chunks = [
        Chunk(chunk_id="a1", document_id="a", text="A fejfájás és szédülés gyakori tünet lehet.", metadata={"source_id": "source-a"}),
        Chunk(chunk_id="a2", document_id="a", text="Más témáról szóló szöveg.", metadata={"source_id": "source-a"}),
        Chunk(chunk_id="b1", document_id="b", text="Fejfájás és szédülés.", metadata={"source_id": "source-b"}),
    ]
    relevant = resolve_relevant_chunk_ids(item, chunks)
    assert "a1" in relevant
    assert "b1" not in relevant


def test_key_fact_coverage_is_bounded() -> None:
    facts = [
        "A fejfájás és szédülés gyakori tünet lehet.",
        "A kezelés módját orvos határozza meg.",
    ]
    score = key_fact_coverage("A fejfájás és a szédülés gyakori tünet. [S1]", facts)
    assert 0.0 <= score <= 1.0
    assert score == 0.5


def test_medical_benchmark_smoke_with_hashing(tmp_path: Path) -> None:
    from rag_engine.evaluation.medical_benchmark import run_rag_benchmark, run_retrieval_benchmark

    source = tmp_path / "sample.md"
    source.write_text(
        "# Tesztállapot\n\nA jellemző tünet a tartós fejfájás és a szédülés. "
        "A kezelés módját egészségügyi szakember határozza meg a vizsgálat alapján.",
        encoding="utf-8",
    )
    source.with_suffix(".md.source.json").write_text(
        json.dumps({"url": "https://example.test/sample", "source_id": "source-a", "title": "Tesztállapot"}),
        encoding="utf-8",
    )
    item = MedicalEvaluationItem(
        id="q1",
        query="Melyek a Tesztállapot jellemző tünetei?",
        question_type="symptoms",
        article_title="Tesztállapot",
        source_id="source-a",
        source_url="https://example.test/sample",
        section="Tünetek",
        evidence_text="A jellemző tünet a tartós fejfájás és a szédülés.",
        expected_key_facts=["A jellemző tünet a tartós fejfájás és a szédülés."],
    )

    retrieval_rows = run_retrieval_benchmark(
        paths=[source],
        items=[item],
        chunking_strategies=["recursive"],
        retrieval_modes=["hybrid"],
        rerankers=["none"],
        embedding_model="hashing",
        embedding_device="cpu",
        vector_device="cpu",
        fallback_embedding=True,
        top_k=3,
        candidate_count=5,
    )
    assert len(retrieval_rows) == 1
    assert retrieval_rows[0].hit_rate_at_k == 1.0

    rag_rows = run_rag_benchmark(
        paths=[source],
        items=[item],
        rag_strategies=["baseline"],
        chunking_strategy="recursive",
        embedding_model="hashing",
        embedding_device="cpu",
        vector_device="cpu",
        llm_provider="dummy",
        fallback_embedding=True,
        top_k=3,
        candidate_count=5,
    )
    assert len(rag_rows) == 1
    assert 0.0 <= rag_rows[0].citation_accuracy <= 1.0
    assert 0.0 <= rag_rows[0].key_fact_coverage <= 1.0
