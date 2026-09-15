from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_support.factories import make_chunk, make_document_record, make_search_hit
from tkip.chunking import chunk_document
from tkip.context import ContextBuilder
from tkip.evaluation import evaluate_answer_record, tool_call_metrics
from tkip.ingestion import changed_documents
from tkip.models import KnowledgeAnswer, ParsedBlock, SourceCitation
from tkip.monitoring import Telemetry, drift_report
from tkip.parsing import parse_document

pytestmark = pytest.mark.unit


def test_markdown_parser_preserves_heading_code_and_paragraph(tmp_path: Path) -> None:
    path = tmp_path / "demo.md"
    path.write_text(
        "# Retrieval\n\nRAG combines retrieval and generation.\n\n```python\nprint('rag')\n```\n",
        encoding="utf-8",
    )
    document = make_document_record(path)

    blocks = parse_document(document)

    assert [block.block_type for block in blocks] == ["heading", "paragraph", "code"]
    assert blocks[1].section == "Retrieval"


def test_html_parser_extracts_structural_block_types(tmp_path: Path) -> None:
    path = tmp_path / "demo.html"
    path.write_text("<h1>Docker</h1><p>Bridge networking.</p><pre>docker ps</pre>", encoding="utf-8")
    document = make_document_record(path)

    blocks = parse_document(document)

    assert [block.block_type for block in blocks] == ["heading", "paragraph", "code"]
    assert all(block.section == "Docker" for block in blocks)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structure_aware", "semantic"])
def test_all_chunking_strategies_return_source_grounded_chunks(strategy: str, tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    document = make_document_record(path)
    blocks = [
        ParsedBlock(document_id=document.document_id, page=1, block_type="heading", text="RAG"),
        ParsedBlock(document_id=document.document_id, page=1, block_type="paragraph", text="Retrieval augmented generation uses relevant external context. " * 8, section="RAG"),
        ParsedBlock(document_id=document.document_id, page=2, block_type="paragraph", text="Reranking improves candidate ordering. " * 6, section="RAG"),
    ]

    chunks = chunk_document(document, blocks, strategy=strategy, size=240, overlap=40)

    assert chunks
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_context_builder_enforces_per_document_limit_and_budget(isolated_config: dict) -> None:
    hits = [
        make_search_hit(make_chunk(chunk_id=f"c{i}", document_id="doc-a", text=f"unique evidence {i} " * 20), rank=i)
        for i in range(1, 5)
    ]

    bundle = ContextBuilder(isolated_config).build(
        hits,
        question="Explain retrieval",
        intent="CONCEPTUAL",
        max_chars=1800,
        max_chunks_per_document=2,
    )

    assert len(bundle["selected_hits"]) <= 2
    assert bundle["chars"] <= 1800
    assert "USER QUESTION" in bundle["prompt"]


def test_changed_documents_ignores_unchanged_checksum(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("hello", encoding="utf-8")
    record = make_document_record(source, checksum="abc")
    state = tmp_path / "indexed.json"
    state.write_text(json.dumps({record.document_id: "abc"}), encoding="utf-8")

    assert changed_documents([record], state) == []

    state.write_text(json.dumps({record.document_id: "different"}), encoding="utf-8")
    assert changed_documents([record], state) == [record]


def test_telemetry_summary_and_feedback_use_isolated_database(isolated_config: dict) -> None:
    telemetry = Telemetry(isolated_config)
    telemetry.log(
        {
            "request_id": "request-1",
            "response_status": "success",
            "total_latency_ms": 100,
            "insufficient_evidence": False,
            "citation_valid": True,
            "total_tokens": 50,
            "estimated_cost_usd": 0.001,
        }
    )
    telemetry.feedback("request-1", True, "useful")

    summary = telemetry.summary()

    assert summary["requests"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["feedback_score"] == 1.0


def test_drift_report_requires_two_windows() -> None:
    report = drift_report([{"query_type": "FACTUAL"}], window=2)

    assert report["status"] == "insufficient_data"
    assert report["required"] == 4


def test_answer_record_does_not_fake_semantic_metrics() -> None:
    citation = SourceCitation(
        document_id="doc-1",
        document_title="Book",
        page=1,
        chunk_id="chunk-1",
        quote_or_evidence="evidence",
    )
    answer = KnowledgeAnswer(answer="Grounded answer", confidence=0.9, sources=[citation])

    metrics = evaluate_answer_record(answer, expected_available=True, context_chunk_ids={"chunk-1"})

    assert metrics["citation_correctness"] == 1.0
    assert metrics["faithfulness"] is None
    assert metrics["answer_correctness"] is None


def test_tool_metrics_penalize_unnecessary_calls() -> None:
    metrics = tool_call_metrics(
        [
            {
                "expected_tool": "search_library",
                "selected_tool": "search_library",
                "arguments_valid": True,
                "execution_success": True,
                "tool_calls": 1,
            },
            {
                "expected_tool": None,
                "selected_tool": "search_library",
                "arguments_valid": True,
                "execution_success": True,
                "tool_calls": 1,
            },
        ]
    )

    assert metrics["tool_selection_accuracy"] == 0.5
    assert metrics["unnecessary_tool_call_rate"] == 0.5
