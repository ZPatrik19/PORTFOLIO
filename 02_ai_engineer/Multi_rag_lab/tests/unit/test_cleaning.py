from rag_engine.ingestion.cleaning import clean_documents, normalize_text
from rag_engine.models import Document


def test_normalize_text_collapses_whitespace_and_html():
    assert normalize_text("  Hello   world <b> x </b>\n\n") == "Hello world x"


def test_clean_documents_deduplicates():
    docs = [
        Document(document_id="a", text="This is a sufficiently long repeated document body.", metadata={}),
        Document(document_id="b", text="This is a sufficiently long repeated document body.", metadata={}),
    ]
    cleaned, stats = clean_documents(docs)
    assert len(cleaned) == 1
    assert stats.duplicates_removed == 1


def test_repeated_pdf_edge_lines_are_removed():
    docs = [
        Document(
            document_id="p1",
            text="Manual Title\nPage body one has enough content for cleaning.\nConfidential",
            metadata={"source": "manual.pdf", "page": 1},
        ),
        Document(
            document_id="p2",
            text="Manual Title\nPage body two also has enough content for cleaning.\nConfidential",
            metadata={"source": "manual.pdf", "page": 2},
        ),
    ]
    cleaned, stats = clean_documents(docs)
    assert len(cleaned) == 2
    assert all("Manual Title" not in d.text and "Confidential" not in d.text for d in cleaned)
    assert stats.repeated_edge_lines_removed == 4


def test_normalize_text_preserves_numeric_comparisons_and_following_content():
    text = (
        "Optimális vérnyomás: < 120 és < 80.\n"
        "## Mikor forduljon orvoshoz?\n"
        "180/120 Hgmm felett sürgős ellátás válhat szükségessé.\n"
        "BMI > 30 kg/m2."
    )

    normalized = normalize_text(text)

    assert "< 120" in normalized
    assert "< 80" in normalized
    assert "## Mikor forduljon orvoshoz?" in normalized
    assert "180/120 Hgmm" in normalized
    assert "BMI > 30" in normalized
