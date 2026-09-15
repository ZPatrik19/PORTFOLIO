from __future__ import annotations

import re
from collections import Counter

TECH_TOPICS = [
    "rag",
    "retrieval",
    "embedding",
    "embeddings",
    "transformer",
    "attention",
    "llm",
    "prompt engineering",
    "context engineering",
    "tool calling",
    "agent",
    "pytorch",
    "tensorflow",
    "machine learning",
    "deep learning",
    "random forest",
    "xgboost",
    "pca",
    "clustering",
    "etl",
    "elt",
    "spark",
    "pyspark",
    "databricks",
    "docker",
    "kubernetes",
    "fastapi",
    "langchain",
    "langgraph",
    "vector database",
    "qdrant",
    "bm25",
    "reranking",
    "monitoring",
    "observability",
    "azure",
    "aws",
    "statistics",
    "data visualization",
    "security",
]

STOP = {
    "the",
    "and",
    "what",
    "how",
    "why",
    "from",
    "with",
    "this",
    "that",
    "into",
    "about",
    "show",
    "explain",
    "mi",
    "az",
    "egy",
    "hogyan",
    "miért",
    "hogy",
    "működik",
    "mutasd",
    "magyarázd",
    "könyveim",
    "alapján",
    "please",
    "keress",
    "find",
    "using",
    "use",
    "my",
    "books",
    "book",
    "nekem",
    "is",
    "van",
    "vagy",
}


def _terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ž0-9_+#.-]{3,}", text.lower())


def analyze_query(question: str, intent: str, language: str) -> dict:
    low = question.lower()
    topics = []
    for topic in TECH_TOPICS:
        if topic in low:
            topics.append(topic)
    if not topics:
        counts = Counter(t for t in _terms(question) if t not in STOP and not t.isdigit())
        topics = [x for x, _ in counts.most_common(4)]

    frameworks = [
        x
        for x in [
            "pytorch",
            "tensorflow",
            "fastapi",
            "langchain",
            "langgraph",
            "docker",
            "kubernetes",
            "databricks",
            "azure",
            "aws",
            "qdrant",
        ]
        if x in low
    ]
    needs_comparison = intent == "COMPARISON" or any(
        x in low for x in ["compare", "hasonlíts", "versus", " vs "]
    )
    needs_code = intent == "CODE_SEARCH" or any(
        x in low for x in ["code", "kód", "implement", "api", "example", "példa"]
    )
    needs_math = any(
        x in low for x in ["math", "matemat", "derive", "képlet", "formula", "equation"]
    )
    needs_public = intent == "PUBLIC_DOC_COMPARISON" or "official" in low or "hivatalos" in low
    needs_learning = intent == "LEARNING" or any(
        x in low for x in ["teach", "taníts", "learn", "tanul"]
    )

    recommended_tools: list[str] = []
    if intent == "METADATA_SEARCH":
        recommended_tools.append("get_document_metadata")
    if needs_code:
        recommended_tools.append("search_code_examples")
    if needs_comparison:
        recommended_tools.append("compare_documents")
    if needs_public:
        recommended_tools.append("search_public_docs")
    if needs_learning:
        recommended_tools.append("create_learning_path")

    if needs_comparison:
        context_strategy = "multi-document diversity"
    elif needs_code:
        context_strategy = "code-biased evidence"
    elif needs_math:
        context_strategy = "definition + derivation + example"
    elif needs_learning:
        context_strategy = "progressive teaching context"
    else:
        context_strategy = "balanced grounded evidence"

    complexity = (
        "high"
        if sum([needs_comparison, needs_code, needs_math, needs_public]) >= 2 or len(question) > 220
        else "medium"
        if len(question) > 90
        else "low"
    )
    specificity = min(
        100,
        45
        + 10 * min(4, len(topics))
        + (10 if frameworks else 0)
        + (10 if len(question.split()) >= 8 else 0),
    )

    return {
        "language": language,
        "intent": intent,
        "topics": topics[:6],
        "frameworks": frameworks,
        "complexity": complexity,
        "needs_comparison": needs_comparison,
        "needs_code": needs_code,
        "needs_math": needs_math,
        "needs_public_docs": needs_public,
        "needs_learning": needs_learning,
        "recommended_tools": recommended_tools,
        "context_strategy": context_strategy,
        "query_specificity_score": specificity,
    }


def evaluate_pipeline(
    *, analysis: dict, selected_hits: list, tool_results: list, ranking: list, citation_valid: bool
) -> dict:
    doc_count = len({h.chunk.document_id for h in selected_hits}) if selected_hits else 0
    context_chars = sum(len(h.chunk.text) for h in selected_hits)
    avg_rerank = 0.0
    vals = [
        float(r.get("reranker_score") or 0)
        for r in ranking[:8]
        if r.get("reranker_score") is not None
    ]
    if vals:
        avg_rerank = sum(vals) / len(vals)

    topic_score = min(
        100, 55 + 10 * len(analysis.get("topics") or []) + (10 if analysis.get("frameworks") else 0)
    )
    context_score = 25
    if selected_hits:
        context_score += 35
    if analysis.get("needs_comparison"):
        if doc_count >= 2:
            context_score += 15
    else:
        # A single highly relevant source can be sufficient for a factual or
        # conceptual question; diversity is mandatory only for comparison.
        context_score += 15
    if 5000 <= context_chars <= 32000:
        context_score += 15
    if citation_valid:
        context_score += 10
    context_score = min(100, context_score)

    expected = set(analysis.get("recommended_tools") or [])
    actual = {
        x.get("tool")
        for x in tool_results
        if isinstance(x, dict) and x.get("tool") and x.get("status") == "success"
    }
    if not expected:
        tool_score = 100 if not actual else 75
    else:
        tool_score = int(100 * len(expected & actual) / max(1, len(expected)))
        if not actual:
            tool_score = max(40, tool_score)

    retrieval_score = min(100, int(55 + 45 * max(0.0, min(1.0, avg_rerank)))) if ranking else 0
    return {
        "topic_analysis_score": int(topic_score),
        "context_analysis_score": int(context_score),
        "tool_calling_score": int(tool_score),
        "retrieval_quality_score": int(retrieval_score),
        "citation_score": 100 if citation_valid else 50,
        "expected_tools": sorted(expected),
        "actual_tools": sorted(actual),
        "context_documents": doc_count,
        "context_chars": context_chars,
        "avg_reranker_score": round(avg_rerank, 4),
    }
