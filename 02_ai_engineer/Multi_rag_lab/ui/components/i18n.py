from __future__ import annotations

"""Központi magyar UI-szótár.

A technikai azonosítók (pl. MRR, nDCG, TTFT, CUDA, FAISS) változatlanok maradnak,
de a felhasználónak szánt címek és magyarázatok magyarul jelennek meg.
"""

LABELS = {
    "quality_latency_tradeoff": "Minőség–késleltetés egyensúly",
    "overall_quality_score": "Összesített minőségi pontszám",
    "efficiency_score": "Hatékonysági pontszám",
    "rag_quality_profile": "RAG stratégiák minőségi profilja",
    "latency_profile": "Válaszidő profil: első token / generálás / teljes idő",
    "citation_accuracy": "Hivatkozási pontosság",
    "citation_coverage": "Hivatkozási lefedettség",
    "citation_source_coverage": "Forráslefedettség",
    "key_fact_coverage": "Kulcstény-lefedettség",
    "context_utilization": "Kontextus-kihasználtság",
    "source_diversity": "Forrásdiverzitás",
    "evidence": "Bizonyíték / forrásrészlet",
    "evidence_plural": "Bizonyítékok / forrásrészletek",
    "retrieval": "Visszakeresés",
    "reranking": "Újrarangsorolás",
    "generation": "Generálás",
    "throughput": "Áteresztőképesség",
    "cold_start": "Hidegindítás",
    "performance": "Teljesítmény",
    "runtime": "Futtatási környezet",
    "run": "Futás",
    "prompt": "Prompt",
    "grounded_prompt": "Forrásokra támaszkodó prompt",
    "answer_evidence": "Válasz és forrásbizonyítékok",
    "raw_results": "Nyers eredmények",
    "benchmark": "Benchmark",
}

METRIC_LABELS = {
    "Citation accuracy": LABELS["citation_accuracy"],
    "Citation coverage": LABELS["citation_coverage"],
    "Citation source coverage": LABELS["citation_source_coverage"],
    "Key-fact coverage": LABELS["key_fact_coverage"],
    "Context utilization": LABELS["context_utilization"],
    "Forrásdiverzitás": LABELS["source_diversity"],
    "Overall score": LABELS["overall_quality_score"],
    "Efficiency score": LABELS["efficiency_score"],
    "Retrieval ms": "Visszakeresés ms",
    "Reranking ms": "Újrarangsorolás ms",
    "Generation ms": "Generálás ms",
    "Mean total ms": "Átlagos teljes idő ms",
    "P50 total ms": "P50 teljes idő ms",
    "P95 total ms": "P95 teljes idő ms",
    "P99 total ms": "P99 teljes idő ms",
    "Token/s": "Token/s",
    "TTFT ms": "TTFT ms",
}

RAG_STRATEGY_HU = {
    "baseline": "Alap RAG",
    "lexical": "Lexikális / BM25 RAG",
    "hybrid": "Hibrid RAG",
    "reranked": "Újrarangsorolt RAG",
    "dense-reranked": "Dense + újrarangsorolt RAG",
    "hyde": "HyDE RAG",
    "multi-hop": "Többlépéses RAG",
    "multi-query": "Több lekérdezéses RAG",
    "query-rewrite": "Lekérdezés-átíró RAG",
    "parent-document": "Szülődokumentum RAG",
    "compression": "Kontextustömörítő RAG",
    "corrective": "Korrekciós RAG",
}

RETRIEVER_HU = {
    "dense": "Dense / szemantikus",
    "bm25": "BM25 / lexikális",
    "hybrid": "Hibrid",
    "hybrid-rrf": "Hibrid / RRF",
    "hybrid-weighted": "Hibrid / súlyozott",
}

RERANKER_HU = {
    "none": "Nincs újrarangsorolás",
    "lexical": "Lexikális újrarangsorolás",
    "cross-encoder": "Cross-Encoder újrarangsorolás",
}


def hu_label(value: str) -> str:
    return METRIC_LABELS.get(value, LABELS.get(value, value))


def rag_strategy_label(key: str, fallback: str | None = None) -> str:
    return RAG_STRATEGY_HU.get(key, fallback or key)


def translate_columns(frame):
    """Return a copy with known user-facing dataframe columns translated to Hungarian."""
    mapping = {
        **METRIC_LABELS,
        "component": "Komponens",
        "device": "Eszköz",
        "total_ms": "Összes idő ms",
        "mean_ms": "Átlag ms",
        "median_ms": "Medián ms",
        "p95_ms": "P95 ms",
        "throughput_per_sec": "Áteresztőképesség/s",
        "workload_size": "Terhelés",
        "cold_start_ms": "Hidegindítás ms",
        "embedding_mode": "Beágyazási mód",
        "embedding_model": "Beágyazási modell",
        "requested_embedding_device": "Kért beágyazási eszköz",
        "requested_vector_device": "Kért vektoros eszköz",
        "requested_reranker_device": "Kért újrarangsoroló eszköz",
        "actual_reranker_device": "Tényleges újrarangsoroló eszköz",
        "chunking": "Darabolás",
        "retriever": "Visszakereső",
        "reranker": "Újrarangsoroló",
        "rag_strategy": "RAG stratégia",
        "questions": "Kérdések",
        "mean_latency_ms": "Átlagos késleltetés ms",
        "p50_latency_ms": "P50 késleltetés ms",
        "p95_latency_ms": "P95 késleltetés ms",
        "p99_latency_ms": "P99 késleltetés ms",
        "queries_per_second": "Lekérdezés/s",
        "recall_at_k": "Recall@K",
        "precision_at_k": "Precision@K",
        "f1_at_k": "F1@K",
        "hit_rate_at_k": "Találati arány@K",
        "mrr": "MRR",
        "map_at_k": "MAP@K",
        "ndcg_at_k": "nDCG@K",
        "source_diversity_at_k": "Forrásdiverzitás@K",
        "duplicate_ratio_at_k": "Duplikációs arány@K",
        "labeling_coverage": "Címkézési lefedettség",
        "citation_accuracy": LABELS["citation_accuracy"],
        "citation_coverage": LABELS["citation_coverage"],
        "citation_source_coverage": LABELS["citation_source_coverage"],
        "key_fact_coverage": LABELS["key_fact_coverage"],
        "context_utilization": LABELS["context_utilization"],
        "mean_total_latency_ms": "Átlagos teljes idő ms",
        "mean_ttft_ms": "Átlagos TTFT ms",
        "mean_tokens_per_second": "Átlagos token/s",
    }
    return frame.rename(columns=mapping)
