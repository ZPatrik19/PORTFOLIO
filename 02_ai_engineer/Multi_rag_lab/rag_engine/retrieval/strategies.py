from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Protocol

from rag_engine.models import RAGResult, RetrievedChunk
from rag_engine.retrieval.fusion import reciprocal_rank_fusion


class RAGPipeline(Protocol):
    name: str
    def answer(self, query: str) -> RAGResult: ...


class BasePipeline:
    name = "base"

    def __init__(self, *, retriever, context_builder, generator, execution_device: str = "cpu", reranker=None, top_k: int = 5, candidate_count: int = 20) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.execution_device = execution_device
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_count = max(candidate_count, top_k)

    def _finish(self, query: str, chunks, *, transformed_queries: list[str] | None = None, retrieval_ms: float = 0.0, rerank_ms: float | None = None, started: float | None = None) -> RAGResult:
        if self.reranker is not None and rerank_ms is None and chunks:
            rerank_started = time.perf_counter()
            chunks = self.reranker.rerank(query, list(chunks), top_k=self.top_k)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            chunks = list(chunks)
        context_started = time.perf_counter()
        build_for_query = getattr(self.context_builder, "build_for_query", None)
        context = build_for_query(query, chunks, top_k=self.top_k) if callable(build_for_query) else self.context_builder.build(chunks[: self.top_k])
        context_ms = (time.perf_counter() - context_started) * 1000
        generation_started = time.perf_counter()
        answer, citations = self.generator.generate(query, context.text)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        llm_metrics = dict(getattr(self.generator.llm, "last_metrics", {}) or {})
        total_ms = (time.perf_counter() - started) * 1000 if started is not None else retrieval_ms + (rerank_ms or 0) + context_ms + generation_ms
        return RAGResult(
            query=query, transformed_queries=transformed_queries or [], answer=answer, citations=citations,
            retrieved_chunks=context.included, retrieval_latency_ms=retrieval_ms, reranking_latency_ms=rerank_ms,
            generation_latency_ms=generation_ms, generation_ttft_ms=float(llm_metrics.get("ttft_ms", 0.0)) if llm_metrics else None,
            output_tokens=int(llm_metrics.get("output_tokens", 0)) if llm_metrics else None,
            tokens_per_second=float(llm_metrics.get("tokens_per_second", 0.0)) if llm_metrics else None,
            total_latency_ms=total_ms, context_tokens=context.tokens, execution_device=self.execution_device, context_text=context.text,
            grounded_prompt=str(getattr(self.generator, "last_prompt", "") or ""),
            grounded_prompt_parts=dict(getattr(self.generator, "last_prompt_parts", {}) or {}),
            trace={
                "retrieval_ms": retrieval_ms, "reranking_ms": rerank_ms or 0.0, "context_ms": context_ms,
                "generation_ms": generation_ms, "ttft_ms": float(llm_metrics.get("ttft_ms", 0.0)) if llm_metrics else 0.0,
                "tokens_per_second": float(llm_metrics.get("tokens_per_second", 0.0)) if llm_metrics else 0.0,
                "output_tokens": int(llm_metrics.get("output_tokens", 0)) if llm_metrics else 0,
                "generation_mode": str(getattr(self.generator, "last_generation_mode", "standard")), "total_ms": total_ms,
            },
        )


class BaselineRAG(BasePipeline):
    name = "baseline"
    def answer(self, query: str):
        started = time.perf_counter(); t = time.perf_counter()
        requested_k = self.candidate_count if self.reranker is not None else self.top_k
        chunks = self.retriever.retrieve(query, top_k=requested_k)
        return self._finish(query, chunks, retrieval_ms=(time.perf_counter()-t)*1000, started=started)


class HybridRAG(BasePipeline):
    name = "hybrid"
    def answer(self, query: str):
        started = time.perf_counter(); t = time.perf_counter()
        requested_k = self.candidate_count if self.reranker is not None else self.top_k
        chunks = self.retriever.retrieve(query, top_k=requested_k, candidate_count=self.candidate_count)
        return self._finish(query, chunks, retrieval_ms=(time.perf_counter()-t)*1000, started=started)


class LexicalRAG(BasePipeline):
    name = "lexical"
    def answer(self, query: str):
        started = time.perf_counter(); t = time.perf_counter()
        requested_k = self.candidate_count if self.reranker is not None else self.top_k
        chunks = self.retriever.retrieve(query, top_k=requested_k)
        return self._finish(query, chunks, retrieval_ms=(time.perf_counter()-t)*1000, started=started)


class RerankedRAG(BasePipeline):
    name = "reranked"
    def answer(self, query: str):
        if self.reranker is None: raise RuntimeError("RerankedRAG requires a reranker")
        started = time.perf_counter(); t = time.perf_counter()
        try: candidates = self.retriever.retrieve(query, top_k=self.candidate_count, candidate_count=self.candidate_count)
        except TypeError: candidates = self.retriever.retrieve(query, top_k=self.candidate_count)
        retrieval_ms = (time.perf_counter()-t)*1000; t = time.perf_counter()
        chunks = self.reranker.rerank(query, candidates, top_k=self.top_k)
        return self._finish(query, chunks, retrieval_ms=retrieval_ms, rerank_ms=(time.perf_counter()-t)*1000, started=started)


class QueryRewriteRAG(BasePipeline):
    name = "query-rewrite"
    def __init__(self, *, rewriter, **kwargs): super().__init__(**kwargs); self.rewriter = rewriter
    def answer(self, query: str):
        started=time.perf_counter(); rewritten=self.rewriter.rewrite(query); t=time.perf_counter(); requested_k=self.candidate_count if self.reranker is not None else self.top_k
        try: chunks=self.retriever.retrieve(rewritten, top_k=requested_k, candidate_count=self.candidate_count)
        except TypeError: chunks=self.retriever.retrieve(rewritten, top_k=requested_k)
        return self._finish(query, chunks, transformed_queries=[rewritten], retrieval_ms=(time.perf_counter()-t)*1000, started=started)


class HyDERAG(BasePipeline):
    name = "hyde"
    def __init__(self, *, hyde_generator, **kwargs): super().__init__(**kwargs); self.hyde_generator=hyde_generator
    def answer(self, query: str):
        started=time.perf_counter(); hypothetical=self.hyde_generator.generate(query); t=time.perf_counter(); requested_k=self.candidate_count if self.reranker is not None else self.top_k
        chunks=self.retriever.retrieve(hypothetical, top_k=requested_k)
        return self._finish(query, chunks, transformed_queries=[hypothetical], retrieval_ms=(time.perf_counter()-t)*1000, started=started)


class MultiQueryRAG(BasePipeline):
    name = "multi-query"
    def __init__(self, *, query_generator, **kwargs): super().__init__(**kwargs); self.query_generator=query_generator
    def answer(self, query: str):
        started=time.perf_counter(); queries=self.query_generator.generate(query); t=time.perf_counter(); result_sets=[]
        for candidate in queries:
            try: result_sets.append(self.retriever.retrieve(candidate, top_k=self.candidate_count, candidate_count=self.candidate_count))
            except TypeError: result_sets.append(self.retriever.retrieve(candidate, top_k=self.candidate_count))
        fusion_k=self.candidate_count if self.reranker is not None else self.top_k
        return self._finish(query, reciprocal_rank_fusion(result_sets, top_k=fusion_k), transformed_queries=queries, retrieval_ms=(time.perf_counter()-t)*1000, started=started)


class MultiHopRAG(BasePipeline):
    name = "multi-hop"
    def __init__(self, *, follow_up_generator, **kwargs): super().__init__(**kwargs); self.follow_up_generator=follow_up_generator
    def answer(self, query: str):
        started=time.perf_counter(); t=time.perf_counter(); first=self.retriever.retrieve(query, top_k=self.candidate_count, candidate_count=self.candidate_count); retrieval_ms=(time.perf_counter()-t)*1000
        evidence="\n".join(item.text for item in first[:min(3,len(first))]); follow_up=self.follow_up_generator.generate(query,evidence); t=time.perf_counter(); second=self.retriever.retrieve(follow_up, top_k=self.candidate_count, candidate_count=self.candidate_count); retrieval_ms+=(time.perf_counter()-t)*1000
        fusion_k=self.candidate_count if self.reranker is not None else self.top_k
        return self._finish(query, reciprocal_rank_fusion([first,second],top_k=fusion_k), transformed_queries=[follow_up], retrieval_ms=retrieval_ms, started=started)


def _compress(query: str, text: str, max_sentences: int = 3) -> str:
    terms=set(re.findall(r"\w+", query.lower())); sentences=re.split(r"(?<=[.!?])\s+", text); ranked=sorted(sentences,key=lambda s:len(terms & set(re.findall(r"\w+", s.lower()))),reverse=True); selected=[s for s in ranked[:max_sentences] if s.strip()]; return " ".join(selected) or text[:700]


class CompressionRAG(BasePipeline):
    name = "compression"
    def answer(self, query: str):
        started=time.perf_counter(); t=time.perf_counter(); chunks=self.retriever.retrieve(query, top_k=self.candidate_count); retrieval_ms=(time.perf_counter()-t)*1000; compressed=[c.model_copy(update={"text":_compress(query,c.text)}) for c in chunks[:self.top_k]]; return self._finish(query, compressed, retrieval_ms=retrieval_ms, started=started)


def lexical_relevance(query: str, chunks) -> float:
    query_terms=set(re.findall(r"\w+", query.lower()))
    if not query_terms or not chunks: return 0.0
    best=0.0
    for chunk in chunks[:5]:
        terms=set(re.findall(r"\w+", chunk.text.lower())); best=max(best,len(query_terms & terms)/len(query_terms))
    return best


class CorrectiveRAG(BasePipeline):
    name = "corrective"
    def __init__(self, *, rewriter, relevance_threshold: float=0.15, relevance_checker: Callable[[str,list],float]=lexical_relevance, max_iterations:int=2, **kwargs):
        super().__init__(**kwargs); self.rewriter=rewriter; self.relevance_threshold=relevance_threshold; self.relevance_checker=relevance_checker; self.max_iterations=min(max_iterations,2)
    def answer(self, query: str):
        started=time.perf_counter(); current=query; transformed=[]; total_retrieval_ms=0.0; chunks=[]
        for _ in range(self.max_iterations):
            t=time.perf_counter()
            try: chunks=self.retriever.retrieve(current, top_k=self.candidate_count if self.reranker is not None else self.top_k, candidate_count=self.candidate_count)
            except TypeError: chunks=self.retriever.retrieve(current, top_k=self.candidate_count if self.reranker is not None else self.top_k)
            total_retrieval_ms+=(time.perf_counter()-t)*1000
            if self.relevance_checker(current,chunks)>=self.relevance_threshold: break
            current=self.rewriter.rewrite(current); transformed.append(current)
        return self._finish(query,chunks,transformed_queries=transformed,retrieval_ms=total_retrieval_ms,started=started)


class ParentDocumentRAG(BasePipeline):
    name = "parent-document"
    def __init__(self, *, parent_chunks: dict[str,object], **kwargs): super().__init__(**kwargs); self.parent_chunks=parent_chunks
    def answer(self, query: str):
        started=time.perf_counter(); t=time.perf_counter(); children=self.retriever.retrieve(query, top_k=self.candidate_count); parents=[]; seen=set()
        for child in children:
            parent_id=child.metadata.get("parent_id"); parent=self.parent_chunks.get(parent_id)
            if parent_id and parent and parent_id not in seen:
                seen.add(parent_id); parents.append(RetrievedChunk(chunk_id=parent.chunk_id,text=parent.text,source=str(parent.metadata.get("source","")),score=child.score,rank=len(parents)+1,metadata=parent.metadata))
        retrieval_ms=(time.perf_counter()-t)*1000; selected=parents[:self.candidate_count] if self.reranker is not None else parents[:self.top_k]; return self._finish(query,selected,retrieval_ms=retrieval_ms,started=started)
