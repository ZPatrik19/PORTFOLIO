from __future__ import annotations

import time

from .citations import citations_from_hits, validate_citations
from .config import load_config
from .context import ContextBuilder
from .costs import estimate_gemini_cost_usd
from .document_media import extract_source_visuals
from .embeddings import create_embedding_provider
from .exceptions import IndexNotReadyError
from .gemini_service import GeminiService
from .guardrails import validate_request
from .index_service import IndexService
from .logging_config import get_logger
from .monitoring import Telemetry
from .multi_index import MultiIndexManager
from .query_analysis import analyze_query, evaluate_pipeline
from .query_understanding import understand_query
from .reranking import rerank
from .runtime_chunking import rechunk_hits
from .tools import LibraryTools
from .utils import ids

LOGGER = get_logger(__name__)


class KnowledgePlatform:
    """Application service coordinating retrieval, generation, validation and telemetry.

    Index construction/loading is delegated to :class:`IndexService`; this class
    owns the request lifecycle and preserves the public API used by the UI/API.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.embedder = create_embedding_provider(self.cfg)
        self.telemetry = Telemetry(self.cfg)
        self.chunks = []
        self.vectors = None
        self.retriever = None
        self._index_service = IndexService(self.cfg, self.embedder)

    def ingest_and_index(self):
        """Build the primary corpus index and update in-memory retrieval state."""

        state, report = self._index_service.build()
        self.embedder = self._index_service.embedder
        self.chunks = state.chunks
        self.vectors = state.vectors
        self.retriever = state.retriever
        return report

    def load_index(self):
        """Load the persisted primary index without reprocessing the corpus."""

        state = self._index_service.load()
        self.embedder = self._index_service.embedder
        self.chunks = state.chunks
        self.vectors = state.vectors
        self.retriever = state.retriever
        return self

    def ensure_ready(self):
        """Ensure a retrieval index is available, building it only when necessary."""

        if self.retriever is not None:
            return
        try:
            self.load_index()
        except IndexNotReadyError:
            self.ingest_and_index()

    def ask(
        self,
        request,
        request_id=None,
        trace_id=None,
        *,
        gemini_api_key: str | None = None,
        progress_callback=None,
    ):
        from .models import AskRequest, KnowledgeAnswer

        req = request if isinstance(request, AskRequest) else AskRequest(**request)
        generated_request_id, generated_trace_id = ids()
        request_id = request_id or generated_request_id
        trace_id = trace_id or generated_trace_id
        start = time.perf_counter()
        pipeline = []

        def stage(name, status="success", duration_ms=0.0, detail=None, metrics=None):
            row = {
                "stage": name,
                "status": status,
                "duration_ms": round(float(duration_ms or 0.0), 2),
                "detail": detail or "",
                "metrics": metrics or {},
            }
            pipeline.append(row)
            if progress_callback:
                try:
                    progress_callback(row, list(pipeline))
                except Exception as exc:
                    LOGGER.debug("Progress callback failed and was ignored: %s", exc)

        # Security/input validation happens before any model call.
        tv = time.perf_counter()
        ok, msg = validate_request(req.question, self.cfg)
        stage(
            "Input guardrails", "success" if ok else "blocked", (time.perf_counter() - tv) * 1000,
            msg if not ok else "Input accepted; exfiltration/injection guardrails passed",
        )
        if not ok:
            ans = KnowledgeAnswer(
                answer=msg,
                confidence=1.0,
                insufficient_evidence=True,
                answer_type="guardrail",
                request_id=request_id,
                trace_id=trace_id,
            )
            ans.diagnostics = {"pipeline_steps": pipeline, "request_id": request_id, "trace_id": trace_id}
            return ans

        tr = time.perf_counter()
        self.ensure_ready()
        stage("Knowledge index", "success", (time.perf_counter() - tr) * 1000, f"{len(self.chunks):,} chunks ready")

        # Persistent chunking/index experiments are independent vector spaces.
        # Never mix vectors from different chunking strategies in one retriever.
        active_retriever = self.retriever
        active_chunks = self.chunks
        index_meta = {"strategy": self.cfg["chunking"].get("strategy", "structure_aware")}
        if req.index_variant != "primary":
            tiv = time.perf_counter()
            try:
                active_chunks, _, active_retriever, index_meta = MultiIndexManager(self.cfg).load(req.index_variant, self.embedder)
                stage(
                    "Index strategy", "success", (time.perf_counter() - tiv) * 1000,
                    f"Persistent '{req.index_variant}' chunking index selected · {len(active_chunks):,} chunks",
                    {"variant": req.index_variant, "chunks": len(active_chunks), **{k: v for k, v in index_meta.items() if k in {"chunk_size", "overlap", "model", "provider_class"}}},
                )
            except Exception as exc:
                active_retriever = self.retriever
                active_chunks = self.chunks
                stage(
                    "Index strategy", "warning", (time.perf_counter() - tiv) * 1000,
                    f"Variant '{req.index_variant}' unavailable; primary index used",
                    {"variant": req.index_variant, "error": str(exc)},
                )
        else:
            stage("Index strategy", "success", 0.0, f"Primary persisted index · {len(active_chunks):,} chunks", {"variant": "primary"})

        # A request-scoped Gemini client allows the UI to accept a session-only API
        # key without mutating the embedding/index configuration or writing secrets.
        gem = GeminiService(self.cfg, api_key=gemini_api_key)

        corrected_question = req.question
        query_review = {
            "original": req.question,
            "corrected": req.question,
            "changed": False,
            "issues": [],
            "reviewer": "disabled",
        }
        if req.prompt_language_check:
            tp = time.perf_counter()
            query_review = gem.polish_query(req.question, req.language)
            corrected_question = (query_review.get("corrected") or req.question).strip()
            stage(
                "Prompt language check",
                "success" if query_review.get("reviewer") == "gemini" else "fallback",
                (time.perf_counter() - tp) * 1000,
                "Language/grammar reviewed without changing technical intent",
                {
                    "changed": bool(query_review.get("changed")),
                    "issues": len(query_review.get("issues") or []),
                    "reviewer": query_review.get("reviewer"),
                },
            )
        else:
            stage("Prompt language check", "skipped", 0.0, "Disabled by user")

        effective_question = corrected_question
        prompt_optimization = {
            "profile": "none", "original": corrected_question, "optimized": corrected_question,
            "changed": False, "improvements": [], "reviewer": "disabled",
        }
        if req.prompt_optimization != "none":
            tpo = time.perf_counter()
            if req.prompt_optimization_use_gemini:
                prompt_optimization = gem.optimize_user_prompt(corrected_question, req.prompt_optimization, req.language)
            else:
                from .prompt_engineering import local_optimize
                prompt_optimization = local_optimize(corrected_question, req.prompt_optimization, req.language)
            effective_question = (prompt_optimization.get("optimized") or corrected_question).strip()
            stage(
                "Advanced prompt engineering",
                "success" if prompt_optimization.get("reviewer") == "gemini" else "fallback",
                (time.perf_counter() - tpo) * 1000,
                f"{prompt_optimization.get('profile_label', req.prompt_optimization)} · {len(prompt_optimization.get('improvements') or [])} optimization(s)",
                {
                    "profile": req.prompt_optimization,
                    "reviewer": prompt_optimization.get("reviewer"),
                    "changed": bool(prompt_optimization.get("changed")),
                },
            )
        else:
            stage("Advanced prompt engineering", "skipped", 0.0, "Raw/corrected question used without advanced rewrite")

        tq = time.perf_counter()
        # Intent/language/retrieval analysis must operate on the user's corrected
        # question, not on an English-heavy prompt-engineering scaffold. The
        # optimized prompt is reserved for generation instructions.
        q = understand_query(corrected_question)
        detected_language = req.language if req.language != "auto" else q["language"]
        query = q["rewritten_query"] if self.cfg["retrieval"].get("query_rewrite") else corrected_question
        stage(
            "Query understanding", "success", (time.perf_counter() - tq) * 1000,
            f"Intent={q['intent']} · language={detected_language}",
            {
                "intent": q["intent"],
                "detected_language": detected_language,
                "original_query": req.question,
                "corrected_query": corrected_question,
                "optimized_query": effective_question,
                "prompt_optimization": req.prompt_optimization,
                "retrieval_query": query,
                "rewrite_enabled": bool(self.cfg["retrieval"].get("query_rewrite")),
            },
        )

        ta = time.perf_counter()
        query_analysis = analyze_query(corrected_question, q["intent"], detected_language)
        stage(
            "Topic & context analysis", "success", (time.perf_counter() - ta) * 1000,
            f"topics={', '.join(query_analysis.get('topics', [])[:3]) or 'general'} · context={query_analysis.get('context_strategy')}",
            query_analysis,
        )

        filters = {}
        if req.source_type:
            filters["source_type"] = req.source_type
        if req.chunk_type:
            filters["chunk_type"] = req.chunk_type
        if req.mode == "code":
            filters["chunk_type"] = "code"

        t0 = time.perf_counter()
        hits = active_retriever.search(query, filters or None)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        raw_hits = [{
            "chunk_id": h.chunk.chunk_id,
            "document_id": h.chunk.document_id,
            "document": h.chunk.title,
            "page": h.chunk.page_start,
            "chapter": h.chunk.chapter,
            "section": h.chunk.section,
            "chunk_type": h.chunk.chunk_type,
            "source_type": h.chunk.source_type,
            "bm25_score": h.bm25_score,
            "dense_score": h.dense_score,
            "hybrid_score": h.hybrid_score,
            "hybrid_rank": i + 1,
            "snippet": h.chunk.text[:420].replace("\n", " "),
        } for i, h in enumerate(hits)]
        stage(
            "Hybrid retrieval", "success" if hits else "warning", retrieval_ms,
            f"BM25 + dense → RRF produced {len(hits)} candidates",
            {
                "candidates": len(hits),
                "bm25_k": self.cfg["retrieval"].get("bm25_k"),
                "dense_k": self.cfg["retrieval"].get("dense_k"),
                "fusion": self.cfg["retrieval"].get("fusion", "rrf"),
            },
        )

        before_filter = len(hits)
        if req.document_ids:
            hits = [h for h in hits if h.chunk.document_id in req.document_ids]
        if req.topic:
            topic = req.topic.lower()
            hits = [
                h for h in hits
                if topic in h.chunk.text.lower()
                or any(topic in k.lower() for k in h.chunk.keywords)
                or (h.chunk.framework and topic in h.chunk.framework.lower())
            ]
        stage(
            "Metadata filters", "success", 0.0, f"{before_filter} → {len(hits)} candidates",
            {
                "document_filter_count": len(req.document_ids),
                "topic": req.topic,
                "source_type": req.source_type,
                "chunk_type": filters.get("chunk_type"),
            },
        )

        rr0 = time.perf_counter()
        hits = rerank(query, hits, self.cfg)
        rerank_ms = (time.perf_counter() - rr0) * 1000
        stage(
            "Reranker", "success" if hits else "warning", rerank_ms,
            f"{len(hits)} candidates retained",
            {
                "enabled": bool(self.cfg["reranking"].get("enabled", True)),
                "provider": self.cfg["reranking"].get("provider"),
                "top_n": self.cfg["reranking"].get("top_n"),
            },
        )
        requested_final_k = int(req.retrieval_final_k or self.cfg["retrieval"].get("final_k", 8))
        hits = hits[:max(requested_final_k, 3)]

        if req.mode == "compare" or q["intent"] == "COMPARISON":
            td = time.perf_counter()
            by_doc = {}
            for h in hits:
                by_doc.setdefault(h.chunk.document_id, []).append(h)
            diversified = []
            depth = 0
            while len(diversified) < requested_final_k and any(depth < len(v) for v in by_doc.values()):
                for did in list(by_doc):
                    if depth < len(by_doc[did]):
                        diversified.append(by_doc[did][depth])
                    if len(diversified) >= requested_final_k:
                        break
                depth += 1
            hits = diversified or hits
            stage(
                "Document diversity", "success", (time.perf_counter() - td) * 1000,
                f"Evidence balanced across {len({h.chunk.document_id for h in hits})} documents",
            )
        else:
            stage("Document diversity", "skipped", 0.0, "Only forced for comparison intent")

        # Preserve the original retrieval ranking for the inspector even when the
        # context is re-chunked below.
        ranking = []
        bm_order = {
            x["chunk_id"]: i + 1 for i, x in enumerate(
                sorted(raw_hits, key=lambda x: x["bm25_score"] if x["bm25_score"] is not None else float("-inf"), reverse=True)
            )
        }
        de_order = {
            x["chunk_id"]: i + 1 for i, x in enumerate(
                sorted(raw_hits, key=lambda x: x["dense_score"] if x["dense_score"] is not None else float("-inf"), reverse=True)
            )
        }
        hy_order = {x["chunk_id"]: x["hybrid_rank"] for x in raw_hits}
        for i, h in enumerate(hits, 1):
            ranking.append({
                "final_rank": i,
                "bm25_rank": bm_order.get(h.chunk.chunk_id),
                "dense_rank": de_order.get(h.chunk.chunk_id),
                "hybrid_rank": hy_order.get(h.chunk.chunk_id),
                "document": h.chunk.title,
                "page": h.chunk.page_start,
                "chapter": h.chunk.chapter,
                "section": h.chunk.section,
                "chunk_id": h.chunk.chunk_id,
                "chunk_type": h.chunk.chunk_type,
                "source_type": h.chunk.source_type,
                "bm25_score": h.bm25_score,
                "dense_score": h.dense_score,
                "hybrid_score": h.hybrid_score,
                "reranker_score": h.reranker_score,
                "snippet": h.chunk.text[:700].replace("\n", " "),
            })

        context_hits = hits
        chunking_meta = {
            "enabled": False,
            "before": len(hits),
            "after": len(hits),
            "strategy": "indexed",
            "size": None,
            "overlap": None,
            "note": "Using the persisted index chunks.",
        }
        if req.runtime_chunking:
            tch = time.perf_counter()
            context_hits, chunking_meta = rechunk_hits(
                hits,
                req.runtime_chunk_strategy,
                req.runtime_chunk_size,
                min(req.runtime_chunk_overlap, max(0, req.runtime_chunk_size - 1)),
                max_output_chunks=max(12, self.cfg["retrieval"].get("hybrid_k", 24)),
            )
            # Re-rank the temporary context chunks after reshaping. Retrieval scores
            # remain priors; this pass only changes context ordering/selection.
            context_hits = rerank(query, context_hits, self.cfg)
            stage(
                "Dynamic chunking", "success", (time.perf_counter() - tch) * 1000,
                f"{chunking_meta['strategy']} · {chunking_meta['size']} chars · overlap {chunking_meta['overlap']} · {chunking_meta['before']} → {chunking_meta['after']}",
                chunking_meta,
            )
        else:
            stage("Dynamic chunking", "skipped", 0.0, "Using persisted index chunks")

        tc = time.perf_counter()
        bundle = ContextBuilder(self.cfg).build(context_hits, effective_question, q["intent"], max_chars=req.context_max_chars)
        context_ms = (time.perf_counter() - tc) * 1000
        stage(
            "Context engineering", "success" if bundle["selected_hits"] else "warning", context_ms,
            f"{len(bundle['selected_hits'])} chunks · {bundle['chars']:,} characters",
            {
                "selected_chunks": len(bundle["selected_hits"]),
                "context_chars": bundle["chars"],
                "documents": len({h.chunk.document_id for h in bundle["selected_hits"]}),
                "max_chars": req.context_max_chars or self.cfg["context"].get("max_chars"),
            },
        )

        citations = citations_from_hits(bundle["selected_hits"], 320)
        tool_results = []
        tool_ms = 0.0
        if gem.available:
            tt = time.perf_counter()
            registry = LibraryTools(active_retriever, active_chunks).registry()
            max_calls = int(req.max_tool_calls if req.max_tool_calls is not None else self.cfg["agent"].get("max_tool_calls", 4))
            tool_results = gem.select_and_execute_tool(corrected_question, registry, max_calls)
            # If Gemini elects not to call a tool for an intent that has an obvious
            # deterministic tool mapping, the backend router may execute one safe
            # allowlisted fallback. This improves reliability without allowing the
            # model to invent arbitrary functions.
            called_now = [x.get("tool") for x in tool_results if isinstance(x, dict) and x.get("tool")]
            recommended = list(query_analysis.get("recommended_tools") or [])
            if not called_now and recommended and max_calls > 0:
                name = recommended[0]
                try:
                    if name in {"search_code_examples", "search_public_docs", "search_library"}:
                        args = {"query": corrected_question, "limit": 5}
                    elif name == "create_learning_path":
                        args = {"topic": (query_analysis.get("topics") or [corrected_question])[0]}
                    elif name == "compare_documents":
                        args = {"query": corrected_question, "document_ids": list(dict.fromkeys(h.chunk.document_id for h in hits[:5]))}
                    elif name == "get_document_metadata" and hits:
                        args = {"document_id": hits[0].chunk.document_id}
                    else:
                        args = {}
                    result = registry.execute(name, args)
                    tool_results.append({"tool": name, "arguments": args, "result": result, "status": "success", "router": "deterministic-fallback"})
                except Exception as exc:
                    tool_results.append({"tool": name, "arguments": locals().get("args", {}), "result": {"error": str(exc)}, "status": "error", "router": "deterministic-fallback"})
            tool_ms = (time.perf_counter() - tt) * 1000
            called = [x.get("tool") for x in tool_results if isinstance(x, dict) and x.get("tool")]
            had_error = any(isinstance(x, dict) and (x.get("tool_error") or x.get("status") == "error") for x in tool_results)
            stage(
                "Tool calling", "warning" if had_error else "success", tool_ms,
                f"{len(called)} backend tool call(s)" if called else "Gemini decided no tool was required",
                {"calls": called, "max_tool_calls": int(req.max_tool_calls if req.max_tool_calls is not None else self.cfg["agent"].get("max_tool_calls", 4))},
            )
            if called:
                tc2 = time.perf_counter()
                bundle = ContextBuilder(self.cfg).build(context_hits, effective_question, q["intent"], tool_results, max_chars=req.context_max_chars)
                stage(
                    "Context refresh", "success", (time.perf_counter() - tc2) * 1000,
                    "Validated tool results merged into grounded context",
                )
        else:
            stage("Tool calling", "skipped", 0.0, "Add a Gemini API key in the UI to enable model-selected tools")

        tg = time.perf_counter()
        ans = gem.generate_structured(
            effective_question,
            bundle,
            citations,
            q["intent"],
            prompt_style=req.prompt_style,
            custom_system_prompt=req.custom_system_prompt,
            custom_instructions=req.custom_instructions,
            answer_language=detected_language,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
        gem_ms = (time.perf_counter() - tg) * 1000
        stage(
            "Gemini generation", "success" if gem.available else "fallback", gem_ms,
            f"{self.cfg['gemini']['model']} · grounded structured synthesis" if gem.available else "Gemini unavailable · retrieval-only backend fallback",
            {"model": self.cfg["gemini"]["model"], "available": gem.available, "prompt_style": req.prompt_style, "backend": gem.last_backend},
        )

        tvc = time.perf_counter()
        proposed_sources = ans.sources or ([] if ans.insufficient_evidence else citations[:4])
        validation = validate_citations(proposed_sources, bundle["selected_hits"])
        citation_ms = (time.perf_counter() - tvc) * 1000
        if not validation["valid"]:
            ans.sources = citations[:4] if not ans.insufficient_evidence else []
            ans.confidence = min(ans.confidence, 0.5)
        elif not ans.sources and citations and not ans.insufficient_evidence:
            ans.sources = citations[:4]
        stage(
            "Citation validator", "success" if validation["valid"] else "warning", citation_ms,
            "All citations resolve to selected context" if validation["valid"] else "Invalid citation(s) removed/repaired",
            {"valid": validation["valid"], "citation_count": len(ans.sources), "errors": validation.get("errors", [])},
        )

        visual_ms = 0.0
        if req.include_source_visual and validation["valid"] and ans.sources and ans.answer_type != "retrieval_only" and not ans.insufficient_evidence:
            tviz = time.perf_counter()
            ans.visuals = extract_source_visuals(ans.sources, active_chunks, max_visuals=1)
            visual_ms = (time.perf_counter() - tviz) * 1000
            stage(
                "Source visual", "success" if ans.visuals else "skipped", visual_ms,
                f"Attached {len(ans.visuals)} trusted source visual(s)" if ans.visuals else "No extractable PDF visual for the selected citation",
                {"visuals": len(ans.visuals)},
            )
        else:
            stage("Source visual", "skipped", 0.0, "Disabled or no validated PDF citation")

        pipeline_evaluation = evaluate_pipeline(
            analysis=query_analysis, selected_hits=bundle["selected_hits"], tool_results=tool_results,
            ranking=ranking, citation_valid=validation["valid"],
        )

        quality_review = {}
        if req.quality_review:
            tqc = time.perf_counter()
            quality_review = gem.review_prompt_and_output(
                original_question=req.question,
                corrected_question=corrected_question,
                final_prompt=gem.last_prompt,
                answer=ans.answer,
                expected_language=detected_language,
                citation_valid=validation["valid"],
                insufficient_evidence=ans.insufficient_evidence,
            )
            stage(
                "Prompt & output QA",
                "success" if quality_review.get("reviewer") == "gemini" else "fallback",
                (time.perf_counter() - tqc) * 1000,
                f"Prompt {quality_review.get('prompt_score', 0)}/100 · Output {quality_review.get('output_score', 0)}/100 · Language {quality_review.get('language_score', 0)}/100",
                {
                    "reviewer": quality_review.get("reviewer"),
                    "prompt_score": quality_review.get("prompt_score"),
                    "output_score": quality_review.get("output_score"),
                    "language_score": quality_review.get("language_score"),
                    "grounding_score": quality_review.get("grounding_score"),
                },
            )
        else:
            stage("Prompt & output QA", "skipped", 0.0, "Disabled by user")

        ans.used_tools = sorted({
            x.get("tool") for x in tool_results
            if isinstance(x, dict) and x.get("tool")
        })
        ans.request_id = request_id
        ans.trace_id = trace_id
        ans.latency_ms = (time.perf_counter() - start) * 1000
        usage = gem.last_usage
        estimated_cost = estimate_gemini_cost_usd(
            usage.get("input_tokens"), usage.get("output_tokens"), self.cfg
        )
        stage(
            "Response ready", "success", 0.0, f"Total latency {ans.latency_ms:.1f} ms",
            {
                "confidence": ans.confidence,
                "insufficient_evidence": ans.insufficient_evidence,
                "citations": len(ans.sources),
                "tools": len(ans.used_tools),
            },
        )

        ans.diagnostics.update({
            "intent": q["intent"],
            "language": detected_language,
            "query_analysis": query_analysis,
            "pipeline_evaluation": pipeline_evaluation,
            "original_query": req.question,
            "corrected_query": corrected_question,
            "optimized_query": effective_question,
            "query_review": query_review,
            "prompt_optimization": prompt_optimization,
            "answer_preset": req.answer_preset,
            "index_variant": req.index_variant,
            "index_metadata": index_meta,
            "retrieval_query": query,
            "retrieval_ms": retrieval_ms,
            "reranking_ms": rerank_ms,
            "context_ms": context_ms,
            "tool_ms": tool_ms,
            "gemini_ms": gem_ms,
            "citation_validation_ms": citation_ms,
            "citation_validation": validation,
            "retrieved": len(hits),
            "context_selected": len(bundle["selected_hits"]),
            "context_chars": bundle["chars"],
            "context_max_chars": req.context_max_chars or self.cfg["context"].get("max_chars"),
            "retrieval_final_k": requested_final_k,
            "max_output_tokens": req.max_output_tokens or self.cfg["gemini"].get("max_output_tokens"),
            "max_tool_calls": req.max_tool_calls if req.max_tool_calls is not None else self.cfg["agent"].get("max_tool_calls"),
            "runtime_chunking": chunking_meta,
            "prompt_style": req.prompt_style,
            "gemini_available": gem.available,
            "gemini_backend": gem.last_backend,
            "gemini_error": gem.last_error,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": estimated_cost,
            "quality_review": quality_review,
            "pipeline_steps": pipeline,
            "request_id": request_id,
            "trace_id": trace_id,
        })
        if req.debug:
            ans.diagnostics["ranking"] = ranking
            ans.diagnostics["retrieved_chunks"] = ranking
            ans.diagnostics["context_chunks"] = [{
                "rank": i + 1,
                "chunk_id": h.chunk.chunk_id,
                "document": h.chunk.title,
                "page": h.chunk.page_start,
                "page_end": h.chunk.page_end,
                "chapter": h.chunk.chapter,
                "section": h.chunk.section,
                "chunk_type": h.chunk.chunk_type,
                "source": h.chunk.source,
                "asset_path": h.chunk.asset_path,
                "chars": len(h.chunk.text),
                "snippet": h.chunk.text[:900].replace("\n", " "),
            } for i, h in enumerate(bundle["selected_hits"])]
            ans.diagnostics["tool_results"] = tool_results
            ans.diagnostics["system_prompt"] = gem.last_system_prompt
            ans.diagnostics["final_prompt"] = gem.last_prompt

        self.telemetry.log({
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/ask",
            "query": req.question,
            "query_type": q["intent"],
            "model": self.cfg["gemini"]["model"],
            "retrieval_method": f"{req.index_variant}:hybrid_rrf",
            "retrieved_chunks": len(hits),
            "retrieval_scores": [h.hybrid_score for h in hits],
            "reranker_scores": [h.reranker_score for h in hits],
            "context_size": bundle["chars"],
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": estimated_cost,
            "tool_calls": ans.used_tools,
            "tool_latency_ms": tool_ms,
            "gemini_latency_ms": gem_ms,
            "retrieval_latency_ms": retrieval_ms,
            "reranking_latency_ms": rerank_ms,
            "context_latency_ms": context_ms,
            "total_latency_ms": ans.latency_ms,
            "citation_count": len(ans.sources),
            "citation_valid": validation["valid"],
            "validation_result": validation,
            "error_type": None,
            "response_status": "success",
            "user_feedback": None,
            "insufficient_evidence": ans.insufficient_evidence,
            "no_answer_numeric": int(ans.insufficient_evidence),
            "top_retrieval_score": hits[0].hybrid_score if hits else 0,
            "runtime_chunking": req.runtime_chunking,
            "prompt_style": req.prompt_style,
            "prompt_optimization": req.prompt_optimization,
            "answer_preset": req.answer_preset,
            "topic_analysis_score": pipeline_evaluation.get("topic_analysis_score"),
            "context_analysis_score": pipeline_evaluation.get("context_analysis_score"),
            "tool_calling_score": pipeline_evaluation.get("tool_calling_score"),
            "retrieval_quality_score": pipeline_evaluation.get("retrieval_quality_score"),
            "index_variant": req.index_variant,
        })
        return ans
