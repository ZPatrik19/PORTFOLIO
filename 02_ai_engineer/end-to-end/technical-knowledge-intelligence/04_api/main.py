from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from tkip.config import load_config
from tkip.exceptions import AuthenticationError, ExternalServiceError, QuotaExceededError
from tkip.gemini_service import GeminiService
from tkip.langchain_adapter import langchain_available
from tkip.logging_config import configure_logging, get_logger
from tkip.models import AskRequest, FeedbackRecord
from tkip.monitoring import drift_report
from tkip.multi_index import INDEX_STRATEGIES, MultiIndexManager
from tkip.orchestration import KnowledgePlatform
from tkip.presets import ANSWER_PRESETS, CHUNK_PRESETS
from tkip.prompt_engineering import PROFILES
from tkip.workflow_graph import workflow_dot, workflow_rows

configure_logging(load_config())
LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared index/platform per API process. This avoids reloading the vector
    # matrix and BM25 corpus on every request and mirrors production ML serving.
    platform = KnowledgePlatform(load_config())
    platform.ensure_ready()
    app.state.platform = platform
    yield
    app.state.platform = None


app = FastAPI(
    title="Technical Knowledge Intelligence Platform",
    version="1.1.1",
    description="Grounded technical-library RAG with hybrid retrieval, prompt experiments, tool calling, evaluation and observability.",
    lifespan=lifespan,
)


@app.exception_handler(QuotaExceededError)
async def quota_error_handler(request: Request, exc: QuotaExceededError) -> JSONResponse:
    LOGGER.warning("Gemini quota exceeded for request %s: %s", getattr(request.state, "request_id", "unknown"), exc)
    return JSONResponse(status_code=429, content={"detail": str(exc), "error_type": "quota_exceeded"})


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    LOGGER.warning("Gemini authentication failed for request %s", getattr(request.state, "request_id", "unknown"))
    return JSONResponse(status_code=401, content={"detail": str(exc), "error_type": "authentication"})


@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError) -> JSONResponse:
    LOGGER.error("External AI service failure for request %s: %s", getattr(request.state, "request_id", "unknown"), exc)
    return JSONResponse(status_code=503, content={"detail": str(exc), "error_type": "external_service"})


@app.middleware("http")
async def request_trace_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.trace_id = request.headers.get("X-Trace-ID") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Trace-ID"] = request.state.trace_id
    return response


def get_platform(request: Request) -> KnowledgePlatform:
    platform = getattr(request.app.state, "platform", None)
    if platform is None:
        raise HTTPException(503, detail="Knowledge platform is not ready")
    return platform


PlatformDep = Annotated[KnowledgePlatform, Depends(get_platform)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(platform: PlatformDep):
    return {
        "status": "ready",
        "chunks": len(platform.chunks),
        "documents": len({c.document_id for c in platform.chunks}),
    }


@app.get("/workflow")
def workflow(kind: str = "full"):
    if kind not in {"full", "query", "indexing"}:
        raise HTTPException(400, detail="kind must be full, query or indexing")
    return {"kind": kind, "steps": workflow_rows(kind), "dot": workflow_dot(kind)}


@app.get("/library")
def library(platform: PlatformDep):
    docs = {}
    for c in platform.chunks:
        docs.setdefault(c.document_id, {
            "document_id": c.document_id,
            "title": c.title,
            "source": c.source,
            "source_type": c.source_type,
            "language": c.language,
            "chunks": 0,
        })
        docs[c.document_id]["chunks"] += 1
    return list(docs.values())


@app.get("/library/statistics")
def library_statistics(platform: PlatformDep):
    return {
        "documents": len({c.document_id for c in platform.chunks}),
        "chunks": len(platform.chunks),
        "languages": sorted({c.language for c in platform.chunks}),
        "chunk_types": sorted({c.chunk_type for c in platform.chunks}),
    }


@app.get("/indexes")
def indexes(platform: PlatformDep):
    return MultiIndexManager(platform.cfg).available()


@app.post("/indexes/build")
def build_index_variants(
    platform: PlatformDep,
    strategies: list[str] | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    force: bool = False,
):
    selected = strategies or list(INDEX_STRATEGIES)
    invalid = [x for x in selected if x not in INDEX_STRATEGIES]
    if invalid:
        raise HTTPException(400, detail=f"Unknown strategies: {invalid}")
    return MultiIndexManager(platform.cfg).build(selected, chunk_size=chunk_size, overlap=overlap, force=force)


@app.post("/documents/index")
def index_documents(platform: PlatformDep):
    return platform.ingest_and_index()


@app.post("/search")
def search(req: AskRequest, platform: PlatformDep):
    if req.index_variant == "primary":
        retriever = platform.retriever
    else:
        _, _, retriever, _ = MultiIndexManager(platform.cfg).load(req.index_variant, platform.embedder)
    hits = retriever.search(req.question)
    return [h.model_dump() for h in hits]


@app.post("/prompt/optimize")
def optimize_prompt(
    req: AskRequest,
    platform: PlatformDep,
    x_gemini_api_key: Annotated[str | None, Header()] = None,
):
    gem = GeminiService(platform.cfg, api_key=x_gemini_api_key)
    return gem.optimize_user_prompt(req.question, req.prompt_optimization, req.language)


@app.get("/prompt/profiles")
def prompt_profiles():
    return PROFILES


@app.get("/presets")
def presets():
    return {
        "answer_presets": ANSWER_PRESETS,
        "chunk_presets": CHUNK_PRESETS,
    }


@app.get("/langchain/status")
def langchain_status():
    return {
        "installed": langchain_available(),
        "role": "optional adapter over the native HybridRetriever; not the source of truth for evaluation",
    }


@app.post("/ask")
def ask(
    req: AskRequest,
    request: Request,
    platform: PlatformDep,
    x_gemini_api_key: Annotated[str | None, Header()] = None,
):
    return platform.ask(
        req,
        request.state.request_id,
        request.state.trace_id,
        gemini_api_key=x_gemini_api_key,
    )


@app.post("/compare")
def compare(
    req: AskRequest,
    request: Request,
    platform: PlatformDep,
    x_gemini_api_key: Annotated[str | None, Header()] = None,
):
    req.mode = "compare"
    return platform.ask(req, request.state.request_id, request.state.trace_id, gemini_api_key=x_gemini_api_key)


@app.post("/learning")
def learning(
    req: AskRequest,
    request: Request,
    platform: PlatformDep,
    x_gemini_api_key: Annotated[str | None, Header()] = None,
):
    req.mode = "learning"
    return platform.ask(req, request.state.request_id, request.state.trace_id, gemini_api_key=x_gemini_api_key)


@app.post("/feedback")
def feedback(req: FeedbackRecord, platform: PlatformDep):
    platform.telemetry.feedback(req.request_id, req.helpful, req.feedback_text)
    return {"stored": True}


@app.get("/metrics")
def metrics(platform: PlatformDep):
    return platform.telemetry.summary()


@app.get("/monitoring/drift")
def drift(platform: PlatformDep):
    return drift_report(platform.telemetry.recent(), platform.cfg["monitoring"].get("drift_window", 50))


@app.post("/evaluation/run")
def evaluation_run(platform: PlatformDep):
    from tkip.evaluation import generate_eval_dataset, run_retrieval_benchmark

    samples = generate_eval_dataset(platform.chunks, 300)
    df = run_retrieval_benchmark(platform.retriever, samples)
    return {
        "samples": len(samples),
        "metrics": df.mean(numeric_only=True).to_dict() if not df.empty else {},
    }
