"""
app.py
------
Production FastAPI application layer for the Indian Legal RAG system.

This module is ONLY the API layer. It does not reimplement retrieval,
reranking, fact extraction, citation validation, or response synthesis —
all of that logic lives in the existing ``src/`` modules and is used
exactly as implemented:

    LegalRAGPipeline      (src/pipeline.py)
    LegalRAGOrchestrator  (src/orchestrator.py)
    FactExtractionAgent   (src/agents/classifier.py)
    CrossEncoderReranker  (src/reranker.py)
    CitationValidator     (src/agents/validator.py)
    ResponseSynthesizer   (src/agents/synthesizer.py)

Every heavyweight object above is constructed exactly once, during the
FastAPI lifespan startup phase, and reused for the lifetime of the
process. No endpoint constructs a new pipeline, retriever, reranker,
or LLM client.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root / src path resolution
# Mirrors the pattern already used in src/pipeline.py and src/orchestrator.py
# so `import config` and `from src.x import Y` both work regardless of how
# this file is launched (python app.py, uvicorn app:app, etc.).
# ---------------------------------------------------------------------------

import sys
from pathlib import Path as _Path

_ROOT_DIR = _Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"

for _path in (str(_ROOT_DIR), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

import config
from pipeline import LegalRAGPipeline
from orchestrator import LegalRAGOrchestrator
from agents.classifier import FactExtractionAgent
from reranker import CrossEncoderReranker
from agents.validator import CitationValidator
from agents.synthesizer import ResponseSynthesizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Loguru with a structured, colourised console sink.

    Reads the log level from ``config.LOG_LEVEL``. File logging reuses
    ``config.LOG_DIR`` / ``config.LOG_ROTATION`` / ``config.LOG_RETENTION``,
    matching every other module in this project.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=getattr(config, "LOG_LEVEL", "DEBUG"),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    config.LOG_DIR.mkdir(exist_ok=True)
    logger.add(
        config.LOG_DIR / "app.log",
        level="DEBUG",
        rotation=config.LOG_ROTATION,
        retention=config.LOG_RETENTION,
        encoding="utf-8",
    )


_configure_logging()


# ---------------------------------------------------------------------------
# Runtime statistics
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeStatistics:
    """Process-lifetime API metrics, updated by the request-logging middleware.

    Attributes:
        start_time: ``time.perf_counter()`` value captured at app startup,
            used to compute uptime.
        total_requests: Total number of requests handled.
        successful_requests: Requests that completed with status < 400.
        failed_requests: Requests that completed with status >= 400.
        total_latency_seconds: Sum of every request's latency, used to
            compute the running average.
        endpoint_counts: Per-path request counts.
    """

    start_time: float = field(default_factory=time.perf_counter)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_seconds: float = 0.0
    endpoint_counts: dict[str, int] = field(default_factory=dict)

    def record(self, path: str, status_code: int, latency_seconds: float) -> None:
        """Record one completed request.

        Args:
            path: The request path (e.g. ``"/chat"``).
            status_code: The HTTP status code returned.
            latency_seconds: Wall-clock request duration in seconds.
        """
        self.total_requests += 1
        if status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_latency_seconds += latency_seconds
        self.endpoint_counts[path] = self.endpoint_counts.get(path, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize current statistics to a plain dict.

        Returns:
            Dict with uptime, request counts, and average latency.
        """
        uptime_seconds = time.perf_counter() - self.start_time
        average_latency_ms = (
            (self.total_latency_seconds / self.total_requests) * 1000
            if self.total_requests
            else 0.0
        )
        return {
            "uptime_seconds": round(uptime_seconds, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_latency_ms": round(average_latency_ms, 2),
            "endpoint_counts": dict(self.endpoint_counts),
        }


_stats = _RuntimeStatistics()


# ---------------------------------------------------------------------------
# Shared component container
# ---------------------------------------------------------------------------


@dataclass
class _Components:
    """Holds every heavyweight component, constructed once at startup.

    Populated by the lifespan handler and read by every endpoint via
    ``request.app.state.components``. Never reconstructed per request.
    """

    pipeline: LegalRAGPipeline | None = None
    orchestrator: LegalRAGOrchestrator | None = None
    classifier: FactExtractionAgent | None = None
    reranker: CrossEncoderReranker | None = None
    validator: CitationValidator | None = None
    synthesizer: ResponseSynthesizer | None = None
    pipeline_ready: bool = False


# ---------------------------------------------------------------------------
# Lifespan — construct everything once, on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise every shared component once at startup, log on shutdown.

    All construction happens here, never inside an endpoint. Pipeline
    initialisation (BM25 build, Qdrant connection, reranker model load)
    is the most expensive step and is logged explicitly so a slow
    startup is visible rather than silently absorbed into the first
    request's latency.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI once startup completes; runs shutdown
        logging after the application stops serving requests.
    """
    logger.info("Application startup beginning…")
    startup_start = time.perf_counter()

    components = _Components()

    try:
        components.pipeline = LegalRAGPipeline()
        components.pipeline.initialize()
        components.pipeline_ready = True
        logger.info("LegalRAGPipeline initialised.")
    except Exception:
        logger.exception("Pipeline initialization failed.")
        raise

    try:
        components.orchestrator = LegalRAGOrchestrator(components.pipeline)
        logger.info("LegalRAGOrchestrator constructed with shared pipeline.")
    except Exception:
        logger.exception("LegalRAGOrchestrator failed to construct.")

    try:
        components.classifier = FactExtractionAgent()
        logger.info("FactExtractionAgent constructed.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("FactExtractionAgent failed to construct.")

    try:
        components.reranker = components.pipeline._reranker

        if components.reranker is None:
            raise RuntimeError("Pipeline failed to initialize reranker.")

        logger.info("Using reranker initialized by LegalRAGPipeline.")
    except Exception:
        logger.exception("Unable to obtain reranker from pipeline.")

    try:
        components.validator = CitationValidator()
        logger.info("CitationValidator constructed.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("CitationValidator failed to construct.")

    try:
        components.synthesizer = ResponseSynthesizer()
        logger.info("ResponseSynthesizer constructed.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("ResponseSynthesizer failed to construct.")

    app.state.components = components

    elapsed = time.perf_counter() - startup_start
    logger.info("Application startup complete in {:.2f}s.", elapsed)

    yield

    logger.info("Application shutdown beginning…")
    logger.info(
        "Final statistics: {}", _stats.to_dict()
    )
    logger.info("Application shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app construction
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Indian Legal RAG API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-id / latency / logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Any) -> Any:
    """Attach a request id, measure latency, and log every request.

    Stores the generated UUID on ``request.state.request_id`` so any
    downstream handler or exception handler can reference the same id,
    and records the completed request into the shared runtime
    statistics container.

    Args:
        request: The incoming Starlette/FastAPI request.
        call_next: The next handler in the middleware chain.

    Returns:
        The response produced by the downstream handler, with an added
        ``X-Request-ID`` header.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    logger.info(
        "Request started: id={} method={} path={}",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - start
        logger.exception(
            "Request raised unhandled exception: id={} path={} latency={:.4f}s error={}",
            request_id,
            request.url.path,
            latency,
            exc,
        )
        _stats.record(request.url.path, 500, latency)
        raise

    latency = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    _stats.record(request.url.path, response.status_code, latency)

    logger.info(
        "Request completed: id={} method={} path={} status={} latency={:.4f}s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _error_payload(request: Request, message: str, details: Any = None) -> dict[str, Any]:
    """Build the standard error response payload.

    Args:
        request: The current request, used to read ``request.state.request_id``.
        message: Human-readable error message.
        details: Optional structured error detail (e.g. validation errors).

    Returns:
        Dict matching the documented error schema.
    """
    request_id = getattr(request.state, "request_id", None)
    payload: dict[str, Any] = {
        "success": False,
        "error": message,
        "request_id": request_id,
    }
    if details is not None:
        payload["details"] = details
    return payload


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return the standard error schema for raised ``HTTPException``s.

    Args:
        request: The current request.
        exc: The raised ``HTTPException``.

    Returns:
        A JSON response matching the documented error schema.
    """
    logger.warning(
        "HTTPException: id={} path={} status={} detail={}",
        getattr(request.state, "request_id", None),
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(request, str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the standard error schema for Pydantic/request validation errors.

    Args:
        request: The current request.
        exc: The raised ``RequestValidationError``.

    Returns:
        A JSON response matching the documented error schema, with
        ``details`` populated from the validator's error list.
    """
    logger.warning(
        "Validation error: id={} path={} errors={}",
        getattr(request.state, "request_id", None),
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_payload(request, "Request validation failed.", exc.errors()),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the standard error schema for any otherwise-unhandled exception.

    This is the final safety net — no exception should ever propagate to
    the client as a bare 500 with no structured body.

    Args:
        request: The current request.
        exc: The unhandled exception.

    Returns:
        A JSON 500 response matching the documented error schema.
    """
    logger.exception(
        "Unhandled exception: id={} path={} error={}",
        getattr(request.state, "request_id", None),
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload(request, "Internal server error.", str(exc)),
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="The user's legal query or incident description.")


class RetrieveRequest(BaseModel):
    """Request body for ``POST /retrieve``."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="The search query.")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Number of candidates to retrieve.")


class ClassifyRequest(BaseModel):
    """Request body for ``POST /classify``."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="The incident description to extract facts from.")


class RerankChunk(BaseModel):
    """A single chunk to be reranked, as expected by ``CrossEncoderReranker.rerank``."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    retrieval_text: str = Field(..., description="Hierarchy-aware text to score against the query.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Full chunk payload.")
    score: float | None = Field(default=None, description="Original retriever score, if available.")
    document: str | None = Field(default=None)
    section: str | None = Field(default=None)
    article: str | None = Field(default=None)


class RerankRequest(BaseModel):
    """Request body for ``POST /rerank``."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="The query to rerank chunks against.")
    chunks: list[RerankChunk] = Field(..., description="Candidate chunks from a prior retrieval step.")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Number of results to return.")


class ValidateRequest(BaseModel):
    """Request body for ``POST /validate``.

    Mirrors the input contract of ``CitationValidator.validate`` exactly:
    ``retrieval_results``, ``selected_quotes``, ``applicable_sections``.
    """

    model_config = ConfigDict(extra="ignore")

    retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    selected_quotes: list[Any] = Field(default_factory=list)
    applicable_sections: list[dict[str, Any]] = Field(default_factory=list)


class SynthesizeRequest(BaseModel):
    """Request body for ``POST /synthesize``.

    Accepts the full pipeline-output-shaped dict expected by
    ``ResponseSynthesizer.generate`` (query, classification/facts,
    applicable_sections, remedy, validation, etc.). Extra fields beyond
    those the synthesizer reads are ignored rather than rejected.
    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1)
    classification: dict[str, Any] = Field(default_factory=dict)
    retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    selected_quotes: list[Any] = Field(default_factory=list)
    applicable_sections: list[dict[str, Any]] = Field(default_factory=list)
    remedy: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dependency-style helpers
# ---------------------------------------------------------------------------


def _get_components(request: Request) -> _Components:
    """Read the shared component container from app state.

    Args:
        request: The current request.

    Returns:
        The ``_Components`` instance populated at startup.
    """
    return request.app.state.components


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    """Return basic service identification.

    Returns:
        Dict with ``service``, ``version``, and ``status``.
    """
    return {
        "service": "Indian Legal RAG API",
        "version": "1.0",
        "status": "running",
    }


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Report health of the pipeline, Qdrant, and Groq.

    Pipeline health reflects whether ``LegalRAGPipeline.initialize()``
    succeeded at startup. Qdrant health is checked live via the
    retriever's own connection (never a new client). Groq health is a
    configuration check only (API key presence) — no live API call is
    made on every health check to avoid burning quota.

    Args:
        request: The current request.

    Returns:
        Dict with ``status``, ``pipeline``, ``qdrant``, and ``groq``.
    """
    components = _get_components(request)

    pipeline_ok = bool(components.pipeline_ready)

    qdrant_ok = False
    if components.pipeline is not None:
        retriever = getattr(components.pipeline, "_retriever", None)
        if retriever is not None:
            try:
                qdrant_ok = bool(retriever.connect())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant health check failed: {}", exc)
                qdrant_ok = False

    groq_ok = bool(config.GROQ_API_KEY)

    overall_status = "healthy" if (pipeline_ok and qdrant_ok and groq_ok) else "degraded"

    return {
        "status": overall_status,
        "pipeline": pipeline_ok,
        "qdrant": qdrant_ok,
        "groq": groq_ok,
    }


@app.post("/chat")
async def chat(request: Request, body: ChatRequest) -> dict[str, Any]:
    """Run the full multi-agent workflow for one user query.

    Delegates entirely to ``LegalRAGOrchestrator.run()`` — the
    orchestrator already coordinates classification, retrieval,
    quote selection, section mapping, remedy advice, citation
    validation, and synthesis.

    Args:
        request: The current request (used for the request id).
        body: The validated request body.

    Returns:
        Dict with ``success``, ``request_id``, ``latency_ms``, and
        ``response``.

    Raises:
        HTTPException: 503 if the orchestrator is unavailable.
    """
    components = _get_components(request)
    if components.orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator is not available.",
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()

    final_state = await _run_in_thread(components.orchestrator.run, body.query)

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "success": True,
        "request_id": request_id,
        "latency_ms": round(latency_ms, 2),
        "response": final_state.get("response", {}),
    }


@app.post("/retrieve")
async def retrieve(request: Request, body: RetrieveRequest) -> dict[str, Any]:
    """Run hybrid retrieval + reranking via the shared pipeline.

    Calls ``pipeline.search(query, retriever_top_k=top_k)`` — never the
    retriever or reranker directly.

    Args:
        request: The current request.
        body: The validated request body.

    Returns:
        Dict with ``results``, ``confidence``, ``filters``,
        ``retrieval_time``, and ``rerank_time`` only (no raw
        orchestration metadata).

    Raises:
        HTTPException: 503 if the pipeline failed to initialise.
    """
    components = _get_components(request)
    if components.pipeline is None or not components.pipeline_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is not available.",
        )

    output = await _run_in_thread(
        components.pipeline.search,
        body.query,
        retriever_top_k=body.top_k,
    )
    output_dict = output.to_dict()

    return {
        "results": output_dict["results"],
        "confidence": output_dict["confidence"],
        "filters": output_dict["filters"],
        "retrieval_time": output_dict["retrieval_time"],
        "rerank_time": output_dict["rerank_time"],
    }


@app.post("/classify")
async def classify(request: Request, body: ClassifyRequest) -> dict[str, Any]:
    """Run fact extraction via ``FactExtractionAgent``.

    Args:
        request: The current request.
        body: The validated request body.

    Returns:
        Dict with ``incident`` (the detected incident types),
        ``entities``, ``search_queries``, and ``actions``.

    Raises:
        HTTPException: 503 if the classifier is unavailable.
    """
    components = _get_components(request)
    if components.classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier is not available.",
        )

    facts = await _run_in_thread(components.classifier.extract, body.query)

    return {
        "incident": facts.get("incident_type", []),
        "entities": facts.get("entities", []),
        "search_queries": facts.get("search_queries", []),
        "actions": facts.get("actions", []),
    }


@app.post("/rerank")
async def rerank(request: Request, body: RerankRequest) -> dict[str, Any]:
    """Rerank candidate chunks via ``CrossEncoderReranker``.

    Args:
        request: The current request.
        body: The validated request body.

    Returns:
        Dict with key ``results`` containing the reranked chunk list.

    Raises:
        HTTPException: 503 if the reranker is unavailable.
    """
    components = _get_components(request)
    if components.reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker is not available.",
        )

    candidates = [chunk.model_dump() for chunk in body.chunks]

    results = await _run_in_thread(
        components.reranker.rerank,
        body.query,
        candidates,
        top_k=body.top_k,
    )

    return {"results": results}


@app.post("/validate")
async def validate(request: Request, body: ValidateRequest) -> dict[str, Any]:
    """Validate citations via ``CitationValidator``.

    Args:
        request: The current request.
        body: The validated request body.

    Returns:
        The validator's PASS/FAIL report dict, unmodified.

    Raises:
        HTTPException: 503 if the validator is unavailable.
    """
    components = _get_components(request)
    if components.validator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validator is not available.",
        )

    pipeline_output = body.model_dump()
    report = await _run_in_thread(components.validator.validate, pipeline_output)
    return report


@app.post("/synthesize")
async def synthesize(request: Request, body: SynthesizeRequest) -> dict[str, Any]:
    """Synthesize the final legal response via ``ResponseSynthesizer``.

    The synthesizer itself enforces that the LLM is only called when
    ``validation.validation_status == "PASS"`` — this endpoint passes
    the request through unmodified and returns exactly what
    ``ResponseSynthesizer.generate`` produces, including its own
    ``{"status": "FAILED", ...}`` shape on validation failure.

    Args:
        request: The current request.
        body: The validated request body.

    Returns:
        The synthesizer's structured response dict, unmodified.

    Raises:
        HTTPException: 503 if the synthesizer is unavailable.
    """
    components = _get_components(request)
    if components.synthesizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synthesizer is not available.",
        )

    pipeline_output = body.model_dump()
    result = await _run_in_thread(components.synthesizer.generate, pipeline_output)
    return result


@app.get("/statistics")
async def statistics() -> dict[str, Any]:
    """Return runtime API metrics accumulated since process startup.

    Returns:
        Dict with uptime, total/successful/failed request counts,
        average latency, and per-endpoint request counts.
    """
    return _stats.to_dict()


# ---------------------------------------------------------------------------
# Async helper — never block the event loop with heavy CPU/IO work
# ---------------------------------------------------------------------------


async def _run_in_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous, potentially CPU/IO-heavy function off the event loop.

    Every call into the existing pipeline/orchestrator/agent modules is
    synchronous code (model inference, Qdrant requests, Groq requests).
    Routing each through ``asyncio.to_thread`` keeps the event loop free
    to serve other requests concurrently.

    Args:
        func: The synchronous callable to run.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.

    Returns:
        Whatever ``func`` returns.
    """
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
