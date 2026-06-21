"""
config.py
---------
Centralised configuration for the Indian Legal RAG pipeline.

Every path, model name, batch size, and tunable constant used by
``src/embedder.py`` (and future modules) is defined here so nothing is
hardcoded inside the pipeline logic itself.

Python 3.11+  |  PEP 8
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

RAW_DIR: Final[Path] = Path("data/raw")
PROCESSED_DIR: Final[Path] = Path("data/processed")
CHUNKED_DIR: Final[Path] = Path("data/chunked")
EMBEDDINGS_DIR: Final[Path] = Path("data/embeddings")
LOG_DIR: Final[Path] = Path("logs")


# ---------------------------------------------------------------------------
# Document registry — shared across chunking and embedding stages
# ---------------------------------------------------------------------------

# label -> chunk filename stem (without .json)
DOCUMENT_CHUNK_STEMS: Final[dict[str, str]] = {
    "BNS": "BNS_chunks",
    "BNSS": "BNSS_chunks",
    "BSA": "BSA_chunks",
    "Constitution": "Constitution_chunks",
}

# label -> embeddings output filename stem (without .json)
DOCUMENT_EMBEDDING_STEMS: Final[dict[str, str]] = {
    "BNS": "BNS_embeddings",
    "BNSS": "BNSS_embeddings",
    "BSA": "BSA_embeddings",
    "Constitution": "Constitution_embeddings",
}

# ---------------------------------------------------------------------------
# Embedding model configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE: Final[int] = 32
NORMALIZE_EMBEDDINGS: Final[bool] = True

# Preference order when auto-detecting compute device.
DEVICE_PREFERENCE: Final[tuple[str, ...]] = ("cuda", "mps", "cpu")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_ROTATION: Final[str] = "10 MB"
LOG_RETENTION: Final[str] = "14 days"


QDRANT_URL: Final[str] = os.getenv("QDRANT_URL", "")

QDRANT_API_KEY: Final[str] = os.getenv("QDRANT_API_KEY", "")

COLLECTION_NAME: Final[str] = os.getenv(
    "COLLECTION_NAME",
    "legal_documents",
)

QDRANT_TIMEOUT: Final[float] = float(
    os.getenv("QDRANT_TIMEOUT", "30")
)

# Qdrant Cloud should usually be reached directly. Some local/dev
# environments inherit dead HTTP(S)_PROXY values, which makes httpx fail
# before it ever reaches Qdrant.
QDRANT_TRUST_ENV: Final[bool] = os.getenv("QDRANT_TRUST_ENV", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ---------------------------------------------------------------------------
# Retrieval configuration (src/retriever.py)
# ---------------------------------------------------------------------------
 
# Final number of results returned by HybridRetriever.retrieve() when its
# top_k argument is omitted.
DEFAULT_TOP_K: Final[int] = int(os.environ.get("DEFAULT_TOP_K", "10"))
 
# Candidate pool sizes pulled from each search backend before fusion.
# Larger values give fusion more to work with at the cost of latency.
DENSE_TOP_K: Final[int] = int(os.environ.get("DENSE_TOP_K", "20"))
BM25_TOP_K: Final[int] = int(os.environ.get("BM25_TOP_K", "20"))
 
# The "k" constant in the Reciprocal Rank Fusion formula:
#   score = 1/(k + dense_rank) + 1/(k + bm25_rank)
# Higher k flattens the curve, reducing the influence of exact rank
# position; 60 is the commonly cited default in RRF literature.
RRF_K: Final[int] = int(os.environ.get("RRF_K", "60"))
 
# "rrf" (default/recommended) combines dense + BM25 by rank position alone,
# requiring no score normalization assumptions. "weighted" instead combines
# independently min-max-normalized dense and BM25 scores — only consulted
# when FUSION_STRATEGY == "weighted".
FUSION_STRATEGY: Final[str] = os.environ.get("FUSION_STRATEGY", "rrf").lower()
 
# Only used when FUSION_STRATEGY == "weighted". Should sum to 1.0, though
# this is not enforced — retriever.py treats them as independent weights.
DENSE_WEIGHT: Final[float] = float(os.environ.get("DENSE_WEIGHT", "0.5"))
BM25_WEIGHT: Final[float] = float(os.environ.get("BM25_WEIGHT", "0.5"))

# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    """Return environment variable or default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Return environment variable as int or default."""
    return int(os.environ.get(key, str(default)))

 
# ===========================================================================
# Reranker
# ===========================================================================

#: CrossEncoder model for reranking retrieved candidates.
RERANKER_MODEL_NAME: Final[str] = os.environ.get(
    "RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"
)

#: Number of top candidates returned after reranking.
RERANKER_TOP_K: Final[int] = int(os.environ.get("RERANKER_TOP_K", "5"))

#: CrossEncoder inference batch size.
RERANKER_BATCH_SIZE: Final[int] = int(os.environ.get("RERANKER_BATCH_SIZE", "32"))

#: Compute device: ``"auto"`` (CUDA → MPS → CPU), ``"cuda"``, ``"mps"``,
#: or ``"cpu"``.
RERANKER_DEVICE: Final[str] = os.environ.get("RERANKER_DEVICE", "auto")

#: Maximum token length passed to the CrossEncoder.
RERANKER_MAX_LENGTH: Final[int] = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))

 
# ===========================================================================
# Pipeline   (used by pipeline.py → LegalRAGPipeline)
# ===========================================================================
 
# Candidates fetched from the retriever and passed into the reranker.
# Should be >= RERANKER_TOP_K so the reranker has a meaningful pool to
# score.  Typical production value: 20.
AGENT_RETRIEVER_TOP_K: Final[int] = int(os.environ.get("AGENT_RETRIEVER_TOP_K", "20"))
 
# Final results returned by pipeline.search() to the agents.
# This is the number of chunks placed in the LLM context window.
AGENT_RERANKER_TOP_K: Final[int] = int(os.environ.get("AGENT_RERANKER_TOP_K", "5"))
 
# Maximum reasoning iterations the LangGraph agent may perform per query.
AGENT_MAX_ITERATIONS: Final[int] = int(os.environ.get("AGENT_MAX_ITERATIONS", "5"))
 
# Minimum confidence threshold below which the pipeline warns that the
# retrieved context may be insufficient.  Not enforced as a hard cutoff
# here — downstream agents decide whether to proceed.
CONFIDENCE_THRESHOLD: Final[float] = float(
    os.environ.get("CONFIDENCE_THRESHOLD", "0.3")
)
 
# Separator inserted between chunks when assembling the LLM context string.
CONTEXT_SEPARATOR: Final[str] = "\n\n---\n\n"

 
# ---------------------------------------------------------------------------
# Orchestrator configuration (src/orchestrator.py)
# ---------------------------------------------------------------------------
 
# Below this confidence, the orchestrator flags the response as low
# confidence (e.g. for the validator/synthesizer agents to add a caveat)
# rather than rejecting it outright -- this module never decides legal
# correctness, only routes and annotates.
CONFIDENCE_THRESHOLD: Final[float] = float(
    os.environ.get("CONFIDENCE_THRESHOLD", "0.5")
)
 
# Per-node timeout (seconds) for agent calls -- prevents one slow/stuck
# agent from hanging the entire workflow indefinitely. None disables.
AGENT_NODE_TIMEOUT_SECONDS: Final[float] = float(
    os.environ.get("AGENT_NODE_TIMEOUT_SECONDS", "30")
)

# ==========================================================
# Groq Configuration
# ==========================================================

GROQ_API_KEY: Final[str] = os.getenv("GROQ_API_KEY", "")

# Backward compatibility
API_KEY: Final[str] = GROQ_API_KEY

GROQ_MODEL_NAME: Final[str] = os.getenv(
    "GROQ_MODEL_NAME",
    "llama-3.3-70b-versatile",
)

# Backward compatibility
MODEL_NAME: Final[str] = GROQ_MODEL_NAME

TEMPERATURE: Final[float] = float(
    os.getenv("TEMPERATURE", "0.2")
)

MAX_TOKENS: Final[int] = int(
    os.getenv("MAX_TOKENS", "1200")
)

TIMEOUT: Final[float] = float(
    os.getenv("TIMEOUT", "30")
)


# ---------------------------------------------------------------------------
# API layer configuration (app.py)
# ---------------------------------------------------------------------------
 
LOG_LEVEL: Final[str] = os.environ.get("LOG_LEVEL", "DEBUG")
 
CORS_ALLOWED_ORIGINS: Final[tuple[str, ...]] = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)
 
API_HOST: Final[str] = os.environ.get("API_HOST", "127.0.0.1")
API_PORT: Final[int] = int(os.environ.get("API_PORT", "8000"))
