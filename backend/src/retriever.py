"""
src/retriever.py
-----------------
Production-grade hybrid retriever for the Indian Legal RAG system.

Responsibilities
----------------
* Combine dense vector search (Qdrant, cosine similarity) with BM25
  keyword search (local, over ``data/chunked/``) into a single ranked
  result set.
* Detect explicit statutory references in the query (e.g. "Section 302",
  "Article 21", "BNSS Section 35", "BNS 303") and turn them into Qdrant
  metadata filters automatically, so exact citations are retrieved by
  direct lookup rather than relying solely on semantic similarity.
* Fuse the two ranked lists with Reciprocal Rank Fusion (configurable),
  deduplicate by ``chunk_hash``, normalize scores, and return the top-K
  chunks in a schema rich enough for a downstream Cross-Encoder reranker
  to consume without another Qdrant lookup.

Does NOT rerank, call an LLM, perform reasoning, generate legal advice,
or use LangChain/LlamaIndex. This module ONLY retrieves relevant legal
chunks.

Expected ``config.py`` attributes (in addition to the ones already used
by ``src/indexer.py``)
---------------------------------------------------------------------
    CHUNKED_DIR: Path           # e.g. Path("data/chunked")
    EMBEDDING_MODEL_NAME: str   # e.g. "BAAI/bge-base-en-v1.5"
    DEFAULT_TOP_K: int          # final results returned, e.g. 10
    DENSE_TOP_K: int            # candidates pulled from Qdrant, e.g. 20
    BM25_TOP_K: int             # candidates pulled from BM25, e.g. 20
    RRF_K: int                  # Reciprocal Rank Fusion constant, e.g. 60
    FUSION_STRATEGY: str        # "rrf" (default/recommended) or "weighted"

Optional (only consulted when FUSION_STRATEGY == "weighted"):
    DENSE_WEIGHT: float         # default 0.5 if absent
    BM25_WEIGHT: float          # default 0.5 if absent

Reuses from indexer.py's existing config surface: COLLECTION_NAME,
QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT, LOG_DIR, LOG_ROTATION,
LOG_RETENTION.

Point IDs vs chunk_hash
------------------------
Duplicate detection across the dense and BM25 result lists is keyed on
``chunk_hash``, never ``chunk_id`` or the Qdrant point ID -- see
``remove_duplicates``.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from loguru import logger
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Lazy/optional heavy imports
# ---------------------------------------------------------------------------
# sentence-transformers, qdrant-client, and rank_bm25 are imported lazily
# inside the methods that need them, mirroring src/indexer.py. This keeps
# the module importable (e.g. for type checking, or by reranker.py) even
# before all three dependencies are installed, and ensures a missing
# dependency degrades one retrieval path gracefully instead of crashing
# the whole module on import.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: BGE models are trained with an asymmetric query/passage instruction
#: scheme -- queries (never indexed passages) must be prefixed with this
#: instruction to get well-calibrated retrieval similarity scores. This is
#: a property of the embedding model family itself, not a deployment
#: setting, so it is kept as a code constant rather than a config value.
_BGE_QUERY_INSTRUCTION: Final[str] = "Represent this sentence for searching relevant passages: "

#: Statute codes recognized by the legal-reference detector.
_DOCUMENT_CODES: Final[tuple[str, ...]] = ("BNSS", "BNS", "BSA")

#: Maps every recognized spelling/alias of a document name to the exact
#: string stored in chunk payloads (``document`` field), as produced by
#: the chunker. The regex-detected "Constitution" must resolve to
#: "Constitution of India" -- the literal value chunker.py writes -- or
#: every filtered query against the Constitution silently returns zero
#: results, in both the BM25 post-filter and the Qdrant MatchValue filter.
_DOCUMENT_CANONICAL: Final[dict[str, str]] = {
    "BNS": "BNS",
    "BNSS": "BNSS",
    "BSA": "BSA",
    "CONSTITUTION": "Constitution of India",
}

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+")

_DOC_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(BNSS|BNS|BSA)\b\s*(?:(?:section|sec\.)\s*)?(\d{1,4}[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)
_ARTICLE_RE: Final[re.Pattern[str]] = re.compile(
    r"\barticle\s+(\d{1,4}[A-Za-z]{0,3})\b", re.IGNORECASE
)
_SECTION_ONLY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:section|sec\.)\s+(\d{1,4}[A-Za-z]{0,3})\b", re.IGNORECASE
)
_DOCUMENT_ONLY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(BNSS|BNS|BSA|Constitution)\b", re.IGNORECASE
)

#: Filter keys the local BM25 path knows how to apply post-hoc, and which
#: also double as ``RetrievedChunk`` attribute names.
_FILTERABLE_FIELDS: Final[tuple[str, ...]] = ("document", "section", "article", "chunk_type")


# ---------------------------------------------------------------------------
# Query preprocessing & legal-reference detection
# ---------------------------------------------------------------------------


def _normalize_query(query: str) -> str:
    """Normalize a query for embedding and BM25 tokenization.

    Lowercases and collapses repeated whitespace only. Deliberately never
    strips or filters characters: numeric tokens such as section/article
    numbers ("303", "21") and short statute codes ("BNS", "BNSS") must
    survive untouched for both dense and keyword retrieval to work.

    Args:
        query: The raw user query.

    Returns:
        The normalized query string.
    """
    return _WHITESPACE_RE.sub(" ", query.strip().lower()).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing/querying.

    Lowercases and splits on non-alphanumeric boundaries while keeping
    every alphanumeric run intact, preserving bare section/article
    numbers and short statute codes as standalone tokens.

    Args:
        text: Raw text to tokenize.

    Returns:
        A list of lowercase tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def _detect_legal_references(query: str) -> dict[str, str]:
    """Detect explicit statutory references and turn them into filters.

    Recognizes patterns such as "Section 302", "Article 21", "BNSS
    Section 35", and "BNS 303", converting them into ``document`` /
    ``section`` / ``article`` filter hints so exact statutory citations
    can be retrieved by direct lookup instead of relying solely on
    semantic similarity -- important in legal text, where a single digit
    of difference (Section 302 vs. 303) changes the offence entirely.

    Detected document codes/names are mapped through
    ``_DOCUMENT_CANONICAL`` so the resulting filter value exactly matches
    what is stored in chunk payloads (e.g. "Constitution" in user text ->
    "Constitution of India" in the data) -- see that constant's docstring
    for why this matters.

    Domain-specific inference: articles are a Constitution-only concept
    in this knowledge base (BNS/BNSS/BSA use "Section"), so an article
    number found with no other named document confidently implies
    ``document="Constitution of India"``.

    Ambiguity safety: if two or more *conflicting* values are found for
    the same filter key in one query (e.g. both "Section 302" and
    "Section 103"), that filter is left unset rather than guessed, since
    an AND-filter on two different values for the same field can only
    ever match zero chunks.

    Args:
        query: The raw, un-normalized user query.

    Returns:
        A dict with zero to three of the keys ``document``, ``section``,
        ``article``, populated only when confidently and unambiguously
        detected.
    """
    documents_found: set[str] = set()
    sections_found: set[str] = set()
    articles_found: set[str] = set()

    for doc_code, section_no in _DOC_SECTION_RE.findall(query):
        documents_found.add(_DOCUMENT_CANONICAL.get(doc_code.upper(), doc_code.upper()))
        sections_found.add(section_no.upper())

    for article_no in _ARTICLE_RE.findall(query):
        articles_found.add(article_no.upper())

    for section_no in _SECTION_ONLY_RE.findall(query):
        sections_found.add(section_no.upper())

    for doc_code in _DOCUMENT_ONLY_RE.findall(query):
        documents_found.add(_DOCUMENT_CANONICAL.get(doc_code.upper(), doc_code.upper()))

    if articles_found and not documents_found:
        documents_found.add(_DOCUMENT_CANONICAL["CONSTITUTION"])

    filters: dict[str, str] = {}

    if len(documents_found) == 1:
        filters["document"] = next(iter(documents_found))
    elif len(documents_found) > 1:
        logger.debug(
            "Ambiguous document references in query ({}); skipping auto document filter.",
            sorted(documents_found),
        )

    if len(sections_found) == 1:
        filters["section"] = next(iter(sections_found))
    elif len(sections_found) > 1:
        logger.debug(
            "Ambiguous section references in query ({}); skipping auto section filter.",
            sorted(sections_found),
        )

    if len(articles_found) == 1:
        filters["article"] = next(iter(articles_found))
    elif len(articles_found) > 1:
        logger.debug(
            "Ambiguous article references in query ({}); skipping auto article filter.",
            sorted(articles_found),
        )

    return filters


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """Unified internal representation of a single candidate chunk.

    Populated from either Qdrant (dense search) or the local BM25 corpus
    (keyword search) -- and potentially both, after deduplication.
    Carries everything a downstream Cross-Encoder reranker needs without
    requiring another Qdrant lookup.

    Attributes:
        chunk_id: Original chunk identifier from the chunking stage.
        chunk_hash: Content-identity hash; the basis for deduplication.
        document: Short document code, e.g. ``"BNS"``, ``"Constitution"``.
        section: Section identifier, if applicable.
        article: Article identifier, if applicable.
        chunk_type: Chunk classification, e.g. ``"clause"``.
        hierarchy: Structural hierarchy metadata as stored by chunking.
        retrieval_text: The hierarchy-aware text used for both embedding
            and BM25 (never the bare ``text`` field).
        payload: The full payload, exactly as stored in Qdrant (or, for
            BM25-only hits, reconstructed to the same shape).
        dense_rank: 1-indexed rank from dense search, or ``None``.
        dense_score: Raw cosine similarity from dense search.
        bm25_rank: 1-indexed rank from BM25 search, or ``None``.
        bm25_score: Raw BM25 score.
        fused_score: Final fused score, populated by the fusion step.
    """

    chunk_id: str
    chunk_hash: str
    document: str
    section: str | None
    article: str | None
    chunk_type: str | None
    hierarchy: Any
    retrieval_text: str
    payload: dict[str, Any]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0

    def to_output(self) -> dict[str, Any]:
        """Serialize into the canonical retriever output schema.

        Returns:
            A dict with keys ``score``, ``document``, ``section``,
            ``article``, ``chunk_type``, ``chunk_id``, ``chunk_hash``,
            ``retrieval_text``, and ``payload``.
        """
        return {
            "score": round(float(self.fused_score), 6),
            "document": self.document,
            "section": self.section,
            "article": self.article,
            "chunk_type": self.chunk_type,
            "chunk_id": self.chunk_id,
            "chunk_hash": self.chunk_hash,
            "retrieval_text": self.retrieval_text,
            "payload": self.payload,
        }


@dataclass
class _RetrievalStats:
    """Mutable counter bag describing a single ``retrieve()`` call."""

    query: str = ""
    auto_filters: dict[str, str] = field(default_factory=dict)
    applied_filters: dict[str, str] = field(default_factory=dict)
    fusion_strategy: str = ""
    dense_candidates: int = 0
    bm25_candidates: int = 0
    deduplicated_candidates: int = 0
    returned_chunks: int = 0
    embedding_time: float = 0.0
    dense_time: float = 0.0
    bm25_time: float = 0.0
    merge_time: float = 0.0
    total_time: float = 0.0


def _matches_filters(chunk: RetrievedChunk, filters: dict[str, str]) -> bool:
    """Check whether a chunk satisfies every key/value pair in ``filters``.

    Args:
        chunk: The candidate chunk.
        filters: Mapping of ``RetrievedChunk`` attribute name to required
            value (only keys in ``_FILTERABLE_FIELDS`` are meaningful).

    Returns:
        ``True`` if every filter key is present on the chunk and equal
        (as strings) to the required value.
    """
    for key, value in filters.items():
        chunk_value = getattr(chunk, key, None)
        if chunk_value is None or str(chunk_value) != str(value):
            return False
    return True


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Hybrid dense + BM25 retriever for the Indian Legal RAG system.

    Usage
    -----
    >>> retriever = HybridRetriever()
    >>> results = retriever.retrieve("What does Section 302 BNS punish?")

    The embedding model, Qdrant client, and BM25 index are each created
    at most once per ``HybridRetriever`` instance and reused for every
    subsequent call to :meth:`retrieve`.
    """

    def __init__(
        self,
        chunked_dir: Path = config.CHUNKED_DIR,
        collection_name: str = config.COLLECTION_NAME,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
        url: str = config.QDRANT_URL,
        api_key: str = config.QDRANT_API_KEY,
        timeout: float = config.QDRANT_TIMEOUT,
        default_top_k: int = config.DEFAULT_TOP_K,
        dense_top_k: int = config.DENSE_TOP_K,
        bm25_top_k: int = config.BM25_TOP_K,
        rrf_k: int = config.RRF_K,
        fusion_strategy: str = config.FUSION_STRATEGY,
        dense_weight: float = getattr(config, "DENSE_WEIGHT", 0.5),
        bm25_weight: float = getattr(config, "BM25_WEIGHT", 0.5),
    ) -> None:
        """Initialise paths, configuration, and logging.

        Neither the embedding model, the Qdrant client, nor the BM25
        index are created here -- they are built lazily on first use (or
        explicitly via :meth:`load_embedding_model` /
        :meth:`load_bm25_index` / :meth:`connect`), so constructing a
        ``HybridRetriever`` never fails due to network, disk, or model
        download issues.

        Args:
            chunked_dir: Directory containing chunk JSON files used to
                build the BM25 index.
            collection_name: Qdrant collection to search.
            embedding_model_name: SentenceTransformers model name/path.
            url: Qdrant Cloud cluster URL.
            api_key: Qdrant Cloud API key.
            timeout: Request timeout in seconds for the Qdrant client.
            default_top_k: Final number of results returned by
                :meth:`retrieve` when its ``top_k`` argument is omitted.
            dense_top_k: Candidates pulled from Qdrant before fusion.
            bm25_top_k: Candidates pulled from BM25 before fusion.
            rrf_k: The ``k`` constant in the Reciprocal Rank Fusion
                formula.
            fusion_strategy: ``"rrf"`` (default/recommended) or
                ``"weighted"``.
            dense_weight: Only used when ``fusion_strategy=="weighted"``.
            bm25_weight: Only used when ``fusion_strategy=="weighted"``.
        """
        self._chunked_dir = chunked_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._default_top_k = default_top_k
        self._dense_top_k = dense_top_k
        self._bm25_top_k = bm25_top_k
        self._rrf_k = rrf_k
        self._fusion_strategy = fusion_strategy
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

        self._client: Any = None  # qdrant_client.QdrantClient, set in connect()
        self._model: Any = None  # SentenceTransformer, set in load_embedding_model()
        self._device: str = "cpu"
        self._bm25: Any = None  # rank_bm25.BM25Okapi, set in build_bm25()
        self._bm25_corpus: list[RetrievedChunk] = []
        self._bm25_loaded: bool = False
        self._last_stats: _RetrievalStats | None = None

        # One-time setup costs, measured once and never re-measured on
        # subsequent queries -- surfaced separately in stats/logging so
        # "Embedding Time" in a query's stats reflects ONLY that query's
        # encode() call, never the (much larger, one-time) model download
        # and weight-loading cost. A fresh process always pays these once;
        # a long-lived process (server, notebook, REPL) pays them exactly
        # once total, no matter how many queries follow.
        self.model_load_time: float = 0.0
        self.bm25_build_time: float = 0.0
        self.qdrant_connect_time: float = 0.0

        self._configure_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_logging() -> None:
        """Configure Loguru with a structured, colourised console sink."""
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
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
            config.LOG_DIR / "retriever.log",
            level="DEBUG",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Qdrant connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Create (or reuse) the Qdrant client connection.

        Returns:
            ``True`` if a client is available and reachable, ``False``
            otherwise.
        """
        if self._client is not None:
            return True

        if not self._url:
            logger.error(
                "QDRANT_URL is not configured. Set it via the QDRANT_URL "
                "environment variable or config.py."
            )
            return False

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            logger.error(
                "qdrant-client is not installed: {}. "
                "Install it with `pip install qdrant-client`.",
                exc,
            )
            return False

        try:
            self._client = QdrantClient(
                url=self._url,
                port=None,
                api_key=self._api_key or None,
                timeout=self._timeout,
                check_compatibility=False,
                prefer_grpc=False,
                trust_env=config.QDRANT_TRUST_ENV,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create Qdrant client: {}", exc)
            self._client = None
            return False

        if not self.health_check():
            logger.error("Qdrant connection established but health check failed.")
            self._client = None
            return False

        logger.info("Connected to Qdrant at '{}'.", self._url)
        return True

    def health_check(self) -> bool:
        """Verify the Qdrant cluster is reachable and responding.

        Returns:
            ``True`` if the cluster responds successfully, ``False``
            otherwise.
        """
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Qdrant health check failed: {}", exc)
            return False

    def _require_client(self) -> bool:
        """Ensure a Qdrant client is available, attempting to connect if not.

        Returns:
            ``True`` if a client is ready to use, ``False`` otherwise.
        """
        if self._client is not None:
            return True
        return self.connect()

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    def detect_device(self) -> str:
        """Detect the best available compute device for the embedding model.

        Preference order: CUDA -> MPS (Apple Silicon) -> CPU. Mirrors the
        same detection logic used in ``src/embedder.py`` so encoding here
        and embedding during indexing run on consistent hardware. Any
        import or runtime failure while probing a device is treated as
        "unavailable" and falls back to the next option.

        This was previously never called -- ``SentenceTransformer(...)``
        was constructed with no ``device`` argument at all, which silently
        pins inference to CPU even when a GPU is present and is a primary
        cause of slow per-query embedding time on GPU-equipped machines.

        Returns:
            One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not importable -- falling back to CPU.")
            return "cpu"

        try:
            if torch.cuda.is_available():
                logger.info("CUDA device detected: {}", torch.cuda.get_device_name(0))
                return "cuda"
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA probe failed: {} -- trying MPS.", exc)

        try:
            if torch.backends.mps.is_available():
                logger.info("Apple MPS device detected.")
                return "mps"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MPS probe failed: {} -- falling back to CPU.", exc)

        logger.info("No GPU/MPS available -- using CPU.")
        return "cpu"

    def load_embedding_model(self) -> bool:
        """Load the SentenceTransformer embedding model exactly once.

        Bottleneck this addresses: model construction (process import of
        torch/transformers plus, on a cold cache, downloading and
        deserializing model weights) is a one-time cost of several to
        tens of seconds. The previous implementation already guarded
        against reloading within a single process (the ``self._model is
        not None`` check below is unchanged), so that part was not the
        bug -- the bug was that this one-time cost was being measured
        together with per-query embedding time in the caller, making
        every query LOOK equally slow in the stats even though only the
        first one actually paid the load cost. ``self.model_load_time``
        now captures it once, here, and ``embed_query`` never touches it
        again.

        Also now selects a compute device (see :meth:`detect_device`)
        instead of leaving it unset, which previously forced CPU-only
        inference unconditionally.

        Returns:
            ``True`` if the model is loaded and ready, ``False`` on
            failure (missing dependency or load error).
        """
        if self._model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error(
                "sentence-transformers is not installed: {}. "
                "Install it with `pip install sentence-transformers`.",
                exc,
            )
            return False

        self._device = self.detect_device()

        try:
            logger.info(
                "Loading embedding model '{}' on device '{}'...",
                self._embedding_model_name,
                self._device,
            )
            load_start = time.perf_counter()
            self._model = SentenceTransformer(self._embedding_model_name, device=self._device)
            self.model_load_time = time.perf_counter() - load_start
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load embedding model '{}': {}", self._embedding_model_name, exc)
            self._model = None
            return False

        logger.info(
            "Embedding model '{}' loaded on '{}' in {:.3f}s.",
            self._embedding_model_name,
            self._device,
            self.model_load_time,
        )
        return True

    def embed_query(self, query: str) -> list[float] | None:
        """Embed a preprocessed query string using the cached model.

        Applies the BGE retrieval instruction prefix (see
        ``_BGE_QUERY_INSTRUCTION``) before encoding, since BAAI/bge
        models are trained with an asymmetric query/passage scheme.

        Args:
            query: The (already preprocessed) query text.

        Returns:
            A dense embedding vector as a list of floats, or ``None`` on
            failure.
        """
        if not self.load_embedding_model():
            return None

        try:
            instructed_query = f"{_BGE_QUERY_INSTRUCTION}{query}"
            vector = self._model.encode(
                instructed_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to embed query: {}", exc)
            return None

        try:
            # .tolist() is a single native numpy call -- faster and more
            # direct than a Python-level `[float(v) for v in vector]` loop,
            # which re-boxes every element through Python's float()
            # constructor one at a time. Negligible in absolute terms for a
            # ~768-dim vector (sub-millisecond either way -- this is NOT
            # where the 20+ seconds was going), but it is the correct,
            # idiomatic conversion now that convert_to_numpy=True guarantees
            # `vector` is always a numpy array, never a torch.Tensor.
            return vector.tolist()
        except AttributeError as exc:
            logger.error("Embedding model returned an unexpected vector type: {}", exc)
            return None

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _chunk_record_from_raw(self, record: dict[str, Any], source_name: str) -> RetrievedChunk | None:
        """Convert one raw chunk-file dict into a ``RetrievedChunk``.

        Defensive against missing/renamed fields, since the chunked-file
        schema is produced by an earlier pipeline stage this module does
        not own. If the record does not already carry a ``payload`` dict
        (mirroring exactly what indexing stores in Qdrant), one is
        reconstructed from the available fields so BM25-only hits are
        indistinguishable downstream from dense hits.

        Args:
            record: A single raw chunk dict loaded from a
                ``data/chunked/`` file.
            source_name: The originating filename, used only for
                logging.

        Returns:
            A populated ``RetrievedChunk`` with no rank/score set yet, or
            ``None`` if the record is missing a required field.
        """
        chunk_id = record.get("chunk_id") or record.get("id")
        chunk_hash = record.get("chunk_hash")
        retrieval_text = record.get("retrieval_text")

        if not chunk_id or not chunk_hash or not retrieval_text:
            logger.debug(
                "Skipping chunk record from {} missing a required field: {}",
                source_name, sorted(record.keys()),
            )
            return None

        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "document": record.get("document", ""),
                "section": record.get("section"),
                "article": record.get("article"),
                "chunk_type": record.get("chunk_type"),
                "hierarchy": record.get("hierarchy"),
                "keywords": record.get("keywords", []),
                "retrieval_text": retrieval_text,
            }

        return RetrievedChunk(
            chunk_id=str(chunk_id),
            chunk_hash=str(chunk_hash),
            document=str(record.get("document") or payload.get("document") or ""),
            section=record.get("section", payload.get("section")),
            article=record.get("article", payload.get("article")),
            chunk_type=record.get("chunk_type", payload.get("chunk_type")),
            hierarchy=record.get("hierarchy", payload.get("hierarchy")),
            retrieval_text=str(retrieval_text),
            payload=payload,
        )

    def build_bm25(self) -> bool:
        """Build the BM25 index from chunk JSON files in ``chunked_dir``.

        Reads every ``*_chunks.json`` file under the configured
        chunked-documents directory -- never sidecar files such as
        ``*_metadata.json`` written by the chunking stage, which contain
        a single summary dict rather than a list of chunk records --
        extracts ``retrieval_text`` -- never ``text``, since the
        retrieval text carries the legal hierarchy prefix -- plus
        surrounding chunk metadata, tokenizes the corpus, and constructs
        a ``BM25Okapi`` index in memory.

        Returns:
            ``True`` if the build process completed (even with zero
            chunks found -- check ``len(self._bm25_corpus)`` to
            distinguish), ``False`` if the chunked-documents directory is
            missing or ``rank_bm25`` is not installed.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            logger.error(
                "rank_bm25 is not installed: {}. Install it with `pip install rank-bm25`.",
                exc,
            )
            return False

        if not self._chunked_dir.exists():
            logger.error("Chunked documents directory does not exist: {}", self._chunked_dir)
            return False

        chunk_files = sorted(self._chunked_dir.glob("*_chunks.json"))
        if not chunk_files:
            logger.warning(
                "No '*_chunks.json' files found in {} (only chunk files are "
                "loaded -- '*_metadata.json' sidecars are intentionally "
                "skipped).",
                self._chunked_dir,
            )

        corpus: list[RetrievedChunk] = []
        tokenized: list[list[str]] = []

        for file_path in tqdm(chunk_files, desc="Loading chunk files", unit="file", leave=False):
            try:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                logger.error("Failed to read chunk file {}: {}", file_path, exc)
                continue
            if not isinstance(raw, list):
                logger.warning("Expected a JSON array in {}, got {}.", file_path, type(raw).__name__)
                continue

            for record in raw:
                if not isinstance(record, dict):
                    continue
                chunk = self._chunk_record_from_raw(record, file_path.name)
                if chunk is None:
                    continue
                corpus.append(chunk)
                tokenized.append(_tokenize(chunk.retrieval_text))

        if not corpus:
            logger.warning("BM25 index built with zero chunks -- BM25 search will return nothing.")
            self._bm25 = None
            self._bm25_corpus = []
            return True

        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = corpus
        logger.info("BM25 index built from {} chunk(s) across {} file(s).", len(corpus), len(chunk_files))
        return True

    def load_bm25_index(self) -> bool:
        """Ensure the BM25 index is built and cached, building it if needed.

        Guarantees the BM25 index is built at most once per process,
        regardless of how many queries are subsequently served.

        Returns:
            ``True`` if a BM25 index (possibly empty) is ready to use,
            ``False`` if building it failed outright.
        """
        if self._bm25_loaded:
            return True
        built = self.build_bm25()
        self._bm25_loaded = built
        return built

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    def _build_filter(self, filters: dict[str, str]) -> Any | None:
        """Build a Qdrant ``Filter`` (AND of equality conditions).

        Args:
            filters: Mapping of payload field name to required value.

        Returns:
            A ``qdrant_client.models.Filter`` instance, or ``None`` if
            ``filters`` is empty or ``qdrant-client`` is unavailable.
        """
        if not filters:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as exc:
            logger.error("qdrant-client models unavailable: {}", exc)
            return None

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()
        ]
        return Filter(must=conditions)

    def dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        qdrant_filter: Any | None = None,
    ) -> list[RetrievedChunk]:
        """Run a dense cosine-similarity search against Qdrant.

        Uses ``query_points`` (the ``search`` method was removed in
        recent qdrant-client/Qdrant Cloud versions).

        Args:
            query_vector: The embedded query vector.
            top_k: Maximum number of points to retrieve.
            qdrant_filter: An optional pre-built
                ``qdrant_client.models.Filter`` applied server-side.

        Returns:
            A rank-ordered list of ``RetrievedChunk`` (rank 1 = best),
            each with ``dense_rank``/``dense_score`` populated. Empty on
            any failure (logged) or if Qdrant is unavailable.
        """
        if not self._require_client():
            return []

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Dense search failed: {}", exc)
            return []

        results: list[RetrievedChunk] = []
        for rank, point in enumerate(getattr(response, "points", []), start=1):
            payload = point.payload or {}
            chunk_hash = payload.get("chunk_hash")
            if not chunk_hash:
                logger.warning("Dense hit '{}' missing chunk_hash in payload -- skipped.", point.id)
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    chunk_hash=str(chunk_hash),
                    document=str(payload.get("document", "")),
                    section=payload.get("section"),
                    article=payload.get("article"),
                    chunk_type=payload.get("chunk_type"),
                    hierarchy=payload.get("hierarchy"),
                    retrieval_text=str(payload.get("retrieval_text", "")),
                    payload=payload,
                    dense_rank=rank,
                    dense_score=float(point.score),
                )
            )
        return results

    def bm25_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run a BM25 keyword search against the cached local corpus.

        Args:
            query: The (already preprocessed) query text.
            top_k: Maximum number of results to return after filtering.
            filters: Optional metadata filters (``document``, ``section``,
                ``article``, ``chunk_type``). Applied locally, after
                scoring, since ``rank_bm25`` has no native filtering --
                Qdrant remains the preferred place for filters whenever
                the dense path can apply them server-side.

        Returns:
            A rank-ordered list of ``RetrievedChunk`` (rank 1 = best),
            each with ``bm25_rank``/``bm25_score`` populated. Empty if
            the index is unavailable, has no chunks, or the query has no
            recognizable tokens.
        """
        if not self.load_bm25_index() or self._bm25 is None or not self._bm25_corpus:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        try:
            raw_scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.error("BM25 scoring failed: {}", exc)
            return []

        ranked_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)

        results: list[RetrievedChunk] = []
        rank = 0
        for index in ranked_indices:
            score = float(raw_scores[index])
            # A BM25 score is exactly 0.0 if and only if none of the query
            # tokens appear in this document at all (every per-token IDF*TF
            # term is mathematically zero) -- this is the correct "no
            # overlap" signal, and such entries are skipped. A small
            # *negative* score can still occur for a document that DOES
            # share tokens with the query, as an artifact of negative IDF
            # on very common terms in a small or skewed corpus; such
            # matches are real and must not be dropped (the previous
            # `score <= 0.0: break` discarded them incorrectly). Note that
            # in a descending sort, negative scores sort AFTER zero, so
            # zero-score entries must be individually skipped (`continue`)
            # rather than used as an early-exit point (`break`) -- breaking
            # on the first zero would stop before ever reaching the
            # negative-but-relevant entries later in the ranking.
            if score == 0.0:
                continue
            chunk = self._bm25_corpus[index]
            if filters and not _matches_filters(chunk, filters):
                continue
            rank += 1
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    chunk_hash=chunk.chunk_hash,
                    document=chunk.document,
                    section=chunk.section,
                    article=chunk.article,
                    chunk_type=chunk.chunk_type,
                    hierarchy=chunk.hierarchy,
                    retrieval_text=chunk.retrieval_text,
                    payload=chunk.payload,
                    bm25_rank=rank,
                    bm25_score=score,
                )
            )
            if rank >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # Score normalization & fusion
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_scores(scores: list[float]) -> list[float]:
        """Min-max normalize a list of raw scores into ``[0, 1]``.

        Used both as a building block for the optional weighted-fusion
        strategy, and to rescale the final fused scores into an
        interpretable ``[0, 1]`` range for the public output schema (raw
        RRF scores are small numbers like ``0.03`` that are not
        meaningful in isolation).

        Args:
            scores: Raw scores in any range, including an empty,
                single-element, or all-equal list.

        Returns:
            Normalized scores in ``[0, 1]``, same length and order as the
            input. If every input score is equal, every output is
            ``1.0`` rather than dividing by zero.
        """
        if not scores:
            return []
        array = np.asarray(scores, dtype=float)
        minimum = float(array.min())
        maximum = float(array.max())
        spread = maximum - minimum
        if spread <= 1e-12:
            return [1.0 for _ in scores]
        return [(float(value) - minimum) / spread for value in array]

    def reciprocal_rank_fusion(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score deduplicated candidates with Reciprocal Rank Fusion.

        For each candidate, ``fused_score = 1/(k + dense_rank) +
        1/(k + bm25_rank)``, where a candidate missing from one list
        contributes ``0`` for that list's term. This rewards chunks that
        rank highly in *both* searches far more than a single very high
        rank in only one -- behaviour a simple average of cosine and
        BM25 scores cannot provide, since those two scores live on
        incomparable scales.

        Args:
            candidates: Deduplicated candidates from
                :meth:`remove_duplicates`.

        Returns:
            The same list, mutated in place with ``fused_score``
            populated on every candidate, and also returned for
            convenience.
        """
        k = self._rrf_k
        for candidate in candidates:
            score = 0.0
            if candidate.dense_rank is not None:
                score += 1.0 / (k + candidate.dense_rank)
            if candidate.bm25_rank is not None:
                score += 1.0 / (k + candidate.bm25_rank)
            candidate.fused_score = score
        return candidates

    def _weighted_fusion(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Alternative fusion: weighted sum of independently normalized scores.

        Selected when ``config.FUSION_STRATEGY == "weighted"`` instead of
        the default ``"rrf"``. Normalizes dense cosine scores and BM25
        scores independently (via :meth:`normalize_scores`) before
        combining them, since they otherwise live on incomparable scales.

        Args:
            candidates: Deduplicated candidates from
                :meth:`remove_duplicates`.

        Returns:
            The same list, mutated in place with ``fused_score``
            populated, and also returned for convenience.
        """
        dense_hashes = [c.chunk_hash for c in candidates if c.dense_score is not None]
        bm25_hashes = [c.chunk_hash for c in candidates if c.bm25_score is not None]
        dense_norm = dict(zip(dense_hashes, self.normalize_scores(
            [c.dense_score for c in candidates if c.dense_score is not None]
        )))
        bm25_norm = dict(zip(bm25_hashes, self.normalize_scores(
            [c.bm25_score for c in candidates if c.bm25_score is not None]
        )))

        for candidate in candidates:
            dense_component = dense_norm.get(candidate.chunk_hash, 0.0)
            bm25_component = bm25_norm.get(candidate.chunk_hash, 0.0)
            candidate.fused_score = (
                self._dense_weight * dense_component + self._bm25_weight * bm25_component
            )
        return candidates

    # ------------------------------------------------------------------
    # Merge & deduplicate
    # ------------------------------------------------------------------

    def merge_results(
        self, dense_results: list[RetrievedChunk], bm25_results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Concatenate dense and BM25 result lists into one candidate pool.

        No deduplication happens here -- a chunk found by both searches
        appears twice at this stage, once from each source, each
        carrying only its own rank/score. :meth:`remove_duplicates`
        consolidates these.

        Args:
            dense_results: Output of :meth:`dense_search`.
            bm25_results: Output of :meth:`bm25_search`.

        Returns:
            The concatenated candidate list.
        """
        return [*dense_results, *bm25_results]

    def remove_duplicates(self, merged: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Collapse duplicate chunks (by ``chunk_hash``) into one record each.

        Identity is based on ``chunk_hash``, never ``chunk_id``: the same
        logical passage could in principle be re-chunked under a
        different ``chunk_id`` while representing identical text, and
        ``chunk_hash`` is the field that captures content identity
        end-to-end through ingestion, embedding, and indexing.

        When a ``chunk_hash`` appears from both searches, the two partial
        records are merged into one surviving record: dense rank/score
        and BM25 rank/score are both preserved (this is what lets fusion
        reward chunks that rank highly in both lists), and the dense
        record's payload/metadata is preferred as authoritative when
        both exist, since it reflects exactly what is live in Qdrant.

        Args:
            merged: The concatenated candidate list from
                :meth:`merge_results`.

        Returns:
            One ``RetrievedChunk`` per distinct ``chunk_hash``, in no
            particular order (sorting happens later, after fusion).
        """
        consolidated: dict[str, RetrievedChunk] = {}

        for candidate in merged:
            existing = consolidated.get(candidate.chunk_hash)
            if existing is None:
                consolidated[candidate.chunk_hash] = candidate
                continue

            if candidate.dense_rank is not None and existing.dense_rank is None:
                existing.dense_rank = candidate.dense_rank
                existing.dense_score = candidate.dense_score
                existing.payload = candidate.payload
                existing.document = candidate.document
                existing.section = candidate.section
                existing.article = candidate.article
                existing.chunk_type = candidate.chunk_type
                existing.hierarchy = candidate.hierarchy
                existing.retrieval_text = candidate.retrieval_text

            if candidate.bm25_rank is not None and existing.bm25_rank is None:
                existing.bm25_rank = candidate.bm25_rank
                existing.bm25_score = candidate.bm25_score

        return list(consolidated.values())

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document: str | None = None,
        section: str | None = None,
        article: str | None = None,
        chunk_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full hybrid retrieval pipeline for a single query.

        Pipeline: validate query -> detect explicit legal references ->
        merge with explicit filter arguments (explicit always wins on
        conflict) -> embed query -> dense search + BM25 search (each
        filtered) -> merge -> deduplicate by ``chunk_hash`` -> fuse
        scores -> normalize -> sort -> truncate to ``top_k``.

        Infrastructure failures (Qdrant unreachable, embedding model
        unavailable, BM25 index unavailable) degrade gracefully -- e.g.
        Qdrant being down falls back to BM25-only retrieval rather than
        failing the whole call. Only a contract violation by the caller
        (an empty query) raises.

        Args:
            query: The raw user query. Must be non-empty after
                stripping.
            top_k: Final number of results to return. Defaults to the
                configured ``default_top_k``.
            document: Optional explicit document filter (e.g. ``"BNS"``,
                ``"Constitution"``). Overrides any auto-detected value.
                Passed through ``_DOCUMENT_CANONICAL`` exactly like
                auto-detected values, so callers can pass either the
                short, conversational name or the literal payload value.
            section: Optional explicit section filter (e.g. ``"303"``).
                Overrides any auto-detected value.
            article: Optional explicit article filter (e.g. ``"21"``).
                Overrides any auto-detected value.
            chunk_type: Optional explicit chunk-type filter (e.g.
                ``"clause"``). Never auto-detected from query text.

        Returns:
            A list of result dicts in the canonical output schema
            (``score``, ``document``, ``section``, ``article``,
            ``chunk_type``, ``chunk_id``, ``chunk_hash``,
            ``retrieval_text``, ``payload``), sorted by descending fused
            score. Empty if no candidates were found from either search
            path.

        Raises:
            ValueError: If ``query`` is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        resolved_top_k = top_k if top_k is not None else self._default_top_k
        stats = _RetrievalStats(query=query)
        overall_start = time.perf_counter()

        auto_filters = _detect_legal_references(query)
        canonical_document = (
            _DOCUMENT_CANONICAL.get(document.upper(), document) if document else None
        )
        explicit_filters = {
            key: value
            for key, value in {
                "document": canonical_document,
                "section": section,
                "article": article,
                "chunk_type": chunk_type,
            }.items()
            if value
        }
        final_filters = {**auto_filters, **explicit_filters}
        stats.auto_filters = auto_filters
        stats.applied_filters = final_filters

        if auto_filters:
            logger.info("Auto-detected legal reference filter(s): {}", auto_filters)
        if final_filters:
            logger.info("Applying filter(s) to retrieval: {}", final_filters)

        normalized_query = _normalize_query(query)

        # Bug fixed here: previously `embed_start` was placed before
        # `embed_query()`, which internally calls `load_embedding_model()`
        # on a cold instance. On the first query in a process, that
        # silently folded the full one-time model-load cost (torch import +
        # weight download/deserialization -- tens of seconds) into
        # `stats.embedding_time`, making every "Embedding Time" figure
        # look equally slow even though only the very first call actually
        # paid it. `model_load_time` is measured once, inside
        # `load_embedding_model`, stored on the instance, and subtracted
        # out here so `stats.embedding_time` always reflects ONLY the
        # `model.encode()` call itself.
        load_time_before = self.model_load_time
        embed_start = time.perf_counter()
        query_vector = self.embed_query(normalized_query)
        wall_embed_time = time.perf_counter() - embed_start
        model_load_time_this_call = self.model_load_time - load_time_before
        stats.model_load_time = model_load_time_this_call
        stats.embedding_time = wall_embed_time - model_load_time_this_call
        if model_load_time_this_call > 0:
            logger.info(
                "Model loaded this call in {:.3f}s (one-time cost; will not "
                "recur for subsequent queries on this instance).",
                model_load_time_this_call,
            )
        if query_vector is None:
            logger.error("Query embedding failed -- aborting retrieval for: '{}'", query)
            stats.total_time = time.perf_counter() - overall_start
            self._last_stats = stats
            self.print_statistics(stats)
            return []

        qdrant_filter = self._build_filter(final_filters)

        # Qdrant connect() is similarly guarded against reconnecting once
        # `self._client` is set -- measured once via `qdrant_connect_time`
        # on the instance, never re-measured here.
        connect_time_before = self.qdrant_connect_time
        dense_start = time.perf_counter()
        if self.connect():
            dense_results = self.dense_search(query_vector, self._dense_top_k, qdrant_filter)
        else:
            logger.warning("Qdrant unavailable -- continuing with BM25-only retrieval.")
            dense_results = []
        wall_dense_time = time.perf_counter() - dense_start
        connect_time_this_call = self.qdrant_connect_time - connect_time_before
        stats.dense_time = wall_dense_time - connect_time_this_call
        stats.dense_candidates = len(dense_results)

        bm25_start = time.perf_counter()
        bm25_results = self.bm25_search(normalized_query, self._bm25_top_k, final_filters)
        stats.bm25_time = time.perf_counter() - bm25_start
        stats.bm25_candidates = len(bm25_results)

        if not dense_results and not bm25_results:
            logger.warning("No candidates found from either search path for: '{}'", query)
            stats.total_time = time.perf_counter() - overall_start
            self._last_stats = stats
            self.print_statistics(stats)
            return []

        merge_start = time.perf_counter()
        merged = self.merge_results(dense_results, bm25_results)
        deduplicated = self.remove_duplicates(merged)
        stats.deduplicated_candidates = len(deduplicated)

        if self._fusion_strategy == "weighted":
            fused = self._weighted_fusion(deduplicated)
        else:
            fused = self.reciprocal_rank_fusion(deduplicated)

        normalized_scores = self.normalize_scores([c.fused_score for c in fused])
        for candidate, normalized_score in zip(fused, normalized_scores):
            candidate.fused_score = normalized_score

        fused.sort(key=lambda c: c.fused_score, reverse=True)
        top_results = fused[:resolved_top_k]
        stats.merge_time = time.perf_counter() - merge_start

        stats.returned_chunks = len(top_results)
        stats.total_time = time.perf_counter() - overall_start
        stats.fusion_strategy = self._fusion_strategy
        self._last_stats = stats
        self.print_statistics(stats)

        return [chunk.to_output() for chunk in top_results]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self, stats: _RetrievalStats) -> None:
        """Print and log a statistics summary for one retrieval run.

        Args:
            stats: The completed ``_RetrievalStats`` for the run.
        """
        print(f"\n{'─' * 46}")
        print(f"  Query              : {stats.query!r}")
        print(f"{'─' * 46}")
        print(f"  Dense Candidates   : {stats.dense_candidates}")
        print(f"  BM25 Candidates    : {stats.bm25_candidates}")
        print(f"  After Dedup        : {stats.deduplicated_candidates}")
        print(f"  Returned Chunks    : {stats.returned_chunks}")
        print(f"  Fusion Strategy    : {stats.fusion_strategy or 'n/a'}")
        print(f"  Applied Filters    : {stats.applied_filters or 'none'}")
        print(f"  Embedding Time     : {stats.embedding_time:.3f}s")
        print(f"  Dense Search Time  : {stats.dense_time:.3f}s")
        print(f"  BM25 Search Time   : {stats.bm25_time:.3f}s")
        print(f"  Merge/Fusion Time  : {stats.merge_time:.3f}s")
        print(f"  Total Time         : {stats.total_time:.3f}s")
        print(f"{'─' * 46}\n")

        logger.info(
            "Retrieval summary — query={!r} dense={} bm25={} deduped={} "
            "returned={} strategy={} filters={} embed={:.3f}s dense_t={:.3f}s "
            "bm25_t={:.3f}s merge={:.3f}s total={:.3f}s",
            stats.query,
            stats.dense_candidates,
            stats.bm25_candidates,
            stats.deduplicated_candidates,
            stats.returned_chunks,
            stats.fusion_strategy,
            stats.applied_filters,
            stats.embedding_time,
            stats.dense_time,
            stats.bm25_time,
            stats.merge_time,
            stats.total_time,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a single ad-hoc retrieval from the command line.

    Example
    -------
    .. code-block:: bash

        python src/retriever.py "What does Section 302 BNS punish?"
    """
    query = " ".join(sys.argv[1:]) or "What does Article 21 of the Constitution guarantee?"
    retriever = HybridRetriever()
    results = retriever.retrieve(query)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
