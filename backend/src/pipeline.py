"""
src/pipeline.py
---------------
Production-grade RAG orchestration pipeline for the Indian Legal RAG system.

Responsibilities
----------------
* Analyse the raw user query with regex-based rules to detect document
  names, section numbers, article numbers, and legal keywords.
* Build metadata filters automatically from detected references.
* Delegate to ``HybridRetriever`` for dense + BM25 + RRF retrieval.
* Delegate to ``CrossEncoderReranker`` for cross-encoder reranking.
* Compute a normalized confidence score for each returned result.
* Return a structured context payload ready for downstream LangGraph agents.

Does NOT call any LLM, generate answers, perform legal reasoning, or
modify the retriever / reranker logic.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root path resolution
# Ensures ``import config`` succeeds regardless of invocation style:
#   python src/pipeline.py
#   python -m src.pipeline
#   import src.pipeline  (from project root)
# ---------------------------------------------------------------------------

import sys
from pathlib import Path as _Path


def _ensure_project_root_on_path() -> None:
    """Add both the project root and src/ to sys.path.

    Guarantees regardless of invocation style:

    * ``import config``           works  — config.py is at the project root
    * ``from retriever import …`` works  — retriever.py is inside src/
    * ``from reranker  import …`` works  — reranker.py  is inside src/

    This file is always  src/pipeline.py, so the layout is fixed:

        LAW-RAG/          ← root_dir  (contains config.py)
        LAW-RAG/src/      ← src_dir   (contains retriever.py, reranker.py)

    Safe to call multiple times — paths are only inserted when absent.
    """
    src_dir  = _Path(__file__).resolve().parent   # …/backend/src
    root_dir = src_dir.parent.parent              # …/LAW-RAG (project root, contains config.py)

    for path_str in (str(root_dir), str(src_dir)):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_project_root_on_path()

import re
import time
from dataclasses import dataclass, field
from typing import Any, Final

from loguru import logger

import config
from retriever import HybridRetriever
from reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Module-level compiled regex patterns — never recompiled
# ---------------------------------------------------------------------------

# Matches "BNS Section 303", "BNSS Sec. 35", "BSA 12", "BNS303" etc.
_RE_DOC_SECTION: Final[re.Pattern[str]] = re.compile(
    r"\b(BNSS|BNS|BSA)\b[\s\-]*(?:(?:section|sec\.?)\s*)?(\d{1,4}[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)

# Matches "Article 21", "Art. 370A"
_RE_ARTICLE: Final[re.Pattern[str]] = re.compile(
    r"\barticle\s+(\d{1,4}[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)

# Matches bare "Section 302" with no preceding document code
_RE_SECTION_ONLY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:section|sec\.?)\s+(\d{1,4}[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)

# Matches explicit document names
_RE_DOCUMENT: Final[re.Pattern[str]] = re.compile(
    r"\b(BNSS|BNS|BSA|Constitution(?:\s+of\s+India)?)\b",
    re.IGNORECASE,
)

# Matches "Chapter IV", "Chapter 2", "Chapter IVA"
_RE_CHAPTER: Final[re.Pattern[str]] = re.compile(
    r"\bchapter\s+([IVXLCDM\d]+[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)

# Legal domain keywords that signal the query's intent — used for
# BM25 boosting metadata and downstream agent routing hints.
_LEGAL_KEYWORDS: Final[frozenset[str]] = frozenset({
    "murder", "culpable homicide", "theft", "robbery", "extortion",
    "assault", "kidnapping", "abduction", "rape", "hurt", "grievous hurt",
    "cheating", "fraud", "forgery", "defamation", "sedition", "bail",
    "arrest", "cognizable", "non-cognizable", "warrant", "summons",
    "fir", "chargesheet", "trial", "acquittal", "conviction", "appeal",
    "fundamental rights", "right to life", "equality", "liberty",
    "freedom of speech", "evidence", "witness", "confession",
    "compensation", "punishment", "imprisonment", "fine", "death penalty",
    "offense", "offence", "accused", "complainant", "victim",
})

# Canonical label mapping for document name normalisation
_DOC_LABEL_MAP: Final[dict[str, str]] = {
    "bns":              "BNS",
    "bnss":             "BNSS",
    "bsa":              "BSA",
    "constitution":     "Constitution of India",
    "constitution of india": "Constitution of India",
}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QueryAnalysis:
    """Structured output of the query analyser.

    Attributes:
        raw_query: The original, unmodified user query.
        normalized_query: Lower-cased, whitespace-collapsed query used
            for pattern matching.
        detected_document: Canonical document label if unambiguously
            detected; empty string otherwise.
        detected_section: Section label if unambiguously detected, in the
            exact ``"Section <N>"`` format produced by the chunking stage
            (e.g. ``"Section 101"``) — never a bare number. See
            ``_format_section_label`` for why this matters: chunker.py
            writes the ``section`` payload field as ``"Section 101"``, not
            ``"101"``, and both the Qdrant ``MatchValue`` filter and the
            BM25 local post-filter (``_matches_filters`` in retriever.py)
            require an exact string match against the stored value.
            Empty string when not detected.
        detected_article: Article number if unambiguously detected;
            empty string otherwise. Unlike ``section``, the chunker stores
            ``article`` as a bare number (e.g. ``"21"``), so no reformatting
            is applied here.
        detected_chapter: Chapter identifier if detected; empty string
            otherwise.
        filters: Metadata filters ready to pass to the retriever.
        legal_keywords: Legal domain keywords found in the query.
    """

    raw_query: str
    normalized_query: str
    detected_document: str = ""
    detected_section: str = ""
    detected_article: str = ""
    detected_chapter: str = ""
    filters: dict[str, str] = field(default_factory=dict)
    legal_keywords: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """A single ranked legal chunk ready for agent consumption.

    Attributes:
        document: Document label, e.g. ``"BNS"``.
        section: Section label, e.g. ``"Section 303"``.
        article: Article number, e.g. ``"21"`` (empty if not applicable).
        chunk_id: Unique chunk identifier.
        chunk_type: Chunk classification, e.g. ``"section"``, ``"clause"``.
        retrieval_score: Normalized fused score from the retriever.
        rerank_score: Raw logit score from the CrossEncoder.
        confidence: Final normalized confidence score in ``[0, 1]``.
        retrieval_text: Hierarchy-aware text embedded and retrieved.
        payload: Full chunk payload for citation and display.
    """

    document: str
    section: str
    article: str
    chunk_id: str
    chunk_type: str
    retrieval_score: float
    rerank_score: float
    confidence: float
    retrieval_text: str
    payload: dict[str, Any]


@dataclass
class PipelineOutput:
    """The complete output of one ``LegalRAGPipeline.search()`` call.

    Attributes:
        query: The original user query.
        filters: Metadata filters that were applied.
        confidence: Highest individual result confidence, or ``0.0``
            when no results were found.
        results: Ordered list of :class:`PipelineResult` objects.
        analysis: Full :class:`QueryAnalysis` for downstream agents.
        retrieval_time: Seconds spent in the hybrid retriever.
        rerank_time: Seconds spent in the CrossEncoder reranker.
        total_time: Wall-clock seconds for the full pipeline call.
    """

    query: str
    filters: dict[str, str]
    confidence: float
    results: list[PipelineResult]
    analysis: QueryAnalysis
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict compatible with the agent schema.

        Returns:
            Dict with keys ``query``, ``filters``, ``confidence``,
            ``results`` (list of result dicts), ``retrieval_time``,
            ``rerank_time``, and ``total_time``.
        """
        return {
            "query": self.query,
            "filters": self.filters,
            "confidence": self.confidence,
            "results": [
                {
                    "document":        r.document,
                    "section":         r.section,
                    "article":         r.article or None,
                    "chunk_id":        r.chunk_id,
                    "chunk_type":      r.chunk_type,
                    "retrieval_score": r.retrieval_score,
                    "rerank_score":    r.rerank_score,
                    "confidence":      r.confidence,
                    "retrieval_text":  r.retrieval_text,
                    "payload":         r.payload,
                }
                for r in self.results
            ],
            "retrieval_time": self.retrieval_time,
            "rerank_time":    self.rerank_time,
            "total_time":     self.total_time,
        }


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------


@dataclass
class _PipelineStats:
    """Mutable counter bag for one ``search()`` call."""

    query: str = ""
    filters: dict[str, str] = field(default_factory=dict)
    retrieval_candidates: int = 0
    reranked_candidates: int = 0
    returned_results: int = 0
    top_confidence: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# Section label canonicalization
# ---------------------------------------------------------------------------


def _format_section_label(raw_section_number: str) -> str:
    """Format a bare section number into the chunker's stored label format.

    chunker.py writes the ``section`` payload field as the literal string
    ``"Section <N>"`` (e.g. ``"Section 101"``), never a bare number like
    ``"101"``. The query analyzer's regexes (``_RE_DOC_SECTION``,
    ``_RE_SECTION_ONLY``) intentionally capture only the bare number, since
    that is what is easiest to extract and disambiguate from free text —
    but passing that bare number straight through as a filter value means
    it can never equal what is actually stored, so every section-filtered
    query would silently return zero results from both the Qdrant
    ``MatchValue`` filter and the BM25 local post-filter in retriever.py
    (``_matches_filters``), no matter how good the underlying search is.

    This mirrors the same fix already applied to ``document`` in
    retriever.py's ``_DOCUMENT_CANONICAL`` map, for the same underlying
    reason: filter values must exactly match the literal strings the
    chunking stage wrote into chunk payloads.

    Args:
        raw_section_number: A bare section identifier as captured by the
            regex group, e.g. ``"101"``, ``"103A"``. Already uppercased by
            the caller.

    Returns:
        The canonical stored label, e.g. ``"Section 101"``.
    """
    return f"Section {raw_section_number}"


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class LegalRAGPipeline:
    """Orchestrates query analysis → retrieval → reranking → context assembly.

    The retriever and reranker are each initialised exactly once and
    reused for every subsequent ``search()`` call.  The pipeline is safe
    to instantiate once at application start-up and share across agents.

    Usage
    -----
    >>> pipeline = LegalRAGPipeline()
    >>> pipeline.initialize()
    >>> output = pipeline.search("What is the punishment for murder under BNS?")
    >>> print(output.to_dict())
    """

    def __init__(
        self,
        retriever_top_k: int | None = None,
        reranker_top_k: int | None = None,
    ) -> None:
        """Initialise configuration and logging.

        Neither the retriever nor the reranker are constructed here —
        call :meth:`initialize` explicitly, or let the first
        :meth:`search` call trigger lazy initialisation.

        Args:
            retriever_top_k: Candidates fetched from the retriever before
                reranking.  Defaults to ``config.AGENT_RETRIEVER_TOP_K``.
            reranker_top_k: Results returned after reranking.  Defaults
                to ``config.AGENT_RERANKER_TOP_K``.
        """
        self._retriever_top_k: int = (
            retriever_top_k
            if retriever_top_k is not None
            else getattr(config, "AGENT_RETRIEVER_TOP_K", config.DENSE_TOP_K)
        )
        self._reranker_top_k: int = (
            reranker_top_k
            if reranker_top_k is not None
            else getattr(config, "AGENT_RERANKER_TOP_K", config.RERANKER_TOP_K)
        )

        self._retriever: HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._initialized: bool = False

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
            config.LOG_DIR / "pipeline.log",
            level="DEBUG",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create and warm the retriever and reranker exactly once.

        Subsequent calls are no-ops.  The retriever loads the BM25 index
        and connects to Qdrant; the reranker loads the CrossEncoder model.
        Both operations are performed here so that the first ``search()``
        call does not pay the initialization cost.

        Raises:
            RuntimeError: If the retriever cannot load its BM25 index.
        """
        if self._initialized:
            return

        logger.info("Initialising LegalRAGPipeline…")

        # ── Retriever ──────────────────────────────────────────────────
        self._retriever = HybridRetriever()
        if not self._retriever.load_bm25_index():
            raise RuntimeError(
                "Failed to build BM25 index from chunked documents. "
                "Ensure data/chunked/ contains valid chunk JSON files."
            )
        # Attempt Qdrant connection — failure is non-fatal (BM25-only mode).
        self._retriever.connect()

        # ── Reranker ───────────────────────────────────────────────────
        self._reranker = CrossEncoderReranker(top_k=self._reranker_top_k)
        # Pre-load the CrossEncoder so the first query is not slow.
        # Non-fatal — reranker falls back to original scores if unavailable.
        self._reranker.load_model()

        self._initialized = True
        logger.info("LegalRAGPipeline initialised successfully.")

    def _ensure_initialized(self) -> None:
        """Call :meth:`initialize` if it has not been called yet."""
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Run regex-based query analysis to detect legal references.

        Detects document codes, section numbers, article numbers, chapter
        identifiers, and legal domain keywords.  Ambiguous references
        (e.g. two different section numbers in one query) are left unset
        rather than guessed.

        Args:
            query: The raw user query.

        Returns:
            A populated :class:`QueryAnalysis` object.
        """
        normalized = re.sub(r"\s+", " ", query.strip()).lower()
        analysis = QueryAnalysis(raw_query=query, normalized_query=normalized)

        # ── 1. Document + section pairs (e.g. "BNS Section 303") ──────
        doc_section_matches = _RE_DOC_SECTION.findall(query)
        docs_from_pairs: set[str] = set()
        sections_from_pairs: set[str] = set()
        for doc_code, section_no in doc_section_matches:
            docs_from_pairs.add(doc_code.upper())
            sections_from_pairs.add(section_no.upper())

        # ── 2. Bare article numbers ────────────────────────────────────
        article_matches = {m.upper() for m in _RE_ARTICLE.findall(query)}

        # ── 3. Bare section numbers (no doc prefix) ────────────────────
        bare_sections = {m.upper() for m in _RE_SECTION_ONLY.findall(query)}
        # Merge with sections found via doc+section pairs.
        all_sections = sections_from_pairs | bare_sections

        # ── 4. Standalone document names ──────────────────────────────
        all_doc_matches: set[str] = set()
        for m in _RE_DOCUMENT.findall(query):
            canonical = _DOC_LABEL_MAP.get(m.lower().strip())
            if canonical:
                all_doc_matches.add(canonical)
        # Add docs detected via doc+section pair scan.
        for code in docs_from_pairs:
            canonical = _DOC_LABEL_MAP.get(code.lower())
            if canonical:
                all_doc_matches.add(canonical)

        # Articles imply Constitution when no other document is specified.
        if article_matches and not all_doc_matches:
            all_doc_matches.add("Constitution of India")

        # ── 5. Chapter ────────────────────────────────────────────────
        chapter_matches = {m.upper() for m in _RE_CHAPTER.findall(query)}

        # ── 6. Legal keywords ─────────────────────────────────────────
        found_keywords = [
            kw for kw in _LEGAL_KEYWORDS if kw in normalized
        ]

        # ── 7. Populate analysis — reject ambiguous values ─────────────
        if len(all_doc_matches) == 1:
            analysis.detected_document = next(iter(all_doc_matches))
        elif len(all_doc_matches) > 1:
            logger.debug(
                "Ambiguous documents in query ({}) — skipping auto document filter.",
                sorted(all_doc_matches),
            )

        if len(all_sections) == 1:
            # Stored as "Section <N>" in chunk payloads — see
            # _format_section_label for why the bare regex capture group
            # must be reformatted before being used as a filter value.
            analysis.detected_section = _format_section_label(next(iter(all_sections)))
        elif len(all_sections) > 1:
            logger.debug(
                "Ambiguous sections in query ({}) — skipping auto section filter.",
                sorted(all_sections),
            )

        if len(article_matches) == 1:
            analysis.detected_article = next(iter(article_matches))
        elif len(article_matches) > 1:
            logger.debug(
                "Ambiguous articles in query ({}) — skipping auto article filter.",
                sorted(article_matches),
            )

        if chapter_matches:
            # Take the first detected chapter; multiple chapters are unusual.
            analysis.detected_chapter = sorted(chapter_matches)[0]

        analysis.legal_keywords = sorted(found_keywords)
        analysis.filters = self.extract_filters(analysis)

        logger.debug(
            "Query analysis — document={!r} section={!r} article={!r} "
            "chapter={!r} keywords={} filters={}",
            analysis.detected_document,
            analysis.detected_section,
            analysis.detected_article,
            analysis.detected_chapter,
            analysis.legal_keywords,
            analysis.filters,
        )

        return analysis

    def extract_filters(self, analysis: QueryAnalysis) -> dict[str, str]:
        """Build a metadata filter dict from a :class:`QueryAnalysis`.

        Only fields that were unambiguously detected are included.  The
        returned dict is passed directly to
        :meth:`retriever.HybridRetriever.retrieve`.

        Args:
            analysis: A populated :class:`QueryAnalysis`.

        Returns:
            Dict with zero to three of the keys ``"document"``,
            ``"section"``, ``"article"``.
        """
        filters: dict[str, str] = {}
        if analysis.detected_document:
            filters["document"] = analysis.detected_document
        if analysis.detected_section:
            filters["section"] = analysis.detected_section
        if analysis.detected_article:
            filters["article"] = analysis.detected_article
        return filters

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        filters: dict[str, str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval via the cached :class:`HybridRetriever`.

        Args:
            query: The raw user query.
            filters: Metadata filters from :meth:`extract_filters`.
            top_k: Number of candidates to retrieve.

        Returns:
            Raw retriever output dicts.  Empty on any failure.

        Raises:
            RuntimeError: If the retriever has not been initialised.
        """
        if self._retriever is None:
            raise RuntimeError(
                "Retriever is not initialised. Call initialize() first."
            )

        document = filters.get("document")
        section  = filters.get("section")
        article  = filters.get("article")

        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
            document=document,
            section=section,
            article=article,
        )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank retrieval candidates via the cached :class:`CrossEncoderReranker`.

        Args:
            query: The raw user query.
            candidates: Raw dicts from :meth:`retrieve`.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked output dicts sorted by ``rerank_score`` descending.
            Falls back to the retriever ordering on any reranker failure.

        Raises:
            RuntimeError: If the reranker has not been initialised.
        """
        if self._reranker is None:
            raise RuntimeError(
                "Reranker is not initialised. Call initialize() first."
            )
        return self._reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        reranked: list[dict[str, Any]],
    ) -> list[float]:
        """Compute a normalized confidence score for each reranked result.

        Strategy
        --------
        Confidence combines the retriever's fused score (``original_score``)
        with the CrossEncoder's logit score (``rerank_score``).

        CrossEncoder logits are unbounded (typically in ``[-10, 10]``).
        They are first passed through a sigmoid to produce a probability in
        ``[0, 1]``, then combined with the retriever score via a weighted
        geometric mean:

            sigmoid(x) = 1 / (1 + exp(-x))
            confidence = sqrt(retrieval_score * sigmoid(rerank_score))

        The geometric mean rewards agreement between both signals: a chunk
        ranked highly by both retrieval *and* the cross-encoder gets a high
        confidence, while a chunk rescued from low retrieval rank by the
        cross-encoder (or vice-versa) is penalized proportionally.

        Results are then normalized to ``[0, 1]`` relative to the batch so
        the top result always reads as ``1.0`` and relative differences are
        preserved.

        Args:
            reranked: Reranker output dicts, sorted by rerank score.

        Returns:
            List of confidence scores in ``[0, 1]``, same length and
            order as *reranked*.
        """
        if not reranked:
            return []

        import math

        raw_confidences: list[float] = []
        for r in reranked:
            retrieval_score = float(r.get("original_score", 0.0))
            rerank_logit    = float(r.get("rerank_score", 0.0))

            # Sigmoid of the cross-encoder logit → probability in [0, 1].
            rerank_prob = 1.0 / (1.0 + math.exp(-rerank_logit))

            # Clamp retrieval score to [0, 1] (it's already normalised by
            # the retriever, but be defensive).
            retrieval_clamped = max(0.0, min(1.0, retrieval_score))

            # Geometric mean — zero if either component is zero.
            if retrieval_clamped > 0.0 and rerank_prob > 0.0:
                combined = math.sqrt(retrieval_clamped * rerank_prob)
            else:
                combined = 0.0

            raw_confidences.append(combined)

        # Normalize so the top result is 1.0.
        max_conf = max(raw_confidences) if raw_confidences else 1.0
        if max_conf <= 1e-12:
            return [0.0] * len(raw_confidences)

        return [round(c / max_conf, 4) for c in raw_confidences]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        retriever_top_k: int | None = None,
        reranker_top_k: int | None = None,
    ) -> PipelineOutput:
        """Run the complete retrieval pipeline for a single user query.

        Pipeline stages
        ---------------
        1. Validate *query*.
        2. Lazy-initialise retriever and reranker if needed.
        3. Analyse query → detect references → build filters.
        4. Hybrid retrieval (dense + BM25 + RRF).
        5. Cross-encoder reranking.
        6. Confidence calculation.
        7. Assemble and return :class:`PipelineOutput`.

        Args:
            query: The raw user query.  Must be non-empty.
            retriever_top_k: Override the configured retriever candidate
                count for this call only.
            reranker_top_k: Override the configured reranker result count
                for this call only.

        Returns:
            A :class:`PipelineOutput` ready for downstream LangGraph agents.
            ``results`` is empty when no candidates were found.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        t_total = time.perf_counter()
        stats = _PipelineStats(query=query)

        resolved_retriever_top_k = retriever_top_k or self._retriever_top_k
        resolved_reranker_top_k  = reranker_top_k  or self._reranker_top_k

        logger.info("Pipeline.search() — query={!r}", query)

        # ── 1. Lazy init ───────────────────────────────────────────────
        self._ensure_initialized()

        # ── 2. Query analysis ──────────────────────────────────────────
        analysis = self.analyze_query(query)
        stats.filters = analysis.filters

        if analysis.filters:
            logger.info("Auto-detected filters: {}", analysis.filters)
        if analysis.legal_keywords:
            logger.info("Legal keywords detected: {}", analysis.legal_keywords)

        # ── 3. Retrieval ───────────────────────────────────────────────
        t_ret = time.perf_counter()
        retrieval_candidates = self.retrieve(
            query=query,
            filters=analysis.filters,
            top_k=resolved_retriever_top_k,
        )
        stats.retrieval_time = time.perf_counter() - t_ret
        stats.retrieval_candidates = len(retrieval_candidates)

        logger.info(
            "Retrieval complete — {} candidate(s) in {:.3f}s.",
            len(retrieval_candidates),
            stats.retrieval_time,
        )

        if not retrieval_candidates:
            logger.warning("No candidates retrieved for query: {!r}", query)
            stats.total_time = time.perf_counter() - t_total
            output = PipelineOutput(
                query=query,
                filters=analysis.filters,
                confidence=0.0,
                results=[],
                analysis=analysis,
                retrieval_time=stats.retrieval_time,
                rerank_time=0.0,
                total_time=stats.total_time,
            )
            self.print_statistics(stats)
            return output

        # ── 4. Reranking ───────────────────────────────────────────────
        t_rer = time.perf_counter()
        reranked = self.rerank(
            query=query,
            candidates=retrieval_candidates,
            top_k=resolved_reranker_top_k,
        )
        stats.rerank_time = time.perf_counter() - t_rer
        stats.reranked_candidates = len(reranked)

        logger.info(
            "Reranking complete — {} result(s) in {:.3f}s.",
            len(reranked),
            stats.rerank_time,
        )

        # ── 5. Confidence ──────────────────────────────────────────────
        confidences = self.calculate_confidence(reranked)
        top_confidence = max(confidences, default=0.0)
        stats.top_confidence = top_confidence

        # ── 6. Assemble results ────────────────────────────────────────
        results: list[PipelineResult] = []
        for item, conf in zip(reranked, confidences):
            payload = item.get("payload") or {}
            results.append(
                PipelineResult(
                    document=str(item.get("document") or ""),
                    section=str(item.get("section") or ""),
                    article=str(item.get("article") or ""),
                    chunk_id=str(item.get("chunk_id") or ""),
                    chunk_type=str(
                        item.get("chunk_type")
                        or payload.get("chunk_type")
                        or ""
                    ),
                    retrieval_score=float(item.get("original_score", 0.0)),
                    rerank_score=float(item.get("rerank_score", 0.0)),
                    confidence=conf,
                    retrieval_text=str(item.get("retrieval_text") or ""),
                    payload=payload,
                )
            )

        stats.returned_results = len(results)
        stats.total_time = time.perf_counter() - t_total

        logger.info(
            "Pipeline.search() complete — {} result(s) | "
            "top_confidence={:.4f} | total={:.3f}s",
            len(results),
            top_confidence,
            stats.total_time,
        )

        output = PipelineOutput(
            query=query,
            filters=analysis.filters,
            confidence=top_confidence,
            results=results,
            analysis=analysis,
            retrieval_time=stats.retrieval_time,
            rerank_time=stats.rerank_time,
            total_time=stats.total_time,
        )
        self.print_statistics(stats)
        return output

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self, stats: _PipelineStats) -> None:
        """Print and log a statistics summary for one :meth:`search` call.

        Args:
            stats: The completed :class:`_PipelineStats` for the call.
        """
        print(f"\n{'─' * 52}")
        print(f"  Pipeline Statistics")
        print(f"{'─' * 52}")
        print(f"  Query                : {stats.query!r}")
        print(f"  Filters              : {stats.filters or 'none'}")
        print(f"  Retrieval Candidates : {stats.retrieval_candidates}")
        print(f"  Reranked Candidates  : {stats.reranked_candidates}")
        print(f"  Returned Results     : {stats.returned_results}")
        print(f"  Top Confidence       : {stats.top_confidence:.4f}")
        print(f"  Retrieval Time       : {stats.retrieval_time:.3f}s")
        print(f"  Rerank Time          : {stats.rerank_time:.3f}s")
        print(f"  Total Time           : {stats.total_time:.3f}s")
        print(f"{'─' * 52}\n")

        logger.info(
            "Pipeline stats — query={!r} filters={} retrieved={} reranked={} "
            "returned={} confidence={:.4f} ret={:.3f}s rer={:.3f}s total={:.3f}s",
            stats.query,
            stats.filters,
            stats.retrieval_candidates,
            stats.reranked_candidates,
            stats.returned_results,
            stats.top_confidence,
            stats.retrieval_time,
            stats.rerank_time,
            stats.total_time,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the pipeline from the command line with a sample query.

    Example
    -------
    .. code-block:: bash

        python src/pipeline.py
        python src/pipeline.py "What does Section 302 BNS say about murder?"
        python -m src.pipeline "Article 21 fundamental rights"
    """
    import json

    query = " ".join(sys.argv[1:]) or (
        "What is the punishment for murder under BNS Section 101?"
    )

    pipeline = LegalRAGPipeline()
    try:
        pipeline.initialize()
        output = pipeline.search(query)
        print(json.dumps(output.to_dict(), indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: {}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()