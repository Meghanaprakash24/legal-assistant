"""
src/agents/quote_selector.py
-----------------------------
Quote Selector Agent for the Indian Legal RAG system.

This module is the third node in the LangGraph multi-agent workflow: it
receives reranked retrieval results from the LegalRAGPipeline and reduces
long legal sections down to the most relevant quoted sentences, using
purely lexical/statistical similarity to the user query. No LLM is used.

Workflow position
------------------
    User Query
        -> Fact Extraction Agent
        -> LegalRAGPipeline
        -> Quote Selector (this module)
        -> Section Mapper
        -> Remedy Advisor
        -> Citation Validator
        -> Synthesizer

Responsibilities
-----------------
* Split each chunk's ``retrieval_text`` into individual sentences.
* Score sentences against the user query using TF-IDF cosine similarity
  (primary), with a lexical-priority boost for statutory signal words.
* Select the top-N quotes per chunk, preserving original sentence order.
* Deduplicate quotes within and across chunks.
* Return a structured list of per-chunk results with selected quotes.

This module MUST NOT
--------------------
* Call any LLM.
* Perform retrieval or reranking.
* Map sections or provide legal advice.
* Mutate or discard the incoming ``payload`` field.

Configuration
-------------
Reads the following attributes from ``config.py`` (all optional; safe
defaults are used if ``config`` is unavailable or an attribute is absent):

* ``TOP_QUOTES``         -- int, default 3   -- quotes kept per chunk.
* ``MAX_QUOTE_LENGTH``   -- int, default 500  -- hard cap per quote (chars).
* ``MIN_SENTENCE_LENGTH``-- int, default 20   -- sentences shorter than this
                            are ignored before scoring.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependencies -- handled gracefully
# ---------------------------------------------------------------------------

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

try:
    from rapidfuzz import fuzz as _fuzz

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAPIDFUZZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional project config
# ---------------------------------------------------------------------------

try:
    import config as _config  # type: ignore[import-not-found]
except ImportError:
    _config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default maximum quotes selected per chunk.
_DEFAULT_TOP_QUOTES: Final[int] = 3

#: Default hard character cap per selected quote.
_DEFAULT_MAX_QUOTE_LENGTH: Final[int] = 500

#: Default minimum sentence character length; shorter sentences are dropped.
_DEFAULT_MIN_SENTENCE_LENGTH: Final[int] = 20

#: Statutory signal words that receive a scoring boost.
_LEGAL_PRIORITY_TERMS: Final[frozenset[str]] = frozenset(
    {
        "shall",
        "punished",
        "liable",
        "imprisonment",
        "fine",
        "whoever",
        "means",
        "includes",
        "provided that",
        "explanation",
        "illustration",
    }
)

#: Regex that matches sentence-ending punctuation for splitting.
_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?;])\s+(?=[A-Z\"\(\[])|(?<=\.\))\s+"
)

#: Regex used to collapse internal whitespace in a sentence.
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

#: Boost applied to a sentence's similarity score for each legal priority
#: term it contains (additive, capped at 1.0 by the normalisation step).
_LEGAL_BOOST_PER_TERM: Final[float] = 0.08

#: Guard to ensure Loguru is configured at most once per process.
_LOGGING_CONFIGURED: bool = False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Loguru once per process (idempotent).

    Multiple agents share the same Python process in the LangGraph graph;
    without the idempotency guard every ``QuoteSelector()`` construction
    would churn handlers.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

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

    if _config is not None:
        try:
            log_dir = _config.LOG_DIR
            log_dir.mkdir(exist_ok=True)
            logger.add(
                log_dir / "quote_selector.log",
                level="DEBUG",
                rotation=getattr(_config, "LOG_ROTATION", "10 MB"),
                retention=getattr(_config, "LOG_RETENTION", "30 days"),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not set up file logging via config.LOG_DIR: {}", exc)

    _LOGGING_CONFIGURED = True


def _read_config(attr: str, default: int) -> int:
    """Safely read an integer from config.py with a fallback default.

    Args:
        attr: Attribute name to look up on the ``config`` module.
        default: Value returned when ``config`` is absent or the
            attribute is missing / not an integer.

    Returns:
        The configured integer value, or ``default``.
    """
    if _config is None:
        return default
    value = getattr(_config, attr, default)
    if not isinstance(value, int) or value <= 0:
        logger.warning(
            "config.{} is not a positive integer ({}); using default {}.",
            attr, value, default,
        )
        return default
    return value


def _has_legal_priority(sentence: str) -> bool:
    """Return True if *sentence* contains at least one statutory signal word.

    Args:
        sentence: A single candidate sentence (any case).

    Returns:
        ``True`` when at least one term from ``_LEGAL_PRIORITY_TERMS``
        appears as a word/phrase in the sentence.
    """
    lower = sentence.lower()
    return any(term in lower for term in _LEGAL_PRIORITY_TERMS)


def _legal_boost(sentence: str) -> float:
    """Compute an additive legal-priority boost for a sentence.

    Each matching statutory term contributes ``_LEGAL_BOOST_PER_TERM``
    to the boost, uncapped here (the caller caps the total score at 1.0).

    Args:
        sentence: A single candidate sentence (any case).

    Returns:
        A non-negative float boost value.
    """
    lower = sentence.lower()
    count = sum(1 for term in _LEGAL_PRIORITY_TERMS if term in lower)
    return count * _LEGAL_BOOST_PER_TERM


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScoredSentence:
    """A sentence with its relevance score and original position.

    Attributes:
        text: The cleaned sentence text.
        score: Composite relevance score in ``[0, ∞)``.
        original_index: Zero-based position in the source
            ``retrieval_text`` sentence list, used to restore order.
    """

    text: str
    score: float
    original_index: int


@dataclass
class ChunkResult:
    """Processed result for a single retrieval chunk.

    Attributes:
        chunk_id: The chunk's unique identifier from the pipeline.
        document: Source document name (e.g. ``"BNS"``).
        section: Section label (e.g. ``"Section 303"``).
        selected_quotes: Up to ``TOP_QUOTES`` selected sentence strings,
            in their original order within the source text.
        quote_count: Length of ``selected_quotes``.
        payload: The original payload dict passed through unchanged.
    """

    chunk_id: str
    document: str
    section: str
    selected_quotes: list[str] = field(default_factory=list)
    quote_count: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the output schema dict.

        Returns:
            A plain dict matching the required output structure.
        """
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "section": self.section,
            "selected_quotes": self.selected_quotes,
            "quote_count": self.quote_count,
            "payload": self.payload,
        }


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class QuoteSelector:
    """Selects the most relevant legal sentences from reranked retrieval results.

    No LLM is used.  Sentence scoring uses one of three backends, tried
    in preference order:

    1. **TF-IDF cosine similarity** (scikit-learn) -- preferred; handles
       legal vocabulary well without requiring a pre-trained model.
    2. **RapidFuzz token-set ratio** -- used when scikit-learn is absent.
    3. **Jaccard overlap on word tokens** -- pure-stdlib fallback; always
       available.

    A legal-priority boost (additive) rewards sentences containing
    statutory signal words (``shall``, ``punished``, ``liable``, etc.).

    Usage
    -----
    >>> selector = QuoteSelector()
    >>> results = selector.select_quotes(retrieval_results, user_query)
    """

    def __init__(self) -> None:
        """Initialise configuration and logging."""
        _configure_logging()

        self._top_quotes: int = _read_config("TOP_QUOTES", _DEFAULT_TOP_QUOTES)
        self._max_quote_length: int = _read_config("MAX_QUOTE_LENGTH", _DEFAULT_MAX_QUOTE_LENGTH)
        self._min_sentence_length: int = _read_config("MIN_SENTENCE_LENGTH", _DEFAULT_MIN_SENTENCE_LENGTH)

        logger.debug(
            "QuoteSelector initialised: TOP_QUOTES={} MAX_QUOTE_LENGTH={} "
            "MIN_SENTENCE_LENGTH={} sklearn={} rapidfuzz={}",
            self._top_quotes,
            self._max_quote_length,
            self._min_sentence_length,
            _SKLEARN_AVAILABLE,
            _RAPIDFUZZ_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def select_quotes(
        self,
        retrieval_results: list[dict[str, Any]],
        user_query: str,
    ) -> list[dict[str, Any]]:
        """Select the most relevant quotes from all reranked retrieval chunks.

        This is the single public method callers invoke.  It iterates
        over every chunk in ``retrieval_results``, delegates per-chunk
        work to ``_process_chunk``, and returns a list of serialised
        :class:`ChunkResult` dicts in the same order as the input.

        Args:
            retrieval_results: List of dicts produced by the pipeline,
                each containing at minimum ``chunk_id``, ``document``,
                ``section``, and ``retrieval_text``.  Extra keys are
                preserved through ``payload``.
            user_query: The original user query string, used as the
                reference for sentence scoring.

        Returns:
            A list of dicts matching the required output schema (see
            :class:`ChunkResult`).  Empty input returns an empty list.
        """
        start_time = time.perf_counter()

        if not retrieval_results:
            logger.warning("select_quotes received an empty retrieval list -- returning [].")
            return []

        if not user_query or not user_query.strip():
            logger.warning("select_quotes received an empty user_query -- scores will be zero.")
            user_query = ""

        logger.info("Processing {} retrieval chunk(s).", len(retrieval_results))

        output: list[dict[str, Any]] = []
        total_sentences = 0
        total_quotes = 0

        for chunk in retrieval_results:
            result, sentence_count = self._process_chunk(chunk, user_query)
            total_sentences += sentence_count
            total_quotes += result.quote_count
            output.append(result.to_dict())

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Quote selection complete in {:.4f}s -- chunks={} sentences_extracted={} "
            "quotes_selected={}",
            elapsed,
            len(retrieval_results),
            total_sentences,
            total_quotes,
        )
        return output

    # ------------------------------------------------------------------
    # Per-chunk processing
    # ------------------------------------------------------------------

    def _process_chunk(
        self,
        chunk: dict[str, Any],
        user_query: str,
    ) -> tuple[ChunkResult, int]:
        """Process a single retrieval chunk into a :class:`ChunkResult`.

        Handles missing or invalid fields gracefully: a chunk with no
        ``retrieval_text`` yields an empty ``ChunkResult`` rather than
        raising.

        Args:
            chunk: A single element from the pipeline's output list.
            user_query: The user query string for similarity scoring.

        Returns:
            A 2-tuple of ``(ChunkResult, number_of_sentences_extracted)``.
        """
        chunk_id = chunk.get("chunk_id", "")
        document = chunk.get("document", "")
        section = chunk.get("section", "")
        payload = chunk.get("payload", {})

        if not isinstance(payload, dict):
            logger.warning(
                "chunk_id='{}' has non-dict payload ({}); replacing with {{}}.",
                chunk_id, type(payload).__name__,
            )
            payload = {}

        retrieval_text = chunk.get("retrieval_text", "")
        if not retrieval_text or not isinstance(retrieval_text, str):
            logger.warning(
                "chunk_id='{}' has missing/invalid retrieval_text -- returning empty quotes.",
                chunk_id,
            )
            return ChunkResult(
                chunk_id=chunk_id,
                document=document,
                section=section,
                selected_quotes=[],
                quote_count=0,
                payload=payload,
            ), 0

        sentences = self.split_sentences(retrieval_text)
        sentence_count = len(sentences)
        logger.debug(
            "chunk_id='{}' ({} {}): {} sentence(s) extracted.",
            chunk_id, document, section, sentence_count,
        )

        if not sentences:
            return ChunkResult(
                chunk_id=chunk_id,
                document=document,
                section=section,
                selected_quotes=[],
                quote_count=0,
                payload=payload,
            ), 0

        scored = self.score_sentences(sentences, user_query)
        quotes = self._pick_top_quotes(scored)
        quotes = self.remove_duplicates(quotes)

        result = ChunkResult(
            chunk_id=chunk_id,
            document=document,
            section=section,
            selected_quotes=quotes,
            quote_count=len(quotes),
            payload=payload,
        )
        logger.debug(
            "chunk_id='{}': {} quote(s) selected from {} sentence(s).",
            chunk_id, result.quote_count, sentence_count,
        )
        return result, sentence_count

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        """Split ``text`` into cleaned, non-empty sentences.

        Uses a lightweight regex split on sentence-ending punctuation
        followed by whitespace and an uppercase letter (or opening
        bracket/quote), combined with a secondary split on newlines, so
        both flowing prose and enumerated statutory provisions are
        handled.

        Sentences shorter than ``MIN_SENTENCE_LENGTH`` characters are
        discarded; all remaining sentences are stripped and have internal
        whitespace collapsed.

        Args:
            text: The raw ``retrieval_text`` string from one chunk.

        Returns:
            A list of cleaned sentence strings.
        """
        if not text or not text.strip():
            return []

        # First split on newlines, then on sentence-ending punctuation.
        raw_lines: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                parts = _SENTENCE_SPLIT_RE.split(line)
                raw_lines.extend(parts)

        sentences: list[str] = []
        for raw in raw_lines:
            cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
            if len(cleaned) >= self._min_sentence_length:
                sentences.append(cleaned)

        return sentences

    # ------------------------------------------------------------------
    # Sentence scoring
    # ------------------------------------------------------------------

    def score_sentences(
        self,
        sentences: list[str],
        user_query: str,
    ) -> list[ScoredSentence]:
        """Score each sentence for relevance to the user query.

        Backend selection (in preference order):

        1. TF-IDF cosine similarity (scikit-learn) if available.
        2. RapidFuzz token-set ratio if available.
        3. Jaccard word-overlap fallback (always available).

        In all cases, a legal-priority boost is added and the final
        score is clamped to ``[0, 1]``.

        Args:
            sentences: Non-empty list of cleaned sentence strings.
            user_query: The user query used as the scoring reference.

        Returns:
            A list of :class:`ScoredSentence` objects in the same order
            as the input.
        """
        if not sentences:
            return []

        if not user_query.strip():
            return [
                ScoredSentence(text=s, score=_legal_boost(s), original_index=i)
                for i, s in enumerate(sentences)
            ]

        if _SKLEARN_AVAILABLE:
            base_scores = self._score_tfidf(sentences, user_query)
        elif _RAPIDFUZZ_AVAILABLE:
            base_scores = self._score_rapidfuzz(sentences, user_query)
        else:
            base_scores = self._score_jaccard(sentences, user_query)

        scored: list[ScoredSentence] = []
        for i, (sentence, base) in enumerate(zip(sentences, base_scores)):
            boosted = min(base + _legal_boost(sentence), 1.0)
            scored.append(ScoredSentence(text=sentence, score=boosted, original_index=i))

        return scored

    def _score_tfidf(
        self,
        sentences: list[str],
        user_query: str,
    ) -> list[float]:
        """Score sentences using TF-IDF cosine similarity.

        The query is appended as the first document so the vectoriser
        builds a vocabulary that includes query terms even when they do
        not appear in any sentence.

        Args:
            sentences: Candidate sentences to score.
            user_query: Reference query string.

        Returns:
            A list of cosine similarity floats (one per sentence),
            normalised to ``[0, 1]``.
        """
        try:
            corpus = [user_query] + sentences
            vectoriser = TfidfVectorizer(
                sublinear_tf=True,
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r"(?u)\b\w+\b",
            )
            tfidf_matrix = vectoriser.fit_transform(corpus)
            query_vector = tfidf_matrix[0]
            sentence_vectors = tfidf_matrix[1:]
            similarities = _sklearn_cosine(query_vector, sentence_vectors).flatten()
            return [float(s) for s in similarities]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TF-IDF scoring failed ({}); falling back to Jaccard.", exc
            )
            return self._score_jaccard(sentences, user_query)

    def _score_rapidfuzz(
        self,
        sentences: list[str],
        user_query: str,
    ) -> list[float]:
        """Score sentences using RapidFuzz token-set ratio.

        Args:
            sentences: Candidate sentences to score.
            user_query: Reference query string.

        Returns:
            A list of normalised similarity floats in ``[0, 1]``.
        """
        scores: list[float] = []
        query_lower = user_query.lower()
        for sentence in sentences:
            ratio = _fuzz.token_set_ratio(query_lower, sentence.lower())
            scores.append(ratio / 100.0)
        return scores

    def _score_jaccard(
        self,
        sentences: list[str],
        user_query: str,
    ) -> list[float]:
        """Score sentences using Jaccard word-token overlap.

        Pure stdlib fallback -- no external dependencies required.

        Args:
            sentences: Candidate sentences to score.
            user_query: Reference query string.

        Returns:
            A list of Jaccard similarity floats in ``[0, 1]``.
        """
        query_tokens = set(re.findall(r"\b\w+\b", user_query.lower()))
        if not query_tokens:
            return [0.0] * len(sentences)

        scores: list[float] = []
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
            if not sentence_tokens:
                scores.append(0.0)
                continue
            intersection = len(query_tokens & sentence_tokens)
            union = len(query_tokens | sentence_tokens)
            scores.append(intersection / union if union else 0.0)
        return scores

    # ------------------------------------------------------------------
    # Quote selection helpers
    # ------------------------------------------------------------------

    def _pick_top_quotes(self, scored: list[ScoredSentence]) -> list[str]:
        """Select the top-N highest-scoring sentences, restoring original order.

        1. Sentences are sorted descending by score.
        2. The top ``TOP_QUOTES`` are retained.
        3. Those survivors are then re-sorted by ``original_index`` so
           the final quotes read in the natural document order.
        4. Each quote is truncated to ``MAX_QUOTE_LENGTH`` characters.

        Args:
            scored: Output of :meth:`score_sentences`.

        Returns:
            A list of up to ``TOP_QUOTES`` quote strings.
        """
        if not scored:
            return []

        by_score = sorted(scored, key=lambda s: s.score, reverse=True)
        top_n = by_score[: self._top_quotes]
        # Restore original document order.
        top_n.sort(key=lambda s: s.original_index)

        quotes: list[str] = []
        for ss in top_n:
            text = ss.text[: self._max_quote_length].strip()
            if text:
                quotes.append(text)
        return quotes

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def remove_duplicates(self, quotes: list[str]) -> list[str]:
        """Remove exact-duplicate and near-duplicate quotes from a list.

        Two quotes are considered near-duplicates if one is a
        case-insensitive prefix or suffix of the other (i.e. the shorter
        is wholly contained in the longer).  In that case, the longer
        quote is kept. Order is preserved.

        Args:
            quotes: Candidate quote strings (may contain duplicates).

        Returns:
            A deduplicated list, order-preserving.
        """
        if not quotes:
            return []

        # Step 1: exact dedup.
        seen_exact: set[str] = set()
        unique: list[str] = []
        for q in quotes:
            key = q.strip().lower()
            if key and key not in seen_exact:
                seen_exact.add(key)
                unique.append(q)

        # Step 2: near-dedup (substring containment).
        survivors: list[str] = []
        for i, candidate in enumerate(unique):
            dominated = False
            for j, other in enumerate(unique):
                if i == j:
                    continue
                if candidate.lower() in other.lower() and len(candidate) < len(other):
                    dominated = True
                    break
            if not dominated:
                survivors.append(candidate)

        return survivors

    # ------------------------------------------------------------------
    # Statistics / diagnostics
    # ------------------------------------------------------------------

    def print_statistics(
        self,
        results: list[dict[str, Any]],
        *,
        indent: int = 2,
    ) -> None:
        """Print a human-readable summary of ``select_quotes`` output.

        Writes to stdout (in addition to the Loguru log).  Intended for
        CLI / test-mode use; not called in the normal workflow.

        Args:
            results: The list of dicts returned by :meth:`select_quotes`.
            indent: JSON indentation level for pretty-printing.
        """
        total_chunks = len(results)
        total_quotes = sum(r.get("quote_count", 0) for r in results)
        print(
            f"\n{'=' * 60}\n"
            f"  QuoteSelector Statistics\n"
            f"  Chunks processed : {total_chunks}\n"
            f"  Total quotes     : {total_quotes}\n"
            f"{'=' * 60}\n"
        )
        for result in results:
            print(
                f"  [{result.get('document', '')} / {result.get('section', '')}]"
                f"  chunk_id={result.get('chunk_id', '')}  "
                f"quotes={result.get('quote_count', 0)}"
            )
            for i, quote in enumerate(result.get("selected_quotes", []), start=1):
                # Truncate for display only.
                display = quote if len(quote) <= 120 else quote[:117] + "..."
                print(f"    {i}. {display}")
        print()
        logger.debug(
            "Statistics: chunks={} total_quotes={}", total_chunks, total_quotes
        )


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

_SAMPLE_RETRIEVAL_RESULTS: Final[list[dict[str, Any]]] = [
    {
        "document": "BNS",
        "section": "Section 303",
        "chunk_id": "bns-303-001",
        "rerank_score": 0.97,
        "retrieval_text": (
            "Whoever commits theft shall be punished with rigorous imprisonment "
            "of either description for a term which may extend to three years. "
            "The offence of theft is defined under Section 303 of the Bharatiya Nyaya Sanhita. "
            "Explanation: A person who dishonestly takes any moveable property out of the "
            "possession of any person without that person's consent is said to commit theft. "
            "The court may also impose a fine in addition to the term of imprisonment. "
            "Illustration: A and B enter into a shop and A distracts the shopkeeper while B "
            "takes a watch without paying."
        ),
        "payload": {"source": "BNS 2023", "page": 102},
    },
    {
        "document": "IPC",
        "section": "Section 392",
        "chunk_id": "ipc-392-001",
        "rerank_score": 0.91,
        "retrieval_text": (
            "Whoever commits robbery shall be punished with rigorous imprisonment "
            "for a term which may extend to ten years, and shall also be liable to fine. "
            "When robbery is committed on the highway between sunset and sunrise, the "
            "imprisonment may extend to fourteen years. "
            "The term 'robbery' includes aggravated forms of theft involving violence or "
            "the threat of violence. "
            "Provided that the court shall consider the gravity of the offence in awarding "
            "the sentence."
        ),
        "payload": {"source": "IPC 1860", "page": 87},
    },
    {
        "document": "BNS",
        "section": "Section 115",
        "chunk_id": "bns-115-001",
        "rerank_score": 0.85,
        "retrieval_text": (
            "Whoever voluntarily causes hurt to any person shall be punished with "
            "imprisonment of either description for a term which may extend to one year, "
            "or with fine which may extend to ten thousand rupees, or with both. "
            "'Hurt' means bodily pain, disease, or infirmity. "
            "Whoever causes grievous hurt shall be punished with imprisonment for up to "
            "seven years and shall also be liable to fine."
        ),
        "payload": {"source": "BNS 2023", "page": 45},
    },
    {
        "document": "BNS",
        "section": "Section 001",
        "chunk_id": "bns-empty-001",
        "rerank_score": 0.60,
        "retrieval_text": "",  # Edge case: empty text.
        "payload": {},
    },
]

_SAMPLE_USER_QUERY: Final[str] = (
    "What is the punishment for theft and robbery under Indian law?"
)


def _run_test_mode() -> None:
    """Run the QuoteSelector against sample data and print results as JSON."""
    selector = QuoteSelector()

    print(f"\nUser Query: {_SAMPLE_USER_QUERY}\n")
    results = selector.select_quotes(_SAMPLE_RETRIEVAL_RESULTS, _SAMPLE_USER_QUERY)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    selector.print_statistics(results)


if __name__ == "__main__":
    _run_test_mode()