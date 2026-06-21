"""
src/agents/validator.py
-----------------------------
Citation Validator Agent (Safety Agent) for the Indian Legal RAG system.

This module is the sixth node in the LangGraph multi-agent workflow: it
receives the outputs of the Retriever, Quote Selector, Section Mapper, and
Remedy Advisor agents and verifies that every legal citation produced by
those upstream agents is actually supported by the retrieved evidence.

Workflow position
------------------
    User Query
        -> Fact Extraction Agent
        -> Hybrid Retrieval (Retriever / Reranker)
        -> Quote Selector
        -> Section Mapper
        -> Remedy Advisor
        -> Citation Validator (this module)
        -> Synthesizer

Responsibilities
-----------------
* Verify that every applicable section referenced by the Section Mapper
  was actually present in the retrieval results (document + section +
  article match).
* Verify that every selected quote is actually found, verbatim or via
  fuzzy match above a configurable threshold, inside the corresponding
  chunk's retrieval text, and belongs to the correct section.
* Verify chunk integrity: chunk_id, chunk_hash, and document must all be
  internally consistent with the retrieval results.
* Compute an overall confidence score for the validated citation set.
* Produce a structured PASS/FAIL validation report for the Synthesizer.

This module MUST NOT
---------------------
* Retrieve documents.
* Rerank documents.
* Call any LLM.
* Generate legal reasoning.
* Generate legal advice.
* Hallucinate citations or assume a section exists without evidence.

It ONLY validates citations against retrieved evidence.

Configuration
-------------
Reads the following attributes from ``config.py`` (all optional; safe
defaults apply when ``config`` is absent or an attribute is missing):

* ``VALIDATOR_SIMILARITY_THRESHOLD``   -- float, default 0.85
* ``VALIDATOR_CONFIDENCE_THRESHOLD``   -- float, default 0.75
* ``VALIDATOR_FUZZY_MATCH_THRESHOLD``  -- float, default 0.80

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Final

from loguru import logger

# ---------------------------------------------------------------------------
# Optional project config
# ---------------------------------------------------------------------------

try:
    import config as _config  # type: ignore[import-not-found]
except ImportError:
    _config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.85
_DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.30
_DEFAULT_FUZZY_MATCH_THRESHOLD: Final[float] = 0.80

_LOGGING_CONFIGURED: bool = False

# Known legal document codes. Used to ensure a section from one document
# (e.g. "BNS") is never validated against a different document
# (e.g. "Constitution").
KNOWN_DOCUMENTS: Final[frozenset[str]] = frozenset(
    {"BNS", "BNSS", "BSA", "CONSTITUTION", "IT ACT", "IPC", "CRPC", "IEA"}
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Loguru once per process (idempotent)."""
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
                log_dir / "validator.log",
                level="DEBUG",
                rotation=getattr(_config, "LOG_ROTATION", "10 MB"),
                retention=getattr(_config, "LOG_RETENTION", "30 days"),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not set up file logging via config.LOG_DIR: {}", exc)

    _LOGGING_CONFIGURED = True


def _read_config_float(attr: str, default: float) -> float:
    """Safely read a float threshold from config.py with a fallback default.

    Args:
        attr: Attribute name on the ``config`` module.
        default: Fallback value.

    Returns:
        The configured float (clamped into [0.0, 1.0]) or ``default`` if
        absent or invalid.
    """
    if _config is None:
        return default
    value = getattr(_config, attr, default)
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        return float(value)
    logger.warning(
        "config.{} is not a valid threshold in [0, 1] ({}); using default {}.",
        attr,
        value,
        default,
    )
    return default


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChunkRecord:
    """Normalised view of a single retrieval result chunk.

    Attributes:
        document: Document code (e.g. "BNS", "BNSS", "Constitution").
        section: Section label (e.g. "Section 303"), if present.
        article: Article label (e.g. "Article 21"), if present.
        chunk_id: Unique identifier of the chunk.
        chunk_hash: Content hash of the chunk, used for integrity checks.
        text: Raw retrieval text of the chunk.
    """

    document: str = ""
    section: str = ""
    article: str = ""
    chunk_id: str = ""
    chunk_hash: str = ""
    text: str = ""

    @property
    def normalised_document(self) -> str:
        """Return the document code normalised for comparison.

        Returns:
            Upper-cased, whitespace-stripped document code.
        """
        return self.document.strip().upper()

    @property
    def normalised_section(self) -> str:
        """Return the section label normalised for comparison.

        Returns:
            Lower-cased, whitespace-stripped section label with internal
            whitespace collapsed.
        """
        return " ".join(self.section.strip().lower().split())


@dataclass
class QuoteValidationResult:
    """Result of validating a single selected quote.

    Attributes:
        quote: The original quote text under validation.
        is_valid: Whether the quote was found (exact or fuzzy) in evidence.
        similarity: Best similarity score against the matched chunk text.
        matched_chunk_id: chunk_id of the chunk the quote matched, if any.
        matched_document: Document code of the matched chunk, if any.
        matched_section: Section label of the matched chunk, if any.
        reason: Human-readable reason when validation fails.
    """

    quote: str
    is_valid: bool
    similarity: float = 0.0
    matched_chunk_id: str | None = None
    matched_document: str | None = None
    matched_section: str | None = None
    reason: str = ""


@dataclass
class SectionValidationResult:
    """Result of validating a single applicable section.

    Attributes:
        document: Document code as referenced by the Section Mapper.
        section: Section label as referenced by the Section Mapper.
        is_valid: Whether the section/document pair was found in evidence.
        matched_chunk_id: chunk_id of the supporting chunk, if any.
        reason: Human-readable reason when validation fails.
    """

    document: str
    section: str
    is_valid: bool
    matched_chunk_id: str | None = None
    reason: str = ""


@dataclass
class ValidationReport:
    """Final structured validation report returned by the validator.

    Attributes:
        validation_status: Either "PASS" or "FAIL".
        validated_sections: Section labels that were successfully validated.
        rejected_sections: Section labels that failed validation.
        validated_quotes: Quote strings that were successfully validated.
        rejected_quotes: Quote strings that failed validation.
        missing_sections: Sections referenced downstream but absent from
            retrieval evidence entirely (a subset of rejected_sections).
        confidence: Overall normalised confidence score in [0, 1].
        reason: Populated with a summary reason when status is "FAIL".
    """

    validation_status: str
    validated_sections: list[str] = field(default_factory=list)
    rejected_sections: list[str] = field(default_factory=list)
    validated_quotes: list[str] = field(default_factory=list)
    rejected_quotes: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the required output schema.

        Returns:
            Dict matching the validator's documented output contract.
        """
        payload: dict[str, Any] = {
            "validation_status": self.validation_status,
            "validated_sections": self.validated_sections,
            "rejected_sections": self.rejected_sections,
            "validated_quotes": self.validated_quotes,
            "missing_sections": self.missing_sections,
            "confidence": round(self.confidence, 4),
        }
        if self.validation_status == "FAIL":
            payload["reason"] = self.reason
        return payload


# ---------------------------------------------------------------------------
# CitationValidator
# ---------------------------------------------------------------------------


class CitationValidator:
    """Validates legal citations strictly against retrieved evidence.

    This is the system's Safety Agent. It performs no retrieval, reranking,
    or LLM calls of any kind, and it never assumes a citation is valid
    without explicit, traceable support in the retrieval results supplied
    to it. Anything not directly supported by evidence is rejected.

    Usage
    -----
    >>> validator = CitationValidator()
    >>> report = validator.validate(pipeline_output)
    >>> print(json.dumps(report, indent=2))
    """

    def __init__(
        self,
        similarity_threshold: float | None = None,
        confidence_threshold: float | None = None,
        fuzzy_match_threshold: float | None = None,
    ) -> None:
        """Initialise the validator with configurable thresholds.

        Args:
            similarity_threshold: Overrides
                ``config.VALIDATOR_SIMILARITY_THRESHOLD`` if provided.
            confidence_threshold: Overrides
                ``config.VALIDATOR_CONFIDENCE_THRESHOLD`` if provided.
            fuzzy_match_threshold: Overrides
                ``config.VALIDATOR_FUZZY_MATCH_THRESHOLD`` if provided.
        """
        _configure_logging()

        self._similarity_threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else _read_config_float(
                "VALIDATOR_SIMILARITY_THRESHOLD", _DEFAULT_SIMILARITY_THRESHOLD
            )
        )
        self._confidence_threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else _read_config_float(
                "VALIDATOR_CONFIDENCE_THRESHOLD", _DEFAULT_CONFIDENCE_THRESHOLD
            )
        )
        self._fuzzy_match_threshold: float = (
            fuzzy_match_threshold
            if fuzzy_match_threshold is not None
            else _read_config_float(
                "VALIDATOR_FUZZY_MATCH_THRESHOLD", _DEFAULT_FUZZY_MATCH_THRESHOLD
            )
        )

        logger.debug(
            "CitationValidator initialised: similarity_threshold={} "
            "confidence_threshold={} fuzzy_match_threshold={}",
            self._similarity_threshold,
            self._confidence_threshold,
            self._fuzzy_match_threshold,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def validate(self, pipeline_output: dict[str, Any]) -> dict[str, Any]:
        """Validate all citations produced by upstream agents.

        Args:
            pipeline_output: Dict expected to contain:

                * ``retrieval_results`` (list[dict]): raw retrieval/rerank
                  output, each with document/section/article/chunk_id/
                  chunk_hash/text-like fields.
                * ``selected_quotes`` (list[dict | str]): quotes chosen by
                  the Quote Selector, each ideally tagged with the
                  supporting document/section/chunk_id.
                * ``applicable_sections`` (list[dict]): sections produced
                  by the Section Mapper, each with document/section.

        Returns:
            Dict matching the validator's documented PASS/FAIL schema.
            This method never raises; on any internal error it degrades
            to a safe FAIL report rather than propagating an exception.
        """
        start_time = time.perf_counter()

        try:
            return self._validate_impl(pipeline_output, start_time)
        except Exception as exc:  # noqa: BLE001
            logger.error("CitationValidator.validate failed unexpectedly: {}", exc)
            elapsed = time.perf_counter() - start_time
            logger.info("CitationValidator aborted after {:.4f}s due to error.", elapsed)
            return ValidationReport(
                validation_status="FAIL",
                reason=f"Internal validation error: {exc}",
                confidence=0.0,
            ).to_dict()

    def _validate_impl(
        self, pipeline_output: dict[str, Any], start_time: float
    ) -> dict[str, Any]:
        """Internal implementation of :meth:`validate`.

        Args:
            pipeline_output: See :meth:`validate`.
            start_time: ``time.perf_counter()`` value captured at entry,
                used for execution-time logging.

        Returns:
            Dict matching the validator's documented PASS/FAIL schema.
        """
        if not isinstance(pipeline_output, dict):
            logger.warning(
                "validate() received non-dict input ({}); failing safe.",
                type(pipeline_output).__name__,
            )
            return ValidationReport(
                validation_status="FAIL",
                reason="Input to validator was not a dict; cannot validate.",
                confidence=0.0,
            ).to_dict()

        raw_retrieval = pipeline_output.get("retrieval_results", []) or []
        raw_quotes = pipeline_output.get("selected_quotes", []) or []
        raw_sections = pipeline_output.get("applicable_sections", []) or []

        chunks = self._build_chunk_index(raw_retrieval)

        if not chunks:
            logger.warning("No retrieval results supplied; nothing can be validated.")
            elapsed = time.perf_counter() - start_time
            logger.info("CitationValidator completed in {:.4f}s (empty evidence).", elapsed)
            return ValidationReport(
                validation_status="FAIL",
                reason="No retrieval evidence was supplied to the validator.",
                confidence=0.0,
            ).to_dict()

        section_results = self.validate_sections(raw_sections, chunks)
        quote_results = self.validate_quotes(raw_quotes, chunks)
        document_issues = self.validate_documents(raw_sections, chunks)
        chunk_issues = self.validate_chunk(raw_retrieval)

        report = self.generate_report(
            section_results=section_results,
            quote_results=quote_results,
            document_issues=document_issues,
            chunk_issues=chunk_issues,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "CitationValidator completed in {:.4f}s -- status={} "
            "validated_sections={} rejected_sections={} "
            "validated_quotes={} rejected_quotes={} confidence={:.4f}",
            elapsed,
            report.validation_status,
            len(report.validated_sections),
            len(report.rejected_sections),
            len(report.validated_quotes),
            len(report.rejected_quotes),
            report.confidence,
        )

        return report.to_dict()

    # ------------------------------------------------------------------
    # Evidence indexing
    # ------------------------------------------------------------------

    def _build_chunk_index(self, raw_retrieval: list[Any]) -> list[ChunkRecord]:
        """Normalise raw retrieval results into ``ChunkRecord`` objects.

        Handles missing or malformed fields gracefully; any chunk that
        cannot yield at least a document or section/text field is skipped
        rather than raising.

        Args:
            raw_retrieval: Raw ``retrieval_results`` list from the pipeline.

        Returns:
            List of normalised ``ChunkRecord`` instances.
        """
        chunks: list[ChunkRecord] = []

        for idx, item in enumerate(raw_retrieval):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping retrieval result at index {}: not a dict ({}).",
                    idx,
                    type(item).__name__,
                )
                continue

            # Accept both legacy `metadata` and the pipeline's `payload`
            # field so retrieval outputs from either retriever or the
            # pipeline are handled consistently.
            meta_src = {}
            if isinstance(item.get("metadata"), dict):
                meta_src = item.get("metadata")
            elif isinstance(item.get("payload"), dict):
                meta_src = item.get("payload")

            document = str(
                item.get("document") or meta_src.get("document") or ""
            ).strip()
            section = str(
                item.get("section") or meta_src.get("section") or ""
            ).strip()
            article = str(
                item.get("article") or meta_src.get("article") or ""
            ).strip()
            chunk_id = str(
                item.get("chunk_id") or meta_src.get("chunk_id") or ""
            ).strip()
            chunk_hash = str(
                item.get("chunk_hash") or meta_src.get("chunk_hash") or ""
            ).strip()
            text = str(
                item.get("text")
                or item.get("retrieval_text")
                or item.get("content")
                or ""
            )

            if not document and not section and not text:
                logger.warning(
                    "Skipping retrieval result at index {}: no usable fields.", idx
                )
                continue

            chunks.append(
                ChunkRecord(
                    document=document,
                    section=section,
                    article=article,
                    chunk_id=chunk_id,
                    chunk_hash=chunk_hash,
                    text=text,
                )
            )

        logger.debug("Built chunk index with {} usable chunk(s).", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Section validation
    # ------------------------------------------------------------------

    def validate_sections(
        self, applicable_sections: list[Any], chunks: list[ChunkRecord]
    ) -> list[SectionValidationResult]:
        """Validate each applicable section against retrieved evidence.

        A section is valid only if a chunk exists in the retrieval
        evidence whose normalised document matches and whose normalised
        section label matches exactly. No section is assumed to exist
        without this explicit support.

        Args:
            applicable_sections: Raw ``applicable_sections`` list from the
                Section Mapper.
            chunks: Normalised retrieval evidence.

        Returns:
            List of ``SectionValidationResult``, one per input section
            (duplicates are validated independently but deduplicated in
            the final report).
        """
        results: list[SectionValidationResult] = []

        for entry in applicable_sections:
            if not isinstance(entry, dict):
                logger.warning(
                    "Skipping malformed applicable_sections entry: {!r}", entry
                )
                continue

            document = str(entry.get("document", "")).strip()
            section = str(entry.get("section", "")).strip()

            if not section:
                results.append(
                    SectionValidationResult(
                        document=document,
                        section=section,
                        is_valid=False,
                        reason="Section label missing from section_mapper output.",
                    )
                )
                continue

            normalised_section = " ".join(section.strip().lower().split())
            normalised_document = document.strip().upper()

            match = next(
                (
                    chunk
                    for chunk in chunks
                    if chunk.normalised_section == normalised_section
                    and (
                        not normalised_document
                        or chunk.normalised_document == normalised_document
                    )
                ),
                None,
            )

            if match is not None:
                results.append(
                    SectionValidationResult(
                        document=document,
                        section=section,
                        is_valid=True,
                        matched_chunk_id=match.chunk_id or None,
                    )
                )
                logger.debug(
                    "Section validated: document='{}' section='{}' chunk_id='{}'",
                    document,
                    section,
                    match.chunk_id,
                )
            else:
                results.append(
                    SectionValidationResult(
                        document=document,
                        section=section,
                        is_valid=False,
                        reason=(
                            f"{section} was referenced but not retrieved."
                            if not document
                            else f"{section} of {document} was referenced but not retrieved."
                        ),
                    )
                )
                logger.warning(
                    "Section rejected: document='{}' section='{}' not found in evidence.",
                    document,
                    section,
                )

        return results

    # ------------------------------------------------------------------
    # Quote validation
    # ------------------------------------------------------------------

    def validate_quotes(
        self, selected_quotes: list[Any], chunks: list[ChunkRecord]
    ) -> list[QuoteValidationResult]:
        """Validate each selected quote against retrieval text.

        Each quote must be found, exactly or via fuzzy match above the
        configured similarity threshold, inside the ``text`` of some
        retrieved chunk. If the quote entry declares an expected
        document/section, the match must also belong to that
        document/section.

        Args:
            selected_quotes: Raw ``selected_quotes`` list from the Quote
                Selector. Entries may be plain strings or dicts with a
                ``quote``/``text`` key plus optional ``document``/
                ``section``/``chunk_id`` hints.
            chunks: Normalised retrieval evidence.

        Returns:
            List of ``QuoteValidationResult``, one per input quote.
        """
        results: list[QuoteValidationResult] = []

        for entry in selected_quotes:
            quote_text: str
            expected_document: str = ""
            expected_section: str = ""
            expected_chunk_id: str = ""

            if isinstance(entry, dict):
                quote_text = str(entry.get("quote") or entry.get("text") or "").strip()
                expected_document = str(entry.get("document", "")).strip()
                expected_section = str(entry.get("section", "")).strip()
                expected_chunk_id = str(entry.get("chunk_id", "")).strip()
            elif isinstance(entry, str):
                quote_text = entry.strip()
            else:
                logger.warning("Skipping malformed selected_quotes entry: {!r}", entry)
                continue

            if not quote_text:
                results.append(
                    QuoteValidationResult(
                        quote=quote_text,
                        is_valid=False,
                        reason="Empty quote text.",
                    )
                )
                continue

            results.append(
                self._validate_single_quote(
                    quote_text,
                    chunks,
                    expected_document=expected_document,
                    expected_section=expected_section,
                    expected_chunk_id=expected_chunk_id,
                )
            )

        return results

    def _validate_single_quote(
        self,
        quote_text: str,
        chunks: list[ChunkRecord],
        expected_document: str = "",
        expected_section: str = "",
        expected_chunk_id: str = "",
    ) -> QuoteValidationResult:
        """Validate a single quote against the candidate chunk pool.

        Args:
            quote_text: The quote string to validate.
            chunks: Normalised retrieval evidence.
            expected_document: Optional document the quote should belong to.
            expected_section: Optional section the quote should belong to.
            expected_chunk_id: Optional chunk_id the quote should come from.

        Returns:
            A single ``QuoteValidationResult``.
        """
        normalised_expected_document = expected_document.strip().upper()
        normalised_expected_section = " ".join(
            expected_section.strip().lower().split()
        )

        candidates = chunks
        if expected_chunk_id:
            narrowed = [c for c in chunks if c.chunk_id == expected_chunk_id]
            candidates = narrowed or chunks
        if normalised_expected_document:
            narrowed = [
                c for c in candidates if c.normalised_document == normalised_expected_document
            ]
            candidates = narrowed or candidates
        if normalised_expected_section:
            narrowed = [
                c for c in candidates if c.normalised_section == normalised_expected_section
            ]
            candidates = narrowed or candidates

        best_similarity = 0.0
        best_chunk: ChunkRecord | None = None

        for chunk in candidates:
            if not chunk.text:
                continue
            if quote_text in chunk.text:
                best_similarity = 1.0
                best_chunk = chunk
                break
            similarity = self._fuzzy_similarity(quote_text, chunk.text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk = chunk

        if best_chunk is not None and best_similarity >= self._fuzzy_match_threshold:
            belongs_to_correct_section = True
            if normalised_expected_section:
                belongs_to_correct_section = (
                    best_chunk.normalised_section == normalised_expected_section
                )

            if not belongs_to_correct_section:
                logger.warning(
                    "Quote matched evidence but in wrong section: quote='{}' "
                    "expected_section='{}' matched_section='{}'",
                    quote_text[:60],
                    expected_section,
                    best_chunk.section,
                )
                return QuoteValidationResult(
                    quote=quote_text,
                    is_valid=False,
                    similarity=best_similarity,
                    matched_chunk_id=best_chunk.chunk_id or None,
                    matched_document=best_chunk.document or None,
                    matched_section=best_chunk.section or None,
                    reason="Quote matched evidence text but under a different section.",
                )

            logger.debug(
                "Quote validated: similarity={:.4f} chunk_id='{}'",
                best_similarity,
                best_chunk.chunk_id,
            )
            return QuoteValidationResult(
                quote=quote_text,
                is_valid=True,
                similarity=best_similarity,
                matched_chunk_id=best_chunk.chunk_id or None,
                matched_document=best_chunk.document or None,
                matched_section=best_chunk.section or None,
            )

        logger.warning(
            "Quote rejected: best_similarity={:.4f} below threshold={:.4f} quote='{}'",
            best_similarity,
            self._fuzzy_match_threshold,
            quote_text[:60],
        )
        return QuoteValidationResult(
            quote=quote_text,
            is_valid=False,
            similarity=best_similarity,
            reason="Quote not found in any retrieved evidence above similarity threshold.",
        )

    @staticmethod
    def _fuzzy_similarity(needle: str, haystack: str) -> float:
        """Compute a best-effort fuzzy similarity of needle within haystack.

        Uses a sliding-window comparison via ``difflib.SequenceMatcher``
        against a same-length window of the haystack centered on the best
        alignment, falling back to a whole-string ratio for short text.

        Args:
            needle: The candidate quote string.
            haystack: The chunk text to search within.

        Returns:
            A similarity score in [0, 1].
        """
        needle_norm = " ".join(needle.split())
        haystack_norm = " ".join(haystack.split())

        if not needle_norm or not haystack_norm:
            return 0.0

        if len(needle_norm) >= len(haystack_norm):
            return SequenceMatcher(None, needle_norm, haystack_norm).ratio()

        window = len(needle_norm)
        step = max(1, window // 4)
        best = 0.0

        for start in range(0, max(1, len(haystack_norm) - window + 1), step):
            segment = haystack_norm[start : start + window]
            ratio = SequenceMatcher(None, needle_norm, segment).ratio()
            if ratio > best:
                best = ratio
            if best >= 0.999:
                break

        # Also check the tail window in case striding skipped past it.
        tail_segment = haystack_norm[-window:]
        tail_ratio = SequenceMatcher(None, needle_norm, tail_segment).ratio()
        return max(best, tail_ratio)

    # ------------------------------------------------------------------
    # Document validation
    # ------------------------------------------------------------------

    def validate_documents(
        self, applicable_sections: list[Any], chunks: list[ChunkRecord]
    ) -> list[str]:
        """Detect document/section mismatches across the citation set.

        Ensures, for example, that a section claimed to belong to "BNS"
        is never silently validated against evidence actually drawn from
        the "Constitution" or another document.

        Args:
            applicable_sections: Raw ``applicable_sections`` list.
            chunks: Normalised retrieval evidence.

        Returns:
            List of human-readable issue descriptions (empty if none).
        """
        issues: list[str] = []

        for entry in applicable_sections:
            if not isinstance(entry, dict):
                continue

            document = str(entry.get("document", "")).strip()
            section = str(entry.get("section", "")).strip()
            if not document or not section:
                continue

            normalised_document = document.strip().upper()
            normalised_section = " ".join(section.strip().lower().split())

            same_section_other_document = [
                chunk
                for chunk in chunks
                if chunk.normalised_section == normalised_section
                and chunk.normalised_document != normalised_document
            ]
            same_section_correct_document = any(
                chunk.normalised_section == normalised_section
                and chunk.normalised_document == normalised_document
                for chunk in chunks
            )

            if same_section_other_document and not same_section_correct_document:
                actual_documents = sorted(
                    {chunk.document for chunk in same_section_other_document if chunk.document}
                )
                issue = (
                    f"{section} was claimed under {document}, but retrieved "
                    f"evidence places it under {', '.join(actual_documents) or 'a different document'}."
                )
                issues.append(issue)
                logger.warning("Document mismatch detected: {}", issue)

            if normalised_document and normalised_document not in KNOWN_DOCUMENTS:
                logger.debug(
                    "Document code '{}' is not in the known document registry; "
                    "proceeding without rejecting on this basis alone.",
                    document,
                )

        return issues

    # ------------------------------------------------------------------
    # Chunk validation
    # ------------------------------------------------------------------

    def validate_chunk(self, raw_retrieval: list[Any]) -> list[str]:
        """Detect chunk-level integrity issues in the raw retrieval results.

        Verifies that chunk_id and chunk_hash are present and that no two
        chunks share the same chunk_id while disagreeing on document or
        chunk_hash (which would indicate corrupt or duplicated metadata).

        Args:
            raw_retrieval: Raw ``retrieval_results`` list from the pipeline.

        Returns:
            List of human-readable issue descriptions (empty if none).
        """
        issues: list[str] = []
        seen: dict[str, dict[str, str]] = {}

        for idx, item in enumerate(raw_retrieval):
            if not isinstance(item, dict):
                continue

            # Support chunk metadata provided either at the top level or
            # inside the candidate's `payload` (pipeline output).
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}

            chunk_id = str(item.get("chunk_id") or payload.get("chunk_id") or "").strip()
            chunk_hash = str(item.get("chunk_hash") or payload.get("chunk_hash") or "").strip()
            document = str(item.get("document") or payload.get("document") or "").strip()

            if not chunk_id:
                issues.append(f"Retrieval result at index {idx} is missing a chunk_id.")
                continue
            if not chunk_hash:
                issues.append(
                    f"Retrieval result for chunk_id='{chunk_id}' is missing a chunk_hash."
                )

            if chunk_id in seen:
                prior = seen[chunk_id]
                if prior.get("chunk_hash") and chunk_hash and prior["chunk_hash"] != chunk_hash:
                    issues.append(
                        f"chunk_id='{chunk_id}' has conflicting chunk_hash values "
                        "across retrieval results (possible corrupt metadata)."
                    )
                if prior.get("document") and document and prior["document"] != document:
                    issues.append(
                        f"chunk_id='{chunk_id}' has conflicting document values "
                        "across retrieval results (possible corrupt metadata)."
                    )
            else:
                seen[chunk_id] = {"chunk_hash": chunk_hash, "document": document}

        if issues:
            for issue in issues:
                logger.warning("Chunk integrity issue: {}", issue)

        return issues

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        section_results: list[SectionValidationResult],
        quote_results: list[QuoteValidationResult],
    ) -> float:
        """Compute an overall normalised confidence score.

        The score blends section validation rate, quote validation rate,
        and average quote similarity, each weighted equally where
        applicable. Categories with no entries are excluded from the
        average rather than penalising the score.

        Args:
            section_results: Per-section validation results.
            quote_results: Per-quote validation results.

        Returns:
            A confidence score normalised to [0, 1].
        """
        components: list[float] = []

        if section_results:
            valid_sections = sum(1 for r in section_results if r.is_valid)
            components.append(valid_sections / len(section_results))

        if quote_results:
            valid_quotes = sum(1 for r in quote_results if r.is_valid)
            components.append(valid_quotes / len(quote_results))

            similarities = [r.similarity for r in quote_results if r.is_valid]
            if similarities:
                components.append(sum(similarities) / len(similarities))

        if not components:
            logger.debug("No section or quote results available; confidence defaults to 0.0.")
            return 0.0

        confidence = sum(components) / len(components)
        confidence = max(0.0, min(1.0, confidence))
        logger.debug("Calculated confidence={:.4f} from {} component(s).", confidence, len(components))
        return confidence

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        section_results: list[SectionValidationResult],
        quote_results: list[QuoteValidationResult],
        document_issues: list[str],
        chunk_issues: list[str],
    ) -> ValidationReport:
        """Assemble the final PASS/FAIL validation report.

        Args:
            section_results: Per-section validation results.
            quote_results: Per-quote validation results.
            document_issues: Document/section mismatch issues, if any.
            chunk_issues: Chunk integrity issues, if any.

        Returns:
            A populated ``ValidationReport``.
        """
        validated_sections = self._dedupe(
            [r.section for r in section_results if r.is_valid and r.section]
        )
        rejected_sections = self._dedupe(
            [r.section for r in section_results if not r.is_valid and r.section]
        )
        missing_sections = self._dedupe(
            [
                r.section
                for r in section_results
                if not r.is_valid and r.section and "not retrieved" in r.reason
            ]
        )

        validated_quotes = self._dedupe(
            [r.quote for r in quote_results if r.is_valid and r.quote]
        )
        rejected_quotes = self._dedupe(
            [r.quote for r in quote_results if not r.is_valid and r.quote]
        )

        confidence = self.calculate_confidence(section_results, quote_results)

        failure_reasons: list[str] = []
        failure_reasons.extend(
            r.reason for r in section_results if not r.is_valid and r.reason
        )
        failure_reasons.extend(document_issues)
        failure_reasons.extend(chunk_issues)
        failure_reasons.extend(
            r.reason for r in quote_results if not r.is_valid and r.reason
        )

        has_hard_failure = bool(document_issues) or bool(chunk_issues)
        below_confidence = confidence < self._confidence_threshold and (
            section_results or quote_results
        )

        if has_hard_failure or below_confidence:
            primary_reason = failure_reasons[0] if failure_reasons else (
                "Overall confidence fell below the configured threshold."
            )
            report = ValidationReport(
                validation_status="FAIL",
                validated_sections=validated_sections,
                rejected_sections=rejected_sections,
                validated_quotes=validated_quotes,
                rejected_quotes=rejected_quotes,
                missing_sections=missing_sections,
                confidence=confidence,
                reason=primary_reason,
            )
        else:
            report = ValidationReport(
                validation_status="PASS",
                validated_sections=validated_sections,
                rejected_sections=rejected_sections,
                validated_quotes=validated_quotes,
                rejected_quotes=rejected_quotes,
                missing_sections=missing_sections,
                confidence=confidence,
            )

        return report

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        """Remove duplicate strings while preserving insertion order.

        Args:
            items: Possibly-duplicate list of strings.

        Returns:
            List with duplicates removed; first occurrence retained.
        """
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalised = item.strip()
            key = normalised.lower()
            if normalised and key not in seen:
                seen.add(key)
                result.append(normalised)
        return result


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

_RETRIEVAL_EVIDENCE: Final[list[dict[str, Any]]] = [
    {
        "document": "BNS",
        "section": "Section 303",
        "chunk_id": "bns-303-001",
        "chunk_hash": "a1b2c3",
        "text": (
            "Whoever commits theft shall be punished with imprisonment of "
            "either description for a term which may extend to three years, "
            "or with fine, or with both."
        ),
    },
    {
        "document": "BNS",
        "section": "Section 331",
        "chunk_id": "bns-331-001",
        "chunk_hash": "d4e5f6",
        "text": (
            "Whoever commits house-trespass shall be punished with "
            "imprisonment of either description for a term which may "
            "extend to one year, or with fine, or with both."
        ),
    },
    {
        "document": "BNS",
        "section": "Section 112",
        "chunk_id": "bns-112-001",
        "chunk_hash": "g7h8i9",
        "text": (
            "Whoever, being a member of an organised crime syndicate, "
            "commits any act of organised crime shall be punished as "
            "provided under this section."
        ),
    },
]

_TEST_SCENARIOS: Final[list[dict[str, Any]]] = [
    {
        "label": "Valid citation (section + quote both supported)",
        "input": {
            "retrieval_results": _RETRIEVAL_EVIDENCE,
            "selected_quotes": [
                {
                    "quote": (
                        "Whoever commits theft shall be punished with "
                        "imprisonment of either description for a term "
                        "which may extend to three years"
                    ),
                    "document": "BNS",
                    "section": "Section 303",
                    "chunk_id": "bns-303-001",
                }
            ],
            "applicable_sections": [{"document": "BNS", "section": "Section 303"}],
        },
    },
    {
        "label": "Invalid citation (section never retrieved)",
        "input": {
            "retrieval_results": _RETRIEVAL_EVIDENCE,
            "selected_quotes": [],
            "applicable_sections": [{"document": "BNS", "section": "Section 307"}],
        },
    },
    {
        "label": "Missing section (empty retrieval evidence)",
        "input": {
            "retrieval_results": [],
            "selected_quotes": [],
            "applicable_sections": [{"document": "BNS", "section": "Section 303"}],
        },
    },
    {
        "label": "Wrong document (section exists, but under a different document)",
        "input": {
            "retrieval_results": _RETRIEVAL_EVIDENCE,
            "selected_quotes": [],
            "applicable_sections": [
                {"document": "Constitution", "section": "Section 303"}
            ],
        },
    },
    {
        "label": "Wrong quote (text not present in any retrieved chunk)",
        "input": {
            "retrieval_results": _RETRIEVAL_EVIDENCE,
            "selected_quotes": [
                {
                    "quote": "This sentence does not appear anywhere in the evidence at all.",
                    "document": "BNS",
                    "section": "Section 303",
                }
            ],
            "applicable_sections": [{"document": "BNS", "section": "Section 303"}],
        },
    },
]


def _run_test_mode() -> None:
    """Run the CitationValidator against sample scenarios and print results."""
    validator = CitationValidator()

    for scenario in _TEST_SCENARIOS:
        label: str = scenario["label"]
        input_data: dict[str, Any] = scenario["input"]

        print(f"\n{'=' * 65}")
        print(f"  Scenario: {label}")
        print(f"{'=' * 65}")
        print("  Input:")
        print(
            json.dumps(input_data, indent=4, ensure_ascii=False).replace("\n", "\n  ")
        )
        print("\n  Output:")
        result = validator.validate(input_data)
        print(json.dumps(result, indent=4, ensure_ascii=False).replace("\n", "\n  "))


if __name__ == "__main__":
    _run_test_mode()