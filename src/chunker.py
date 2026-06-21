"""
src/chunker.py
--------------
Production-grade hierarchy-aware chunker for the Indian Legal RAG system.

Responsibilities
----------------
* Load normalised JSON records from  data/processed/
* Decompose every legal section / article into a hierarchy of typed chunks
* Preserve Act → Chapter → Section → Clause → Sub-Clause → Explanation
  → Illustration relationships exactly
* Emit deterministic, hash-verified Chunk objects
* Save per-document chunk JSON and metadata JSON to  data/chunked/

Does NOT generate embeddings, call Qdrant, call any LLM, or perform retrieval.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator

from loguru import logger

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

PROCESSED_DIR: Final[Path] = Path("data/processed")
CHUNKED_DIR: Final[Path] = Path("data/chunked")

# Maps document label → (processed filename stem, output filename stem, chunker method)
_DOCUMENT_MAP: Final[dict[str, tuple[str, str, str]]] = {
    "BNS":          ("BNS_processed",          "BNS_chunks",          "chunk_bns"),
    "BNSS":         ("BNSS_processed",          "BNSS_chunks",         "chunk_bnss"),
    "BSA":          ("BSA_processed",           "BSA_chunks",          "chunk_bsa"),
    "Constitution": ("Indian_Constitution_2024_processed",  "Constitution_chunks", "chunk_constitution"),
}

# ---------------------------------------------------------------------------
# Stopwords for keyword extraction (lightweight, no external dependency)
# ---------------------------------------------------------------------------

_STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "any", "each",
    "every", "all", "both", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "as", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "them", "their", "which", "who",
    "whom", "what", "when", "where", "how", "there", "here", "then",
    "also", "same", "under", "section", "clause", "act", "provided",
    "provided that", "herein", "hereof", "hereby", "thereof", "therefore",
    "whereas", "aforesaid", "said", "such", "above",
})

# ---------------------------------------------------------------------------
# Compiled regex patterns — compiled once at module load, never again
# ---------------------------------------------------------------------------

# Top-level numeric clauses: (1), (2), (3) … possibly followed by body text.
_RE_TOP_CLAUSE: Final[re.Pattern[str]] = re.compile(
    r"(?m)(?:^|\n)\s*(\(\d+\))\s*"
)

# Alphabetic sub-clauses: (a), (b), (c) …
_RE_ALPHA_CLAUSE: Final[re.Pattern[str]] = re.compile(
    r"(?m)(?:^|\n)\s*(\([a-z]\))\s*"
)

# Roman-numeral sub-sub-clauses: (i), (ii), (iii), (iv), (v) …
_RE_ROMAN_CLAUSE: Final[re.Pattern[str]] = re.compile(
    r"(?m)(?:^|\n)\s*(\((?:i{1,3}|iv|vi{0,3}|ix|x{1,3})\))\s*"
)

# A single token: letter/digit sequences (for keyword extraction).
_RE_TOKEN: Final[re.Pattern[str]] = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]{2,}\b")

# Normalise runs of whitespace inside text.
_RE_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s{2,}")

# Strip purely numeric or punctuation-only lines (noise).
_RE_NOISE_LINE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[\d\s\.\-–—,;:\"\'()]+\s*$"
)

# ---------------------------------------------------------------------------
# Chunk type labels
# ---------------------------------------------------------------------------

CHUNK_TYPE_SECTION: Final[str] = "section"
CHUNK_TYPE_CLAUSE: Final[str] = "clause"
CHUNK_TYPE_EXPLANATION: Final[str] = "explanation"
CHUNK_TYPE_ILLUSTRATION: Final[str] = "illustration"
CHUNK_TYPE_ARTICLE: Final[str] = "article"


# ---------------------------------------------------------------------------
# Chunk dataclass — canonical output schema
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single hierarchical chunk ready for embedding and retrieval.

    String fields are always ``str`` (never ``None``).
    ``section_no`` and ``page`` may be ``None`` when not applicable.
    """

    chunk_id: str
    document: str
    chapter: str
    part: str
    section: str
    section_no: int | None
    article: str
    clause: str
    title: str
    text: str
    retrieval_text: str
    hierarchy: list[str]
    parent_chunk_id: str
    chunk_type: str
    keywords: list[str]
    chunk_hash: str
    version: int = 1
    page: int | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-document statistics accumulator
# ---------------------------------------------------------------------------


@dataclass
class _DocStats:
    """Mutable counter bag threaded through a single document's chunking run."""

    document: str
    sections: int = 0
    clauses: int = 0
    explanations: int = 0
    illustrations: int = 0
    articles: int = 0
    duplicates: int = 0
    total_text_len: int = 0
    total_clause_len: int = 0
    processing_time: float = 0.0

    @property
    def total_chunks(self) -> int:
        """Sum of all chunk type counters."""
        return (
            self.sections
            + self.clauses
            + self.explanations
            + self.illustrations
            + self.articles
        )


# ---------------------------------------------------------------------------
# Main chunker class
# ---------------------------------------------------------------------------


class LegalChunker:
    """Hierarchy-aware chunker for Indian legal documents.

    Usage
    -----
    >>> chunker = LegalChunker()
    >>> chunker.load_processed_documents()   # loads all four acts
    >>> chunker.chunk_document("BNS")        # chunk a single document
    # or run everything at once:
    >>> for label in ("BNS", "BNSS", "BSA", "Constitution"):
    ...     chunker.chunk_document(label)
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        chunked_dir: Path = CHUNKED_DIR,
    ) -> None:
        """Initialise directory paths and configure Loguru.

        Args:
            processed_dir: Directory containing ``*_processed.json`` files.
            chunked_dir: Directory where chunk and metadata JSON is written.
        """
        self._processed_dir = processed_dir
        self._chunked_dir = chunked_dir
        # Loaded records keyed by document label, e.g. "BNS".
        self._documents: dict[str, list[dict[str, Any]]] = {}
        self._configure_logging()
        self._chunked_dir.mkdir(parents=True, exist_ok=True)

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
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "chunker.log",
            level="DEBUG",
            rotation="10 MB",
            retention="14 days",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_processed_documents(self) -> dict[str, int]:
        """Load all processed JSON files from *processed_dir*.

        Returns:
            Mapping of document label → number of records loaded.
        """
        summary: dict[str, int] = {}
        for label, (stem, _, _) in _DOCUMENT_MAP.items():
            path = self._processed_dir / f"{stem}.json"
            records = self._load_json_file(path, label)
            if records is not None:
                self._documents[label] = records
                summary[label] = len(records)
                logger.info(
                    "Loaded '{}' — {} record(s).", label, len(records)
                )
        return summary

    def chunk_document(self, label: str) -> list[Chunk]:
        """Chunk a single document by its label.

        If the document has not been loaded yet it is loaded automatically.

        Args:
            label: One of ``"BNS"``, ``"BNSS"``, ``"BSA"``,
                ``"Constitution"``.

        Returns:
            List of validated ``Chunk`` objects written to disk.
        """
        if label not in _DOCUMENT_MAP:
            logger.error("Unknown document label '{}'. Skipping.", label)
            return []

        _, _, method_name = _DOCUMENT_MAP[label]

        if label not in self._documents:
            stem, *_ = _DOCUMENT_MAP[label]
            path = self._processed_dir / f"{stem}.json"
            records = self._load_json_file(path, label)
            if records is None:
                return []
            self._documents[label] = records

        chunker_method = getattr(self, method_name)
        return chunker_method()

    # ── Per-document chunker entry points ──────────────────────────────

    def chunk_bns(self) -> list[Chunk]:
        """Chunk all BNS sections.

        Returns:
            Validated ``Chunk`` list persisted to disk.
        """
        return self._chunk_act("BNS")

    def chunk_bnss(self) -> list[Chunk]:
        """Chunk all BNSS sections.

        Returns:
            Validated ``Chunk`` list persisted to disk.
        """
        return self._chunk_act("BNSS")

    def chunk_bsa(self) -> list[Chunk]:
        """Chunk all BSA sections.

        Returns:
            Validated ``Chunk`` list persisted to disk.
        """
        return self._chunk_act("BSA")

    def chunk_constitution(self) -> list[Chunk]:
        """Chunk all Constitution articles.

        Returns:
            Validated ``Chunk`` list persisted to disk.
        """
        label = "Constitution"
        records = self._documents.get(label, [])
        if not records:
            logger.warning("No records found for '{}'.", label)
            return []

        t0 = time.perf_counter()
        stats = _DocStats(document=label)
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        for record in records:
            chunks = self._chunk_constitution_record(record, stats, seen_ids)
            all_chunks.extend(chunks)

        stats.processing_time = time.perf_counter() - t0
        self.save_chunks(all_chunks, label)
        self.save_metadata(stats, all_chunks)
        self.print_statistics(stats)
        return all_chunks

    # ------------------------------------------------------------------
    # Clause extraction
    # ------------------------------------------------------------------

    def extract_clauses(self, text: str) -> list[tuple[str, str]]:
        """Split section body text into labelled clause segments.

        The method detects three nesting levels in order of precedence:

        1. Numeric top-level clauses — ``(1)``, ``(2)``, ``(3)`` …
        2. Alphabetic sub-clauses    — ``(a)``, ``(b)``, ``(c)`` …
        3. Roman sub-sub-clauses     — ``(i)``, ``(ii)``, ``(iii)`` …

        Nested clauses are **kept inside** their parent clause text; a
        clause split occurs only at the *same nesting level*.

        Args:
            text: Raw section body text.

        Returns:
            List of ``(clause_label, clause_text)`` tuples.
            Returns an empty list when no clause markers are found.
        """
        if not text:
            return []

        # Determine which level of clause is present.
        has_numeric = bool(_RE_TOP_CLAUSE.search(text))
        has_alpha = bool(_RE_ALPHA_CLAUSE.search(text))
        has_roman = bool(_RE_ROMAN_CLAUSE.search(text))

        if has_numeric:
            return self._split_by_pattern(text, _RE_TOP_CLAUSE)
        if has_alpha:
            return self._split_by_pattern(text, _RE_ALPHA_CLAUSE)
        if has_roman:
            return self._split_by_pattern(text, _RE_ROMAN_CLAUSE)

        return []

    # ------------------------------------------------------------------
    # Chunk constructors
    # ------------------------------------------------------------------

    def create_parent_chunk(
        self,
        record: dict[str, Any],
        chunk_id: str,
        stats: _DocStats,
    ) -> Chunk:
        """Build the top-level (section) chunk for an Act record.

        The parent chunk contains the full section body text verbatim.
        Explanations and illustrations are intentionally excluded from the
        parent text; they receive their own child chunks.

        Args:
            record: Normalised ingestion record dict.
            chunk_id: Pre-computed deterministic chunk ID.
            stats: Mutable statistics accumulator (updated in-place).

        Returns:
            A fully populated ``Chunk`` object.
        """
        document = _s(record.get("document"))
        chapter  = _s(record.get("chapter"))
        part     = _s(record.get("part"))
        section  = _s(record.get("section"))
        section_no = record.get("section_no")
        title    = _s(record.get("title"))
        text     = _clean(record.get("text", ""))
        source   = _s(record.get("source"))
        page     = record.get("page")

        hierarchy = self.build_hierarchy(
            document=document,
            part=part,
            chapter=chapter,
            section=section,
            article="",
            clause="",
        )
        retrieval_text = self.generate_retrieval_text(
            document=document,
            part=part,
            chapter=chapter,
            section=section,
            article="",
            clause="",
            title=title,
            chunk_type=CHUNK_TYPE_SECTION,
            text=text,
        )
        keywords = self.extract_keywords(title + " " + text)
        chunk_hash = self.compute_chunk_hash(retrieval_text)

        stats.sections += 1
        stats.total_text_len += len(text)

        return Chunk(
            chunk_id=chunk_id,
            document=document,
            chapter=chapter,
            part=part,
            section=section,
            section_no=_int_or_none(section_no),
            article="",
            clause="",
            title=title,
            text=text,
            retrieval_text=retrieval_text,
            hierarchy=hierarchy,
            parent_chunk_id="",
            chunk_type=CHUNK_TYPE_SECTION,
            keywords=keywords,
            chunk_hash=chunk_hash,
            version=1,
            page=_int_or_none(page),
            source=source,
        )

    def create_clause_chunk(
        self,
        record: dict[str, Any],
        parent_chunk_id: str,
        clause_label: str,
        clause_text: str,
        clause_index: int,
        stats: _DocStats,
    ) -> Chunk:
        """Build a clause-level child chunk.

        Args:
            record: Normalised ingestion record dict.
            parent_chunk_id: ID of the parent section chunk.
            clause_label: Label such as ``"(1)"``, ``"(a)"``, ``"(ii)"``.
            clause_text: Body text of this clause.
            clause_index: 1-based positional index within the section.
            stats: Mutable statistics accumulator (updated in-place).

        Returns:
            A fully populated ``Chunk`` object.
        """
        document   = _s(record.get("document"))
        chapter    = _s(record.get("chapter"))
        part       = _s(record.get("part"))
        section    = _s(record.get("section"))
        section_no = record.get("section_no")
        title      = _s(record.get("title"))
        source     = _s(record.get("source"))
        page       = record.get("page")

        doc_tag = _doc_tag(document)
        sec_tag = _sec_tag(section_no)
        chunk_id = f"{doc_tag}_{sec_tag}_C{clause_index}"

        text = _clean(clause_text)

        hierarchy = self.build_hierarchy(
            document=document,
            part=part,
            chapter=chapter,
            section=section,
            article="",
            clause=clause_label,
        )
        retrieval_text = self.generate_retrieval_text(
            document=document,
            part=part,
            chapter=chapter,
            section=section,
            article="",
            clause=clause_label,
            title=title,
            chunk_type=CHUNK_TYPE_CLAUSE,
            text=text,
        )
        keywords = self.extract_keywords(title + " " + text)
        chunk_hash = self.compute_chunk_hash(retrieval_text)

        stats.clauses += 1
        stats.total_clause_len += len(text)

        return Chunk(
            chunk_id=chunk_id,
            document=document,
            chapter=chapter,
            part=part,
            section=section,
            section_no=_int_or_none(section_no),
            article="",
            clause=clause_label,
            title=title,
            text=text,
            retrieval_text=retrieval_text,
            hierarchy=hierarchy,
            parent_chunk_id=parent_chunk_id,
            chunk_type=CHUNK_TYPE_CLAUSE,
            keywords=keywords,
            chunk_hash=chunk_hash,
            version=1,
            page=_int_or_none(page),
            source=source,
        )

    def create_explanation_chunks(
        self,
        record: dict[str, Any],
        parent_chunk_id: str,
        stats: _DocStats,
    ) -> list[Chunk]:
        """Build one chunk per explanation string in *record*.

        Args:
            record: Normalised ingestion record dict.
            parent_chunk_id: ID of the parent section chunk.
            stats: Mutable statistics accumulator (updated in-place).

        Returns:
            List of explanation ``Chunk`` objects (may be empty).
        """
        explanations: list[Any] = record.get("explanations") or []
        return [
            self._build_auxiliary_chunk(
                record=record,
                parent_chunk_id=parent_chunk_id,
                text=_clean(str(exp)),
                index=idx + 1,
                suffix="EXP",
                chunk_type=CHUNK_TYPE_EXPLANATION,
                stats=stats,
            )
            for idx, exp in enumerate(explanations)
            if _clean(str(exp))
        ]

    def create_illustration_chunks(
        self,
        record: dict[str, Any],
        parent_chunk_id: str,
        stats: _DocStats,
    ) -> list[Chunk]:
        """Build one chunk per illustration string in *record*.

        Args:
            record: Normalised ingestion record dict.
            parent_chunk_id: ID of the parent section chunk.
            stats: Mutable statistics accumulator (updated in-place).

        Returns:
            List of illustration ``Chunk`` objects (may be empty).
        """
        illustrations: list[Any] = record.get("illustrations") or []
        return [
            self._build_auxiliary_chunk(
                record=record,
                parent_chunk_id=parent_chunk_id,
                text=_clean(str(ill)),
                index=idx + 1,
                suffix="ILL",
                chunk_type=CHUNK_TYPE_ILLUSTRATION,
                stats=stats,
            )
            for idx, ill in enumerate(illustrations)
            if _clean(str(ill))
        ]

    # ------------------------------------------------------------------
    # Retrieval text & hierarchy
    # ------------------------------------------------------------------

    def generate_retrieval_text(  # noqa: PLR0913
        self,
        document: str,
        part: str,
        chapter: str,
        section: str,
        article: str,
        clause: str,
        title: str,
        chunk_type: str,
        text: str,
    ) -> str:
        """Assemble the retrieval-optimised string that the embedding model
        will encode.

        The format is::

            Document: <document>
            Part: <part>           (omitted when empty)
            Chapter: <chapter>     (omitted when empty)
            Section: <section>     (omitted when empty)
            Article: <article>     (omitted when empty)
            Clause: <clause>       (omitted when empty)
            Title: <title>         (omitted when empty)
            Type: <chunk_type>
            <text>

        Args:
            document: Act name, e.g. ``"BNS"``.
            part: Part label (may be empty).
            chapter: Chapter label (may be empty).
            section: Section label, e.g. ``"Section 303"``.
            article: Article label, e.g. ``"21"`` (may be empty).
            clause: Clause label, e.g. ``"(2)"`` (may be empty).
            title: Section or article title.
            chunk_type: One of the ``CHUNK_TYPE_*`` constants.
            text: The chunk body text.

        Returns:
            Multi-line retrieval string.  Never empty.
        """
        lines: list[str] = []
        if document:
            lines.append(f"Document: {document}")
        if part:
            lines.append(f"Part: {part}")
        if chapter:
            lines.append(f"Chapter: {chapter}")
        if section:
            lines.append(f"Section: {section}")
        if article:
            lines.append(f"Article: {article}")
        if clause:
            lines.append(f"Clause: {clause}")
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Type: {chunk_type}")
        if text:
            lines.append(text)
        return "\n".join(lines)

    def extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from *text* without using an LLM.

        Strategy
        --------
        1. Tokenise with a simple word-boundary regex.
        2. Lowercase all tokens.
        3. Remove stopwords.
        4. Remove tokens shorter than 3 characters.
        5. Deduplicate while preserving first-occurrence order.

        Args:
            text: Combined title + body text for keyword extraction.

        Returns:
            Ordered, deduplicated list of meaningful keyword strings.
        """
        if not text:
            return []
        tokens = _RE_TOKEN.findall(text)
        seen: set[str] = set()
        keywords: list[str] = []
        for tok in tokens:
            lower = tok.lower()
            if lower in _STOPWORDS or len(lower) < 3:
                continue
            if lower not in seen:
                seen.add(lower)
                keywords.append(lower)
        return keywords

    def build_hierarchy(
        self,
        document: str,
        part: str,
        chapter: str,
        section: str,
        article: str,
        clause: str,
    ) -> list[str]:
        """Build the ordered hierarchy breadcrumb for a chunk.

        Only non-empty components are included.

        Args:
            document: Act or document name.
            part: Part label (may be empty).
            chapter: Chapter label (may be empty).
            section: Section label (may be empty).
            article: Article label (may be empty).
            clause: Clause label (may be empty).

        Returns:
            Ordered list of non-empty breadcrumb strings.
        """
        components = [document, part, chapter, section, article, clause]
        return [c for c in components if c]

    def compute_chunk_hash(self, retrieval_text: str) -> str:
        """Compute a deterministic SHA-256 hash of *retrieval_text*.

        Args:
            retrieval_text: The assembled retrieval string.

        Returns:
            Lowercase hex digest string (64 characters).
        """
        return hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_chunk(
        self,
        chunk: Chunk,
        seen_ids: set[str],
    ) -> bool:
        """Validate a single chunk against the required quality constraints.

        Rejection criteria
        ------------------
        * ``text`` is empty.
        * ``retrieval_text`` is empty.
        * ``chunk_id`` is already present in *seen_ids* (duplicate).
        * ``source`` is empty.
        * ``parent_chunk_id`` is empty for non-section / non-article chunks.

        Args:
            chunk: The ``Chunk`` to validate.
            seen_ids: Mutable set of previously accepted chunk IDs.

        Returns:
            ``True`` if the chunk passes all checks; ``False`` otherwise.
            Warnings are logged for every rejected chunk.
        """
        if not chunk.text:
            logger.warning(
                "Rejected chunk '{}' — empty text.", chunk.chunk_id
            )
            return False

        if not chunk.retrieval_text:
            logger.warning(
                "Rejected chunk '{}' — empty retrieval_text.", chunk.chunk_id
            )
            return False

        if chunk.chunk_id in seen_ids:
            logger.warning(
                "Rejected chunk '{}' — duplicate chunk_id.", chunk.chunk_id
            )
            return False

        if not chunk.source:
            logger.warning(
                "Rejected chunk '{}' — missing source.", chunk.chunk_id
            )
            return False

        if not chunk.title:
            logger.warning(
                "Rejected chunk '{}' — missing title.", chunk.chunk_id
            )
            return False

        if (
            chunk.chunk_type not in (CHUNK_TYPE_SECTION, CHUNK_TYPE_ARTICLE)
            and not chunk.parent_chunk_id
        ):
            logger.warning(
                "Rejected chunk '{}' — missing parent_chunk_id for type '{}'.",
                chunk.chunk_id,
                chunk.chunk_type,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk], label: str) -> Path:
        """Serialise *chunks* to ``<label>_chunks.json`` in *chunked_dir*.

        Args:
            chunks: Validated ``Chunk`` objects to persist.
            label: Document label, e.g. ``"BNS"``.

        Returns:
            Path to the written file.
        """
        _, output_stem, _ = _DOCUMENT_MAP[label]
        output_path = self._chunked_dir / f"{output_stem}.json"
        payload = [c.to_dict() for c in chunks]
        _write_json(output_path, payload)
        logger.info(
            "Saved chunks: {} ({} chunk(s))", output_path, len(chunks)
        )
        return output_path

    def save_metadata(self, stats: _DocStats, chunks: list[Chunk]) -> Path:
        """Serialise document-level metadata to ``<label>_metadata.json``.

        Args:
            stats: Completed ``_DocStats`` for the document.
            chunks: The full validated chunk list (used for averages).

        Returns:
            Path to the written metadata file.
        """
        label = stats.document
        _, output_stem, _ = _DOCUMENT_MAP[label]
        meta_path = self._chunked_dir / f"{output_stem.replace('_chunks', '_metadata')}.json"

        total = stats.total_chunks
        avg_chunk_len = (
            round(stats.total_text_len / max(stats.sections + stats.articles, 1), 1)
        )
        avg_clause_len = (
            round(stats.total_clause_len / max(stats.clauses, 1), 1)
        )

        metadata: dict[str, Any] = {
            "document": label,
            "total_records": stats.sections + stats.articles,
            "parent_chunks": stats.sections,
            "clause_chunks": stats.clauses,
            "explanation_chunks": stats.explanations,
            "illustration_chunks": stats.illustrations,
            "article_chunks": stats.articles,
            "total_chunks": total,
            "average_chunk_length": avg_chunk_len,
            "average_clause_length": avg_clause_len,
            "duplicate_count": stats.duplicates,
            "processing_time_seconds": round(stats.processing_time, 4),
        }
        _write_json(meta_path, metadata)
        logger.info("Saved metadata: {}", meta_path)
        return meta_path

    def print_statistics(self, stats: _DocStats) -> None:
        """Print a formatted statistics table to stdout.

        Args:
            stats: Completed ``_DocStats`` for the document.
        """
        doc = stats.document
        print(f"\n{'─' * 40}")
        print(f"  {doc}")
        print(f"{'─' * 40}")
        if stats.articles:
            print(f"  Articles      : {stats.articles}")
        if stats.sections:
            print(f"  Sections      : {stats.sections}")
        if stats.clauses:
            print(f"  Clauses       : {stats.clauses}")
        if stats.explanations:
            print(f"  Explanations  : {stats.explanations}")
        if stats.illustrations:
            print(f"  Illustrations : {stats.illustrations}")
        if stats.duplicates:
            print(f"  Duplicates    : {stats.duplicates}  (skipped)")
        print(f"  Total Chunks  : {stats.total_chunks}")
        print(f"  Time          : {stats.processing_time:.2f}s")
        print(f"{'─' * 40}\n")

    # ------------------------------------------------------------------
    # Private — act chunking core
    # ------------------------------------------------------------------

    def _chunk_act(self, label: str) -> list[Chunk]:
        """Run the full section-level chunking pipeline for an Act document.

        Each record produces:
        * One parent section chunk
        * N clause chunks   (extracted from body text)
        * M explanation chunks
        * K illustration chunks

        Args:
            label: Document label, e.g. ``"BNS"``.

        Returns:
            Validated ``Chunk`` list persisted to disk.
        """
        records = self._documents.get(label, [])
        if not records:
            logger.warning("No records found for '{}'.", label)
            return []

        t0 = time.perf_counter()
        stats = _DocStats(document=label)
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        for record in records:
            chunks = self._chunk_act_record(record, stats, seen_ids)
            all_chunks.extend(chunks)

        stats.processing_time = time.perf_counter() - t0
        self.save_chunks(all_chunks, label)
        self.save_metadata(stats, all_chunks)
        self.print_statistics(stats)
        return all_chunks

    def _chunk_act_record(
        self,
        record: dict[str, Any],
        stats: _DocStats,
        seen_ids: set[str],
    ) -> list[Chunk]:
        """Decompose a single Act section record into its chunk hierarchy.

        Args:
            record: Normalised ingestion record dict.
            stats: Mutable statistics accumulator.
            seen_ids: Mutable set of accepted chunk IDs for dedup checking.

        Returns:
            List of validated chunks for this record (parent + children).
        """
        document   = _s(record.get("document"))
        section_no = record.get("section_no")
        doc_tag    = _doc_tag(document)
        sec_tag    = _sec_tag(section_no)
        parent_id  = f"{doc_tag}_{sec_tag}"

        chunks: list[Chunk] = []

        # ── Parent (section) chunk ─────────────────────────────────────
        parent = self.create_parent_chunk(record, parent_id, stats)
        if self._accept(parent, seen_ids, stats):
            seen_ids.add(parent.chunk_id)
            chunks.append(parent)
        else:
            # If the parent is rejected, skip all children to avoid orphans.
            return []

        # ── Clause chunks ──────────────────────────────────────────────
        body_text = _s(record.get("text"))
        clauses = self.extract_clauses(body_text)
        for idx, (clause_label, clause_text) in enumerate(clauses, start=1):
            c_chunk = self.create_clause_chunk(
                record=record,
                parent_chunk_id=parent_id,
                clause_label=clause_label,
                clause_text=clause_text,
                clause_index=idx,
                stats=stats,
            )
            if self._accept(c_chunk, seen_ids, stats):
                seen_ids.add(c_chunk.chunk_id)
                chunks.append(c_chunk)

        # ── Explanation chunks ─────────────────────────────────────────
        for exp_chunk in self.create_explanation_chunks(record, parent_id, stats):
            if self._accept(exp_chunk, seen_ids, stats):
                seen_ids.add(exp_chunk.chunk_id)
                chunks.append(exp_chunk)

        # ── Illustration chunks ────────────────────────────────────────
        for ill_chunk in self.create_illustration_chunks(record, parent_id, stats):
            if self._accept(ill_chunk, seen_ids, stats):
                seen_ids.add(ill_chunk.chunk_id)
                chunks.append(ill_chunk)

        return chunks

    # ------------------------------------------------------------------
    # Private — Constitution chunking core
    # ------------------------------------------------------------------

    def _chunk_constitution_record(
        self,
        record: dict[str, Any],
        stats: _DocStats,
        seen_ids: set[str],
    ) -> list[Chunk]:
        """Build a single article chunk from a Constitution record.

        Constitution articles are not further split into clauses — they
        become atomic article-level chunks.

        Args:
            record: Normalised Constitution ingestion record.
            stats: Mutable statistics accumulator.
            seen_ids: Mutable set of accepted chunk IDs.

        Returns:
            List containing zero or one ``Chunk`` objects.
        """
        document = _s(record.get("document"))
        part     = _s(record.get("part"))
        chapter  = _s(record.get("chapter"))
        article  = _s(record.get("article"))
        title    = _s(record.get("title"))
        text     = _clean(record.get("text", ""))
        source   = _s(record.get("source"))
        page     = record.get("page")

        chunk_id = f"CONST_ART{article}" if article else f"CONST_UNKNOWN_{stats.articles}"

        hierarchy = self.build_hierarchy(
            document=document,
            part=part,
            chapter=chapter,
            section="",
            article=f"Article {article}" if article else "",
            clause="",
        )
        retrieval_text = self.generate_retrieval_text(
            document=document,
            part=part,
            chapter=chapter,
            section="",
            article=f"Article {article}" if article else "",
            clause="",
            title=title,
            chunk_type=CHUNK_TYPE_ARTICLE,
            text=text,
        )
        keywords = self.extract_keywords(title + " " + text)
        chunk_hash = self.compute_chunk_hash(retrieval_text)

        chunk = Chunk(
            chunk_id=chunk_id,
            document=document,
            chapter=chapter,
            part=part,
            section="",
            section_no=None,
            article=article,
            clause="",
            title=title,
            text=text,
            retrieval_text=retrieval_text,
            hierarchy=hierarchy,
            parent_chunk_id="",
            chunk_type=CHUNK_TYPE_ARTICLE,
            keywords=keywords,
            chunk_hash=chunk_hash,
            version=1,
            page=_int_or_none(page),
            source=source,
        )

        if self._accept(chunk, seen_ids, stats):
            seen_ids.add(chunk.chunk_id)
            stats.articles += 1
            return [chunk]
        return []

    # ------------------------------------------------------------------
    # Private — auxiliary chunk builder (explanations & illustrations)
    # ------------------------------------------------------------------

    def _build_auxiliary_chunk(
        self,
        record: dict[str, Any],
        parent_chunk_id: str,
        text: str,
        index: int,
        suffix: str,
        chunk_type: str,
        stats: _DocStats,
    ) -> Chunk:
        """Build an explanation or illustration chunk.

        Args:
            record: Normalised ingestion record dict.
            parent_chunk_id: ID of the parent section chunk.
            text: Cleaned body text for this chunk.
            index: 1-based index within its type list.
            suffix: ``"EXP"`` or ``"ILL"``.
            chunk_type: ``CHUNK_TYPE_EXPLANATION`` or ``CHUNK_TYPE_ILLUSTRATION``.
            stats: Mutable statistics accumulator (updated in-place).

        Returns:
            A fully populated ``Chunk`` object.
        """
        document   = _s(record.get("document"))
        chapter    = _s(record.get("chapter"))
        part       = _s(record.get("part"))
        section    = _s(record.get("section"))
        section_no = record.get("section_no")
        title      = _s(record.get("title"))
        source     = _s(record.get("source"))
        page       = record.get("page")

        doc_tag  = _doc_tag(document)
        sec_tag  = _sec_tag(section_no)
        chunk_id = f"{doc_tag}_{sec_tag}_{suffix}{index}"

        hierarchy = self.build_hierarchy(
            document=document,
            part=part,
            chapter=chapter,
            section=section,
            article="",
            clause="",
        )
        retrieval_text = self.generate_retrieval_text(
            document=document,
            part=part,
            chapter=chapter,
            section=section,
            article="",
            clause="",
            title=title,
            chunk_type=chunk_type,
            text=text,
        )
        keywords = self.extract_keywords(title + " " + text)
        chunk_hash = self.compute_chunk_hash(retrieval_text)

        if chunk_type == CHUNK_TYPE_EXPLANATION:
            stats.explanations += 1
        else:
            stats.illustrations += 1

        return Chunk(
            chunk_id=chunk_id,
            document=document,
            chapter=chapter,
            part=part,
            section=section,
            section_no=_int_or_none(section_no),
            article="",
            clause="",
            title=title,
            text=text,
            retrieval_text=retrieval_text,
            hierarchy=hierarchy,
            parent_chunk_id=parent_chunk_id,
            chunk_type=chunk_type,
            keywords=keywords,
            chunk_hash=chunk_hash,
            version=1,
            page=_int_or_none(page),
            source=source,
        )

    # ------------------------------------------------------------------
    # Private — clause splitter
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_pattern(
        text: str,
        pattern: re.Pattern[str],
    ) -> list[tuple[str, str]]:
        """Split *text* into labelled segments at every match of *pattern*.

        Text appearing before the first match is treated as preamble and
        associated with the label ``"preamble"``.

        Args:
            text: Full section body text to split.
            pattern: Compiled regex whose group(1) is the clause label.

        Returns:
            List of ``(label, body_text)`` tuples.  Only pairs where
            ``body_text`` is non-empty after stripping are returned.
        """
        segments: list[tuple[str, str]] = []
        matches = list(pattern.finditer(text))
        if not matches:
            return segments

        # Preamble: text before the first clause marker.
        preamble = text[: matches[0].start()].strip()
        if preamble:
            segments.append(("preamble", preamble))

        for i, match in enumerate(matches):
            label = match.group(1)
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                segments.append((label, body))

        return segments

    # ------------------------------------------------------------------
    # Private — chunk acceptance gate
    # ------------------------------------------------------------------

    def _accept(
        self,
        chunk: Chunk,
        seen_ids: set[str],
        stats: _DocStats,
    ) -> bool:
        """Validate *chunk* and update duplicate counter on rejection.

        Args:
            chunk: The ``Chunk`` to validate.
            seen_ids: Mutable set of previously accepted chunk IDs.
            stats: Updated when a duplicate is detected.

        Returns:
            ``True`` if the chunk passes validation.
        """
        ok = self.validate_chunk(chunk, seen_ids)
        if not ok:
            if chunk.chunk_id in seen_ids:
                stats.duplicates += 1
        return ok

    # ------------------------------------------------------------------
    # Private — JSON file loader
    # ------------------------------------------------------------------

    def _load_json_file(
        self,
        path: Path,
        label: str,
    ) -> list[dict[str, Any]] | None:
        """Load a JSON array from *path*, logging errors without crashing.

        Args:
            path: Absolute path to the JSON file.
            label: Document label used in log messages.

        Returns:
            Parsed list of dicts, or ``None`` on any error.
        """
        if not path.exists():
            logger.error("Processed file not found for '{}': {}", label, path)
            return None
        if path.stat().st_size == 0:
            logger.warning("Empty processed file for '{}': {}", label, path)
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in {}: {}", path, exc)
            return None
        except UnicodeDecodeError as exc:
            logger.error("Encoding error in {}: {}", path, exc)
            return None
        if not isinstance(raw, list):
            logger.error(
                "Expected JSON array in {}, got {}.", path, type(raw).__name__
            )
            return None
        return raw


# ---------------------------------------------------------------------------
# Module-level helpers — not exposed on the class (pure functions)
# ---------------------------------------------------------------------------


def _s(value: Any) -> str:
    """Coerce *value* to a stripped str; always returns ``str``, never ``None``.

    Args:
        value: Arbitrary value.

    Returns:
        Non-null string.
    """
    if value is None:
        return ""
    return str(value).strip()


def _clean(value: Any) -> str:
    """Normalise whitespace in a string value.

    Args:
        value: Raw string value (may be ``None``).

    Returns:
        Whitespace-normalised string; never ``None``.
    """
    if value is None:
        return ""
    text = _RE_WHITESPACE.sub(" ", str(value))
    return text.strip()


def _int_or_none(value: Any) -> int | None:
    """Convert *value* to ``int`` or return ``None`` on failure.

    Args:
        value: Raw value from a record dict.

    Returns:
        Integer or ``None``.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _doc_tag(document: str) -> str:
    """Derive a short uppercase document tag used in chunk IDs.

    Args:
        document: Full document name, e.g. ``"BNS"``.

    Returns:
        Uppercase tag, e.g. ``"BNS"``.
    """
    return document.upper().replace(" ", "_")


def _sec_tag(section_no: Any) -> str:
    """Format *section_no* into a zero-padded section tag.

    Args:
        section_no: Raw section number (int, str, or None).

    Returns:
        Tag string such as ``"SEC303"`` or ``"SEC_UNKNOWN"``.
    """
    n = _int_or_none(section_no)
    return f"SEC{n}" if n is not None else "SEC_UNKNOWN"


def _write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* as indented JSON.

    Args:
        path: Destination file path.
        payload: JSON-serialisable object.
    """
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.error("Failed to write {}: {}", path, exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full chunking pipeline from the command line.

    Example
    -------
    .. code-block:: bash

        python src/chunker.py
    """
    chunker = LegalChunker()
    chunker.load_processed_documents()
    for label in ("BNS", "BNSS", "BSA", "Constitution"):
        chunker.chunk_document(label)


if __name__ == "__main__":
    main() 