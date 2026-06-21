"""
src/ingestion.py
----------------
Production-grade document ingestion for the Indian Legal RAG system.

Responsibilities
----------------
* Scan  data/raw/  for supported files (.pdf, .json)
* Parse each file with the appropriate parser
* Normalise every record to the canonical LegalRecord schema
* Write processed JSON to  data/processed/

Does NOT perform chunking, embedding, vector indexing, or retrieval.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import fitz  # PyMuPDF
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR: Final[Path] = Path("data/raw")
PROCESSED_DIR: Final[Path] = Path("data/processed")
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf", ".json"})

_JSON_ACT_MAP: Final[dict[str, str]] = {
    "bns": "normalize_bns",
    "bnss": "normalize_bnss",
    "bsa": "normalize_bsa",
}

# ---------------------------------------------------------------------------
# Constitution parsing — compiled patterns (applied per-page, not whole-doc)
# ---------------------------------------------------------------------------

# "PART I", "PART IV A", "PART XIV", "PART IXA" etc.
_RE_PART: Final[re.Pattern[str]] = re.compile(
    r"^PART\s+([IVXLCDM]+[A-Z]?)\s*$",
    re.MULTILINE,
)

# "CHAPTER I", "CHAPTER II", "CHAPTER I-A" etc.
_RE_CHAPTER: Final[re.Pattern[str]] = re.compile(
    r"^CHAPTER\s+([IVXLCDM\d]+[\w\-]*)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Article header: line starting with digits (optionally followed by one
# capital letter), then a period, then the title text.
# Examples: "1. Name and territory of the Union."
#           "370A. Temporary provisions ..."
#           "51A. Fundamental duties."
_RE_ARTICLE_LINE: Final[re.Pattern[str]] = re.compile(
    r"^(\d+[A-Z]?)\.\s+(.+)$",
    re.MULTILINE,
)

# Lines that are almost certainly noise: lone page numbers, short roman
# numerals used as section counters, blank lines etc.
_RE_NOISE_LINE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d{1,4}|[ivxlcdmIVXLCDM]{1,6}|—|-+|_+|\*+|\.+)\s*$"
)

# Table-of-contents lines typically look like "... 12" or "…….12".
_RE_TOC_LINE: Final[re.Pattern[str]] = re.compile(
    r"[.\u2026]{3,}\s*\d+\s*$|^\s*\d+\s*[.\u2026]{3,}"
)

# Typical header / footer fragments found in Indian legal PDFs.
_NOISE_FRAGMENTS: Final[frozenset[str]] = frozenset({
    "the constitution of india",
    "constitution of india",
    "ministry of law",
    "government of india",
    "table of contents",
    "contents",
    "schedule",
    "appendix",
    "annexure",
    "lok sabha",
    "rajya sabha",
})

# ---------------------------------------------------------------------------
# Canonical output schema
# ---------------------------------------------------------------------------


@dataclass
class LegalRecord:
    """Canonical schema for every normalised legal record.

    String fields are always ``str`` (never ``None``).
    Integer fields may be ``None`` when not applicable.
    List fields default to empty lists.
    """

    document: str = ""
    chapter: str = ""
    part: str = ""
    section: str = ""
    section_no: int | None = None
    article: str = ""
    title: str = ""
    text: str = ""
    explanations: list[str] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)
    page: int | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` suitable for JSON serialisation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal Constitution page-level state
# ---------------------------------------------------------------------------


@dataclass
class _PageState:
    """Mutable state threaded through the page-by-page Constitution parser."""

    current_part: str = ""
    current_chapter: str = ""
    # Article being assembled across pages.
    open_article_no: str = ""
    open_article_title: str = ""
    open_article_page: int | None = None
    open_article_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main ingestor class
# ---------------------------------------------------------------------------


class LegalDocumentIngestor:
    """Ingest, parse, and normalise raw Indian legal documents.

    Usage
    -----
    >>> ingestor = LegalDocumentIngestor()
    >>> ingestor.ingest_directory()           # process everything in data/raw/
    >>> ingestor.ingest_file(Path("..."))     # process a single file

    The class is intentionally stateless between invocations so it can be
    instantiated once and reused.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        processed_dir: Path = PROCESSED_DIR,
    ) -> None:
        """Initialise directory paths and configure Loguru.

        Args:
            raw_dir: Directory containing raw source documents.
            processed_dir: Directory where normalised JSON files are written.
        """
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self._configure_logging()
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_logging() -> None:
        """Configure Loguru with a structured, colourised format."""
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
            log_dir / "ingestion.log",
            level="DEBUG",
            rotation="10 MB",
            retention="14 days",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_directory(self) -> dict[str, int]:
        """Scan *raw_dir* and ingest every supported file.

        Returns:
            Mapping of filename → number of records written.
        """
        if not self._raw_dir.exists():
            logger.error("Raw directory does not exist: {}", self._raw_dir)
            return {}

        files = sorted(self._raw_dir.iterdir())
        if not files:
            logger.warning("No files found in {}", self._raw_dir)
            return {}

        logger.info("Scanning {} for legal documents …", self._raw_dir)
        summary: dict[str, int] = {}

        for file_path in files:
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file: {}", file_path.name)
                continue
            logger.info("File discovered: {}", file_path.name)
            summary[file_path.name] = self.ingest_file(file_path)

        total = sum(summary.values())
        logger.info(
            "Ingestion complete — {} file(s), {} total records.",
            len(summary),
            total,
        )
        return summary

    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single file, normalise it, and persist the output.

        Args:
            file_path: Path to the source document.

        Returns:
            Number of records written; 0 on any failure.
        """
        t0 = time.perf_counter()
        records: list[LegalRecord] = []

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                records = self.parse_pdf(file_path)
            elif suffix == ".json":
                records = self.parse_json(file_path)
            else:
                logger.warning("Unsupported extension '{}' — skipped.", suffix)
                return 0
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error while processing {}: {}", file_path.name, exc
            )
            return 0

        if not records:
            logger.warning("No records extracted from {}.", file_path.name)
            return 0

        written = self.save_processed(records, stem=file_path.stem)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Finished {} in {:.2f}s — {} record(s) written.",
            file_path.name,
            elapsed,
            written,
        )
        return written

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def parse_pdf(self, file_path: Path) -> list[LegalRecord]:
        """Parse a PDF and dispatch to the correct normaliser.

        Args:
            file_path: Path to the PDF document.

        Returns:
            List of normalised ``LegalRecord`` objects.
        """
        if file_path.stat().st_size == 0:
            logger.warning("Empty PDF file: {}", file_path.name)
            return []

        try:
            doc = fitz.open(str(file_path))
        except fitz.FileDataError as exc:
            logger.error("Corrupt PDF {}: {}", file_path.name, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open PDF {}: {}", file_path.name, exc)
            return []

        logger.debug(
            "Opened PDF '{}' — {} page(s).", file_path.name, doc.page_count
        )

        stem_lower = file_path.stem.lower()
        if "constitution" in stem_lower:
            logger.info("Parser selected: ConstitutionParser → {}", file_path.name)
            records = self.normalize_constitution(doc, source=file_path.name)
        else:
            logger.warning(
                "Unknown PDF type '{}' — falling back to ConstitutionParser.",
                file_path.name,
            )
            records = self.normalize_constitution(doc, source=file_path.name)

        doc.close()
        logger.info(
            "PDF '{}' → {} record(s) extracted.", file_path.name, len(records)
        )
        return records

    def parse_json(self, file_path: Path) -> list[LegalRecord]:
        """Parse a structured JSON legal file and dispatch to a normaliser.

        The ``act`` field in the first record determines the normaliser.

        Args:
            file_path: Path to the JSON document.

        Returns:
            List of normalised ``LegalRecord`` objects.
        """
        if file_path.stat().st_size == 0:
            logger.warning("Empty JSON file: {}", file_path.name)
            return []

        try:
            text = file_path.read_text(encoding="utf-8")
            raw: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in {}: {}", file_path.name, exc)
            return []
        except UnicodeDecodeError as exc:
            logger.error("Encoding error in {}: {}", file_path.name, exc)
            return []

        if not isinstance(raw, list):
            logger.error(
                "Expected JSON array in {}, got {}.",
                file_path.name,
                type(raw).__name__,
            )
            return []

        if not raw:
            logger.warning("JSON array is empty in {}.", file_path.name)
            return []

        act_key = self._safe_str(raw[0].get("act")).lower()
        normaliser_name = _JSON_ACT_MAP.get(act_key)

        if normaliser_name is None:
            logger.error(
                "Unknown act '{}' in {} — cannot normalise.",
                act_key,
                file_path.name,
            )
            return []

        logger.info(
            "Parser selected: {} → {}", normaliser_name, file_path.name
        )
        normaliser = getattr(self, normaliser_name)
        records = normaliser(raw, source=file_path.name)
        logger.info(
            "JSON '{}' ({}) → {} record(s) extracted.",
            file_path.name,
            act_key.upper(),
            len(records),
        )
        return records

    # ------------------------------------------------------------------
    # Normalisers — JSON acts
    # ------------------------------------------------------------------

    def normalize_bns(
        self,
        raw_records: list[dict[str, Any]],
        source: str = "BNS.json",
    ) -> list[LegalRecord]:
        """Normalise Bharatiya Nyaya Sanhita (BNS) records.

        Args:
            raw_records: Parsed list of raw JSON dicts.
            source: Originating filename.

        Returns:
            List of normalised ``LegalRecord`` objects.
        """
        return self._normalize_json_act(raw_records, "BNS", source)

    def normalize_bnss(
        self,
        raw_records: list[dict[str, Any]],
        source: str = "BNSS.json",
    ) -> list[LegalRecord]:
        """Normalise Bharatiya Nagarik Suraksha Sanhita (BNSS) records.

        Args:
            raw_records: Parsed list of raw JSON dicts.
            source: Originating filename.

        Returns:
            List of normalised ``LegalRecord`` objects.
        """
        return self._normalize_json_act(raw_records, "BNSS", source)

    def normalize_bsa(
        self,
        raw_records: list[dict[str, Any]],
        source: str = "BSA.json",
    ) -> list[LegalRecord]:
        """Normalise Bharatiya Sakshya Adhiniyam (BSA) records.

        Args:
            raw_records: Parsed list of raw JSON dicts.
            source: Originating filename.

        Returns:
            List of normalised ``LegalRecord`` objects.
        """
        return self._normalize_json_act(raw_records, "BSA", source)

    # ------------------------------------------------------------------
    # Constitution PDF normaliser — page-by-page
    # ------------------------------------------------------------------

    def normalize_constitution(
        self,
        doc: fitz.Document,
        source: str = "Constitution.pdf",
    ) -> list[LegalRecord]:
        """Parse the Constitution of India PDF into per-article records.

        Design
        ------
        * Pages are processed **one at a time** — the full document is never
          concatenated into a single string.
        * Structural state (current PART, CHAPTER) is threaded through a
          ``_PageState`` object that survives across page boundaries.
        * An "open article" accumulates lines across pages until a new article
          header or end-of-document is encountered.
        * TOC pages, index pages, headers, footers, and noise lines are
          filtered before any pattern matching.

        Args:
            doc: An open ``fitz.Document`` instance.
            source: Originating filename stored in every record.

        Returns:
            List of normalised ``LegalRecord`` objects, one per article.
        """
        records: list[LegalRecord] = []
        seen_articles: set[str] = set()
        state = _PageState()

        for page_no in range(doc.page_count):
            page_obj = doc[page_no]
            try:
                raw_text: str = page_obj.get_text("text") or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not extract text from page {}: {}", page_no + 1, exc
                )
                continue

            clean_lines = self._clean_page_lines(raw_text)
            if not clean_lines:
                continue

            # Detect if this looks like a TOC / index page and skip it.
            if self._is_toc_page(clean_lines):
                logger.debug(
                    "Skipping TOC/index page {} in '{}'.", page_no + 1, source
                )
                continue

            self._process_constitution_page(
                lines=clean_lines,
                page_no=page_no + 1,  # 1-based
                state=state,
                records=records,
                seen_articles=seen_articles,
                source=source,
            )

        # Flush the last open article.
        if state.open_article_no:
            self._flush_article(state, records, seen_articles, source)

        logger.debug(
            "Constitution normaliser extracted {} article record(s).",
            len(records),
        )
        return records

    # ------------------------------------------------------------------
    # Constitution — internal helpers
    # ------------------------------------------------------------------

    def _process_constitution_page(
        self,
        lines: list[str],
        page_no: int,
        state: _PageState,
        records: list[LegalRecord],
        seen_articles: set[str],
        source: str,
    ) -> None:
        """Process a single cleaned page of the Constitution.

        Updates *state* in-place.  Completed articles are appended to
        *records*.

        Args:
            lines: Cleaned, noise-filtered lines from this page.
            page_no: 1-based page number.
            state: Mutable parser state threaded across pages.
            records: Accumulator for completed ``LegalRecord`` objects.
            seen_articles: Set of article numbers already emitted (dedup).
            source: Originating filename for provenance.
        """
        i = 0
        while i < len(lines):
            line = lines[i]

            # ── Structural markers ──────────────────────────────────────
            part_match = _RE_PART.match(line)
            if part_match:
                state.current_part = part_match.group(1).strip()
                state.current_chapter = ""  # new PART resets chapter
                logger.debug("Part transition → '{}' on page {}.",
                             state.current_part, page_no)
                i += 1
                continue

            chapter_match = _RE_CHAPTER.match(line)
            if chapter_match:
                state.current_chapter = chapter_match.group(1).strip()
                logger.debug("Chapter transition → '{}' on page {}.",
                             state.current_chapter, page_no)
                i += 1
                continue

            # ── Article header ──────────────────────────────────────────
            art_match = _RE_ARTICLE_LINE.match(line)
            if art_match:
                article_no = art_match.group(1).strip()
                article_title = self._normalise_text(art_match.group(2))

                # Flush the previously open article before starting a new one.
                if state.open_article_no:
                    self._flush_article(state, records, seen_articles, source)

                state.open_article_no = article_no
                state.open_article_title = article_title
                state.open_article_page = page_no
                state.open_article_lines = []
                i += 1
                continue

            # ── Body line belonging to the current open article ─────────
            if state.open_article_no:
                normalised = self._normalise_text(line)
                if normalised:
                    state.open_article_lines.append(normalised)

            i += 1

    def _flush_article(
        self,
        state: _PageState,
        records: list[LegalRecord],
        seen_articles: set[str],
        source: str,
    ) -> None:
        """Finalise and validate an open article, then append it to *records*.

        Resets the open-article fields in *state* regardless of whether the
        record was accepted.

        Args:
            state: Mutable parser state.
            records: Accumulator for completed ``LegalRecord`` objects.
            seen_articles: Set of article numbers already emitted.
            source: Originating filename.
        """
        article_no = state.open_article_no
        title = state.open_article_title
        body = self._collapse_lines(state.open_article_lines)

        # Reset state immediately so partial failures don't leave stale data.
        state.open_article_no = ""
        state.open_article_title = ""
        state.open_article_page = None
        state.open_article_lines = []

        # ── Validation ──────────────────────────────────────────────────
        if not title:
            logger.warning(
                "Skipping article '{}' in '{}' — missing title.",
                article_no, source,
            )
            return

        if not body:
            logger.warning(
                "Skipping article '{}' ('{}') in '{}' — empty text.",
                article_no, title, source,
            )
            return

        if article_no in seen_articles:
            logger.warning(
                "Skipping duplicate article '{}' in '{}'.",
                article_no, source,
            )
            return

        seen_articles.add(article_no)
        records.append(LegalRecord(
            document="Constitution of India",
            part=state.current_part,
            chapter=state.current_chapter,
            article=article_no,
            title=title,
            text=body,
            section="",
            section_no=None,
            explanations=[],
            illustrations=[],
            page=state.open_article_page,
            source=source,
        ))

    # ------------------------------------------------------------------
    # Normaliser — shared JSON act logic
    # ------------------------------------------------------------------

    def _normalize_json_act(
        self,
        raw_records: list[dict[str, Any]],
        document_label: str,
        source: str,
    ) -> list[LegalRecord]:
        """Shared normalisation logic for BNS, BNSS, and BSA JSON files.

        ``explanations`` and ``illustrations`` are preserved as separate
        lists and are **never** concatenated into the body.

        Args:
            raw_records: Raw dicts loaded from JSON.
            document_label: Human-readable act name, e.g. ``"BNS"``.
            source: Originating filename for provenance.

        Returns:
            List of validated ``LegalRecord`` objects.
        """
        records: list[LegalRecord] = []
        seen_sections: set[int] = set()
        skipped = 0

        for idx, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                logger.warning(
                    "[{}] Record {} is not a dict — skipped.", source, idx
                )
                skipped += 1
                continue

            section_no = self._safe_int(raw.get("section_no"),
                                        context=f"{source}[{idx}]")
            title = self._safe_str(raw.get("title"))
            body = self._safe_str(raw.get("body"))

            # ── Validation ──────────────────────────────────────────────
            if not title:
                logger.warning(
                    "[{}] Section {} at index {} missing title — skipped.",
                    source, section_no, idx,
                )
                skipped += 1
                continue

            if not body:
                logger.warning(
                    "[{}] Section {} ('{}') at index {} has empty text — skipped.",
                    source, section_no, title, idx,
                )
                skipped += 1
                continue

            if section_no is not None and section_no in seen_sections:
                logger.warning(
                    "[{}] Duplicate section_no {} at index {} — skipped.",
                    source, section_no, idx,
                )
                skipped += 1
                continue

            if section_no is not None:
                seen_sections.add(section_no)

            section_label = f"Section {section_no}" if section_no is not None else ""

            records.append(LegalRecord(
                document=document_label,
                chapter=self._safe_str(raw.get("chapter")),
                part="",
                section=section_label,
                section_no=section_no,
                article="",
                title=title,
                text=body,
                explanations=self._safe_list(
                    raw.get("explanations"),
                    context=f"{source}[{idx}].explanations",
                ),
                illustrations=self._safe_list(
                    raw.get("illustrations"),
                    context=f"{source}[{idx}].illustrations",
                ),
                page=None,
                source=source,
            ))

        if skipped:
            logger.warning(
                "[{}] {} record(s) skipped due to validation failures.",
                source, skipped,
            )
        return records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_processed(self, records: list[LegalRecord], stem: str) -> int:
        """Serialise normalised records to ``<stem>_processed.json``.

        Args:
            records: Normalised ``LegalRecord`` objects.
            stem: Filename stem of the source document.

        Returns:
            Number of records written; 0 on failure.
        """
        if not records:
            logger.warning(
                "save_processed called with zero records for '{}'.", stem
            )
            return 0

        output_path = self._processed_dir / f"{stem}_processed.json"
        payload = [r.to_dict() for r in records]

        try:
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error(
                "Failed to write {}: {}", output_path, exc
            )
            return 0

        logger.info("Saved file: {} ({} record(s))", output_path, len(records))
        return len(records)

    # ------------------------------------------------------------------
    # Text cleaning utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Normalise Unicode, collapse whitespace, and strip the result.

        Args:
            text: Raw string from PDF or JSON.

        Returns:
            Cleaned string; always ``str``, never ``None``.
        """
        if not text:
            return ""
        # NFC normalisation fixes composed/decomposed character mismatches.
        text = unicodedata.normalize("NFC", text)
        # Replace non-breaking spaces and other whitespace variants with space.
        text = re.sub(r"[\u00a0\u2002-\u200b\u202f\u3000]", " ", text)
        # Collapse runs of spaces.
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def _collapse_lines(lines: list[str]) -> str:
        """Join body lines, collapsing blank lines to a single blank line.

        Args:
            lines: Sequence of normalised body text lines.

        Returns:
            Single string with collapsed blank lines; stripped.
        """
        joined = "\n".join(lines)
        # Collapse 3+ consecutive newlines to exactly 2.
        collapsed = re.sub(r"\n{3,}", "\n\n", joined)
        return collapsed.strip()

    def _clean_page_lines(self, raw_text: str) -> list[str]:
        """Extract, normalise, and filter lines from a single page's text.

        Removes:
        * Noise-only lines (lone page numbers, rule lines, etc.)
        * Known header/footer fragments
        * TOC indicator lines

        Args:
            raw_text: Raw text string extracted by PyMuPDF from one page.

        Returns:
            List of clean, non-empty lines.
        """
        lines: list[str] = []
        for raw_line in raw_text.splitlines():
            line = self._normalise_text(raw_line)
            if not line:
                continue
            if _RE_NOISE_LINE.match(line):
                continue
            if _RE_TOC_LINE.search(line):
                continue
            if line.lower() in _NOISE_FRAGMENTS:
                continue
            lines.append(line)
        return lines

    @staticmethod
    def _is_toc_page(lines: list[str]) -> bool:
        """Heuristic: return ``True`` if the majority of lines look like TOC.

        A page is considered a TOC/index page when more than 50 % of its
        lines contain trailing page-number patterns or known TOC phrasing.

        Args:
            lines: Cleaned lines from a single page.

        Returns:
            ``True`` if the page is likely a TOC or index page.
        """
        if not lines:
            return False
        toc_hits = sum(
            1 for ln in lines
            if _RE_TOC_LINE.search(ln)
            or ln.lower() in _NOISE_FRAGMENTS
        )
        return toc_hits / len(lines) > 0.5

    # ------------------------------------------------------------------
    # Type-coercion helpers — never return None for strings
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_str(value: Any) -> str:
        """Coerce *value* to a stripped string; returns ``""`` not ``None``.

        Args:
            value: Arbitrary value from a source record.

        Returns:
            Non-null string.
        """
        if value is None:
            return ""
        normalised = unicodedata.normalize("NFC", str(value))
        return normalised.strip()

    @staticmethod
    def _safe_int(value: Any, context: str = "") -> int | None:
        """Coerce *value* to an integer; returns ``None`` on failure.

        Args:
            value: Arbitrary value from a source record.
            context: Field origin for logging.

        Returns:
            Integer or ``None``.
        """
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "Cannot convert section_no={!r} to int at {}.", value, context
            )
            return None

    @staticmethod
    def _safe_list(value: Any, context: str = "") -> list[Any]:
        """Return *value* as a list; wraps scalars, returns ``[]`` for None.

        Args:
            value: Arbitrary value from a source record.
            context: Field origin for logging.

        Returns:
            List (never ``None``).
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        logger.warning(
            "Expected list for {} but got {} — wrapping.", context, type(value).__name__
        )
        return [value]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full ingestion pipeline from the command line.

    Example
    -------
    .. code-block:: bash

        python src/ingestion.py
    """
    ingestor = LegalDocumentIngestor()
    summary = ingestor.ingest_directory()

    if summary:
        logger.info("Ingestion summary:")
        for filename, count in summary.items():
            logger.info("  {}: {} record(s)", filename, count)
    else:
        logger.warning("No files were successfully processed.")


if __name__ == "__main__":
    main()
