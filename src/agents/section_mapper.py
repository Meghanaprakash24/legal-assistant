"""
src/agents/section_mapper.py
-----------------------------
Section Mapper Agent for the Indian Legal RAG system.

This module is the fourth node in the LangGraph multi-agent workflow: it
receives structured facts from the Fact Extraction Agent and selected quotes
from the Quote Selector Agent, then determines which retrieved legal
provisions are actually applicable to the incident. Every decision is driven
by configurable mappings and scoring arithmetic -- no LLM is used.

Workflow position
------------------
    User Query
        -> Fact Extraction Agent
        -> LegalRAGPipeline
        -> Quote Selector
        -> Section Mapper (this module)
        -> Remedy Advisor
        -> Citation Validator
        -> Synthesizer

Responsibilities
-----------------
* Maintain a configurable ``SECTION_REGISTRY`` mapping incident types and
  legal keywords to known statutory provisions.
* Score each candidate section against the incident facts using a weighted
  combination of: registry match, keyword overlap with the user query,
  retrieval score, rerank score, and quote text relevance.
* Rank surviving candidates by confidence and remove duplicates.
* Return a structured ``applicable_sections`` list for downstream agents.

This module MUST NOT
--------------------
* Call any LLM.
* Perform retrieval or reranking.
* Provide legal advice or determine guilt.
* Mutate incoming facts or quotes.

Configuration
-------------
Reads the following attributes from ``config.py`` (all optional; safe
defaults apply when ``config`` is absent or an attribute is missing):

* ``SECTION_MAPPER_MIN_CONFIDENCE``  -- float, default 0.30 -- sections
  below this threshold are dropped from the output.
* ``SECTION_MAPPER_MAX_RESULTS``     -- int,   default 10   -- maximum
  number of sections returned.
* ``SECTION_MAPPER_KEYWORD_WEIGHT``  -- float, default 0.30
* ``SECTION_MAPPER_RETRIEVAL_WEIGHT``-- float, default 0.20
* ``SECTION_MAPPER_RERANK_WEIGHT``   -- float, default 0.20
* ``SECTION_MAPPER_QUOTE_WEIGHT``    -- float, default 0.30

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
# Optional project config
# ---------------------------------------------------------------------------

try:
    import config as _config  # type: ignore[import-not-found]
except ImportError:
    _config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_DEFAULT_MIN_CONFIDENCE: Final[float] = 0.12
_DEFAULT_MAX_RESULTS: Final[int] = 10
_DEFAULT_KEYWORD_WEIGHT: Final[float] = 0.20
_DEFAULT_RETRIEVAL_WEIGHT: Final[float] = 0.10
_DEFAULT_RERANK_WEIGHT: Final[float] = 0.10
_DEFAULT_QUOTE_WEIGHT: Final[float] = 0.20

#: Guard to ensure Loguru is configured at most once per process.
_LOGGING_CONFIGURED: bool = False

# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------
#
# Each entry maps one or more incident types and synonymous keywords to a
# statutory provision. Adding a new offence means adding a record here; no
# control-flow changes are needed.
#
# Schema per record:
#   "document"       -- canonical corpus name (e.g. "BNS", "IPC", "POCSO")
#   "section"        -- human-readable section label
#   "incident_types" -- list of incident-type labels (from classifier.py)
#   "keywords"       -- list of additional lexical triggers
#   "reason_template"-- f-string template; ``{incident_type}`` is interpolated
#                       at runtime with the first matching incident type.

SECTION_REGISTRY: Final[list[dict[str, Any]]] = [
    # ------------------------------------------------------------------
    # Theft / Robbery / Extortion
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 303",
        "incident_types": ["theft"],
        "keywords": ["steal", "stolen", "stole", "theft", "movable property", "dishonestly takes"],
        "reason_template": "Matches {incident_type} of movable property.",
    },
    {
        "document": "BNS",
        "section": "Section 309",
        "incident_types": ["robbery"],
        "keywords": ["rob", "robbed", "robbery", "force", "fear", "instant death"],
        "reason_template": "Matches {incident_type} involving force or threat.",
    },
    {
        "document": "BNS",
        "section": "Section 310",
        "incident_types": ["robbery", "dacoity"],
        "keywords": ["dacoity", "dacoits", "gang robbery", "five or more"],
        "reason_template": "Matches gang-related {incident_type} (dacoity).",
    },
    {
        "document": "BNS",
        "section": "Section 308",
        "incident_types": ["extortion"],
        "keywords": ["extort", "extorted", "extortion", "wrongful fear", "compel"],
        "reason_template": "Matches {incident_type} by inducing fear.",
    },
    # ------------------------------------------------------------------
    # House trespass / Breaking and entering
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 329",
        "incident_types": ["house trespass"],
        "keywords": ["trespass", "trespassing", "enter", "enters", "criminal trespass"],
        "reason_template": "Matches criminal trespass / {incident_type}.",
    },
    {
        "document": "BNS",
        "section": "Section 331",
        "incident_types": ["house trespass"],
        "keywords": ["house trespass", "break into", "broke into", "dwelling house"],
        "reason_template": "Matches {incident_type} of a dwelling house.",
    },
    {
        "document": "BNS",
        "section": "Section 332",
        "incident_types": ["house trespass"],
        "keywords": ["lurking", "lurk", "house breaking", "night", "housebreaking"],
        "reason_template": "Matches house-breaking by night for {incident_type}.",
    },
    # ------------------------------------------------------------------
    # Assault / Hurt / Grievous hurt
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 115",
        "incident_types": ["assault", "hurt"],
        "keywords": ["hurt", "bodily pain", "voluntarily causes hurt", "injury"],
        "reason_template": "Matches voluntary causing of hurt ({incident_type}).",
    },
    {
        "document": "BNS",
        "section": "Section 116",
        "incident_types": ["assault", "grievous hurt"],
        "keywords": ["grievous hurt", "fracture", "disfigure", "emasculation", "permanent"],
        "reason_template": "Matches grievous hurt ({incident_type}).",
    },
    {
        "document": "BNS",
        "section": "Section 131",
        "incident_types": ["assault"],
        "keywords": ["assault", "criminal force", "apprehension", "gesture"],
        "reason_template": "Matches assault / use of criminal force ({incident_type}).",
    },
    # ------------------------------------------------------------------
    # Murder / Culpable homicide
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 101",
        "incident_types": ["murder"],
        "keywords": ["murder", "kill", "killed", "killing", "death", "causes death"],
        "reason_template": "Matches {incident_type}.",
    },
    {
        "document": "BNS",
        "section": "Section 105",
        "incident_types": ["murder", "culpable homicide"],
        "keywords": ["culpable homicide", "not amounting to murder", "intention", "knowledge"],
        "reason_template": "Matches culpable homicide not amounting to murder ({incident_type}).",
    },
    # ------------------------------------------------------------------
    # Kidnapping / Abduction
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 137",
        "incident_types": ["kidnapping"],
        "keywords": ["kidnap", "kidnapped", "kidnapping", "minor", "lawful guardian"],
        "reason_template": "Matches {incident_type} from lawful guardianship.",
    },
    {
        "document": "BNS",
        "section": "Section 138",
        "incident_types": ["kidnapping"],
        "keywords": ["abduct", "abducted", "abduction", "compel", "induce"],
        "reason_template": "Matches abduction ({incident_type}).",
    },
    # ------------------------------------------------------------------
    # Cheating / Fraud / Forgery
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 318",
        "incident_types": ["cheating"],
        "keywords": ["cheat", "cheated", "cheating", "deceive", "fraud", "fraudulently"],
        "reason_template": "Matches {incident_type} by deception.",
    },
    {
        "document": "BNS",
        "section": "Section 319",
        "incident_types": ["cheating"],
        "keywords": ["cheating by personation", "impersonate", "personates"],
        "reason_template": "Matches cheating by personation ({incident_type}).",
    },
    {
        "document": "BNS",
        "section": "Section 336",
        "incident_types": ["forgery"],
        "keywords": ["forge", "forged", "forging", "forgery", "false document"],
        "reason_template": "Matches {incident_type} of a document.",
    },
    {
        "document": "BNS",
        "section": "Section 340",
        "incident_types": ["forgery"],
        "keywords": ["using forged document", "fraudulent use", "produces before"],
        "reason_template": "Matches using a forged document ({incident_type}).",
    },
    {
        "document": "BNS",
        "section": "Section 357",
        "incident_types": ["counterfeiting"],
        "keywords": ["counterfeit", "counterfeit coin", "fake currency", "currency notes"],
        "reason_template": "Matches {incident_type} of currency.",
    },
    # ------------------------------------------------------------------
    # Criminal intimidation / Threatening
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 351",
        "incident_types": ["criminal intimidation"],
        "keywords": ["threaten", "threatened", "threatening", "intimidate", "fear of injury"],
        "reason_template": "Matches {incident_type}.",
    },
    # ------------------------------------------------------------------
    # Arson / Mischief
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 324",
        "incident_types": ["mischief"],
        "keywords": ["mischief", "damage", "destroy", "wrongful loss", "diminish value"],
        "reason_template": "Matches {incident_type} causing wrongful loss.",
    },
    {
        "document": "BNS",
        "section": "Section 326",
        "incident_types": ["arson"],
        "keywords": ["fire", "arson", "burn", "burnt", "burned", "set fire"],
        "reason_template": "Matches {incident_type} (mischief by fire).",
    },
    # ------------------------------------------------------------------
    # Sexual offences
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 64",
        "incident_types": ["rape"],
        "keywords": ["rape", "sexual assault", "penetration", "without consent"],
        "reason_template": "Matches {incident_type}.",
    },
    {
        "document": "BNS",
        "section": "Section 74",
        "incident_types": ["assault"],
        "keywords": ["outrage modesty", "assault on woman", "indecent assault"],
        "reason_template": "Matches assault on woman with intent to outrage modesty.",
    },
    # ------------------------------------------------------------------
    # Cyber crime
    # ------------------------------------------------------------------
    {
        "document": "IT Act",
        "section": "Section 43",
        "incident_types": ["cyber crime"],
        "keywords": ["hack", "hacked", "unauthorized access", "computer", "data theft"],
        "reason_template": "Matches {incident_type} (unauthorized computer access).",
    },
    {
        "document": "IT Act",
        "section": "Section 66",
        "incident_types": ["cyber crime"],
        "keywords": ["cyber crime", "dishonestly", "fraudulently", "computer related offence"],
        "reason_template": "Matches {incident_type} under the IT Act.",
    },
    {
        "document": "IT Act",
        "section": "Section 66C",
        "incident_types": ["cyber crime", "identity theft"],
        "keywords": ["identity theft", "password", "electronic signature", "unique identification"],
        "reason_template": "Matches {incident_type} (identity theft).",
    },
    # ------------------------------------------------------------------
    # Bribery / Corruption
    # ------------------------------------------------------------------
    {
        "document": "Prevention of Corruption Act",
        "section": "Section 7",
        "incident_types": ["bribery"],
        "keywords": ["bribe", "bribed", "bribing", "gratification", "public servant"],
        "reason_template": "Matches {incident_type} of a public servant.",
    },
    # ------------------------------------------------------------------
    # Smuggling
    # ------------------------------------------------------------------
    {
        "document": "Customs Act",
        "section": "Section 135",
        "incident_types": ["smuggling"],
        "keywords": ["smuggle", "smuggled", "smuggling", "contraband", "import", "export"],
        "reason_template": "Matches {incident_type} of prohibited goods.",
    },
    # ------------------------------------------------------------------
    # Criminal breach of trust
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 316",
        "incident_types": ["criminal breach of trust"],
        "keywords": [
            "criminal breach of trust", "entrusted", "dishonestly misappropriates",
            "converts to own use",
        ],
        "reason_template": "Matches {incident_type}.",
    },
    # ------------------------------------------------------------------
    # Receiving stolen property
    # ------------------------------------------------------------------
    {
        "document": "BNS",
        "section": "Section 317",
        "incident_types": ["receiving stolen property"],
        "keywords": ["receive stolen", "stolen property", "dishonestly receives", "retains"],
        "reason_template": "Matches {incident_type}.",
    },
    # ------------------------------------------------------------------
    # Constitutional articles / Fundamental Rights
    # ------------------------------------------------------------------
    {
        "document": "Constitution",
        "section": "Article 14",
        "incident_types": ["article 14", "fundamental right", "fundamental rights", "right to equality"],
        "keywords": ["article 14", "equality", "equal protection", "before law", "equal protection of laws", "rule of law"],
        "reason_template": "Matches right to equality under {incident_type}.",
    },
    {
        "document": "Constitution",
        "section": "Article 19",
        "incident_types": ["article 19", "fundamental right", "fundamental rights", "freedom of speech", "freedom of expression"],
        "keywords": ["article 19", "freedom of speech", "freedom of expression", "free speech", "expression", "assemble"],
        "reason_template": "Matches freedom of speech and expression ({incident_type}).",
    },
    {
        "document": "Constitution",
        "section": "Article 21",
        "incident_types": ["article 21", "fundamental right", "fundamental rights", "right to life", "personal liberty"],
        "keywords": ["article 21", "life", "liberty", "personal liberty", "right to life", "protection of life", "due process"],
        "reason_template": "Matches protection of life and personal liberty ({incident_type}).",
    },
    {
        "document": "Constitution",
        "section": "Article 32",
        "incident_types": ["article 32", "fundamental right", "fundamental rights", "habeas corpus", "writ petition"],
        "keywords": ["article 32", "writ", "habeas corpus", "supreme court", "enforcement", "fundamental rights enforcement"],
        "reason_template": "Matches right to approach Supreme Court for {incident_type}.",
    },
    {
        "document": "Constitution",
        "section": "Article 226",
        "incident_types": ["article 226", "fundamental right", "writ petition"],
        "keywords": ["article 226", "high court", "writ", "mandamus", "certiorari", "prohibition"],
        "reason_template": "Matches High Court writ jurisdiction ({incident_type}).",
    },
    # ------------------------------------------------------------------
    # Consumer Rights
    # ------------------------------------------------------------------
    {
        "document": "Consumer Protection Act",
        "section": "Section 2",
        "incident_types": ["consumer rights"],
        "keywords": ["consumer", "consumer rights", "defective product", "deficient service", "unfair trade"],
        "reason_template": "Matches {incident_type} under the Consumer Protection Act.",
    },
    # ------------------------------------------------------------------
    # Anticipatory bail / BNSS
    # ------------------------------------------------------------------
    {
        "document": "BNSS",
        "section": "Section 482",
        "incident_types": ["anticipatory bail"],
        "keywords": ["anticipatory bail", "bail before arrest", "fear of arrest", "pre-arrest bail"],
        "reason_template": "Matches application for {incident_type}.",
    },
    {
        "document": "BNSS",
        "section": "Section 479",
        "incident_types": ["bail"],
        "keywords": ["bail", "release on bail", "bail application", "undertrail", "undertrial"],
        "reason_template": "Matches application for {incident_type}.",
    },
]

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Loguru once per process (idempotent).

    Safe to call from multiple agent constructors in the same process.
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
                log_dir / "section_mapper.log",
                level="DEBUG",
                rotation=getattr(_config, "LOG_ROTATION", "10 MB"),
                retention=getattr(_config, "LOG_RETENTION", "30 days"),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not set up file logging via config.LOG_DIR: {}", exc)

    _LOGGING_CONFIGURED = True


def _read_config_float(attr: str, default: float) -> float:
    """Safely read a float from config.py with a fallback default.

    Args:
        attr: Attribute name on the ``config`` module.
        default: Fallback value.

    Returns:
        The configured float, or ``default`` if absent/invalid.
    """
    if _config is None:
        return default
    value = getattr(_config, attr, default)
    try:
        value = float(value)
        if 0.0 <= value <= 1.0:
            return value
    except (TypeError, ValueError):
        pass
    logger.warning("config.{} is not a float in [0,1] ({}); using default {}.", attr, value, default)
    return default


def _read_config_int(attr: str, default: int) -> int:
    """Safely read a positive integer from config.py with a fallback default.

    Args:
        attr: Attribute name on the ``config`` module.
        default: Fallback value.

    Returns:
        The configured int, or ``default`` if absent/invalid.
    """
    if _config is None:
        return default
    value = getattr(_config, attr, default)
    if isinstance(value, int) and value > 0:
        return value
    logger.warning("config.{} is not a positive int ({}); using default {}.", attr, value, default)
    return default


def _tokenize(text: str) -> set[str]:
    """Lowercase word-token set for lexical overlap calculations.

    Args:
        text: Any string.

    Returns:
        Set of lowercase word tokens.
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Args:
        set_a: First token set.
        set_b: Second token set.

    Returns:
        Jaccard coefficient in ``[0, 1]``.
    """
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CandidateSection:
    """An intermediate scored section candidate.

    Attributes:
        document: Corpus name (e.g. ``"BNS"``).
        section: Section label (e.g. ``"Section 303"``).
        reason: Human-readable reason string.
        confidence: Composite confidence score in ``[0, 1]``.
        matched_incident_types: Incident types that triggered this candidate.
    """

    document: str
    section: str
    reason: str
    confidence: float
    matched_incident_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the required output schema dict.

        Returns:
            Dict with keys ``document``, ``section``, ``reason``,
            ``confidence``.
        """
        return {
            "document": self.document,
            "section": self.section,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class SectionMapper:
    """Maps retrieved legal evidence to applicable statutory provisions.

    No LLM is used. Scoring is a weighted combination of:

    * **Registry match** -- whether the section's incident types or keywords
      overlap with the extracted facts (the primary signal).
    * **Keyword overlap** -- Jaccard similarity between the section's
      configured keywords and the combined text of all selected quotes.
    * **Retrieval score** -- the pipeline's raw retrieval score for the
      chunk whose metadata matched this section (if available).
    * **Rerank score** -- the reranker's score for the same chunk.
    * **Quote relevance** -- Jaccard similarity between the section's
      keywords and every selected quote, averaged.

    All four component weights are configurable via ``config.py`` and must
    sum to ≤ 1.0; the registry-match component uses the remaining weight.

    Usage
    -----
    >>> mapper = SectionMapper()
    >>> result = mapper.map_sections(facts, selected_quotes)
    """

    def __init__(
        self,
        registry: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialise the mapper with configuration and registry.

        Args:
            registry: Optional override of ``SECTION_REGISTRY``.  Pass a
                custom list to inject test data or extend the built-in
                registry without editing the module.
        """
        _configure_logging()

        self._min_confidence: float = _read_config_float(
            "SECTION_MAPPER_MIN_CONFIDENCE", _DEFAULT_MIN_CONFIDENCE
        )
        self._max_results: int = _read_config_int(
            "SECTION_MAPPER_MAX_RESULTS", _DEFAULT_MAX_RESULTS
        )
        self._keyword_weight: float = _read_config_float(
            "SECTION_MAPPER_KEYWORD_WEIGHT", _DEFAULT_KEYWORD_WEIGHT
        )
        self._retrieval_weight: float = _read_config_float(
            "SECTION_MAPPER_RETRIEVAL_WEIGHT", _DEFAULT_RETRIEVAL_WEIGHT
        )
        self._rerank_weight: float = _read_config_float(
            "SECTION_MAPPER_RERANK_WEIGHT", _DEFAULT_RERANK_WEIGHT
        )
        self._quote_weight: float = _read_config_float(
            "SECTION_MAPPER_QUOTE_WEIGHT", _DEFAULT_QUOTE_WEIGHT
        )

        self._registry: list[dict[str, Any]] = (
            registry if registry is not None else SECTION_REGISTRY
        )

        logger.debug(
            "SectionMapper initialised: min_confidence={} max_results={} "
            "weights=(keyword={} retrieval={} rerank={} quote={})",
            self._min_confidence,
            self._max_results,
            self._keyword_weight,
            self._retrieval_weight,
            self._rerank_weight,
            self._quote_weight,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def map_sections(
        self,
        facts: dict[str, Any],
        selected_quotes: list[dict[str, Any]],
        retrieval_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Determine applicable legal provisions from facts and quotes.

        Production-grade grounding strategy (in priority order):

        1. **Metadata extraction** (PRIMARY): Read ``document``, ``section``,
           and ``article`` fields directly from retrieved chunk dicts.  These
           fields are populated by the chunker/indexer and are 100% grounded
           in the indexed corpus — no LLM inference involved.
        2. **Registry scoring** (FALLBACK): Used only when metadata extraction
           yields zero candidates (e.g. chunks lack explicit section labels).

        Args:
            facts: Structured dict from the Fact Extraction Agent.
            selected_quotes: Per-chunk dicts from the Quote Selector Agent.
            retrieval_results: Raw retrieval dicts from the pipeline, each
                containing ``document``, ``section``, ``article``,
                ``confidence``, and ``retrieval_text``.  When provided,
                metadata extraction is attempted first.

        Returns:
            A dict with key ``"applicable_sections"`` whose value is a list
            of section dicts (``document``, ``section``, ``reason``,
            ``confidence``), sorted descending by confidence and capped at
            ``MAX_RESULTS``.
        """
        start_time = time.perf_counter()

        if not isinstance(facts, dict):
            logger.warning("map_sections received non-dict facts; returning empty result.")
            return {"applicable_sections": []}

        incident_types: list[str] = facts.get("incident_type", []) or []
        fact_keywords: list[str] = facts.get("keywords", []) or []
        fact_actions: list[str] = facts.get("actions", []) or []

        logger.info(
            "Mapping sections for incident_types={} keywords={} actions={}",
            incident_types,
            fact_keywords,
            fact_actions,
        )

        # ── PRIMARY PATH: extract sections directly from chunk metadata ────
        metadata_candidates = self._extract_sections_from_metadata(
            retrieval_results or []
        )
        logger.info(
            "{} section(s) extracted directly from retrieval metadata.",
            len(metadata_candidates),
        )

        if metadata_candidates:
            # Metadata extraction succeeded — use these sections exclusively.
            # They are grounded in the indexed corpus; no registry guess-work needed.
            candidates = metadata_candidates
            logger.info(
                "Using metadata-extracted sections only (skipping SECTION_REGISTRY)."
            )
        else:
            # ── FALLBACK PATH: SECTION_REGISTRY scoring ────────────────────
            logger.info(
                "No section metadata in retrieved chunks — falling back to SECTION_REGISTRY."
            )
            quote_texts: list[str] = self._collect_quote_texts(selected_quotes)
            combined_quote_tokens: set[str] = _tokenize(" ".join(quote_texts))
            section_score_map: dict[str, dict[str, float]] = (
                self._build_section_score_map(selected_quotes)
            )
            fact_signal_tokens: set[str] = _tokenize(
                " ".join(incident_types + fact_keywords + fact_actions)
            )
            candidates = []
            for entry in self._registry:
                candidate = self._evaluate_entry(
                    entry=entry,
                    incident_types=incident_types,
                    fact_signal_tokens=fact_signal_tokens,
                    combined_quote_tokens=combined_quote_tokens,
                    quote_texts=quote_texts,
                    section_score_map=section_score_map,
                )
                if candidate is not None:
                    candidates.append(candidate)

        logger.debug("{} candidate section(s) before dedup/threshold.", len(candidates))

        candidates = self.remove_duplicates(candidates)
        ranked = self.rank_sections(candidates)

        # Apply confidence threshold and result cap.
        filtered = [c for c in ranked if c.confidence >= self._min_confidence]
        capped = filtered[: self._max_results]

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Section mapping complete in {:.4f}s -- {} applicable section(s) returned.",
            elapsed,
            len(capped),
        )

        return {"applicable_sections": [c.to_dict() for c in capped]}

    def _extract_sections_from_metadata(
        self,
        retrieval_results: list[dict[str, Any]],
    ) -> list[CandidateSection]:
        """Extract applicable sections directly from retrieved chunk metadata.

        This is the production-grade primary path.  Every retrieved chunk
        already carries ``document``, ``section``, and ``article`` fields
        populated at index time — no LLM reasoning is needed.

        Deduplication is by ``(document, section)`` key; the highest-
        confidence chunk for each unique section is kept.

        Args:
            retrieval_results: List of retrieval result dicts, each
                expected to contain ``document`` (str), ``section`` (str),
                ``article`` (str), and ``confidence`` (float).

        Returns:
            List of :class:`CandidateSection` objects, one per unique
            ``(document, section)`` pair found in the chunk metadata.
        """
        if not retrieval_results:
            return []

        best: dict[str, CandidateSection] = {}

        for chunk in retrieval_results:
            if not isinstance(chunk, dict):
                continue

            doc = str(chunk.get("document") or "").strip()
            section = str(chunk.get("section") or "").strip()
            article = str(chunk.get("article") or "").strip()
            chunk_confidence = float(chunk.get("confidence") or chunk.get("rerank_score") or 0.5)

            # Constitution chunks store article number (e.g. "21") separately;
            # normalise to section label "Article 21".
            if article and (not section or section.lower() in ("", "none")):
                section = f"Article {article}"
                if not doc or doc.upper() in ("", "NONE"):
                    doc = "Constitution"

            if not doc or not section:
                continue

            # Clamp to [0.70, 0.98] — metadata-extracted sections are always
            # highly reliable, so we set a generous floor.
            confidence = max(0.70, min(0.98, chunk_confidence + 0.10))

            key = f"{doc}|{section}".lower()
            existing = best.get(key)
            if existing is None or confidence > existing.confidence:
                best[key] = CandidateSection(
                    document=doc,
                    section=section,
                    reason=(
                        f"Section {section} of {doc} extracted directly from "
                        "retrieved chunk metadata — fully grounded in indexed corpus."
                    ),
                    confidence=confidence,
                    matched_incident_types=[],
                )

        return list(best.values())

    # ------------------------------------------------------------------
    # Registry evaluation
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self,
        entry: dict[str, Any],
        incident_types: list[str],
        fact_signal_tokens: set[str],
        combined_quote_tokens: set[str],
        quote_texts: list[str],
        section_score_map: dict[str, dict[str, float]],
    ) -> CandidateSection | None:
        """Evaluate a single registry entry and produce a scored candidate.

        Returns ``None`` if the entry has no match signal at all (i.e. the
        registry-match component is zero AND the keyword-overlap with the
        fact signals is zero).

        Args:
            entry: One record from ``SECTION_REGISTRY``.
            incident_types: Incident types from the classifier.
            fact_signal_tokens: Combined token set of incident types +
                keywords + actions.
            combined_quote_tokens: Combined token set of all selected quotes.
            quote_texts: Individual quote strings for per-quote scoring.
            section_score_map: ``{section_key: {"retrieval": float,
                "rerank": float}}`` built from quote chunk metadata.

        Returns:
            A :class:`CandidateSection` or ``None``.
        """
        doc = entry.get("document", "")
        section = entry.get("section", "")
        entry_incident_types: list[str] = entry.get("incident_types", [])
        entry_keywords: list[str] = entry.get("keywords", [])
        reason_template: str = entry.get("reason_template", "Matches retrieved evidence.")

        registry_score, matched_types = self._registry_match_score(
            entry_incident_types, entry_keywords, incident_types, fact_signal_tokens
        )

        # Early exit: no signal at all from this entry.
        if registry_score == 0.0:
            return None

        confidence = self.calculate_confidence(
            registry_score=registry_score,
            entry_keywords=entry_keywords,
            combined_quote_tokens=combined_quote_tokens,
            quote_texts=quote_texts,
            section_key=f"{doc}|{section}",
            section_score_map=section_score_map,
        )

        # Build reason string.
        first_match = matched_types[0] if matched_types else (entry_incident_types[0] if entry_incident_types else "incident")
        reason = reason_template.format(incident_type=first_match)

        return CandidateSection(
            document=doc,
            section=section,
            reason=reason,
            confidence=confidence,
            matched_incident_types=matched_types,
        )

    def _registry_match_score(
        self,
        entry_incident_types: list[str],
        entry_keywords: list[str],
        query_incident_types: list[str],
        fact_signal_tokens: set[str],
    ) -> tuple[float, list[str]]:
        """Compute the registry-match component of the confidence score.

        Two sub-signals are combined:

        * **Incident-type hit** (weight 0.60 of this component): at least
          one incident type from the registry entry matches the classifier
          output.
        * **Keyword overlap** (weight 0.40 of this component): Jaccard
          overlap between the entry's keyword list and the combined fact
          signal tokens.

        Args:
            entry_incident_types: Incident types listed in the registry entry.
            entry_keywords: Keywords listed in the registry entry.
            query_incident_types: Incident types from the classifier.
            fact_signal_tokens: Token set of all fact signals.

        Returns:
            A 2-tuple of ``(registry_score, matched_incident_types)``.
        """
        query_types_lower = {t.lower() for t in query_incident_types}
        matched = [
            t for t in entry_incident_types if t.lower() in query_types_lower
        ]
        type_hit = 1.0 if matched else 0.0

        entry_kw_tokens = _tokenize(" ".join(entry_keywords))
        kw_overlap = _jaccard(entry_kw_tokens, fact_signal_tokens)

        registry_score = 0.60 * type_hit + 0.40 * kw_overlap
        return registry_score, matched

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        registry_score: float,
        entry_keywords: list[str],
        combined_quote_tokens: set[str],
        quote_texts: list[str],
        section_key: str,
        section_score_map: dict[str, dict[str, float]],
    ) -> float:
        """Compute a composite confidence score for one candidate section.

        The four configurable weights (``SECTION_MAPPER_KEYWORD_WEIGHT``,
        ``SECTION_MAPPER_RETRIEVAL_WEIGHT``, ``SECTION_MAPPER_RERANK_WEIGHT``,
        ``SECTION_MAPPER_QUOTE_WEIGHT``) determine how much each signal
        contributes.  The remaining weight (1 − sum of the four) is
        assigned to the registry-match component so total weights always
        equal 1.0.

        Args:
            registry_score: Output of :meth:`_registry_match_score`.
            entry_keywords: Keywords from the registry entry.
            combined_quote_tokens: Token set of all selected quotes.
            quote_texts: Individual quote strings.
            section_key: ``"document|section"`` key into ``section_score_map``.
            section_score_map: Retrieval / rerank scores by section key.

        Returns:
            A confidence score in ``[0, 1]``.
        """
        configured_weights_sum = (
            self._keyword_weight
            + self._retrieval_weight
            + self._rerank_weight
            + self._quote_weight
        )
        registry_weight = max(0.0, 1.0 - configured_weights_sum)

        # 1. Registry match component.
        registry_component = registry_weight * registry_score

        # 2. Keyword overlap with combined quotes.
        kw_tokens = _tokenize(" ".join(entry_keywords))
        keyword_overlap = _jaccard(kw_tokens, combined_quote_tokens)
        keyword_component = self._keyword_weight * keyword_overlap

        # 3. Retrieval score from chunk metadata.
        scores = section_score_map.get(section_key, {})
        retrieval_score = min(float(scores.get("retrieval", 0.0)), 1.0)
        retrieval_component = self._retrieval_weight * retrieval_score

        # 4. Rerank score from chunk metadata.
        rerank_score = min(float(scores.get("rerank", 0.0)), 1.0)
        rerank_component = self._rerank_weight * rerank_score

        # 5. Average per-quote relevance.
        quote_relevance = self.score_match(kw_tokens, quote_texts)
        quote_component = self._quote_weight * quote_relevance

        confidence = (
            registry_component
            + keyword_component
            + retrieval_component
            + rerank_component
            + quote_component
        )
        return round(min(max(confidence, 0.0), 1.0), 4)

    def score_match(
        self,
        entry_kw_tokens: set[str],
        quote_texts: list[str],
    ) -> float:
        """Compute average Jaccard relevance between entry keywords and quotes.

        Args:
            entry_kw_tokens: Pre-tokenised keyword set for the registry entry.
            quote_texts: Individual selected quote strings.

        Returns:
            Mean Jaccard similarity in ``[0, 1]``; returns 0.0 for empty input.
        """
        if not entry_kw_tokens or not quote_texts:
            return 0.0
        scores = [_jaccard(entry_kw_tokens, _tokenize(qt)) for qt in quote_texts]
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Ranking & deduplication
    # ------------------------------------------------------------------

    def rank_sections(
        self,
        candidates: list[CandidateSection],
    ) -> list[CandidateSection]:
        """Sort candidates descending by confidence.

        Args:
            candidates: Unordered list of :class:`CandidateSection` objects.

        Returns:
            A new list sorted highest-confidence first.
        """
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)

    def remove_duplicates(
        self,
        candidates: list[CandidateSection],
    ) -> list[CandidateSection]:
        """Remove duplicate sections, keeping the highest-confidence instance.

        Two candidates are considered duplicates if they share the same
        ``(document, section)`` pair.

        Args:
            candidates: Possibly-duplicate candidate list.

        Returns:
            A deduplicated list.  When duplicates exist, the one with the
            higher confidence score is retained.
        """
        seen: dict[tuple[str, str], CandidateSection] = {}
        for candidate in candidates:
            key = (candidate.document, candidate.section)
            if key not in seen or candidate.confidence > seen[key].confidence:
                seen[key] = candidate
        return list(seen.values())

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_quote_texts(selected_quotes: list[dict[str, Any]]) -> list[str]:
        """Flatten all selected quote strings from the Quote Selector output.

        Args:
            selected_quotes: List of per-chunk dicts from the Quote Selector.

        Returns:
            A flat list of individual quote strings.
        """
        texts: list[str] = []
        for chunk in selected_quotes:
            if not isinstance(chunk, dict):
                continue
            for quote in chunk.get("selected_quotes", []):
                if isinstance(quote, str) and quote.strip():
                    texts.append(quote.strip())
        return texts

    @staticmethod
    def _build_section_score_map(
        selected_quotes: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Build a ``{document|section: {retrieval, rerank}}`` lookup.

        When the same section appears in multiple chunks, the highest
        retrieval and rerank scores are retained.

        Args:
            selected_quotes: List of per-chunk dicts from the Quote Selector.
                Each dict may contain ``document``, ``section``,
                ``rerank_score``, and/or ``retrieval_score`` / ``score``.

        Returns:
            A dict keyed by ``"document|section"`` with sub-dict
            ``{"retrieval": float, "rerank": float}``.
        """
        score_map: dict[str, dict[str, float]] = {}
        for chunk in selected_quotes:
            if not isinstance(chunk, dict):
                continue
            doc = chunk.get("document", "")
            section = chunk.get("section", "")
            if not doc or not section:
                continue
            key = f"{doc}|{section}"
            retrieval = float(
                chunk.get("retrieval_score", chunk.get("score", 0.0)) or 0.0
            )
            rerank = float(chunk.get("rerank_score", 0.0) or 0.0)
            if key not in score_map:
                score_map[key] = {"retrieval": retrieval, "rerank": rerank}
            else:
                score_map[key]["retrieval"] = max(score_map[key]["retrieval"], retrieval)
                score_map[key]["rerank"] = max(score_map[key]["rerank"], rerank)
        return score_map


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

_SAMPLE_FACTS_THEFT: Final[dict[str, Any]] = {
    "incident_type": ["theft", "house trespass"],
    "entities": ["phone", "house", "last night"],
    "actions": ["steal", "break into"],
    "victims": [],
    "locations": ["house"],
    "objects": ["phone"],
    "search_queries": ["theft", "house trespass", "steal phone"],
    "keywords": ["theft", "house trespass", "movable property"],
    "confidence": 0.88,
}

_SAMPLE_FACTS_ASSAULT: Final[dict[str, Any]] = {
    "incident_type": ["assault", "murder"],
    "entities": ["knife", "Mohan"],
    "actions": ["attack", "murder"],
    "victims": ["Mohan"],
    "locations": [],
    "objects": ["knife"],
    "search_queries": ["assault knife", "murder"],
    "keywords": ["assault", "murder", "grievous hurt"],
    "confidence": 0.91,
}

_SAMPLE_QUOTES_THEFT: Final[list[dict[str, Any]]] = [
    {
        "chunk_id": "bns-303-001",
        "document": "BNS",
        "section": "Section 303",
        "selected_quotes": [
            "Whoever commits theft shall be punished with rigorous imprisonment.",
            "Explanation: A person who dishonestly takes any moveable property commits theft.",
        ],
        "quote_count": 2,
        "rerank_score": 0.97,
        "retrieval_score": 0.88,
        "payload": {},
    },
    {
        "chunk_id": "bns-331-001",
        "document": "BNS",
        "section": "Section 331",
        "selected_quotes": [
            "Whoever commits house trespass shall be liable to fine or imprisonment.",
            "Breaking into a dwelling house with intent to commit an offence therein.",
        ],
        "quote_count": 2,
        "rerank_score": 0.91,
        "retrieval_score": 0.82,
        "payload": {},
    },
]

_SAMPLE_QUOTES_ASSAULT: Final[list[dict[str, Any]]] = [
    {
        "chunk_id": "bns-115-001",
        "document": "BNS",
        "section": "Section 115",
        "selected_quotes": [
            "Whoever voluntarily causes hurt shall be punished with imprisonment.",
        ],
        "quote_count": 1,
        "rerank_score": 0.89,
        "retrieval_score": 0.80,
        "payload": {},
    },
    {
        "chunk_id": "bns-101-001",
        "document": "BNS",
        "section": "Section 101",
        "selected_quotes": [
            "Whoever commits murder shall be punished with death or imprisonment for life.",
        ],
        "quote_count": 1,
        "rerank_score": 0.95,
        "retrieval_score": 0.92,
        "payload": {},
    },
]


def _run_test_mode() -> None:
    """Run the SectionMapper against sample theft and assault queries."""
    mapper = SectionMapper()

    scenarios: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = [
        ("Theft + House Trespass", _SAMPLE_FACTS_THEFT, _SAMPLE_QUOTES_THEFT),
        ("Assault + Murder", _SAMPLE_FACTS_ASSAULT, _SAMPLE_QUOTES_ASSAULT),
    ]

    for label, facts, quotes in scenarios:
        print(f"\n{'=' * 60}")
        print(f"  Scenario: {label}")
        print(f"{'=' * 60}")
        result = mapper.map_sections(facts, quotes)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _run_test_mode()