"""
src/agents/classifier.py
-------------------------
Deterministic Fact Extraction Agent for the Indian Legal RAG system.

Despite the filename, this is NOT a machine-learning classifier and
trains nothing. It is the first node in the LangGraph multi-agent
workflow: it converts a user's free-text legal incident description into
structured search information for the retrieval pipeline that follows.

Workflow position
------------------
    User Query
        -> Fact Extraction Agent (this module)
        -> LegalRAGPipeline
        -> Quote Selector
        -> Section Mapper
        -> Remedy Advisor
        -> Citation Validator
        -> Synthesizer

Responsibilities
-----------------
* Extract typed entities (locations, objects, weapons, victims, etc.).
* Extract recognized action verbs/phrases.
* Map actions (and direct legal-keyword mentions) to incident types.
* Expand the incident into multiple diverse retrieval queries.
* Assemble a legal-keyword list and a deterministic confidence score.

This module MUST NOT retrieve documents, call any LLM, call Qdrant,
perform legal reasoning, determine guilt, or identify exact statutory
sections -- it only converts free text into structured search
information for later stages to act on.

Technique: lightweight, fully deterministic NLP -- regex, configurable
keyword/phrase dictionaries, and phrase matching, with an OPTIONAL spaCy
NER enhancement for person-name detection (tried lazily; the agent is
fully functional without spaCy installed). No LLM is used anywhere in
this module.

Backward compatibility
-----------------------
The previous placeholder exposed a module-level ``classify(query) ->
dict`` function for ``src/orchestrator.py`` to call. That function is
preserved below as a thin wrapper around ``FactExtractionAgent.extract``,
so existing call sites keep working -- but it now returns the full
structured schema described in ``FactExtractionAgent.extract`` rather
than the old placeholder's ``{"category": ...}`` shape. If
``orchestrator.py`` pattern-matches on the old shape, it will need a
small update to consume ``incident_type`` / ``search_queries`` etc.
instead; the function name and single-string argument are unchanged.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from loguru import logger

try:
    import config  # type: ignore[import-not-found]
except ImportError:
    config = None  # This agent is self-contained; config.py is optional.

# ---------------------------------------------------------------------------
# Entity type constants
# ---------------------------------------------------------------------------

_LOCATION: Final[str] = "LOCATION"
_PROPERTY: Final[str] = "PROPERTY"
_WEAPON: Final[str] = "WEAPON"
_VEHICLE: Final[str] = "VEHICLE"
_MONEY: Final[str] = "MONEY"
_DOCUMENT: Final[str] = "DOCUMENT"
_ANIMAL: Final[str] = "ANIMAL"
_ORGANIZATION: Final[str] = "ORGANIZATION"
_GOVERNMENT_OFFICE: Final[str] = "GOVERNMENT_OFFICE"
_COURT: Final[str] = "COURT"
_POLICE: Final[str] = "POLICE"
_VICTIM: Final[str] = "VICTIM"
_ACCUSED: Final[str] = "ACCUSED"
_PERSON: Final[str] = "PERSON"

#: Entity types that represent a place -> feed ``extract_locations``.
_LOCATION_TYPES: Final[frozenset[str]] = frozenset({_LOCATION, _COURT, _POLICE, _GOVERNMENT_OFFICE})
#: Entity types that represent a tangible thing -> feed ``extract_objects``.
_OBJECT_TYPES: Final[frozenset[str]] = frozenset(
    {_PROPERTY, _WEAPON, _VEHICLE, _MONEY, _DOCUMENT, _ANIMAL, _ORGANIZATION}
)

# ---------------------------------------------------------------------------
# Configurable dictionaries -- entities, actions, incident types, keywords
# ---------------------------------------------------------------------------
#
# Every behavioural decision this agent makes is driven by a lookup into
# one of these dictionaries, never by an "if word == ..." chain. Adding a
# new synonym, action, or incident type means editing data here, not
# control flow below.

_ENTITY_LEXICON: Final[dict[str, tuple[str, ...]]] = {
    _LOCATION: (
        "house", "home", "flat", "apartment", "shop", "store", "market", "street",
        "road", "highway", "school", "college", "university", "office", "factory",
        "warehouse", "field", "farm", "village", "city", "town", "border", "airport",
        "railway station", "bus stand", "park", "forest", "temple", "mosque", "church",
        "hotel", "restaurant", "bar", "atm", "parking lot", "platform", "bridge",
    ),
    _COURT: ("court", "tribunal", "magistrate court", "high court", "supreme court"),
    _POLICE: ("police station", "police", "constable", "inspector", "fir"),
    _GOVERNMENT_OFFICE: (
        "government office", "municipal office", "collector office", "tehsil",
        "panchayat office", "ministry", "passport office", "rto",
    ),
    _PROPERTY: (
        "phone", "mobile", "mobile phone", "smartphone", "laptop", "computer",
        "tablet", "jewelry", "jewellery", "gold", "necklace", "ring", "earrings",
        "wallet", "purse", "bag", "luggage", "television", "tv", "camera",
        "watch", "furniture", "land", "plot", "crop",
    ),
    _WEAPON: (
        "knife", "gun", "pistol", "rifle", "sword", "axe", "iron rod", "rod",
        "stick", "acid", "explosive", "bomb", "blade", "machete", "hammer", "chain",
    ),
    _VEHICLE: (
        "car", "bike", "motorcycle", "motorbike", "scooter", "scooty", "truck",
        "auto rickshaw", "auto", "rickshaw", "bus", "van", "tractor", "bicycle", "cycle",
    ),
    _DOCUMENT: (
        "signature", "passport", "aadhaar card", "aadhaar", "pan card", "cheque",
        "will", "contract", "agreement", "certificate", "license", "licence",
        "deed", "stamp paper", "voter id", "id card",
    ),
    _ANIMAL: ("dog", "cow", "cattle", "buffalo", "goat", "horse", "pet", "livestock"),
    _ORGANIZATION: ("company", "firm", "bank", "ngo", "trust", "society", "cooperative"),
}

#: Additional standalone keyword fallback for money mentions that aren't
#: a numeric amount (the numeric case is handled by ``_MONEY_RE``).
_MONEY_KEYWORDS: Final[tuple[str, ...]] = ("cash", "money", "amount")

_MONEY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\u20b9|rs\.?|inr|rupees?)\s?[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s?(?:rupees?|rs\.?|inr)\b",
    re.IGNORECASE,
)

_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|yesterday|today|tonight"
    r"|last\s+(?:night|week|month|year)"
    r"|this\s+(?:morning|afternoon|evening|week)"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{2,4})?)\b",
    re.IGNORECASE,
)
_TIME_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)|morning|afternoon|evening|night|midnight|noon|dawn|dusk)\b",
    re.IGNORECASE,
)

#: Surface forms (synonyms/inflections) recognized for each canonical
#: action. Matching any surface form yields the canonical label.
_ACTION_SURFACE_FORMS: Final[dict[str, tuple[str, ...]]] = {
    "steal": ("steal", "stole", "stolen", "stealing", "theft of"),
    "rob": ("rob", "robbed", "robbing", "robbery of", "held up"),
    "break into": ("break into", "broke into", "breaking into", "broken into", "forced entry into"),
    "hit": ("hit", "hits", "hitting", "beat", "beaten", "beating"),
    "attack": ("attack", "attacked", "attacking"),
    "murder": ("murder", "murdered", "murdering", "killed", "killing"),
    "kidnap": ("kidnap", "kidnapped", "kidnapping", "abduct", "abducted", "abduction"),
    "cheat": ("cheat", "cheated", "cheating", "defraud", "defrauded", "fraud"),
    "forge": ("forge", "forged", "forging", "forgery"),
    "threaten": ("threaten", "threatened", "threatening", "intimidate", "intimidated", "intimidation"),
    "extort": ("extort", "extorted", "extorting", "extortion"),
    "assault": ("assault", "assaulted", "assaulting"),
    "rape": ("rape", "raped", "sexual assault", "sexually assaulted"),
    "burn": ("burn", "burned", "burnt", "burning", "set fire", "arson"),
    "damage": ("damage", "damaged", "damaging", "vandalize", "vandalized", "vandalism"),
    "hack": ("hack", "hacked", "hacking", "cyber attack", "unauthorized access"),
    "blackmail": ("blackmail", "blackmailed", "blackmailing"),
    "smuggle": ("smuggle", "smuggled", "smuggling"),
    "bribe": ("bribe", "bribed", "bribing", "bribery"),
    "counterfeit": ("counterfeit", "counterfeited", "counterfeiting", "fake currency", "fake documents"),
}

#: Maps each canonical action to the incident type(s) it implies.
_ACTION_TO_INCIDENT_TYPE: Final[dict[str, tuple[str, ...]]] = {
    "steal": ("theft",),
    "rob": ("robbery",),
    "break into": ("house trespass",),
    "hit": ("assault",),
    "attack": ("assault",),
    "murder": ("murder",),
    "kidnap": ("kidnapping",),
    "cheat": ("cheating",),
    "forge": ("forgery",),
    "threaten": ("criminal intimidation",),
    "extort": ("extortion",),
    "assault": ("assault",),
    "rape": ("rape",),
    "burn": ("arson",),
    "damage": ("mischief",),
    "hack": ("cyber crime",),
    "blackmail": ("criminal intimidation", "extortion"),
    "smuggle": ("smuggling",),
    "bribe": ("bribery",),
    "counterfeit": ("counterfeiting",),
}

#: A short, compact noun form per action, used to build alternate
#: "<location> <noun>" style query phrasing in ``expand_queries``.
_ACTION_COMPACT_NOUNS: Final[dict[str, str]] = {
    "steal": "theft",
    "rob": "robbery",
    "break into": "breaking",
    "hit": "assault",
    "attack": "attack",
    "murder": "murder",
    "kidnap": "kidnapping",
    "cheat": "fraud",
    "forge": "forgery",
    "threaten": "intimidation",
    "extort": "extortion",
    "assault": "assault",
    "rape": "assault",
    "burn": "arson",
    "damage": "vandalism",
    "hack": "hacking",
    "blackmail": "blackmail",
    "smuggle": "smuggling",
    "bribe": "bribery",
    "counterfeit": "counterfeiting",
}

#: Master legal-concept vocabulary. Serves double duty: (1) any phrase
#: mentioned verbatim in the query is treated as a directly-named
#: incident type even with no matching action verb, and (2) the same
#: vocabulary populates the final ``keywords`` output field. Extend via
#: the ``extra_keywords`` constructor argument rather than editing this
#: tuple directly, to keep the agent configurable without code changes.
_LEGAL_KEYWORDS: Final[tuple[str, ...]] = (
    "theft", "robbery", "house trespass", "criminal intimidation", "assault",
    "kidnapping", "forgery", "counterfeiting", "cheating", "extortion", "murder",
    "culpable homicide", "hurt", "grievous hurt", "criminal breach of trust",
    "receiving stolen property", "cyber crime", "identity theft", "arson",
    "mischief", "bribery", "smuggling", "rape",
    "fundamental right", "fundamental rights", "right to life", "personal liberty",
    "right to equality", "freedom of speech", "freedom of expression",
    "article 14", "article 19", "article 21", "article 32", "article 226",
    "habeas corpus", "writ petition", "consumer rights", "anticipatory bail",
)

#: Context cues used to decide whether a detected person mention is the
#: accused or the victim of the incident.
_ACCUSED_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(accused|suspect|attacker|perpetrator|culprit|offender)\b", re.IGNORECASE
)
_VICTIM_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:was|were|got)\s+(?:attacked|robbed|killed|murdered|injured|hurt|beaten|stabbed|"
    r"kidnapped|abducted|cheated|defrauded|threatened|raped|assaulted|extorted|blackmailed)\b",
    re.IGNORECASE,
)

#: Capitalized English words that are *not* person names, used to filter
#: the regex-based person-detection fallback (used when spaCy is
#: unavailable). Deliberately conservative: false negatives (missing a
#: real name) are preferred over false positives here, since this
#: agent's output feeds further automated reasoning.
_NON_NAME_CAPITALIZED_WORDS: Final[frozenset[str]] = frozenset(
    {
        "someone", "somebody", "anyone", "anybody", "everyone", "nobody",
        "i", "my", "me", "the", "a", "an", "he", "she", "they", "we", "it",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "police", "court", "fir", "section", "article", "today", "yesterday",
        "last", "this", "constitution",
    }
)
_CAPITALIZED_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z][a-z]+\b")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

_LOGGING_CONFIGURED: bool = False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Loguru once per process (idempotent across instances).

    Multiple agents (this one, the retriever, the indexer, ...) may be
    instantiated within the same LangGraph process; without the
    idempotency guard, each instantiation would call ``logger.remove()``
    and re-add handlers, churning file handles and duplicating output.
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

    if config is not None:
        try:
            log_dir = config.LOG_DIR
            log_dir.mkdir(exist_ok=True)
            logger.add(
                log_dir / "classifier.log",
                level="DEBUG",
                rotation=getattr(config, "LOG_ROTATION", "10 MB"),
                retention=getattr(config, "LOG_RETENTION", "30 days"),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 -- file logging is a nice-to-have
            logger.debug("Could not set up file logging via config.LOG_DIR: {}", exc)

    _LOGGING_CONFIGURED = True


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for matching purposes.

    Args:
        text: Raw input text.

    Returns:
        The normalized text.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _looks_non_english(text: str) -> bool:
    """Heuristically flag text that is unlikely to be English.

    Not a real language detector -- only checks whether a large share of
    alphabetic characters fall outside the basic Latin range, which
    catches Devanagari/Arabic/Tamil/etc. script so a warning can be
    logged. Rule-based English dictionaries will simply fail to match
    such text; this only affects logging, never control flow.

    Args:
        text: Raw input text.

    Returns:
        ``True`` if more than 30% of alphabetic characters are non-Latin.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for ch in letters if ord(ch) > 0x024F)
    return (non_latin / len(letters)) > 0.3


def _compile_lexicon(lexicon: dict[str, tuple[str, ...]]) -> dict[str, re.Pattern[str]]:
    """Compile a ``{entity_type: phrases}`` dict into per-type regexes.

    Phrases within each type are sorted longest-first so multi-word
    phrases (e.g. "police station") win over shorter ones that happen to
    be substrings (e.g. "police"), when both could otherwise match.

    Args:
        lexicon: Mapping of entity type to a tuple of phrases.

    Returns:
        Mapping of entity type to a compiled, case-insensitive
        alternation pattern.
    """
    compiled: dict[str, re.Pattern[str]] = {}
    for entity_type, phrases in lexicon.items():
        ordered = sorted(set(phrases), key=len, reverse=True)
        escaped = [re.escape(phrase) for phrase in ordered]
        compiled[entity_type] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return compiled


def _compile_phrase_dict(phrases: Iterable[str]) -> dict[str, re.Pattern[str]]:
    """Compile a flat iterable of phrases into one regex per phrase.

    Args:
        phrases: Phrases to compile (e.g. the legal-keyword vocabulary).

    Returns:
        Mapping of phrase to its compiled, case-insensitive,
        word-bounded pattern.
    """
    return {phrase: re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE) for phrase in phrases}


def _build_action_patterns() -> dict[str, re.Pattern[str]]:
    """Compile each canonical action's surface forms into one regex.

    Returns:
        Mapping of canonical action label to its compiled pattern.
    """
    patterns: dict[str, re.Pattern[str]] = {}
    for canonical, surface_forms in _ACTION_SURFACE_FORMS.items():
        ordered = sorted(set(surface_forms), key=len, reverse=True)
        escaped = [re.escape(form) for form in ordered]
        patterns[canonical] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return patterns


def _regex_person_candidates(raw_query: str) -> list[tuple[str, int, int]]:
    """Fallback person-name detection used when spaCy is unavailable.

    Treats any capitalized word not in ``_NON_NAME_CAPITALIZED_WORDS`` as
    a candidate proper noun. A heuristic, not true NER.

    Args:
        raw_query: The original, case-preserved query text.

    Returns:
        A list of ``(matched_text, start_index, end_index)`` tuples.
    """
    candidates: list[tuple[str, int, int]] = []
    for match in _CAPITALIZED_WORD_RE.finditer(raw_query):
        word = match.group(0)
        if word.lower() in _NON_NAME_CAPITALIZED_WORDS:
            continue
        candidates.append((word, match.start(), match.end()))
    return candidates


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedEntity:
    """A single typed entity span found in the query.

    Attributes:
        text: The matched surface text, as it appeared in the source
            (lowercased for dictionary-matched entities; original case
            preserved for person names and money amounts).
        entity_type: One of the entity-type constants, e.g. ``"WEAPON"``.
    """

    text: str
    entity_type: str


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class FactExtractionAgent:
    """Deterministic, rule-based fact-extraction agent.

    Converts a free-text legal incident description into the structured
    schema consumed by the retrieval pipeline: incident types, typed
    entities, actions, victims, locations, objects, expanded search
    queries, legal keywords, and a confidence score.

    No machine-learning model and no LLM is used. Every decision is
    driven by configurable dictionaries (``_ENTITY_LEXICON``,
    ``_ACTION_SURFACE_FORMS``, ``_ACTION_TO_INCIDENT_TYPE``,
    ``_LEGAL_KEYWORDS``) compiled once per instance, plus an optional
    spaCy NER enhancement for person-name detection that is tried lazily
    and falls back to a conservative regex heuristic if spaCy (or its
    English model) is not installed.

    Usage
    -----
    >>> agent = FactExtractionAgent()
    >>> result = agent.extract("Someone broke into my house and stole my phone.")
    """

    def __init__(self, extra_keywords: Sequence[str] | None = None) -> None:
        """Build all compiled-regex resources once for this instance.

        Args:
            extra_keywords: Optional additional legal-keyword phrases to
                merge with the built-in ``_LEGAL_KEYWORDS`` vocabulary,
                without editing module-level constants. Each phrase
                participates in both direct incident-type detection and
                the final ``keywords`` output field.
        """
        _configure_logging()

        self._action_patterns = _build_action_patterns()
        self._entity_patterns = _compile_lexicon(_ENTITY_LEXICON)
        self._money_keyword_patterns = _compile_phrase_dict(_MONEY_KEYWORDS)

        merged_keywords: dict[str, None] = dict.fromkeys(_LEGAL_KEYWORDS)
        if extra_keywords:
            merged_keywords.update(dict.fromkeys(extra_keywords))
        self._legal_keyword_patterns = _compile_phrase_dict(merged_keywords.keys())

        self._spacy_nlp: Any = None
        self._spacy_unavailable: bool = False

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def extract_entities(self, raw_query: str, normalized_query: str | None = None) -> list[ExtractedEntity]:
        """Extract every typed entity mentioned in the query.

        Combines dictionary/phrase matching (locations, property,
        weapons, vehicles, documents, animals, organizations,
        institutions), numeric money-amount detection, and person
        detection (victim/accused/unclear) in one pass.

        Args:
            raw_query: The original, case-preserved query text.
            normalized_query: Optional pre-computed
                ``_normalize_text(raw_query)``; computed if not given.

        Returns:
            A list of ``ExtractedEntity``, in no particular dedup order
            (callers needing a flat, deduplicated view should use
            :meth:`extract_locations`, :meth:`extract_objects`, or the
            agent's internal flattening in :meth:`extract`).
        """
        if normalized_query is None:
            normalized_query = _normalize_text(raw_query)

        entities: list[ExtractedEntity] = []

        for entity_type, pattern in self._entity_patterns.items():
            for match in pattern.finditer(normalized_query):
                entities.append(ExtractedEntity(text=match.group(0), entity_type=entity_type))

        for match in _MONEY_RE.finditer(raw_query):
            entities.append(ExtractedEntity(text=match.group(0).strip(), entity_type=_MONEY))
        for _, pattern in self._money_keyword_patterns.items():
            for match in pattern.finditer(normalized_query):
                entities.append(ExtractedEntity(text=match.group(0), entity_type=_MONEY))

        entities.extend(self._extract_persons(raw_query))

        return entities

    def _extract_persons(self, raw_query: str) -> list[ExtractedEntity]:
        """Detect person mentions and classify each as victim/accused/unclear.

        Tries spaCy NER first if available; otherwise uses the
        conservative capitalized-word regex fallback. Either way, role
        classification uses the same nearby-context regex heuristics,
        since spaCy's generic PERSON label does not itself distinguish
        victims from the accused.

        Args:
            raw_query: The original, case-preserved query text.

        Returns:
            A deduplicated list of classified person ``ExtractedEntity``
            objects (entity_type one of ``VICTIM``, ``ACCUSED``,
            ``PERSON``).
        """
        candidates = self._spacy_person_names(raw_query)
        if candidates is None:
            candidates = _regex_person_candidates(raw_query)

        classified: list[ExtractedEntity] = []
        seen: set[str] = set()
        for name, start, end in candidates:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            window = raw_query[max(0, start - 40): min(len(raw_query), end + 40)]
            if _ACCUSED_CONTEXT_RE.search(window):
                classified.append(ExtractedEntity(text=name, entity_type=_ACCUSED))
            elif _VICTIM_CONTEXT_RE.search(window):
                classified.append(ExtractedEntity(text=name, entity_type=_VICTIM))
            else:
                classified.append(ExtractedEntity(text=name, entity_type=_PERSON))
        return classified

    def _spacy_person_names(self, raw_query: str) -> list[tuple[str, int, int]] | None:
        """Attempt spaCy-based PERSON entity detection.

        Lazily imports and loads spaCy's small English model on first
        use, caching the result (success or failure) on the instance so
        the (potentially slow) import/load is attempted at most once.

        Args:
            raw_query: The original, case-preserved query text.

        Returns:
            A list of ``(text, start_char, end_char)`` tuples for each
            spaCy PERSON entity, or ``None`` if spaCy / its English model
            is unavailable, or NER otherwise fails -- signalling the
            caller to use the regex fallback instead.
        """
        if self._spacy_unavailable:
            return None

        if self._spacy_nlp is None:
            try:
                import spacy
            except ImportError:
                logger.debug("spaCy not installed -- using regex-based person detection.")
                self._spacy_unavailable = True
                return None
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except Exception as exc:  # noqa: BLE001 -- any model-load failure falls back
                logger.debug(
                    "spaCy English model unavailable ({}) -- using regex-based person detection.", exc
                )
                self._spacy_unavailable = True
                return None

        try:
            doc = self._spacy_nlp(raw_query)
        except Exception as exc:  # noqa: BLE001 -- never let NER crash extraction
            logger.warning("spaCy NER failed on query -- falling back to regex: {}", exc)
            return None

        return [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]

    @staticmethod
    def _unique_texts(entities: list[ExtractedEntity], allowed_types: frozenset[str]) -> list[str]:
        """Filter entities by type and return deduplicated surface text.

        Args:
            entities: Entities to filter.
            allowed_types: Entity types to keep.

        Returns:
            Deduplicated text values, in first-seen order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for entity in entities:
            key = entity.text.lower()
            if entity.entity_type in allowed_types and key not in seen:
                seen.add(key)
                result.append(entity.text)
        return result

    def extract_locations(self, entities: list[ExtractedEntity]) -> list[str]:
        """Filter entities down to place-like types.

        Args:
            entities: Output of :meth:`extract_entities`.

        Returns:
            Deduplicated location surface strings, in first-seen order.
        """
        return self._unique_texts(entities, _LOCATION_TYPES)

    def extract_objects(self, entities: list[ExtractedEntity]) -> list[str]:
        """Filter entities down to tangible-thing types.

        Args:
            entities: Output of :meth:`extract_entities`.

        Returns:
            Deduplicated object surface strings, in first-seen order.
        """
        return self._unique_texts(entities, _OBJECT_TYPES)

    def _extract_victims(self, entities: list[ExtractedEntity]) -> list[str]:
        """Filter entities down to those classified as victims.

        Args:
            entities: Output of :meth:`extract_entities`.

        Returns:
            Deduplicated victim names, in first-seen order. Note this is
            deliberately conservative -- an implicit first-person victim
            (e.g. "my house") is never inferred as a named victim; only
            explicit person mentions in a victim context are included.
        """
        return self._unique_texts(entities, frozenset({_VICTIM}))

    def extract_dates(self, raw_query: str) -> list[str]:
        """Extract date and time mentions (absolute or relative).

        Args:
            raw_query: The original, case-preserved query text.

        Returns:
            Deduplicated date/time surface strings, in first-seen order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for pattern in (_DATE_RE, _TIME_RE):
            for match in pattern.finditer(raw_query):
                text = match.group(0)
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    result.append(text)
        return result

    # ------------------------------------------------------------------
    # Actions & incident types
    # ------------------------------------------------------------------

    def extract_actions(self, normalized_query: str) -> list[str]:
        """Detect recognized action verbs/phrases in the query.

        Args:
            normalized_query: Lowercased, whitespace-normalized query.

        Returns:
            Canonical action labels (e.g. ``"steal"``, ``"break into"``),
            deduplicated and ordered by first occurrence in the text.
        """
        found: list[tuple[int, str]] = []
        for canonical, pattern in self._action_patterns.items():
            match = pattern.search(normalized_query)
            if match:
                found.append((match.start(), canonical))
        found.sort(key=lambda pair: pair[0])
        return [canonical for _, canonical in found]

    def detect_incident_types(self, actions: list[str], normalized_query: str) -> list[str]:
        """Map detected actions, and any directly-named legal concepts, to incident types.

        Two detection paths are merged: (1) every recognized action's
        configured incident type(s), via ``_ACTION_TO_INCIDENT_TYPE``,
        and (2) any legal-keyword phrase mentioned verbatim in the query
        with no corresponding action verb (e.g. a user typing "criminal
        breach of trust" directly).

        Args:
            actions: Output of :meth:`extract_actions`.
            normalized_query: Lowercased, whitespace-normalized query.

        Returns:
            Deduplicated incident type labels, in first-seen order
            (action-derived types first, then directly-named ones).
        """
        incident_types: list[str] = []
        seen: set[str] = set()

        for action in actions:
            for incident_type in _ACTION_TO_INCIDENT_TYPE.get(action, ()):
                if incident_type not in seen:
                    seen.add(incident_type)
                    incident_types.append(incident_type)

        for keyword, pattern in self._legal_keyword_patterns.items():
            if keyword not in seen and pattern.search(normalized_query):
                seen.add(keyword)
                incident_types.append(keyword)

        return incident_types

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_queries(
        self,
        incident_types: list[str],
        actions: list[str],
        locations: list[str],
        objects_: list[str],
    ) -> list[str]:
        """Generate diverse retrieval queries from the extracted facts.

        This is the most important method in the agent: retrieval
        quality downstream depends on having multiple differently
        phrased queries rather than one, since legal text uses varied
        phrasing for the same offence (e.g. "house trespass" vs.
        "breaking into a house"). Each combination of incident type,
        location, object, and action contributes a candidate phrasing;
        duplicates are dropped.

        Args:
            incident_types: Output of :meth:`detect_incident_types`.
            actions: Output of :meth:`extract_actions`.
            locations: Output of :meth:`extract_locations`.
            objects_: Output of :meth:`extract_objects`.

        Returns:
            A deduplicated, order-preserving list of candidate search
            query strings, each intended to be sent independently to the
            retriever.
        """
        queries: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            normalized = " ".join(candidate.split()).strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                queries.append(normalized)

        for incident_type in incident_types:
            add(incident_type)

        for location in locations:
            for incident_type in incident_types:
                add(f"{location} {incident_type}")
            for action in actions:
                add(f"{action} {location}")
                compact_noun = _ACTION_COMPACT_NOUNS.get(action)
                if compact_noun:
                    add(f"{location} {compact_noun}")

        for obj in objects_:
            for incident_type in incident_types:
                add(f"{incident_type} of {obj}")
            for action in actions:
                add(f"{action} {obj}")

        if len(incident_types) >= 2:
            add(" ".join(incident_types))

        return queries

    # ------------------------------------------------------------------
    # Keywords & confidence
    # ------------------------------------------------------------------

    def _collect_keywords(self, incident_types: list[str], normalized_query: str) -> list[str]:
        """Assemble the final legal-keyword list for retrieval expansion.

        Args:
            incident_types: Output of :meth:`detect_incident_types`.
            normalized_query: Lowercased, whitespace-normalized query.

        Returns:
            Every detected incident type plus any configured legal
            keyword phrase mentioned verbatim in the query, deduplicated
            and order-preserving.
        """
        keywords: list[str] = []
        seen: set[str] = set()

        for incident_type in incident_types:
            key = incident_type.lower()
            if key not in seen:
                seen.add(key)
                keywords.append(incident_type)

        for keyword, pattern in self._legal_keyword_patterns.items():
            if keyword not in seen and pattern.search(normalized_query):
                seen.add(keyword)
                keywords.append(keyword)

        return keywords

    def calculate_confidence(
        self,
        normalized_query: str,
        actions: list[str],
        incident_types: list[str],
        entities: list[str],
    ) -> float:
        """Compute a deterministic confidence score for one extraction.

        A weighted combination of four independent signals: whether any
        action was recognized (0.30), whether any incident type was
        resolved (0.30), how many distinct entities were found (up to
        0.20, partial credit capped at 3 entities), and the query's
        length (up to 0.20, partial credit capped at 8 words) -- a
        one-word query is inherently less reliable to extract from than
        a full sentence, regardless of what was found in it.

        Args:
            normalized_query: Lowercased, whitespace-normalized query.
            actions: Output of :meth:`extract_actions`.
            incident_types: Output of :meth:`detect_incident_types`.
            entities: The flattened entity list.

        Returns:
            A confidence score in ``[0, 1]``, rounded to two decimals.
        """
        action_signal = 0.30 if actions else 0.0
        incident_signal = 0.30 if incident_types else 0.0
        entity_signal = 0.20 * min(len(entities) / 3.0, 1.0)
        word_count = len(normalized_query.split())
        length_signal = 0.20 * min(word_count / 8.0, 1.0)

        confidence = action_signal + incident_signal + entity_signal + length_signal
        return round(min(max(confidence, 0.0), 1.0), 2)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Build the zero-signal result returned for empty/failed input."""
        return {
            "incident_type": [],
            "entities": [],
            "actions": [],
            "victims": [],
            "locations": [],
            "objects": [],
            "search_queries": [],
            "keywords": [],
            "confidence": 0.0,
        }

    @staticmethod
    def _flatten_entities(entities: list[ExtractedEntity], dates: list[str]) -> list[str]:
        """Flatten typed entities and date/time mentions into one list.

        Args:
            entities: Output of :meth:`extract_entities`.
            dates: Output of :meth:`extract_dates`.

        Returns:
            A deduplicated, order-preserving flat list of every
            extracted surface string (locations, objects, persons,
            dates/times alike).
        """
        seen: set[str] = set()
        flat: list[str] = []
        for entity in entities:
            key = entity.text.lower()
            if key not in seen:
                seen.add(key)
                flat.append(entity.text)
        for date_text in dates:
            key = date_text.lower()
            if key not in seen:
                seen.add(key)
                flat.append(date_text)
        return flat

    def extract(self, query: str) -> dict[str, Any]:
        """Extract structured facts from a raw legal incident description.

        Never raises -- empty input, very short input, likely-non-English
        input, and any unexpected internal error all degrade to a
        logged warning/error plus a safe, zero-signal result, since this
        agent sits at the front of an automated multi-agent workflow
        that must never crash on user input.

        Args:
            query: The raw user incident description, e.g. "Someone
                broke into my house and stole my phone."

        Returns:
            A dict with keys ``incident_type``, ``entities``, ``actions``,
            ``victims``, ``locations``, ``objects``, ``search_queries``,
            ``keywords``, and ``confidence`` (float in ``[0, 1]``).
        """
        start_time = time.perf_counter()

        if not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query received -- returning empty extraction result.")
            return self._empty_result()

        normalized_query = _normalize_text(query)
        word_count = len(normalized_query.split())
        if word_count < 3:
            logger.warning(
                "Very short query ({} word(s)): '{}' -- extraction may be unreliable.", word_count, query
            )
        if _looks_non_english(query):
            logger.warning(
                "Query may not be in English: '{}' -- rule-based extraction may be unreliable.", query
            )

        try:
            actions = self.extract_actions(normalized_query)
            typed_entities = self.extract_entities(query, normalized_query)
            locations = self.extract_locations(typed_entities)
            objects_ = self.extract_objects(typed_entities)
            victims = self._extract_victims(typed_entities)
            dates = self.extract_dates(query)
            incident_types = self.detect_incident_types(actions, normalized_query)
            entities_flat = self._flatten_entities(typed_entities, dates)
            search_queries = self.expand_queries(incident_types, actions, locations, objects_)
            keywords = self._collect_keywords(incident_types, normalized_query)
            confidence = self.calculate_confidence(normalized_query, actions, incident_types, entities_flat)
        except Exception as exc:  # noqa: BLE001 -- this agent must never crash the workflow
            logger.exception("Unexpected error during fact extraction: {}", exc)
            return self._empty_result()

        if not entities_flat:
            logger.warning("No entities found for query: '{}'", query)

        result: dict[str, Any] = {
            "incident_type": incident_types,
            "entities": entities_flat,
            "actions": actions,
            "victims": victims,
            "locations": locations,
            "objects": objects_,
            "search_queries": search_queries,
            "keywords": keywords,
            "confidence": confidence,
        }

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Extraction complete in {:.4f}s — incident_types={} actions={} entities={} "
            "queries={} confidence={}",
            elapsed, incident_types, actions, entities_flat, len(search_queries), confidence,
        )
        return result


# ---------------------------------------------------------------------------
# Backward-compatible functional entry point
# ---------------------------------------------------------------------------

#: A single shared agent instance, since building it has a small fixed
#: cost (compiling the regex dictionaries) that should not be repeated
#: per call. Safe to reuse across calls -- the agent holds no per-query
#: mutable state between invocations.
_DEFAULT_AGENT: Final[FactExtractionAgent] = FactExtractionAgent()


def classify(query: str) -> dict[str, Any]:
    """Backward-compatible functional entry point for orchestrator.py.

    Wraps :class:`FactExtractionAgent` behind the original placeholder's
    function signature so existing callers do not need to change their
    import or call site. See the module docstring's "Backward
    compatibility" section for what changed in the return shape.

    Args:
        query: The raw user incident description.

    Returns:
        The structured extraction dict produced by
        :meth:`FactExtractionAgent.extract`.
    """
    return _DEFAULT_AGENT.extract(query)


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------


def _run_test_mode() -> None:
    """Run the agent against a fixed set of example incidents and print JSON."""
    test_queries = [
        "Someone broke into my house and stole my phone.",
        "Mohan was attacked with a knife.",
        "My bike was stolen.",
        "Someone forged my signature.",
    ]

    agent = FactExtractionAgent()
    for test_query in test_queries:
        print(f"\nQuery: {test_query}")
        result = agent.extract(test_query)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _run_test_mode()