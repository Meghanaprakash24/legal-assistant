"""
src/agents/remedy_advisor.py
-----------------------------
Remedy Advisor Agent for the Indian Legal RAG system.

This module is the fifth node in the LangGraph multi-agent workflow: it
receives the mapped applicable sections from the Section Mapper Agent and
returns structured, general legal procedural guidance.

Workflow position
------------------
    User Query
        -> Fact Extraction Agent
        -> LegalRAGPipeline
        -> Quote Selector
        -> Section Mapper
        -> Remedy Advisor (this module)
        -> Citation Validator
        -> Synthesizer

Responsibilities
-----------------
* Maintain configurable mappings from incident types to recommended
  actions, required documents, and standard procedures.
* Aggregate and deduplicate guidance across multiple incident types.
* Return a structured output dict for downstream agents.

This module MUST NOT
--------------------
* Call any LLM.
* Determine guilt or predict court outcomes.
* Provide personalised legal advice.
* Recommend specific lawyers.
* Estimate punishment or sentence length.
* Replace a qualified legal professional.

Configuration
-------------
Reads the following attributes from ``config.py`` (all optional; safe
defaults apply when ``config`` is absent or an attribute is missing):

* ``REMEDY_ADVISOR_MAX_ACTIONS``    -- int, default 20
* ``REMEDY_ADVISOR_MAX_DOCUMENTS``  -- int, default 15
* ``REMEDY_ADVISOR_MAX_NOTES``      -- int, default 10

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import sys
import time
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

_DEFAULT_MAX_ACTIONS: Final[int] = 20
_DEFAULT_MAX_DOCUMENTS: Final[int] = 15
_DEFAULT_MAX_NOTES: Final[int] = 10

_LOGGING_CONFIGURED: bool = False

# ---------------------------------------------------------------------------
# Remedy mappings
# ---------------------------------------------------------------------------
#
# Each entry is keyed by a normalised incident-type label (lowercase).
# Values are dicts with:
#   "actions"   -- list[str]  recommended immediate actions
#   "documents" -- list[str]  documents required
#   "procedure" -- list[str]  general procedural steps (overrides default)
#   "notes"     -- list[str]  important domain-specific notes
#
# The GENERAL_PROCEDURE and IMPORTANT_NOTES lists are always appended
# after merging incident-specific entries.

REMEDY_MAPPINGS: Final[dict[str, dict[str, list[str]]]] = {
    # ------------------------------------------------------------------
    # Property offences
    # ------------------------------------------------------------------
    "theft": {
        "actions": [
            "File an FIR at the nearest police station immediately.",
            "Preserve any physical or digital evidence of the theft.",
            "Record witness information if any persons observed the incident.",
            "Prepare a list of stolen items with descriptions and estimated values.",
            "Cooperate fully with the investigating officer.",
        ],
        "documents": [
            "Identity proof (Aadhaar, Passport, or Voter ID)",
            "Proof of ownership of stolen property (receipts, invoices, photographs)",
            "List of stolen items with descriptions",
            "Witness details and contact information (if any)",
            "Any available CCTV footage or photographs",
        ],
        "procedure": [],
        "notes": [
            "Retain all receipts and invoices related to stolen goods as ownership evidence.",
        ],
    },
    "robbery": {
        "actions": [
            "Contact the police immediately by calling 100 or visiting the nearest station.",
            "Seek immediate medical attention if injured during the robbery.",
            "Do not disturb the crime scene until police arrive.",
            "Note the description, direction of escape, and any vehicle details of the accused.",
            "Preserve any physical evidence such as discarded items.",
        ],
        "documents": [
            "Identity proof",
            "Medical examination report (if injured)",
            "List of stolen items and estimated values",
            "Witness details (if any)",
            "CCTV footage or photographs (if available)",
        ],
        "procedure": [],
        "notes": [
            "Robbery involves force or threat; immediate police reporting is essential.",
            "Seek medical care even for minor injuries and obtain documentation.",
        ],
    },
    "dacoity": {
        "actions": [
            "Report to the police immediately; dacoity is a serious cognizable offence.",
            "Seek medical attention for all injured persons.",
            "Preserve the crime scene until police arrive.",
            "Compile a list of persons who were present and can serve as witnesses.",
            "Note descriptions of all accused and direction of flight.",
        ],
        "documents": [
            "Identity proof",
            "Medical examination reports for all injured persons",
            "List of stolen items and values",
            "Witness statements",
            "CCTV footage or any other recordings",
        ],
        "procedure": [],
        "notes": [
            "Dacoity involves five or more persons; mention the number of accused in the FIR.",
        ],
    },
    "extortion": {
        "actions": [
            "Report the extortion to the police immediately.",
            "Preserve all communications — messages, emails, call recordings — as evidence.",
            "Do not comply with demands before consulting law enforcement.",
            "Note the identity or contact details of the accused if known.",
        ],
        "documents": [
            "Identity proof",
            "Screenshots or printouts of threatening messages or emails",
            "Call recordings or logs (if available)",
            "Bank statements if any payment was made under duress",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [
            "Do not delete any threatening communications; they constitute crucial evidence.",
        ],
    },
    "house trespass": {
        "actions": [
            "File an FIR at the nearest police station.",
            "Do not touch or disturb items at the site of trespass.",
            "Document the scene with photographs or video immediately.",
            "Note the time, date, and description of the accused.",
            "Collect contact details of any witnesses.",
        ],
        "documents": [
            "Identity proof",
            "Proof of ownership or tenancy of the property",
            "Photographs or video of the trespass scene",
            "Witness details (if any)",
            "Any prior trespass complaints or notices served",
        ],
        "procedure": [],
        "notes": [
            "Retain property documents such as sale deed, rent agreement, or utility bills.",
        ],
    },
    "mischief": {
        "actions": [
            "File an FIR or a written complaint at the police station.",
            "Photograph or video-record the damaged property before any repairs.",
            "Obtain a damage assessment or repair estimate from a qualified professional.",
            "Note the identity of the accused or witnesses if available.",
        ],
        "documents": [
            "Identity proof",
            "Photographs of damaged property",
            "Damage assessment report or repair estimate",
            "Proof of ownership of damaged property",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [],
    },
    "arson": {
        "actions": [
            "Ensure safety of all persons and call fire services immediately.",
            "Call the police and file an FIR.",
            "Preserve the fire scene for forensic investigation.",
            "Seek medical attention for any burn injuries.",
            "Document the extent of damage with photographs.",
        ],
        "documents": [
            "Identity proof",
            "Medical reports for any injuries",
            "Photographs of fire damage",
            "Fire department report",
            "Proof of ownership of property",
            "Insurance documents (if applicable)",
        ],
        "procedure": [],
        "notes": [
            "Do not disturb the origin point of the fire; forensic examination is essential.",
        ],
    },
    "criminal breach of trust": {
        "actions": [
            "File an FIR or a written complaint with supporting documents.",
            "Preserve all agreements, contracts, and correspondence as evidence.",
            "Prepare an account of property entrusted and the alleged misappropriation.",
        ],
        "documents": [
            "Identity proof",
            "Trust deed, power of attorney, or agreement under which property was entrusted",
            "Correspondence evidencing the breach",
            "Bank statements or financial records",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [],
    },
    "receiving stolen property": {
        "actions": [
            "Report the matter to the police with details of the property.",
            "Do not further transfer or conceal the property.",
            "Preserve any transaction records, receipts, or communications.",
        ],
        "documents": [
            "Identity proof",
            "Transaction records or receipts",
            "Communications with the person from whom property was received",
            "Proof of purchase price and circumstances of acquisition",
        ],
        "procedure": [],
        "notes": [],
    },

    # ------------------------------------------------------------------
    # Violent offences
    # ------------------------------------------------------------------
    "assault": {
        "actions": [
            "Seek immediate medical examination and treatment.",
            "File an FIR at the nearest police station.",
            "Photograph injuries before treatment where medically safe to do so.",
            "Identify and record witness information.",
            "Preserve any clothing or objects with evidence of the assault.",
        ],
        "documents": [
            "Identity proof",
            "Medical examination report (MLC — Medico-Legal Case)",
            "Photographs of injuries",
            "Witness details",
            "Any CCTV footage or recordings",
        ],
        "procedure": [],
        "notes": [
            "Request a Medico-Legal Case (MLC) at the hospital; it is admissible as evidence.",
        ],
    },
    "hurt": {
        "actions": [
            "Obtain immediate medical treatment.",
            "File an FIR or a written complaint.",
            "Preserve the MLC report and all medical records.",
            "Record witness information.",
        ],
        "documents": [
            "Identity proof",
            "Medical examination report (MLC)",
            "Photographs of injuries",
            "Witness details",
        ],
        "procedure": [],
        "notes": [
            "Request a Medico-Legal Case (MLC) at the hospital.",
        ],
    },
    "grievous hurt": {
        "actions": [
            "Obtain immediate medical treatment; this is a serious offence.",
            "File an FIR immediately; grievous hurt is a cognizable offence.",
            "Preserve all medical records including X-rays and specialist reports.",
            "Do not delay reporting as evidence may degrade over time.",
        ],
        "documents": [
            "Identity proof",
            "Medical examination report (MLC)",
            "Specialist reports (orthopaedic, ophthalmological, etc. as applicable)",
            "Photographs of injuries",
            "Witness details",
        ],
        "procedure": [],
        "notes": [
            "Grievous hurt includes permanent disfigurement, fracture, or loss of organ function.",
        ],
    },
    "murder": {
        "actions": [
            "Call emergency services (112) and the police (100) immediately.",
            "Do not touch, move, or disturb the body or crime scene.",
            "Secure the area to prevent unauthorised access.",
            "Note details of any witnesses and preserve their contact information.",
            "Cooperate fully with the investigating officer and forensic team.",
        ],
        "documents": [
            "Identity proof of complainant",
            "Post-mortem report (arranged by police/magistrate)",
            "Witness statements",
            "Any CCTV footage or photographic evidence",
            "Crime scene preservation records",
        ],
        "procedure": [],
        "notes": [
            "Murder is cognizable and non-bailable; police must register an FIR.",
            "Do not attempt to clean or disturb the crime scene before forensic examination.",
        ],
    },
    "culpable homicide": {
        "actions": [
            "Report to the police immediately.",
            "Preserve the scene until law enforcement arrives.",
            "Ensure post-mortem examination is conducted through official channels.",
            "Collect witness information.",
        ],
        "documents": [
            "Identity proof",
            "Post-mortem report",
            "Witness statements",
            "Any available evidence of the incident",
        ],
        "procedure": [],
        "notes": [],
    },
    "kidnapping": {
        "actions": [
            "Contact the police immediately; do not delay in kidnapping cases.",
            "Preserve all communications from the accused (messages, calls, ransom notes).",
            "Do not negotiate with kidnappers without police guidance.",
            "Provide the police with recent photographs and description of the missing person.",
            "Contact the missing person's mobile operator to assist in tracing.",
        ],
        "documents": [
            "Identity proof of complainant",
            "Recent photographs of the missing person",
            "Contact details and description of the missing person",
            "Any ransom communications or notes",
            "Details of last known location and time",
        ],
        "procedure": [],
        "notes": [
            "Kidnapping of a minor is a cognizable offence; police must act immediately.",
            "Do not pay ransom without informing and coordinating with police.",
        ],
    },
    "rape": {
        "actions": [
            "Seek immediate medical attention at the nearest hospital.",
            "File an FIR at the nearest police station or Women's Help Desk.",
            "Do not bathe, change clothes, or wash before medical examination to preserve forensic evidence.",
            "Request that the medical examination be conducted by a female doctor where possible.",
            "Contact the National Commission for Women helpline: 7827170170.",
        ],
        "documents": [
            "Identity proof",
            "Medical examination report (forensic evidence collection)",
            "Clothing worn at the time of the incident (preserved in a paper bag)",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [
            "The victim has the right to record the statement before a female magistrate.",
            "The identity of the victim is protected by law and must not be disclosed.",
            "Free legal aid is available to the victim through the District Legal Services Authority.",
        ],
    },
    "criminal intimidation": {
        "actions": [
            "File a written complaint or FIR at the police station.",
            "Preserve all threatening communications as evidence.",
            "Inform trusted family members or colleagues of the threats received.",
        ],
        "documents": [
            "Identity proof",
            "Screenshots or printouts of threatening messages",
            "Call recordings or logs",
            "Witness details",
        ],
        "procedure": [],
        "notes": [],
    },

    # ------------------------------------------------------------------
    # Cyber crimes
    # ------------------------------------------------------------------
    "cyber crime": {
        "actions": [
            "Report the offence at the National Cyber Crime Reporting Portal: cybercrime.gov.in.",
            "File an FIR at the nearest police station or Cyber Crime Cell.",
            "Preserve screenshots, URLs, emails, and all digital evidence immediately.",
            "Do not delete or overwrite any relevant data, logs, or communications.",
            "Contact your bank immediately if financial fraud is involved.",
            "Change passwords and secure affected accounts.",
        ],
        "documents": [
            "Identity proof",
            "Screenshots of fraudulent communications or transactions",
            "Bank statements showing unauthorised transactions (if applicable)",
            "Email headers or message metadata",
            "Device details: make, model, and IMEI (for mobile devices)",
            "URLs and IP addresses involved (if known)",
        ],
        "procedure": [],
        "notes": [
            "Report financial cyber fraud immediately to the bank and call 1930 (Cyber Fraud Helpline).",
            "Do not click any links or respond to the accused after the incident.",
            "Preserve device logs before performing any factory reset.",
        ],
    },
    "identity theft": {
        "actions": [
            "Report to the Cyber Crime Cell and file an FIR.",
            "Report at cybercrime.gov.in immediately.",
            "Notify your bank and financial institutions to freeze accounts if required.",
            "Change all passwords and enable two-factor authentication on affected accounts.",
            "Preserve all evidence of misuse of your identity.",
        ],
        "documents": [
            "Identity proof",
            "Evidence of misuse (screenshots, transaction records, communications)",
            "Bank statements showing unauthorised activity",
            "Correspondence with service providers regarding the identity misuse",
        ],
        "procedure": [],
        "notes": [
            "Request a credit freeze from credit bureaus to prevent further financial fraud.",
        ],
    },

    # ------------------------------------------------------------------
    # Financial crimes
    # ------------------------------------------------------------------
    "cheating": {
        "actions": [
            "File an FIR at the nearest police station.",
            "Preserve all contracts, agreements, and correspondence with the accused.",
            "Compile a clear timeline of events and financial transactions.",
            "Contact your bank if cheating involved financial transactions.",
        ],
        "documents": [
            "Identity proof",
            "Contracts, agreements, or memoranda of understanding",
            "Financial transaction records and bank statements",
            "Communications (emails, messages, letters) with the accused",
            "Receipts or invoices involved in the fraud",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [],
    },
    "bribery": {
        "actions": [
            "Report to the Anti-Corruption Bureau (ACB) or Central Vigilance Commission (CVC).",
            "Do not pay a bribe; consider reporting before payment with police assistance.",
            "Preserve evidence of the demand — recordings, messages, or witnesses.",
            "File a complaint under the Prevention of Corruption Act.",
        ],
        "documents": [
            "Identity proof",
            "Evidence of demand for bribe (recordings, messages)",
            "Witness details",
            "Details of the public servant involved: name, designation, department",
        ],
        "procedure": [],
        "notes": [
            "ACB can assist in laying a trap to catch the accused in the act of accepting the bribe.",
            "Reporting bribery is protected; complainants should not fear legal repercussions.",
        ],
    },
    "counterfeiting": {
        "actions": [
            "Report to the police immediately; do not circulate the counterfeit currency.",
            "Surrender the counterfeit notes to the police or a bank.",
            "Preserve all counterfeit items as evidence.",
            "Note the source from whom the counterfeit currency was received.",
        ],
        "documents": [
            "Identity proof",
            "Counterfeit currency or items (surrender to police)",
            "Statement regarding the circumstances of receipt",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [
            "Knowingly circulating counterfeit currency is an offence; surrender it immediately.",
        ],
    },
    "smuggling": {
        "actions": [
            "Report to Customs authorities or the Directorate of Revenue Intelligence (DRI).",
            "Do not handle or move the smuggled goods.",
            "Preserve evidence of the smuggling operation.",
        ],
        "documents": [
            "Identity proof",
            "Evidence of smuggling activity",
            "Details of persons involved and routes used (if known)",
            "Any documents related to the goods",
        ],
        "procedure": [],
        "notes": [
            "Customs authorities have independent jurisdiction; coordinate with them alongside police.",
        ],
    },

    # ------------------------------------------------------------------
    # Forgery / Document fraud
    # ------------------------------------------------------------------
    "forgery": {
        "actions": [
            "File an FIR at the nearest police station.",
            "Preserve the forged document and the original (if available) as evidence.",
            "Report to the relevant authority whose documents have been forged (bank, government office, court).",
            "Do not use or further circulate the forged document.",
        ],
        "documents": [
            "Identity proof",
            "Original document (if available)",
            "Forged document",
            "Evidence establishing that the document is forged (handwriting reports, digital metadata)",
            "Witness details (if any)",
        ],
        "procedure": [],
        "notes": [
            "A forensic handwriting or document examination may be required; cooperate with investigators.",
        ],
    },

    # ------------------------------------------------------------------
    # Women and child offences
    # ------------------------------------------------------------------
    "harassment": {
        "actions": [
            "File a complaint with the police or Women's Help Desk.",
            "Report workplace harassment to the Internal Complaints Committee (ICC) under the POSH Act.",
            "Preserve all evidence of harassment: messages, emails, photographs.",
            "Contact the women's helpline: 181.",
        ],
        "documents": [
            "Identity proof",
            "Evidence of harassment (screenshots, recordings, witness statements)",
            "Details of the accused: name, designation, and contact information",
            "Medical or psychological assessment report (if applicable)",
        ],
        "procedure": [],
        "notes": [
            "The identity of the complainant in sexual harassment cases is protected by law.",
            "A complaint to the ICC must be filed within three months of the incident.",
        ],
    },

    # ------------------------------------------------------------------
    # Traffic offences
    # ------------------------------------------------------------------
    "traffic offence": {
        "actions": [
            "Report the accident to the nearest police station or traffic police.",
            "Seek immediate medical attention for injured persons.",
            "Do not move vehicles involved in a fatal accident before police arrive.",
            "Note the vehicle registration number, driver details, and witnesses.",
            "File an insurance claim with the relevant insurer.",
        ],
        "documents": [
            "Identity proof",
            "Driving licence",
            "Vehicle registration certificate (RC)",
            "Insurance documents",
            "Medical reports for injured persons",
            "Witness details",
            "Photographs of the accident scene",
        ],
        "procedure": [],
        "notes": [
            "Good Samaritans who assist accident victims are protected under Motor Vehicles Act provisions.",
        ],
    },

    # ------------------------------------------------------------------
    # Evidence-related offences
    # ------------------------------------------------------------------
    "evidence tampering": {
        "actions": [
            "Report the tampering to the investigating officer or court immediately.",
            "Preserve any undisturbed evidence and document what has been tampered.",
            "File a complaint if a public servant is involved in the tampering.",
        ],
        "documents": [
            "Identity proof",
            "Documentation of original evidence and evidence of tampering",
            "Witness details",
            "Any recordings or photographs showing tampering",
        ],
        "procedure": [],
        "notes": [
            "Evidence tampering is a serious offence that can prejudice the outcome of a case.",
        ],
    },
}

# ---------------------------------------------------------------------------
# General procedure applied to all incidents
# ---------------------------------------------------------------------------

GENERAL_PROCEDURE: Final[list[str]] = [
    "Incident is reported to the police.",
    "Police register an FIR if the offence is cognizable.",
    "For non-cognizable offences, a complaint may be filed before a Magistrate.",
    "Investigation begins: scene visit, evidence collection, witness statements.",
    "Police may arrest the accused if warranted.",
    "A charge sheet (final report) may be filed before the appropriate court.",
    "Court takes cognizance and issues summons or warrant as applicable.",
    "Trial proceeds: charges framed, evidence recorded, arguments heard.",
    "Court delivers judgment.",
]

# ---------------------------------------------------------------------------
# Important notes appended to every response
# ---------------------------------------------------------------------------

IMPORTANT_NOTES: Final[list[str]] = [
    "This information is provided for general educational and informational purposes only.",
    "It does not constitute legal advice and must not be relied upon as such.",
    "Laws and procedures may vary by state and jurisdiction within India.",
    "Consult a qualified legal professional (advocate) for advice specific to your situation.",
    "Free legal aid is available through District Legal Services Authorities (DLSA) for eligible persons.",
]

# ---------------------------------------------------------------------------
# Fallback guidance for unknown offences
# ---------------------------------------------------------------------------

_DEFAULT_ACTIONS: Final[list[str]] = [
    "File an FIR at the nearest police station.",
    "Preserve any physical or digital evidence.",
    "Record witness information if available.",
    "Cooperate with the investigating officer.",
    "Consult a qualified legal professional.",
]

_DEFAULT_DOCUMENTS: Final[list[str]] = [
    "Identity proof",
    "Written description of the incident with date, time, and location",
    "Available physical or digital evidence",
    "Witness details (if any)",
]


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
                log_dir / "remedy_advisor.log",
                level="DEBUG",
                rotation=getattr(_config, "LOG_ROTATION", "10 MB"),
                retention=getattr(_config, "LOG_RETENTION", "30 days"),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not set up file logging via config.LOG_DIR: {}", exc)

    _LOGGING_CONFIGURED = True


def _read_config_int(attr: str, default: int) -> int:
    """Safely read a positive integer from config.py with a fallback default.

    Args:
        attr: Attribute name on the ``config`` module.
        default: Fallback value.

    Returns:
        The configured int, or ``default`` if absent or invalid.
    """
    if _config is None:
        return default
    value = getattr(_config, attr, default)
    if isinstance(value, int) and value > 0:
        return value
    logger.warning(
        "config.{} is not a positive int ({}); using default {}.", attr, value, default
    )
    return default


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class RemedyResult:
    """Structured remedy guidance result.

    Attributes:
        recommended_actions: Prioritised list of actions the complainant
            should take.
        procedure: General legal procedural steps.
        documents_required: Documents to gather and preserve.
        important_notes: Mandatory disclaimers and important guidance.
    """

    recommended_actions: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    documents_required: list[str] = field(default_factory=list)
    important_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        """Serialise to the required output schema.

        Returns:
            Dict with keys ``recommended_actions``, ``procedure``,
            ``documents_required``, and ``important_notes``.
        """
        return {
            "recommended_actions": self.recommended_actions,
            "procedure": self.procedure,
            "documents_required": self.documents_required,
            "important_notes": self.important_notes,
        }


# ---------------------------------------------------------------------------
# RemedyAdvisor
# ---------------------------------------------------------------------------


class RemedyAdvisor:
    """Provides general legal procedural guidance from mapped incident types.

    All guidance is derived from configurable static mappings; no LLM is
    invoked. The advisor aggregates guidance across multiple incident types,
    deduplicates entries, and always appends the standard procedural steps
    and mandatory disclaimers.

    Usage
    -----
    >>> advisor = RemedyAdvisor()
    >>> result = advisor.recommend(section_mapper_output)
    >>> print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        remedy_mappings: dict[str, dict[str, list[str]]] | None = None,
        general_procedure: list[str] | None = None,
        important_notes: list[str] | None = None,
    ) -> None:
        """Initialise the advisor with mappings and configuration.

        Args:
            remedy_mappings: Optional override of ``REMEDY_MAPPINGS``. Useful
                for testing or extending the built-in registry without editing
                the module.
            general_procedure: Optional override of ``GENERAL_PROCEDURE``.
            important_notes: Optional override of ``IMPORTANT_NOTES``.
        """
        _configure_logging()

        self._max_actions: int = _read_config_int(
            "REMEDY_ADVISOR_MAX_ACTIONS", _DEFAULT_MAX_ACTIONS
        )
        self._max_documents: int = _read_config_int(
            "REMEDY_ADVISOR_MAX_DOCUMENTS", _DEFAULT_MAX_DOCUMENTS
        )
        self._max_notes: int = _read_config_int(
            "REMEDY_ADVISOR_MAX_NOTES", _DEFAULT_MAX_NOTES
        )

        self._mappings: dict[str, dict[str, list[str]]] = (
            remedy_mappings if remedy_mappings is not None else REMEDY_MAPPINGS
        )
        self._general_procedure: list[str] = (
            general_procedure if general_procedure is not None else list(GENERAL_PROCEDURE)
        )
        self._important_notes: list[str] = (
            important_notes if important_notes is not None else list(IMPORTANT_NOTES)
        )

        logger.debug(
            "RemedyAdvisor initialised: max_actions={} max_documents={} max_notes={} "
            "mapping_keys={}",
            self._max_actions,
            self._max_documents,
            self._max_notes,
            sorted(self._mappings.keys()),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recommend(self, section_mapper_output: dict[str, Any]) -> dict[str, list[str]]:
        """Generate general legal procedural guidance from mapped sections.

        Args:
            section_mapper_output: Dict from the Section Mapper Agent,
                expected to contain ``"incident_type"`` (list[str]) and/or
                ``"applicable_sections"`` (list[dict]).  Both keys are
                optional; the advisor gracefully handles missing or empty
                values.

        Returns:
            Dict with keys:

            * ``recommended_actions``
            * ``procedure``
            * ``documents_required``
            * ``important_notes``
        """
        start_time = time.perf_counter()

        if not isinstance(section_mapper_output, dict):
            logger.warning(
                "recommend() received non-dict input ({}); returning safe defaults.",
                type(section_mapper_output).__name__,
            )
            return self._safe_default()

        incident_types: list[str] = section_mapper_output.get("incident_type", []) or []
        applicable_sections: list[dict[str, Any]] = (
            section_mapper_output.get("applicable_sections", []) or []
        )

        logger.info(
            "RemedyAdvisor.recommend called: incident_types={} applicable_sections={}",
            incident_types,
            [s.get("section") for s in applicable_sections if isinstance(s, dict)],
        )

        # Normalise incident types to lowercase stripped strings.
        normalised_types: list[str] = [
            t.strip().lower() for t in incident_types if isinstance(t, str) and t.strip()
        ]

        # If no incident types provided, attempt to infer from section metadata.
        if not normalised_types:
            normalised_types = self._infer_incident_types(applicable_sections)

        if not normalised_types:
            logger.warning("No incident types available; returning safe defaults.")
            return self._safe_default()

        logger.debug("Resolved incident types for guidance: {}", normalised_types)

        actions = self.recommend_actions(normalised_types)
        documents = self.recommend_documents(normalised_types)
        procedure = self.recommend_procedure(normalised_types)
        notes = self.recommend_notes(normalised_types)

        result = RemedyResult(
            recommended_actions=actions,
            procedure=procedure,
            documents_required=documents,
            important_notes=notes,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "RemedyAdvisor completed in {:.4f}s -- actions={} documents={} notes={}",
            elapsed,
            len(actions),
            len(documents),
            len(notes),
        )

        return result.to_dict()

    # ------------------------------------------------------------------
    # Guidance assembly methods
    # ------------------------------------------------------------------

    def recommend_actions(self, incident_types: list[str]) -> list[str]:
        """Aggregate recommended actions for all incident types.

        Actions from each matching incident type are merged in order, then
        deduplicated while preserving insertion order.  If no incident type
        matches the mapping, default actions are returned.

        Args:
            incident_types: Normalised (lowercase) incident type strings.

        Returns:
            Deduplicated list of recommended action strings, capped at
            ``REMEDY_ADVISOR_MAX_ACTIONS``.
        """
        raw: list[str] = []
        matched_any = False

        for incident in incident_types:
            mapping = self._mappings.get(incident)
            if mapping:
                matched_any = True
                raw.extend(mapping.get("actions", []))
                logger.debug("Actions sourced for incident type '{}'.", incident)
            else:
                logger.warning("No action mapping found for incident type '{}'.", incident)

        if not matched_any:
            logger.warning("No mappings matched; using default actions.")
            raw = list(_DEFAULT_ACTIONS)

        deduped = self.remove_duplicates(raw)
        capped = deduped[: self._max_actions]
        logger.debug("recommend_actions: {} actions after dedup/cap.", len(capped))
        return capped

    def recommend_documents(self, incident_types: list[str]) -> list[str]:
        """Aggregate required documents for all incident types.

        Args:
            incident_types: Normalised (lowercase) incident type strings.

        Returns:
            Deduplicated list of required document strings, capped at
            ``REMEDY_ADVISOR_MAX_DOCUMENTS``.
        """
        raw: list[str] = []
        matched_any = False

        for incident in incident_types:
            mapping = self._mappings.get(incident)
            if mapping:
                matched_any = True
                raw.extend(mapping.get("documents", []))
                logger.debug("Documents sourced for incident type '{}'.", incident)

        if not matched_any:
            raw = list(_DEFAULT_DOCUMENTS)

        deduped = self.remove_duplicates(raw)
        capped = deduped[: self._max_documents]
        logger.debug("recommend_documents: {} documents after dedup/cap.", len(capped))
        return capped

    def recommend_procedure(self, incident_types: list[str]) -> list[str]:
        """Return the general legal procedure, extended with incident-specific steps.

        Incident-specific procedure steps (if defined in the mapping) are
        prepended before the general procedural steps.

        Args:
            incident_types: Normalised (lowercase) incident type strings.

        Returns:
            Ordered, deduplicated list of procedural step strings.
        """
        incident_specific: list[str] = []

        for incident in incident_types:
            mapping = self._mappings.get(incident)
            if mapping:
                incident_specific.extend(mapping.get("procedure", []))

        combined = self.remove_duplicates(incident_specific + self._general_procedure)
        logger.debug("recommend_procedure: {} steps.", len(combined))
        return combined

    def recommend_notes(self, incident_types: list[str]) -> list[str]:
        """Aggregate important notes for all incident types.

        Incident-specific notes are prepended, followed by the mandatory
        disclaimer notes.

        Args:
            incident_types: Normalised (lowercase) incident type strings.

        Returns:
            Deduplicated list of note strings, capped at
            ``REMEDY_ADVISOR_MAX_NOTES``.
        """
        incident_notes: list[str] = []

        for incident in incident_types:
            mapping = self._mappings.get(incident)
            if mapping:
                incident_notes.extend(mapping.get("notes", []))

        combined = self.remove_duplicates(incident_notes + self._important_notes)
        capped = combined[: self._max_notes]
        logger.debug("recommend_notes: {} notes after dedup/cap.", len(capped))
        return capped

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def remove_duplicates(items: list[str]) -> list[str]:
        """Remove duplicate strings while preserving insertion order.

        Comparison is case-insensitive and strips surrounding whitespace.

        Args:
            items: Possibly-duplicate list of strings.

        Returns:
            List with duplicates removed; first occurrence is retained.
        """
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            normalised = item.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(item.strip())
        return result

    def _infer_incident_types(
        self, applicable_sections: list[dict[str, Any]]
    ) -> list[str]:
        """Attempt to infer incident types from applicable section metadata.

        Checks each section's ``reason`` or ``section`` fields against the
        mapping keys using simple substring matching.

        Args:
            applicable_sections: Section dicts from the Section Mapper output.

        Returns:
            List of inferred incident type strings (may be empty).
        """
        inferred: list[str] = []
        mapping_keys = list(self._mappings.keys())

        for sec in applicable_sections:
            if not isinstance(sec, dict):
                continue
            combined_text = " ".join(
                filter(None, [sec.get("section", ""), sec.get("reason", "")])
            ).lower()
            for key in mapping_keys:
                if key in combined_text and key not in inferred:
                    inferred.append(key)

        logger.debug("Inferred incident types from sections: {}", inferred)
        return inferred

    def _safe_default(self) -> dict[str, list[str]]:
        """Return safe default guidance when inputs are invalid or empty.

        Returns:
            Dict with default actions, general procedure, default documents,
            and important notes.
        """
        return RemedyResult(
            recommended_actions=list(_DEFAULT_ACTIONS),
            procedure=list(self._general_procedure),
            documents_required=list(_DEFAULT_DOCUMENTS),
            important_notes=list(self._important_notes),
        ).to_dict()


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

_TEST_SCENARIOS: Final[list[dict[str, Any]]] = [
    {
        "label": "Theft + House Trespass",
        "input": {
            "incident_type": ["theft", "house trespass"],
            "applicable_sections": [
                {"document": "BNS", "section": "Section 303", "confidence": 0.96},
                {"document": "BNS", "section": "Section 331", "confidence": 0.91},
            ],
        },
    },
    {
        "label": "Assault",
        "input": {
            "incident_type": ["assault"],
            "applicable_sections": [
                {"document": "BNS", "section": "Section 131", "confidence": 0.89},
            ],
        },
    },
    {
        "label": "Cyber Crime",
        "input": {
            "incident_type": ["cyber crime"],
            "applicable_sections": [
                {"document": "IT Act", "section": "Section 66", "confidence": 0.92},
                {"document": "IT Act", "section": "Section 66C", "confidence": 0.85},
            ],
        },
    },
    {
        "label": "Forgery",
        "input": {
            "incident_type": ["forgery"],
            "applicable_sections": [
                {"document": "BNS", "section": "Section 336", "confidence": 0.88},
            ],
        },
    },
    {
        "label": "Unknown Offence (graceful fallback)",
        "input": {
            "incident_type": ["unknown_offence_xyz"],
            "applicable_sections": [],
        },
    },
    {
        "label": "Empty Input (safe default)",
        "input": {},
    },
]


def _run_test_mode() -> None:
    """Run the RemedyAdvisor against sample scenarios and print results."""
    advisor = RemedyAdvisor()

    for scenario in _TEST_SCENARIOS:
        label: str = scenario["label"]
        input_data: dict[str, Any] = scenario["input"]

        print(f"\n{'=' * 65}")
        print(f"  Scenario: {label}")
        print(f"{'=' * 65}")
        print("  Input:")
        print(
            json.dumps(input_data, indent=4, ensure_ascii=False)
            .replace("\n", "\n  ")
        )
        print("\n  Output:")
        result = advisor.recommend(input_data)
        print(
            json.dumps(result, indent=4, ensure_ascii=False)
            .replace("\n", "\n  ")
        )


if __name__ == "__main__":
    _run_test_mode()