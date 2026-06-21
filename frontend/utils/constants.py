"""
utils/constants.py
------------------
Application-wide constants for the Indian Legal RAG System frontend.

This module contains only pure data constants — no logic, no imports from
application modules, no Streamlit calls.  Import what you need:

    from utils.constants import LEGAL_DOCUMENTS, PIPELINE_STAGES, ICONS

Python 3.11+  |  PEP 8
"""

from __future__ import annotations

from typing import Final

# ===========================================================================
# Legal document registry
# ===========================================================================

LEGAL_DOCUMENTS: Final[list[dict[str, str]]] = [
    {
        "id":          "BNS",
        "label":       "Bharatiya Nyaya Sanhita",
        "short":       "BNS",
        "year":        "2023",
        "description": (
            "The principal criminal code of India replacing the Indian "
            "Penal Code, 1860. Covers offences against persons, property, "
            "the state, and public order."
        ),
        "icon":        "📖",
    },
    {
        "id":          "BNSS",
        "label":       "Bharatiya Nagarik Suraksha Sanhita",
        "short":       "BNSS",
        "year":        "2023",
        "description": (
            "The code of criminal procedure replacing CrPC, 1973. "
            "Governs arrest, bail, trial procedure, and sentencing."
        ),
        "icon":        "⚖️",
    },
    {
        "id":          "BSA",
        "label":       "Bharatiya Sakshya Adhiniyam",
        "short":       "BSA",
        "year":        "2023",
        "description": (
            "The law of evidence replacing the Indian Evidence Act, 1872. "
            "Governs admissibility, relevancy, and proof of facts."
        ),
        "icon":        "📋",
    },
    {
        "id":          "Constitution",
        "label":       "Constitution of India",
        "short":       "Constitution",
        "year":        "1950",
        "description": (
            "The supreme law of India. Defines fundamental rights, directive "
            "principles, the structure of government, and constitutional remedies."
        ),
        "icon":        "🏛️",
    },
]

# Quick lookup: short_code → full label
DOCUMENT_LABEL: Final[dict[str, str]] = {
    doc["id"]: doc["label"] for doc in LEGAL_DOCUMENTS
}

# Quick lookup: short_code → icon
DOCUMENT_ICON: Final[dict[str, str]] = {
    doc["id"]: doc["icon"] for doc in LEGAL_DOCUMENTS
}

# ===========================================================================
# Chunk types
# ===========================================================================

CHUNK_TYPES: Final[dict[str, str]] = {
    "section":      "Section",
    "clause":       "Clause",
    "explanation":  "Explanation",
    "illustration": "Illustration",
    "article":      "Article",
}

# ===========================================================================
# Pipeline stage definitions
# ===========================================================================

PIPELINE_STAGES: Final[list[dict[str, str]]] = [
    {
        "id":          "query_analysis",
        "label":       "Query Analysis",
        "description": "Parse the user query; detect section/article references and legal keywords.",
        "icon":        "🔍",
        "agent":       "Pipeline",
    },
    {
        "id":          "classification",
        "label":       "Classification",
        "description": "Identify the offence category, affected parties, and relevant documents.",
        "icon":        "🏷️",
        "agent":       "Classifier Agent",
    },
    {
        "id":          "retrieval",
        "label":       "Hybrid Retrieval",
        "description": "Run Dense (Qdrant) + BM25 search; fuse results with Reciprocal Rank Fusion.",
        "icon":        "📡",
        "agent":       "Retriever",
    },
    {
        "id":          "reranking",
        "label":       "CrossEncoder Reranking",
        "description": "Score every (query, passage) pair with BAAI/bge-reranker; return top-K.",
        "icon":        "🎯",
        "agent":       "Reranker",
    },
    {
        "id":          "quote_selection",
        "label":       "Quote Selection",
        "description": "Extract the most legally relevant verbatim passages from top chunks.",
        "icon":        "💬",
        "agent":       "Quote Selector Agent",
    },
    {
        "id":          "section_mapping",
        "label":       "Section Mapping",
        "description": "Map retrieved chunks to applicable BNS/BNSS/BSA/Constitution sections.",
        "icon":        "🗺️",
        "agent":       "Section Mapper Agent",
    },
    {
        "id":          "remedy_advice",
        "label":       "Remedy Advice",
        "description": "Generate FIR guidance, bail eligibility, and procedural next steps.",
        "icon":        "💊",
        "agent":       "Remedy Advisor Agent",
    },
    {
        "id":          "validation",
        "label":       "Citation Validation",
        "description": "Verify every citation against retrieved evidence; PASS or FAIL.",
        "icon":        "✅",
        "agent":       "Validator Agent",
    },
    {
        "id":          "synthesis",
        "label":       "Response Synthesis",
        "description": "LLM assembles the final structured legal response from validated evidence.",
        "icon":        "✍️",
        "agent":       "Synthesizer Agent",
    },
]

# Quick lookup: stage id → label
STAGE_LABEL: Final[dict[str, str]] = {s["id"]: s["label"] for s in PIPELINE_STAGES}

# Quick lookup: stage id → icon
STAGE_ICON: Final[dict[str, str]] = {s["id"]: s["icon"] for s in PIPELINE_STAGES}

# ===========================================================================
# Status labels and colours
# ===========================================================================

class Status:
    """Centralised status constants consumed by badge components."""

    PASS     = "PASS"
    FAIL     = "FAIL"
    RUNNING  = "RUNNING"
    PENDING  = "PENDING"
    SUCCESS  = "SUCCESS"
    ERROR    = "ERROR"
    ONLINE   = "online"
    DEGRADED = "degraded"
    OFFLINE  = "offline"
    UNKNOWN  = "unknown"


# Maps a Status value to the CSS class suffix used by .lex-badge-* rules
STATUS_BADGE_CLASS: Final[dict[str, str]] = {
    Status.PASS:     "success",
    Status.SUCCESS:  "success",
    Status.ONLINE:   "success",
    Status.FAIL:     "error",
    Status.ERROR:    "error",
    Status.OFFLINE:  "error",
    Status.RUNNING:  "info",
    Status.PENDING:  "neutral",
    Status.DEGRADED: "warning",
    Status.UNKNOWN:  "neutral",
}

# Human-readable labels for display
STATUS_LABEL: Final[dict[str, str]] = {
    Status.PASS:     "PASS",
    Status.FAIL:     "FAIL",
    Status.RUNNING:  "Running",
    Status.PENDING:  "Pending",
    Status.SUCCESS:  "Success",
    Status.ERROR:    "Error",
    Status.ONLINE:   "Online",
    Status.DEGRADED: "Degraded",
    Status.OFFLINE:  "Offline",
    Status.UNKNOWN:  "Unknown",
}

# ===========================================================================
# Icon library  (emoji-based, zero external dependency)
# ===========================================================================

class Icons:
    """Single source of truth for every emoji / unicode icon used in the UI."""

    # Navigation
    DASHBOARD    = "📊"
    CHAT         = "💬"
    EVIDENCE     = "📑"
    PIPELINE     = "🔄"
    API_STATUS   = "🔌"
    SETTINGS     = "⚙️"
    ABOUT        = "ℹ️"
    SAVED        = "🔖"

    # Legal
    LAW          = "⚖️"
    SECTION      = "§"
    ARTICLE      = "📌"
    DOCUMENT     = "📄"
    CONSTITUTION = "🏛️"
    CITATION     = "🔗"
    CASE         = "📁"

    # Status
    SUCCESS      = "✅"
    WARNING      = "⚠️"
    ERROR        = "❌"
    INFO         = "ℹ️"
    LOADING      = "⏳"
    ONLINE       = "🟢"
    OFFLINE      = "🔴"
    DEGRADED     = "🟡"

    # Actions
    SEARCH       = "🔍"
    SEND         = "➤"
    COPY         = "📋"
    EXPAND       = "⬇️"
    COLLAPSE     = "⬆️"
    REFRESH      = "🔃"
    DOWNLOAD     = "⬇️"
    CLEAR        = "🗑️"

    # Pipeline agents
    CLASSIFIER   = "🏷️"
    RETRIEVER    = "📡"
    RERANKER     = "🎯"
    QUOTE        = "💬"
    MAPPER       = "🗺️"
    REMEDY       = "💊"
    VALIDATOR    = "🛡️"
    SYNTHESIZER  = "✍️"

    # Confidence
    HIGH_CONF    = "🔵"
    MED_CONF     = "🟡"
    LOW_CONF     = "🔴"

    # Misc
    ROBOT        = "🤖"
    USER         = "👤"
    CLOCK        = "⏱️"
    STAR         = "⭐"
    LOCK         = "🔒"
    OPEN         = "🔓"

# ===========================================================================
# Confidence level labels
# ===========================================================================

CONFIDENCE_LEVELS: Final[dict[str, dict[str, str]]] = {
    "high":   {"label": "High Confidence",   "class": "success", "icon": Icons.HIGH_CONF},
    "medium": {"label": "Medium Confidence", "class": "warning", "icon": Icons.MED_CONF},
    "low":    {"label": "Low Confidence",    "class": "error",   "icon": Icons.LOW_CONF},
}

def confidence_level(score: float) -> str:
    """Return the confidence level key for a normalised score in [0, 1].

    Args:
        score: Normalised confidence score.

    Returns:
        One of ``"high"``, ``"medium"``, or ``"low"``.
    """
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"

# ===========================================================================
# API endpoint slugs  (appended to config.API_BASE_URL)
# ===========================================================================

class Endpoints:
    """Backend REST API endpoint paths."""

    HEALTH      = "/health"
    CHAT        = "/chat"
    RETRIEVE    = "/retrieve"
    STATS       = "/statistics"
    PIPELINE    = "/pipeline/status"
    QDRANT      = "/qdrant/status"
    GROQ        = "/groq/status"

# ===========================================================================
# Session state keys
# ===========================================================================

class SessionKeys:
    """Keys used in st.session_state throughout the application."""

    CHAT_HISTORY      = "chat_history"
    CURRENT_PAGE      = "current_page"
    LAST_QUERY        = "last_query"
    LAST_RESPONSE     = "last_response"
    LAST_EVIDENCE     = "last_evidence"
    PIPELINE_STATUS   = "pipeline_status"
    PIPELINE_STAGES   = "pipeline_stages_state"
    TOTAL_QUERIES     = "total_queries"
    AVG_LATENCY       = "avg_latency"
    IS_LOADING        = "is_loading"
    SIDEBAR_COLLAPSED = "sidebar_collapsed"

# ===========================================================================
# Sample / demo data
# ===========================================================================

DEMO_QUERIES: Final[list[str]] = [
    "What is the punishment for murder?",
    "What sections apply if someone breaks into my house and steals my phone?",
    "How do I file an FIR?",
    "What is the difference between theft and robbery?",
]