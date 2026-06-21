
"""
config.py
---------
Central configuration for the AI-Powered Indian Legal RAG System frontend.

All constants consumed by pages, components, and utilities are defined
here so nothing is hardcoded inside UI logic.  Import this module at the
top of every Streamlit page:

    from config import APP_NAME, API_BASE_URL, PRIMARY_COLOR, ...

Python 3.11+  |  PEP 8
"""

from __future__ import annotations

import os
from pathlib import Path

# ===========================================================================
# Application identity
# ===========================================================================

APP_NAME: str = "LexAI – Indian Legal Research Assistant"
"""Full display name shown in the browser tab and page header."""

APP_ICON: str = "⚖️"
"""Emoji icon used as the Streamlit page favicon."""

APP_DESCRIPTION: str = (
    "An AI-powered legal research assistant built on Hybrid RAG "
    "(Dense + BM25), Qdrant vector search, CrossEncoder reranking, "
    "and a LangGraph multi-agent pipeline grounded in the Indian "
    "legal corpus: BNS, BNSS, BSA, and the Constitution of India."
)
"""One-paragraph description rendered on the Home and About pages."""

FOOTER_TEXT: str = (
    "⚠️ This tool is for informational and educational purposes only. "
    "It does not constitute legal advice. Consult a qualified advocate "
    "for advice specific to your situation."
)
"""Disclaimer footer rendered at the bottom of every page."""

# ===========================================================================
# Backend API
# ===========================================================================

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
"""
Base URL of the FastAPI backend.

Override at runtime via the ``API_BASE_URL`` environment variable so the
frontend works in local dev, Docker Compose, and cloud deployments without
a code change::

    export API_BASE_URL=https://api.mydeployment.com
"""

API_TIMEOUT: int = int(os.environ.get("API_TIMEOUT", "30"))
"""HTTP request timeout in seconds for all backend calls."""

# ===========================================================================
# Color palette  (hex strings)
# ===========================================================================

PRIMARY_COLOR: str = "#0F172A"
"""Deep navy — main brand colour, used for the sidebar and headings."""

SECONDARY_COLOR: str = "#1E293B"
"""Slightly lighter navy — secondary panels and card headers."""

ACCENT_COLOR: str = "#2563EB"
"""Royal blue — CTA buttons, active states, and links."""

BACKGROUND_COLOR: str = "#F8FAFC"
"""Off-white page background — keeps contrast without pure white glare."""

CARD_COLOR: str = "#FFFFFF"
"""Pure white card surfaces."""

BORDER_COLOR: str = "#E5E7EB"
"""Light grey borders for cards, dividers, and table edges."""

TEXT_PRIMARY: str = "#111827"
"""Near-black for headings and primary body copy."""

TEXT_SECONDARY: str = "#6B7280"
"""Mid-grey for captions, meta-text, and helper labels."""

TEXT_INVERSE: str = "#F9FAFB"
"""Near-white text used on dark backgrounds (sidebar, primary buttons)."""

SUCCESS_COLOR: str = "#16A34A"
"""Green — PASS status badges, positive metrics."""

WARNING_COLOR: str = "#D97706"
"""Amber — partial-confidence badges and soft warnings."""

ERROR_COLOR: str = "#DC2626"
"""Red — FAIL status badges, error messages, and critical alerts."""

INFO_COLOR: str = "#0284C7"
"""Sky blue — informational banners and neutral highlights."""

# ===========================================================================
# Layout
# ===========================================================================

SIDEBAR_WIDTH: int = 280
"""Target sidebar width in pixels (applied via CSS; Streamlit's own
sidebar width is approximate)."""

CARD_RADIUS: int = 10
"""Border-radius in pixels applied to all card components."""

MAX_CONTENT_WIDTH: int = 1200
"""Maximum width of the main content area in pixels."""

# ===========================================================================
# Navigation & routing
# ===========================================================================

DEFAULT_PAGE: str = "Dashboard"
"""The page rendered when the app first loads (must match a sidebar label)."""

SIDEBAR_PAGES: list[str] = [
    "Dashboard",
    "Legal Assistant",
    "Evidence Explorer",
    "Pipeline Monitor",
    "API Status",
    "About",
]
"""Ordered list of top-level navigation pages shown in the sidebar."""

# ===========================================================================
# Chat / conversation
# ===========================================================================

MAX_CHAT_HISTORY: int = 50
"""Maximum number of conversation turns retained in session state.
Oldest turns are dropped when the limit is exceeded."""

TYPING_INDICATOR_DELAY: float = 0.05
"""Seconds per chunk when streaming a simulated typing animation."""

SUGGESTED_QUESTIONS: list[str] = [
    "What is the punishment for murder under BNS Section 101?",
    "Explain the right to life under Article 21 of the Constitution.",
    "What sections apply if someone breaks into my house and steals?",
    "What is the difference between culpable homicide and murder in BNS?",
    "How do I file an FIR under BNSS?",
    "What constitutes cheating under the Bharatiya Nyaya Sanhita?",
    "Explain the provisions for bail under BNSS.",
    "What evidence is admissible under the Bharatiya Sakshya Adhiniyam?",
]
"""Pre-populated suggested queries shown on the Chat page."""

# ===========================================================================
# Confidence thresholds
# ===========================================================================

CONFIDENCE_HIGH: float = 0.75
"""Confidence ≥ this value renders a GREEN badge."""

CONFIDENCE_MEDIUM: float = 0.50
"""Confidence ≥ this value (and < HIGH) renders an AMBER badge."""
# Confidence < CONFIDENCE_MEDIUM renders a RED badge.

# ===========================================================================
# File paths
# ===========================================================================

ROOT_DIR: Path = Path(__file__).resolve().parent
STYLES_DIR: Path = ROOT_DIR / "styles"
ASSETS_DIR: Path = ROOT_DIR / "assets"
LOGO_PATH: Path = ASSETS_DIR / "logo" / "logo.svg"
CSS_PATH: Path = STYLES_DIR / "theme.css"
