"""
components/sidebar.py
---------------------
Reusable sidebar component for the Indian Legal RAG System.

Usage
-----
    from components.sidebar import render_sidebar

    selected_page = render_sidebar()

The function renders the complete sidebar and returns the name of the
page the user has selected.  Pass ``selected_page`` to your page router
in the main app file.

Design notes
------------
* All navigation items and status indicators are defined as module-level
  constants so they can be updated without touching render logic.
* Status values are fetched through the shared API client, so every page
  uses the same backend URL, timeout handling, and health normalization.
* CSS is injected inline via ``st.markdown(unsafe_allow_html=True)`` so
  this component is self-contained and works without the global theme.css
  being loaded first.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Try to import project config; fall back to safe defaults so this module
# works even when run in isolation during development.
# ---------------------------------------------------------------------------

try:
    from config import (
        APP_NAME,
        APP_ICON,
        API_BASE_URL,
        PRIMARY_COLOR,
        ACCENT_COLOR,
    )
except ImportError:
    import os as _os
    APP_NAME      = "LexAI – Indian Legal Research"
    APP_ICON      = "⚖️"
    API_BASE_URL  = _os.environ.get("API_URL", _os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    PRIMARY_COLOR = "#0F172A"
    ACCENT_COLOR  = "#2563EB"

try:
    from utils.constants import SessionKeys
except ImportError:
    class SessionKeys:  # type: ignore[no-redef]
        CURRENT_PAGE = "current_page"

try:
    from services.api import api_client
except ImportError:
    api_client = None  # type: ignore[assignment]

# ===========================================================================
# Navigation items
# ===========================================================================

#: Each entry: (icon, label, page_key).
#: ``page_key`` is the string stored in session_state and returned by
#: ``render_sidebar()``.  It matches the title used in your page router.
NAV_ITEMS: list[tuple[str, str, str]] = [
    ("🏠", "Home",               "Home"),
    ("💬", "Legal Assistant",    "Legal Assistant"),
    ("📑", "Retrieved Evidence", "Retrieved Evidence"),
    ("📚", "Citations",          "Citations"),
    ("📊", "Dashboard",          "Dashboard"),
    ("🖥️", "System Status",      "System Status"),
    ("⚙️", "Settings",           "Settings"),
    ("ℹ️", "About",              "About"),
]

# ===========================================================================
# Quick-action search shortcuts
# ===========================================================================

#: Each entry: (icon, label, document_key).
#: ``document_key`` is passed to the Legal Assistant page as a pre-filter.
QUICK_ACTIONS: list[tuple[str, str, str]] = [
    ("🏛️", "Search Constitution", "Constitution"),
    ("⚖️", "Search BNS",         "BNS"),
    ("📖", "Search BNSS",        "BNSS"),
    ("📄", "Search BSA",         "BSA"),
]

# ===========================================================================
# Backend service definitions
# ===========================================================================

#: Each entry: (icon, display_label, service_key).
#: ``service_key`` maps to the key returned by ``_fetch_backend_status()``.
BACKEND_SERVICES: list[tuple[str, str, str]] = [
    ("🚀", "FastAPI",   "fastapi"),
    ("🤖", "Groq",      "groq"),
    ("🗄️", "Qdrant",    "qdrant"),
    ("🔄", "Pipeline",  "pipeline"),
]

# ===========================================================================
# System information
# ===========================================================================

SYSTEM_INFO: list[tuple[str, str]] = [
    ("Version",          "1.0.0"),
    ("Environment",      "Development"),
    ("Backend",          "FastAPI"),
    ("Vector Database",  "Qdrant"),
    ("LLM",              "Groq / Llama 3.3"),
]

# ===========================================================================
# Status display helpers
# ===========================================================================

#: Maps a status string to (dot_colour, label, badge_colour).
_STATUS_META: dict[str, tuple[str, str, str]] = {
    "online":       ("🟢", "Online",       "#DCFCE7"),
    "initializing": ("🟡", "Initializing", "#FEF9C3"),
    "offline":      ("🔴", "Offline",      "#FEE2E2"),
    "unknown":      ("⚪", "Unknown",      "#F1F5F9"),
}


# ===========================================================================
# Backend status  (mocked — replace with real API call)
# ===========================================================================


def _fetch_backend_status() -> dict[str, str]:
    """Return the operational status of each backend service.

    Uses ``services.api.api_client`` so sidebar health checks share the
    same URL, timeout, logging, and normalization path as the dashboard.

    Returns:
        Dict mapping service key → status string.  Valid status strings
        are ``"online"``, ``"initializing"``, ``"offline"``, ``"unknown"``.
    """
    # ── MOCKED ────────────────────────────────────────────────────────────
    if api_client is None:
        return {key: "offline" for key in ("fastapi", "groq", "qdrant", "pipeline")}

    status = api_client.get_backend_status()
    return {
        key: "online" if status.get(key) else "offline"
        for key in ("fastapi", "groq", "qdrant", "pipeline")
    }
    # ── END MOCK ──────────────────────────────────────────────────────────


# ===========================================================================
# Sidebar-scoped CSS
# ===========================================================================


def _inject_sidebar_css() -> None:
    """Inject minimal sidebar-scoped CSS.

    Keeps the sidebar visually consistent even when the global theme.css
    has not been loaded.  Rules target Streamlit's own sidebar container
    and the custom ``.lex-`` classes used by this component.
    """
    st.markdown(
        """
        <style>
        /* ── Sidebar shell ──────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid rgba(255,255,255,0.07);
        }
        [data-testid="stSidebar"] * {
            color: #F9FAFB !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.10) !important;
        }

        /* ── Logo block ─────────────────────────────────────────────── */
        .lex-sb-logo-icon {
            font-size: 2rem;
            line-height: 1;
        }
        .lex-sb-app-name {
            font-size: 1.0625rem;
            font-weight: 700;
            color: #F9FAFB !important;
            letter-spacing: -0.01em;
        }
        .lex-sb-app-sub {
            font-size: 0.75rem;
            color: #94A3B8 !important;
            margin-top: 2px;
            line-height: 1.4;
        }

        /* ── Section headings ───────────────────────────────────────── */
        .lex-sb-section {
            font-size: 0.6875rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #64748B !important;
            margin: 0.25rem 0 0.5rem;
        }

        /* ── Nav items ──────────────────────────────────────────────── */
        .lex-nav-btn {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            width: 100%;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            border: none;
            background: transparent;
            color: #CBD5E1 !important;
            text-align: left;
            transition: background 120ms ease, color 120ms ease;
        }
        .lex-nav-btn:hover {
            background: rgba(255,255,255,0.07);
            color: #F9FAFB !important;
        }
        .lex-nav-btn.active {
            background: #2563EB;
            color: #FFFFFF !important;
            font-weight: 600;
        }

        /* ── Quick-action buttons ───────────────────────────────────── */
        .lex-qa-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8125rem;
            font-weight: 500;
            color: #94A3B8 !important;
            cursor: pointer;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            width: 100%;
            text-align: left;
            margin-bottom: 0.25rem;
            transition: background 120ms ease, color 120ms ease;
        }
        .lex-qa-btn:hover {
            background: rgba(37,99,235,0.20);
            color: #93C5FD !important;
            border-color: rgba(37,99,235,0.40);
        }

        /* ── Status row ─────────────────────────────────────────────── */
        .lex-status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.375rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.8125rem;
        }
        .lex-status-row:last-child {
            border-bottom: none;
        }
        .lex-status-svc {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            color: #CBD5E1 !important;
        }
        .lex-status-pill {
            font-size: 0.6875rem;
            font-weight: 600;
            padding: 0.1rem 0.45rem;
            border-radius: 20px;
            letter-spacing: 0.02em;
        }

        /* ── System info table ──────────────────────────────────────── */
        .lex-info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .lex-info-row:last-child { border-bottom: none; }
        .lex-info-key   { color: #64748B !important; }
        .lex-info-value { color: #CBD5E1 !important; font-weight: 500; }

        /* ── Footer ─────────────────────────────────────────────────── */
        .lex-sb-footer {
            text-align: center;
            font-size: 0.6875rem;
            color: #475569 !important;
            padding: 0.75rem 0 0.25rem;
        }

        /* Hide Streamlit's default radio button bullets inside sidebar */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
            gap: 0 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Private render helpers
# ===========================================================================


def _render_logo() -> None:
    """Render the app logo block at the top of the sidebar."""
    st.markdown(
        f"""
        <div style="padding: 0.25rem 0 1rem;">
            <div class="lex-sb-logo-icon">{APP_ICON}</div>
            <div class="lex-sb-app-name">Indian Legal AI</div>
            <div class="lex-sb-app-sub">AI Powered Legal Research Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_navigation(current_page: str) -> str:
    """Render the navigation section and return the selected page.

    Uses ``st.radio`` (hidden via CSS) so Streamlit tracks the state
    natively, while custom HTML buttons provide the visual appearance.

    Args:
        current_page: The page key currently active in session state.

    Returns:
        The page key the user selected.
    """
    st.markdown('<p class="lex-sb-section">Navigation</p>', unsafe_allow_html=True)

    # Build option labels with icons embedded so st.radio stays consistent.
    options = [f"{icon}  {label}" for icon, label, _ in NAV_ITEMS]
    page_keys = [key for _, _, key in NAV_ITEMS]
    current_index = page_keys.index(current_page) if current_page in page_keys else 0

    # st.radio provides the state management; CSS hides the default radio UI.
    selected_label = st.radio(
        label="navigation",
        options=options,
        index=current_index,
        label_visibility="collapsed",
        key="sb_nav_radio",
    )

    # Derive the page key from the selected label.
    selected_index = options.index(selected_label) if selected_label in options else 0
    return page_keys[selected_index]


def _render_quick_actions() -> None:
    """Render the quick-action document search buttons.

    Clicking a quick-action navigates to the Legal Assistant page with a
    pre-selected document filter stored in session state.
    """
    st.divider()
    st.markdown('<p class="lex-sb-section">Quick Actions</p>', unsafe_allow_html=True)

    for icon, label, doc_key in QUICK_ACTIONS:
        if st.button(
            f"{icon}  {label}",
            key=f"qa_{doc_key}",
            use_container_width=True,
        ):
            # Store the pre-filter and navigate to the assistant page.
            st.session_state["quick_action_document"] = doc_key
            st.session_state[SessionKeys.CURRENT_PAGE] = "Legal Assistant"
            st.rerun()


def _render_backend_status() -> None:
    """Render the backend service status panel with live (mocked) indicators."""
    st.divider()
    st.markdown(
        '<p class="lex-sb-section">Backend Status</p>',
        unsafe_allow_html=True,
    )

    statuses = _fetch_backend_status()

    # Build HTML rows for all services.
    rows_html = ""
    for svc_icon, svc_label, svc_key in BACKEND_SERVICES:
        raw_status = statuses.get(svc_key, "unknown")
        dot, status_text, pill_bg = _STATUS_META.get(
            raw_status, _STATUS_META["unknown"]
        )
        # Pick a readable text colour that contrasts against the pill background.
        text_colours = {
            "#DCFCE7": "#15803D",
            "#FEF9C3": "#92400E",
            "#FEE2E2": "#991B1B",
            "#F1F5F9": "#475569",
        }
        pill_fg = text_colours.get(pill_bg, "#475569")

        rows_html += f"""
        <div class="lex-status-row">
            <span class="lex-status-svc">{svc_icon} {svc_label}</span>
            <span class="lex-status-pill"
                  style="background:{pill_bg};color:{pill_fg};">
                {dot} {status_text}
            </span>
        </div>
        """

    st.markdown(
        f'<div style="padding:0.25rem 0;">{rows_html}</div>',
        unsafe_allow_html=True,
    )

    # Manual refresh button.
    if st.button("🔃  Refresh Status", key="sb_refresh_status", use_container_width=True):
        st.rerun()


def _render_system_info() -> None:
    """Render the system information table."""
    st.divider()
    st.markdown(
        '<p class="lex-sb-section">System Information</p>',
        unsafe_allow_html=True,
    )

    rows_html = "".join(
        f"""
        <div class="lex-info-row">
            <span class="lex-info-key">{key}</span>
            <span class="lex-info-value">{value}</span>
        </div>
        """
        for key, value in SYSTEM_INFO
    )
    st.markdown(
        f'<div style="padding:0.25rem 0 0.5rem;">{rows_html}</div>',
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    """Render the sidebar footer disclaimer."""
    st.divider()
    st.markdown(
        """
        <div class="lex-sb-footer">
            Made for<br>
            <strong style="color:#CBD5E1 !important;">
                Indian Legal AI Assistant
            </strong><br>
            <span style="font-size:0.625rem;color:#334155 !important;">
                ⚠️ Not legal advice
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Public API
# ===========================================================================


def render_sidebar() -> str:
    """Render the complete application sidebar.

    Call this function at the top of your main app file before any page
    routing logic::

        from components.sidebar import render_sidebar

        selected_page = render_sidebar()

        if selected_page == "Home":
            ...

    The function:

    1. Injects sidebar-scoped CSS.
    2. Renders the logo and app identity block.
    3. Renders the navigation radio group and returns the selected page.
    4. Renders quick-action document search buttons.
    5. Renders the live backend status panel (mocked; 30-second cache).
    6. Renders the system information table.
    7. Renders the footer.

    Session state
    -------------
    * ``SessionKeys.CURRENT_PAGE`` — updated here when a quick-action
      changes the page; the caller should also keep it updated.

    Returns:
        The page key (str) the user has selected, e.g. ``"Home"``,
        ``"Legal Assistant"``, ``"Dashboard"``, etc.
        Always returns a valid value from :data:`NAV_ITEMS`.
    """
    # Initialise current page in session state on first run.
    if SessionKeys.CURRENT_PAGE not in st.session_state:
        st.session_state[SessionKeys.CURRENT_PAGE] = NAV_ITEMS[0][2]  # "Home"

    with st.sidebar:
        # ── CSS ───────────────────────────────────────────────────────
        _inject_sidebar_css()

        # ── Logo ──────────────────────────────────────────────────────
        _render_logo()
        st.divider()

        # ── Navigation ────────────────────────────────────────────────
        selected_page = _render_navigation(
            st.session_state[SessionKeys.CURRENT_PAGE]
        )

        # Keep session state in sync so quick-actions can override it.
        if selected_page != st.session_state.get(SessionKeys.CURRENT_PAGE):
            st.session_state[SessionKeys.CURRENT_PAGE] = selected_page

        # ── Quick Actions ─────────────────────────────────────────────
        _render_quick_actions()

        # ── Backend Status ────────────────────────────────────────────
        _render_backend_status()

        # ── System Info ───────────────────────────────────────────────
        _render_system_info()

        # ── Footer ────────────────────────────────────────────────────
        _render_footer()

    return selected_page
