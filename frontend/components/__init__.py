"""
components
==========
Shared, reusable Streamlit UI building blocks for the Indian Legal AI
Assistant frontend.

This package contains **no business logic** and makes **no direct HTTP
calls** beyond the existing ``services.api.api_client`` — it only renders
UI. Every existing or future page can import from here instead of
duplicating CSS, status cards, loading states, or export buttons.

Modules
-------
theme
    Global CSS injection (cards, buttons, headers, tables, sidebar,
    metrics, expanders, forms, tabs, animations) and the light/dark/
    system theme switcher.
notifications
    Toast-style success/warning/error/info helpers, including the
    standard backend/Groq/Qdrant connectivity notifications.
loading
    Spinners, skeleton cards, and loading placeholders for metrics,
    charts, tables, and chat responses.
errors
    Beautiful empty/error-state cards: backend offline, API error,
    timeout, no results, empty search, no citations, no retrieved
    documents, no statistics.
session
    Session bookkeeping helpers (start time, query counter, message
    counter) and the Session Information display block.
caching
    Thin ``st.cache_data`` / ``st.cache_resource`` wrappers around the
    real ``api_client`` calls (health, statistics) so pages share one
    cache instead of re-fetching on every rerun.
exports
    Download-button helpers for exporting chat transcripts, retrieved
    evidence, and statistics in their respective formats.
shortcuts
    Keyboard shortcut wiring (Ctrl+Enter, Ctrl+L, Ctrl+R) via a small
    injected JS listener that posts back into Streamlit.

Usage
-----
    from components.theme import inject_global_theme, theme_switcher
    from components.notifications import notify_success, notify_backend_status
    from components.loading import skeleton_cards, loading_spinner
    from components.errors import render_backend_offline, render_no_results
    from components.session import render_session_info, increment_query_count
    from components.caching import get_cached_health, get_cached_statistics
    from components.exports import export_chat_buttons, export_evidence_buttons
    from components.shortcuts import inject_keyboard_shortcuts
"""

from __future__ import annotations