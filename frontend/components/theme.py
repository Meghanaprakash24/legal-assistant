"""
components/theme.py
====================
Global CSS theme and theme-switcher for the Indian Legal AI Assistant.

Call :func:`inject_global_theme` once near the top of every page (after
``st.set_page_config``) to apply the shared corporate look: white
background, dark-blue primary accent, light-gray borders, rounded
corners, soft shadows, and smooth hover/animation transitions across
cards, buttons, headers, tables, the sidebar, metrics, expanders, forms,
containers, columns, tabs, and progress bars.

This module owns the **visual theme only**. It does not fetch data and
does not know about any specific page's content.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

ThemeName = Literal["Light", "Dark", "System"]

THEME_STATE_KEY: str = "global_theme"
DEFAULT_THEME: ThemeName = "System"


# =============================================================================
# Color tokens
# =============================================================================

_LIGHT_TOKENS: dict[str, str] = {
    "--bg": "#FFFFFF",
    "--bg-secondary": "#F5F7FB",
    "--text": "#1A1D29",
    "--text-muted": "#6B7280",
    "--accent": "#0B3D91",
    "--accent-light": "#1E5BB8",
    "--border": "#E2E6EE",
    "--shadow": "rgba(11, 61, 145, 0.08)",
    "--success-bg": "#DCFCE7",
    "--success-text": "#15803D",
    "--warning-bg": "#FEF3C7",
    "--warning-text": "#B45309",
    "--error-bg": "#FEE2E2",
    "--error-text": "#B91C1C",
    "--info-bg": "#DBEAFE",
    "--info-text": "#1E40AF",
}

_DARK_TOKENS: dict[str, str] = {
    "--bg": "#0F1419",
    "--bg-secondary": "#1A2030",
    "--text": "#E5E7EB",
    "--text-muted": "#9CA3AF",
    "--accent": "#3B82F6",
    "--accent-light": "#60A5FA",
    "--border": "#2D3548",
    "--shadow": "rgba(0, 0, 0, 0.35)",
    "--success-bg": "#0F2E1B",
    "--success-text": "#4ADE80",
    "--warning-bg": "#3A2A0A",
    "--warning-text": "#FBBF24",
    "--error-bg": "#3A1213",
    "--error-text": "#F87171",
    "--info-bg": "#11243F",
    "--info-text": "#60A5FA",
}


def get_active_theme() -> ThemeName:
    """Return the theme currently selected in session state (defaulting
    to System on first run)."""
    return st.session_state.get(THEME_STATE_KEY, DEFAULT_THEME)


def _resolve_tokens(theme: ThemeName) -> dict[str, str]:
    """Resolve a theme name to its CSS variable token set.

    "System" falls back to the light token set for server-rendered CSS
    purposes, but is paired with a ``prefers-color-scheme`` media query
    so the browser's own preference still applies.
    """
    if theme == "Dark":
        return _DARK_TOKENS
    return _LIGHT_TOKENS


def theme_switcher(*, location: Literal["sidebar", "inline"] = "sidebar") -> ThemeName:
    """Render a Light / Dark / System theme selector and persist the choice.

    Args:
        location: Where to render the selector — the sidebar (default)
            or inline in the main page flow.

    Returns:
        The currently selected theme name.
    """
    if THEME_STATE_KEY not in st.session_state:
        st.session_state[THEME_STATE_KEY] = DEFAULT_THEME

    container = st.sidebar if location == "sidebar" else st
    options: tuple[ThemeName, ...] = ("Light", "Dark", "System")
    current = st.session_state[THEME_STATE_KEY]

    selected = container.selectbox(
        "🎨 Theme",
        options=options,
        index=options.index(current) if current in options else 2,
        key="theme_switcher_select",
    )
    st.session_state[THEME_STATE_KEY] = selected
    return selected


def inject_global_theme() -> None:
    """Inject the full shared CSS theme for the active session theme.

    Safe to call on every page render — Streamlit replaces the prior
    ``<style>`` block rather than stacking duplicates visually, and the
    cost of re-injecting a small CSS string is negligible.
    """
    theme = get_active_theme()
    tokens = _resolve_tokens(theme)
    token_css = "\n".join(f"{key}: {value};" for key, value in tokens.items())

    dark_media_query = (
        """
        @media (prefers-color-scheme: dark) {
            :root { %s }
        }
        """
        % "\n".join(f"{k}: {v};" for k, v in _DARK_TOKENS.items())
        if theme == "System"
        else ""
    )

    st.markdown(
        f"""
        <style>
        :root {{
            {token_css}
        }}
        {dark_media_query}

        /* ---------- Global typography & background ---------- */
        .stApp {{
            background-color: var(--bg);
            color: var(--text);
        }}
        h1, h2, h3, h4 {{
            color: var(--accent);
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        /* ---------- Cards ---------- */
        .app-card {{
            background-color: var(--bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px var(--shadow);
            transition: box-shadow 0.2s ease, transform 0.15s ease;
        }}
        .app-card:hover {{
            box-shadow: 0 4px 14px var(--shadow);
        }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            transition: all 0.18s ease !important;
            font-weight: 600 !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 10px var(--shadow);
            border-color: var(--accent) !important;
        }}
        .stButton > button[kind="primary"] {{
            background-color: var(--accent) !important;
            border-color: var(--accent) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: var(--accent-light) !important;
            border-color: var(--accent-light) !important;
        }}
        .stDownloadButton > button {{
            border-radius: 10px !important;
            transition: all 0.18s ease !important;
        }}
        .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 10px var(--shadow);
        }}

        /* ---------- Metrics ---------- */
        div[data-testid="stMetric"] {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.9rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow);
        }}

        /* ---------- Expanders ---------- */
        div[data-testid="stExpander"] {{
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }}

        /* ---------- Forms / inputs ---------- */
        div[data-testid="stForm"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.25rem;
            background-color: var(--bg-secondary);
        }}
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            border-radius: 8px !important;
        }}

        /* ---------- Tabs ---------- */
        button[data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0 !important;
            font-weight: 600 !important;
        }}

        /* ---------- Tables / dataframes ---------- */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            border-right: 1px solid var(--border);
        }}

        /* ---------- Progress bars ---------- */
        div[data-testid="stProgress"] > div > div {{
            background-color: var(--accent) !important;
            transition: width 0.3s ease;
        }}

        /* ---------- Fade-in animation for cards/content ---------- */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .app-card, div[data-testid="stMetric"] {{
            animation: fadeInUp 0.25s ease-out;
        }}

        /* ---------- Status pills (shared across pages) ---------- */
        .status-pill {{
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .pill-success {{ background-color: var(--success-bg); color: var(--success-text); }}
        .pill-warning {{ background-color: var(--warning-bg); color: var(--warning-text); }}
        .pill-error {{ background-color: var(--error-bg); color: var(--error-text); }}
        .pill-info {{ background-color: var(--info-bg); color: var(--info-text); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: str | None = None, icon: str = "") -> None:
    """Open a shared ``.app-card`` container, optionally with a heading."""
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    if title:
        st.markdown(f"### {icon} {title}".strip())


def card_close() -> None:
    """Close a shared ``.app-card`` container."""
    st.markdown("</div>", unsafe_allow_html=True)