"""
components/accessibility.py
============================
Accessibility helpers for the Indian Legal AI Assistant.

Provides a small settings panel (high contrast, large fonts) backed by
``st.session_state``, plus a helper for emitting screen-reader-only
labels alongside icon-only buttons or custom HTML controls.

This module deliberately keeps to CSS-level adjustments — it does not
attempt to re-implement full ARIA semantics for native Streamlit
widgets, which already carry reasonable default accessibility roles.
"""

from __future__ import annotations

import streamlit as st

HIGH_CONTRAST_KEY: str = "a11y_high_contrast"
LARGE_FONT_KEY: str = "a11y_large_font"


def render_accessibility_controls(*, location: str = "sidebar") -> None:
    """Render High Contrast and Large Font toggles and persist the choice.

    Args:
        location: "sidebar" (default) or "inline".
    """
    container = st.sidebar if location == "sidebar" else st
    container.markdown("##### ♿ Accessibility")

    st.session_state.setdefault(HIGH_CONTRAST_KEY, False)
    st.session_state.setdefault(LARGE_FONT_KEY, False)

    st.session_state[HIGH_CONTRAST_KEY] = container.toggle(
        "High Contrast",
        value=st.session_state[HIGH_CONTRAST_KEY],
        help="Increases text/background contrast and border visibility.",
        key="a11y_high_contrast_toggle",
    )
    st.session_state[LARGE_FONT_KEY] = container.toggle(
        "Large Fonts",
        value=st.session_state[LARGE_FONT_KEY],
        help="Increases base font size across the app.",
        key="a11y_large_font_toggle",
    )


def inject_accessibility_css() -> None:
    """Apply the currently selected accessibility settings as CSS.

    Call this after :func:`render_accessibility_controls` (or with the
    defaults, if controls haven't been rendered on this page) so every
    page respects the session-wide accessibility preferences.
    """
    high_contrast = st.session_state.get(HIGH_CONTRAST_KEY, False)
    large_font = st.session_state.get(LARGE_FONT_KEY, False)

    css_rules = []

    if high_contrast:
        css_rules.append(
            """
            .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
            .app-card, div[data-testid="stMetric"], div[data-testid="stExpander"] {
                border-width: 2px !important;
                border-color: #000000 !important;
            }
            a, .stButton > button { outline: 2px solid transparent; }
            .stButton > button:focus-visible, a:focus-visible {
                outline: 3px solid #0B3D91 !important;
                outline-offset: 2px;
            }
            """
        )

    if large_font:
        css_rules.append(
            """
            html, body, .stApp, p, span, div, label { font-size: 1.15rem !important; }
            h1 { font-size: 2.3rem !important; }
            h2 { font-size: 1.9rem !important; }
            h3 { font-size: 1.5rem !important; }
            """
        )

    if css_rules:
        st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)


def sr_only_label(text: str) -> None:
    """Render a visually-hidden label for screen readers only.

    Use immediately before/after an icon-only button or custom widget
    that has no visible text label of its own.

    Args:
        text: The accessible name to announce to screen readers.
    """
    st.markdown(
        f"""
        <span style="
            position: absolute;
            width: 1px; height: 1px;
            padding: 0; margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )