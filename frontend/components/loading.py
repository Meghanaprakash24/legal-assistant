"""
components/loading.py
======================
Loading-state placeholders for the Indian Legal AI Assistant.

Streamlit reruns the whole script on every interaction, so "loading
state" here means: a context manager around an API call (spinner), or a
short-lived skeleton placeholder rendered immediately before a slower
section populates. None of these functions make network calls — callers
wrap their own ``api_client`` calls with them.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st


@contextmanager
def loading_spinner(message: str = "Loading...") -> Iterator[None]:
    """Thin, named wrapper around ``st.spinner`` for call-site clarity.

    Usage:
        with loading_spinner("Fetching statistics..."):
            result = api_client.statistics()
    """
    with st.spinner(message):
        yield


def skeleton_cards(count: int = 3, *, columns: int = 3, height_px: int = 110) -> None:
    """Render ``count`` pulsing skeleton placeholder cards.

    Call this immediately before a slower data fetch, then overwrite the
    same area (e.g. via ``st.empty()``) once real content is ready.

    Args:
        count: Number of skeleton cards to render.
        columns: Number of columns to lay them out in.
        height_px: Height of each skeleton card in pixels.
    """
    _inject_skeleton_css()
    cols = st.columns(columns)
    for idx in range(count):
        with cols[idx % columns]:
            st.markdown(
                f'<div class="skeleton-card" style="height:{height_px}px;"></div>',
                unsafe_allow_html=True,
            )


def loading_metrics(count: int = 4) -> None:
    """Render ``count`` skeleton placeholders shaped like ``st.metric`` cards."""
    _inject_skeleton_css()
    cols = st.columns(count)
    for idx in range(count):
        with cols[idx]:
            st.markdown(
                """
                <div class="skeleton-card" style="height:78px;">
                    <div class="skeleton-line" style="width:55%; height:0.7rem; margin-top:0.6rem;"></div>
                    <div class="skeleton-line" style="width:35%; height:1.1rem; margin-top:0.5rem;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def loading_chart(height_px: int = 320) -> None:
    """Render a single skeleton placeholder shaped like a chart panel."""
    _inject_skeleton_css()
    st.markdown(
        f'<div class="skeleton-card" style="height:{height_px}px;"></div>',
        unsafe_allow_html=True,
    )


def loading_table(rows: int = 5, columns: int = 4) -> None:
    """Render a skeleton placeholder shaped like a data table.

    Args:
        rows: Number of skeleton rows.
        columns: Number of skeleton columns per row.
    """
    _inject_skeleton_css()
    st.markdown('<div class="skeleton-card" style="padding:0.9rem;">', unsafe_allow_html=True)
    for _ in range(rows):
        cols = st.columns(columns)
        for c in cols:
            with c:
                st.markdown('<div class="skeleton-line" style="height:0.8rem;"></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def loading_chat_response() -> None:
    """Render an animated 'thinking' placeholder for an in-flight chat reply."""
    _inject_skeleton_css()
    st.markdown(
        """
        <div class="skeleton-card" style="padding:1rem 1.25rem; display:flex; align-items:center; gap:0.6rem;">
            <span class="typing-dot"></span>
            <span class="typing-dot" style="animation-delay:0.15s;"></span>
            <span class="typing-dot" style="animation-delay:0.3s;"></span>
            <span style="color:var(--text-muted, #6B7280); font-size:0.9rem;">Generating response…</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_skeleton_css() -> None:
    """Inject skeleton/typing-indicator CSS once per render.

    Re-injecting this small style block on every call is cheap and
    avoids requiring callers to remember a separate setup step.
    """
    st.markdown(
        """
        <style>
        .skeleton-card {
            background: linear-gradient(
                90deg,
                var(--bg-secondary, #F5F7FB) 25%,
                rgba(11, 61, 145, 0.06) 37%,
                var(--bg-secondary, #F5F7FB) 63%
            );
            background-size: 400% 100%;
            animation: skeleton-shimmer 1.4s ease infinite;
            border: 1px solid var(--border, #E2E6EE);
            border-radius: 12px;
            margin-bottom: 0.6rem;
        }
        .skeleton-line {
            background: linear-gradient(
                90deg,
                var(--bg-secondary, #F5F7FB) 25%,
                rgba(11, 61, 145, 0.08) 37%,
                var(--bg-secondary, #F5F7FB) 63%
            );
            background-size: 400% 100%;
            animation: skeleton-shimmer 1.4s ease infinite;
            border-radius: 6px;
            margin-bottom: 0.4rem;
        }
        @keyframes skeleton-shimmer {
            0% { background-position: 100% 50%; }
            100% { background-position: 0 50%; }
        }
        .typing-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: var(--accent, #0B3D91);
            display: inline-block;
            animation: typing-bounce 1s infinite ease-in-out;
        }
        @keyframes typing-bounce {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )