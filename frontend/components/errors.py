"""
components/errors.py
=====================
Beautiful, reusable error and empty-state cards for the Indian Legal AI
Assistant.

Every function here renders a self-contained, non-crashing UI block for
a specific failure or empty-data condition. None of these raise — they
are meant to be the *replacement* content shown instead of a stack
trace or a blank page when something upstream fails or returns nothing.
"""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st


def _state_card(
    *,
    icon: str,
    title: str,
    message: str,
    tone: str = "error",
    detail: Optional[str] = None,
    retry_label: Optional[str] = None,
    on_retry: Optional[Callable[[], None]] = None,
    key: Optional[str] = None,
) -> None:
    """Shared renderer for all error/empty states in this module.

    Args:
        icon: Large leading emoji/icon.
        title: Short, bold headline.
        message: One or two sentence explanation.
        tone: "error" | "warning" | "info" — controls accent color.
        detail: Optional technical detail shown in a collapsed expander.
        retry_label: If provided, renders a retry button with this label.
        on_retry: Callback invoked when the retry button is pressed. If
            omitted but ``retry_label`` is set, defaults to ``st.rerun()``.
        key: Optional unique key for the retry button (needed if this
            card is rendered more than once on the same page).
    """
    tone_colors = {
        "error": ("#FEE2E2", "#B91C1C"),
        "warning": ("#FEF3C7", "#B45309"),
        "info": ("#DBEAFE", "#1E40AF"),
    }
    bg, fg = tone_colors.get(tone, tone_colors["error"])

    st.markdown(
        f"""
        <div style="
            background-color: {bg}1A;
            border: 1px solid {bg};
            border-radius: 16px;
            padding: 2rem 1.75rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <div style="font-size: 2.4rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {fg}; margin-bottom: 0.35rem;">
                {title}
            </div>
            <div style="color: #4B5563; font-size: 0.95rem; max-width: 480px; margin: 0 auto;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if detail:
        with st.expander("Technical details"):
            st.code(detail)

    if retry_label:
        _, center, _ = st.columns([1, 1, 1])
        with center:
            if st.button(retry_label, type="primary", use_container_width=True, key=key):
                if on_retry is not None:
                    on_retry()
                else:
                    st.rerun()


def render_backend_offline(detail: Optional[str] = None, *, key: str = "retry_backend_offline") -> None:
    """Beautiful 'Backend Offline' state with a retry button."""
    _state_card(
        icon="🔴",
        title="Backend Offline",
        message="The Indian Legal AI backend isn't responding right now. "
        "It may be starting up, restarting, or temporarily down.",
        tone="error",
        detail=detail,
        retry_label="🔁 Retry Connection",
        key=key,
    )


def render_api_error(detail: Optional[str] = None, *, key: str = "retry_api_error") -> None:
    """Beautiful generic API error state with a retry button."""
    _state_card(
        icon="⚠️",
        title="API Error",
        message="Something went wrong while talking to the backend. "
        "This is usually temporary — try again in a moment.",
        tone="error",
        detail=detail,
        retry_label="🔁 Try Again",
        key=key,
    )


def render_timeout(detail: Optional[str] = None, *, key: str = "retry_timeout") -> None:
    """Beautiful request-timeout state with a retry button."""
    _state_card(
        icon="⏱️",
        title="Request Timed Out",
        message="The backend took too long to respond. The pipeline may "
        "be under heavy load — please try again.",
        tone="warning",
        detail=detail,
        retry_label="🔁 Retry",
        key=key,
    )


def render_no_results(query: Optional[str] = None) -> None:
    """Beautiful 'No Results' state for an empty retrieval/search response."""
    message = (
        f'No relevant legal provisions were found for "{query}". '
        "Try rephrasing your query or using more specific legal terms."
        if query
        else "No relevant legal provisions were found for this query. "
        "Try rephrasing or using more specific legal terms."
    )
    _state_card(icon="🔍", title="No Results Found", message=message, tone="info")


def render_empty_search() -> None:
    """Beautiful prompt shown before the user has searched for anything."""
    _state_card(
        icon="⚖️",
        title="Ask a Legal Question",
        message="Type a question above to search Indian legal provisions, "
        "case-relevant sections, and grounded citations.",
        tone="info",
    )


def render_no_citations() -> None:
    """Beautiful 'No Citations' empty state."""
    _state_card(
        icon="❝",
        title="No Citations Available",
        message="This response doesn't have any validated citations yet. "
        "Citations appear once the response passes validation.",
        tone="info",
    )


def render_no_retrieved_documents() -> None:
    """Beautiful 'No Retrieved Documents' empty state."""
    _state_card(
        icon="📄",
        title="No Documents Retrieved",
        message="No source documents were retrieved for this query. "
        "This can happen if the knowledge base has no matching content.",
        tone="info",
    )


def render_no_statistics() -> None:
    """Beautiful 'No Statistics' empty state."""
    _state_card(
        icon="📊",
        title="No Statistics Available",
        message="Runtime statistics aren't available yet. They populate "
        "once the backend has handled at least one request.",
        tone="info",
    )