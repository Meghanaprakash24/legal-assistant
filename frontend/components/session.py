"""
components/session.py
======================
Session-lifetime bookkeeping for the Indian Legal AI Assistant.

Tracks values that only exist on the frontend (session start time,
queries asked this session, conversation length) in ``st.session_state``.
This module does not call the backend — for backend-reported uptime, see
``components.caching.get_cached_statistics``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import streamlit as st

SESSION_ID_KEY: str = "shared_session_id"
SESSION_START_KEY: str = "shared_session_start_time"
QUERY_COUNT_KEY: str = "shared_query_count"
CONVERSATION_LENGTH_KEY: str = "shared_conversation_length"


def ensure_session_initialized() -> None:
    """Initialize session bookkeeping keys on first use. Idempotent —
    safe to call at the top of every page."""
    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())[:8]
    if SESSION_START_KEY not in st.session_state:
        st.session_state[SESSION_START_KEY] = datetime.now()
    if QUERY_COUNT_KEY not in st.session_state:
        st.session_state[QUERY_COUNT_KEY] = 0
    if CONVERSATION_LENGTH_KEY not in st.session_state:
        st.session_state[CONVERSATION_LENGTH_KEY] = 0


def get_session_id() -> str:
    """Return this session's short identifier, initializing if needed."""
    ensure_session_initialized()
    return st.session_state[SESSION_ID_KEY]


def get_session_start_time() -> datetime:
    """Return when this session started, initializing if needed."""
    ensure_session_initialized()
    return st.session_state[SESSION_START_KEY]


def increment_query_count() -> int:
    """Increment and return the number of queries asked this session.

    Call this once per user-submitted query (e.g. in the Legal Assistant
    page, right before calling ``api_client.chat(...)``).
    """
    ensure_session_initialized()
    st.session_state[QUERY_COUNT_KEY] += 1
    return st.session_state[QUERY_COUNT_KEY]


def set_conversation_length(message_count: int) -> None:
    """Set the current conversation length (total chat messages).

    Call this with ``len(st.session_state.messages)`` (or equivalent)
    whenever the chat history changes, so the Session Information block
    always reflects the live conversation.
    """
    ensure_session_initialized()
    st.session_state[CONVERSATION_LENGTH_KEY] = message_count


def estimate_session_memory_kb() -> float:
    """Best-effort estimate of this session's in-memory footprint, in KB.

    This sums the approximate size of everything currently held in
    ``st.session_state`` via ``sys.getsizeof`` on each value. It is a
    rough, frontend-only estimate (Python's shallow ``getsizeof`` does
    not follow nested references) — useful as a relative indicator, not
    an exact memory profile.
    """
    import sys

    total_bytes = 0
    for value in st.session_state.values():
        try:
            total_bytes += sys.getsizeof(value)
        except Exception:  # noqa: BLE001
            continue
    return round(total_bytes / 1024, 1)


def format_duration(start: datetime) -> str:
    """Format the elapsed time since ``start`` as e.g. '3h 21m' or '4m'."""
    elapsed = datetime.now() - start
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def render_session_info(*, columns: int = 5) -> None:
    """Render the standard Section 7 'Session Information' metric row.

    Displays: Current Session (id), Session Start Time, Queries Asked,
    Conversation Length, and an estimated Memory Usage figure.

    Args:
        columns: Number of metric columns to lay the values out in.
    """
    ensure_session_initialized()

    session_id = get_session_id()
    start_time = get_session_start_time()
    query_count = st.session_state[QUERY_COUNT_KEY]
    conversation_length = st.session_state[CONVERSATION_LENGTH_KEY]
    memory_kb = estimate_session_memory_kb()

    cols = st.columns(columns)
    cols[0].metric("Current Session", session_id)
    cols[1].metric("Session Start Time", start_time.strftime("%H:%M:%S"))
    cols[2].metric("Queries Asked", query_count)
    cols[3].metric("Conversation Length", conversation_length)
    cols[4].metric("Memory Usage", f"{memory_kb:,.1f} KB")