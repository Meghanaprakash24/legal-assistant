"""
components/notifications.py
============================
Toast-style notifications for the Indian Legal AI Assistant.

Wraps Streamlit's native ``st.toast`` (with a graceful fallback to
``st.success`` / ``st.warning`` / ``st.error`` / ``st.info`` on older
Streamlit versions that don't have ``st.toast``) so every page raises
notifications the same way, with the same icons and wording for common
backend-connectivity events.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

NotificationKind = Literal["success", "warning", "error", "info"]

_ICONS: dict[NotificationKind, str] = {
    "success": "✅",
    "warning": "⚠️",
    "error": "🔴",
    "info": "ℹ️",
}

_HAS_TOAST: bool = hasattr(st, "toast")


def _dispatch(kind: NotificationKind, message: str) -> None:
    """Send a message via st.toast if available, else the matching alert."""
    icon = _ICONS[kind]
    if _HAS_TOAST:
        try:
            st.toast(message, icon=icon)
            return
        except Exception:  # noqa: BLE001 - fall through to alert box
            pass

    fallback = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }[kind]
    fallback(f"{icon} {message}")


def notify_success(message: str) -> None:
    """Show a success toast/alert."""
    _dispatch("success", message)


def notify_warning(message: str) -> None:
    """Show a warning toast/alert."""
    _dispatch("warning", message)


def notify_error(message: str) -> None:
    """Show an error toast/alert."""
    _dispatch("error", message)


def notify_info(message: str) -> None:
    """Show an informational toast/alert."""
    _dispatch("info", message)


# =============================================================================
# Standard connectivity notifications
#
# These wrap the four specific events called out in the spec so every page
# announces backend/Groq/Qdrant connectivity changes with identical wording.
# =============================================================================


def notify_backend_connected() -> None:
    """Standard 'Backend Connected' notification."""
    notify_success("Backend Connected")


def notify_backend_offline() -> None:
    """Standard 'Backend Offline' notification."""
    notify_error("Backend Offline")


def notify_groq_connected() -> None:
    """Standard 'Groq Connected' notification."""
    notify_success("Groq Connected")


def notify_qdrant_connected() -> None:
    """Standard 'Qdrant Connected' notification."""
    notify_success("Qdrant Connected")


def notify_backend_status(*, pipeline_ok: bool, qdrant_ok: bool, groq_ok: bool) -> None:
    """Fire the appropriate set of connectivity toasts from one /health read.

    Intended to be called once per fresh health check (e.g. right after
    ``api_client.health()``), so pages don't have to hand-wire each
    individual notify_* call.

    Args:
        pipeline_ok: Whether the backend/pipeline itself is reachable.
        qdrant_ok: Whether Qdrant reported healthy.
        groq_ok: Whether Groq reported healthy (configured).
    """
    if pipeline_ok:
        notify_backend_connected()
    else:
        notify_backend_offline()

    if qdrant_ok:
        notify_qdrant_connected()
    else:
        notify_warning("Qdrant Offline")

    if groq_ok:
        notify_groq_connected()
    else:
        notify_warning("Groq Offline")