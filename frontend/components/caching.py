"""
components/caching.py
======================
Shared ``st.cache_data`` / ``st.cache_resource`` wrappers for the Indian
Legal AI Assistant.

Streamlit reruns the entire script on every interaction, so without
caching, every widget click on every page would re-issue a fresh
``GET /health`` or ``GET /statistics`` call. These wrappers give every
page a short, shared cache window instead.

Only the two real, side-effect-free GET endpoints exposed by
``services/api.py`` (``health()`` and ``statistics()``) are wrapped here.
``chat``/``retrieve``/``classify``/``rerank``/``validate`` are POST calls
with query-specific results and are intentionally NOT cached — caching a
chat response by argument would mean a repeated question silently
returns a stale answer instead of hitting the live pipeline.

TTLs are short (a few seconds) by design: this is about collapsing
*redundant* calls within the same rerun/interaction burst, not about
serving long-stale system status.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

try:
    from services.api import LegalAPIClient, api_client as _default_api_client  # type: ignore
except ImportError:
    LegalAPIClient = None  # type: ignore
    _default_api_client = None  # type: ignore

HEALTH_TTL_SECONDS: int = 5
STATISTICS_TTL_SECONDS: int = 5
CONFIG_TTL_SECONDS: int = 300


@st.cache_resource(show_spinner=False)
def get_shared_api_client() -> Any:
    """Return the process-wide shared API client as an ``st.cache_resource``.

    ``services.api`` already constructs one module-level ``api_client``
    instance (a persistent ``requests.Session``), which is the correct
    pattern. This wrapper exists so pages that want the *resource-cache*
    guarantee explicitly (e.g. "never construct a second client") can
    depend on this function instead of importing the module-level
    singleton directly. Both return the same underlying object.

    Returns:
        The shared ``LegalAPIClient`` instance, or ``None`` if
        ``services.api`` could not be imported.
    """
    return _default_api_client


@st.cache_data(ttl=HEALTH_TTL_SECONDS, show_spinner=False)
def get_cached_health() -> dict[str, Any]:
    """Cached wrapper around ``api_client.health()`` (GET /health).

    Returns:
        The standard response envelope:
        ``{"success": bool, "data": {...}}`` or
        ``{"success": False, "error": str, "status_code": int | None}``.
    """
    client = get_shared_api_client()
    if client is None:
        return {"success": False, "error": "services.api.api_client not available", "status_code": None}
    return client.health()


@st.cache_data(ttl=STATISTICS_TTL_SECONDS, show_spinner=False)
def get_cached_statistics() -> dict[str, Any]:
    """Cached wrapper around ``api_client.statistics()`` (GET /statistics).

    Returns:
        The standard response envelope (see :func:`get_cached_health`).
    """
    client = get_shared_api_client()
    if client is None:
        return {"success": False, "error": "services.api.api_client not available", "status_code": None}
    return client.statistics()


@st.cache_data(ttl=CONFIG_TTL_SECONDS, show_spinner=False)
def get_cached_config_snapshot() -> dict[str, Any]:
    """Cached, read-only snapshot of the handful of ``config.py`` values
    that are safe to surface in the UI (no secrets).

    A long TTL is appropriate here since these are process-startup
    constants that don't change during a running session.

    Returns:
        Dict of non-secret configuration values, or an empty dict if
        ``config`` could not be imported.
    """
    try:
        import config as app_config  # type: ignore
    except ImportError:
        return {}

    return {
        "embedding_model": getattr(app_config, "EMBEDDING_MODEL_NAME", None),
        "reranker_model": getattr(app_config, "RERANKER_MODEL_NAME", None),
        "reranker_top_k": getattr(app_config, "RERANKER_TOP_K", None),
        "groq_model": getattr(app_config, "GROQ_MODEL_NAME", None),
        "temperature": getattr(app_config, "TEMPERATURE", None),
        "max_tokens": getattr(app_config, "MAX_TOKENS", None),
        "collection_name": getattr(app_config, "COLLECTION_NAME", None),
        "default_top_k": getattr(app_config, "DEFAULT_TOP_K", None),
        "dense_top_k": getattr(app_config, "DENSE_TOP_K", None),
        "bm25_top_k": getattr(app_config, "BM25_TOP_K", None),
        "fusion_strategy": getattr(app_config, "FUSION_STRATEGY", None),
        "confidence_threshold": getattr(app_config, "CONFIDENCE_THRESHOLD", None),
        "api_host": getattr(app_config, "API_HOST", None),
        "api_port": getattr(app_config, "API_PORT", None),
    }


def clear_all_caches() -> None:
    """Clear every cache populated by this module.

    Intended for a "Refresh Configuration" / "Reset Cache" quick action
    button — forces the next read to hit the backend again instead of
    serving a stale cached value.
    """
    get_cached_health.clear()
    get_cached_statistics.clear()
    get_cached_config_snapshot.clear()