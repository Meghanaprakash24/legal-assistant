"""
services/api.py
================
Reusable API service layer for the Indian Legal AI Assistant Streamlit
frontend.

This module is responsible *only* for talking to the FastAPI backend.
It contains no UI code whatsoever. Every Streamlit page should import
the shared client instance:

    from services.api import api_client

    result = api_client.chat("What is Section 420 IPC?")
    if result["success"]:
        st.write(result["data"])
    else:
        st.error(result["error"])

Design goals
------------
- Single persistent ``requests.Session`` for connection re-use.
- Centralized error handling (connection errors, timeouts, HTTP errors,
  JSON decoding errors) via private ``_get`` / ``_post`` helpers.
- Automatic retry with fixed backoff for transient failures.
- A consistent response envelope for every public method:

    {"success": True, "data": <parsed JSON>}

  or

    {"success": False, "error": "<message>", "status_code": <int | None>}

- Structured logging of every request (URL, method, latency, status
  code) and of any errors encountered.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, JSONDecodeError, RequestException, Timeout

from config import API_BASE_URL

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logger = logging.getLogger("legal_ai_frontend.api")
if not logger.handlers:
    # Avoid attaching duplicate handlers if this module is re-imported
    # (e.g. Streamlit's hot-reload) within the same interpreter session.
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_TIMEOUT_SECONDS: int = 120
MAX_RETRIES: int = 1
RETRY_BACKOFF_SECONDS: float = 1.0

# Standard response envelope type alias for readability.
APIResponse = Dict[str, Any]


class LegalAPIClient:
    """
    Thin, reusable HTTP client for the Indian Legal AI Assistant backend.

    The client wraps a persistent :class:`requests.Session` and exposes
    one method per backend endpoint. Every method returns a normalized
    dictionary envelope (see module docstring) so that calling UI code
    never needs to handle raw exceptions or inconsistent payload shapes.

    Attributes:
        base_url: Root URL of the FastAPI backend, sourced from
            ``config.API_BASE_URL``. Never hardcoded.
        timeout: Per-request timeout, in seconds.
        session: Persistent ``requests.Session`` used for all calls.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the API client.

        Args:
            base_url: Override for the backend base URL. Defaults to
                ``config.API_BASE_URL`` when not provided.
            timeout: Request timeout in seconds. Defaults to 30.
        """
        self.base_url: str = (base_url or API_BASE_URL).rstrip("/")
        self.timeout: int = timeout
        self.session: requests.Session = requests.Session()
        logger.info("LegalAPIClient initialized with base_url=%s", self.base_url)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_url(self, endpoint: str) -> str:
        """
        Build a fully-qualified URL from an endpoint path.

        Args:
            endpoint: Path such as ``"/health"``.

        Returns:
            The absolute URL, e.g. ``"http://localhost:8000/health"``.
        """
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """
        Execute an HTTP request with retry-and-backoff logic, normalizing
        all outcomes (success or failure) into the standard response
        envelope.

        Retries occur on connection errors and timeouts, which are
        considered transient. HTTP error status codes are NOT retried,
        since they typically indicate a deterministic client/server
        issue (e.g. 400, 404, 422) rather than a transient network
        problem.

        Args:
            method: HTTP verb, ``"GET"`` or ``"POST"``.
            url: Fully-qualified request URL.
            json_body: Optional JSON-serializable request payload.

        Returns:
            Standard response envelope dict.
        """
        last_error: str = "Unknown error"
        last_status_code: Optional[int] = None

        for attempt in range(1, MAX_RETRIES + 1):
            start_time = time.monotonic()
            try:
                logger.info("Requested URL: %s", url)
                response: Response = self.session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=self.timeout,
                )
                latency_ms = (time.monotonic() - start_time) * 1000

                logger.info(
                    "%s %s -> status=%s latency=%.1fms attempt=%d/%d",
                    method,
                    url,
                    response.status_code,
                    latency_ms,
                    attempt,
                    MAX_RETRIES,
                )
                logger.info("HTTP status code: %s", response.status_code)
                logger.info("Raw JSON received: %s", response.text)

                response.raise_for_status()

                try:
                    data = response.json()
                except JSONDecodeError as exc:
                    logger.error(
                        "JSON decode error for %s %s status=%s raw=%r: %s",
                        method,
                        url,
                        response.status_code,
                        response.text,
                        exc,
                    )
                    return {
                        "success": False,
                        "error": f"Invalid JSON received from backend: {exc}",
                        "status_code": response.status_code,
                    }

                logger.info("Parsed JSON: %s", data)
                return {"success": True, "data": data}

            except HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                detail = self._extract_http_error_detail(exc.response)
                logger.error(
                    "HTTP error on %s %s: status=%s detail=%s",
                    method,
                    url,
                    status_code,
                    detail,
                )
                # HTTP errors are deterministic (e.g. validation errors,
                # not found) -> do not retry, fail fast.
                return {
                    "success": False,
                    "error": detail,
                    "status_code": status_code,
                }

            except Timeout as exc:
                last_error = (
                    f"Request to {url} timed out after {self.timeout}s. "
                    "Confirm the FastAPI backend is responsive and try again."
                )
                last_status_code = None
                logger.warning(
                    "Timeout on %s %s (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

            except RequestsConnectionError as exc:
                last_error = (
                    f"Connection error while reaching backend at {url}: {exc}. "
                    f"Expected FastAPI at {self.base_url}."
                )
                last_status_code = None
                logger.warning(
                    "Connection error on %s %s (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

            except RequestException as exc:
                # Catch-all for any other requests-related failure.
                last_error = f"Request to {url} failed: {exc}"
                last_status_code = None
                logger.error(
                    "Unexpected request exception on %s %s (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

            # Backoff before next retry, unless this was the last attempt.
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)

        logger.error(
            "All %d attempts failed for %s %s: %s", MAX_RETRIES, method, url, last_error
        )
        return {
            "success": False,
            "error": last_error,
            "status_code": last_status_code,
        }

    @staticmethod
    def _extract_http_error_detail(response: Optional[Response]) -> str:
        """
        Attempt to extract a human-readable error message from a failed
        HTTP response body (FastAPI typically returns ``{"detail": ...}``).

        Args:
            response: The ``requests.Response`` object, if available.

        Returns:
            A descriptive error string.
        """
        if response is None:
            return "HTTP error with no response object available."

        try:
            payload = response.json()
            if isinstance(payload, dict) and "detail" in payload:
                return f"Backend error ({response.status_code}): {payload['detail']}"
            return f"Backend error ({response.status_code}): {payload}"
        except JSONDecodeError:
            return f"Backend error ({response.status_code}): {response.text}"

    def _get(self, endpoint: str) -> APIResponse:
        """
        Perform a GET request against the backend with retry handling.

        Args:
            endpoint: API path, e.g. ``"/health"``.

        Returns:
            Standard response envelope dict.
        """
        url = self._build_url(endpoint)
        return self._request_with_retry("GET", url)

    def get(self, endpoint: str) -> APIResponse:
        """Public GET wrapper that keeps pages on the shared API client."""
        return self._get(endpoint)

    def _post(self, endpoint: str, json_body: Dict[str, Any]) -> APIResponse:
        """
        Perform a POST request against the backend with retry handling.

        Args:
            endpoint: API path, e.g. ``"/chat"``.
            json_body: JSON-serializable request payload.

        Returns:
            Standard response envelope dict.
        """
        url = self._build_url(endpoint)
        return self._request_with_retry("POST", url, json_body=json_body)

    def post(self, endpoint: str, json_body: Dict[str, Any]) -> APIResponse:
        """Public POST wrapper that keeps pages on the shared API client."""
        return self._post(endpoint, json_body=json_body)

    # ------------------------------------------------------------------ #
    # Public endpoint methods
    # ------------------------------------------------------------------ #
    def health(self) -> APIResponse:
        """
        Check backend health.

        Endpoint:
            GET /health

        Returns:
            Standard response envelope containing backend status data.
        """
        result = self._get("/health")
        if not result.get("success"):
            logger.error("Health check failed: %s", result.get("error"))
            return result

        data = result.get("data")
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": f"Unexpected health payload type: {type(data).__name__}",
                "status_code": result.get("status_code"),
            }

        normalized = self.normalize_health_payload(data)
        logger.info("Normalized JSON: %s", normalized)
        return {
            "success": True,
            "data": normalized,
            "status_code": result.get("status_code", 200),
        }

    def statistics(self) -> APIResponse:
        """
        Retrieve system statistics from the backend.

        Endpoint:
            GET /statistics

        Returns:
            Standard response envelope containing statistics data.
        """
        return self._get("/statistics")

    def chat(self, query: str) -> APIResponse:
        """
        Send a chat query to the legal assistant pipeline.

        Endpoint:
            POST /chat

        Args:
            query: The user's natural-language legal question.

        Returns:
            Standard response envelope containing the chat response data.
        """
        return self._post("/chat", {"query": query})

    def retrieve(self, query: str, top_k: int = 10) -> APIResponse:
        """
        Retrieve relevant legal document chunks for a query.

        Endpoint:
            POST /retrieve

        Args:
            query: The search query.
            top_k: Number of top results to retrieve. Defaults to 10.

        Returns:
            Standard response envelope containing retrieved chunks.
        """
        return self._post("/retrieve", {"query": query, "top_k": top_k})

    def classify(self, query: str) -> APIResponse:
        """
        Classify a legal query (e.g. by domain, intent, or act).

        Endpoint:
            POST /classify

        Args:
            query: The text to classify.

        Returns:
            Standard response envelope containing classification data.
        """
        return self._post("/classify", {"query": query})

    def rerank(self, query: str, chunks: List[Any]) -> APIResponse:
        """
        Rerank a list of retrieved chunks against a query.

        Endpoint:
            POST /rerank

        Args:
            query: The original search query.
            chunks: List of candidate chunks to rerank (shape defined by
                backend; typically a list of dicts).

        Returns:
            Standard response envelope containing reranked chunks.
        """
        return self._post("/rerank", {"query": query, "chunks": chunks})

    def validate(self, response_text: str, citations: List[Any]) -> APIResponse:
        """
        Validate a generated response against its cited sources.

        Endpoint:
            POST /validate

        Args:
            response_text: The generated answer text to validate.
            citations: List of citation objects/strings supporting the
                response.

        Returns:
            Standard response envelope containing validation results.
        """
        return self._post(
            "/validate",
            {"response_text": response_text, "citations": citations},
        )

    # ------------------------------------------------------------------ #
    # Convenience / status helper methods
    # ------------------------------------------------------------------ #
    def backend_online(self) -> bool:
        """
        Quick boolean check for whether the backend is reachable and
        healthy.

        Returns:
            True if the ``/health`` call succeeded, False otherwise.
        """
        result = self.health()
        if not result.get("success"):
            return False
        data = result.get("data", {})
        if not isinstance(data, dict):
            return False
        components = data.get("components", {})
        return bool(
            isinstance(components, dict)
            and (components.get("fastapi") or data.get("status") == "healthy")
        )

    def get_backend_status(self) -> Dict[str, bool]:
        """
        Build a normalized component-level status dictionary describing
        which backend subsystems are reported as healthy.

        Uses:
            GET /health

        The backend's ``/health`` payload is expected to expose nested
        component statuses (e.g. ``{"components": {"fastapi": "ok", ...}}``
        or flat boolean fields). This method tries to be tolerant of
        either shape and falls back to ``False`` for components it
        cannot find.

        Returns:
            Dictionary in the form::

                {
                    "fastapi": True,
                    "groq": True,
                    "qdrant": True,
                    "pipeline": True,
                }
        """
        default_status: Dict[str, bool] = {
            "fastapi": False,
            "groq": False,
            "qdrant": False,
            "pipeline": False,
        }

        result = self.health()
        if not result.get("success"):
            logger.warning(
                "get_backend_status: health check failed (%s)", result.get("error")
            )
            return default_status

        data = result.get("data", {})
        if not isinstance(data, dict):
            logger.warning("get_backend_status: unexpected health payload shape.")
            return default_status

        components = data.get("components") or data.get("services") or data

        for key in ("fastapi", "groq", "qdrant", "pipeline"):
            value = components.get(key) if isinstance(components, dict) else None
            default_status[key] = self._coerce_to_bool(value)

        return default_status

    @classmethod
    def normalize_health_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize varied backend health payloads into the dashboard shape.

        The current backend returns a flat shape:
        ``{"status": "healthy", "pipeline": true, "qdrant": true, "groq": true}``.
        Frontend widgets expect:
        ``{"components": {"fastapi": true, "groq": true, "qdrant": true, "pipeline": true}}``.
        The original response is preserved under ``raw`` for diagnostics.
        """
        if not isinstance(payload, dict):
            payload = {}

        components = payload.get("components") or payload.get("services") or {}
        if not isinstance(components, dict):
            components = {}

        status_value = payload.get("status") or components.get("fastapi")
        fastapi_ok = cls._coerce_to_bool(status_value) or bool(payload)

        normalized_components = {
            "fastapi": fastapi_ok,
            "groq": cls._first_health_bool(payload, components, "groq", "llm"),
            "qdrant": cls._first_health_bool(payload, components, "qdrant", "vector_db"),
            "pipeline": cls._first_health_bool(payload, components, "pipeline", "rag_pipeline"),
        }

        status = str(payload.get("status") or "").strip().lower()
        if not status:
            status = "healthy" if all(normalized_components.values()) else "degraded"

        normalized = dict(payload)
        normalized["status"] = status
        normalized["components"] = normalized_components
        normalized["fastapi"] = normalized_components["fastapi"]
        normalized["groq"] = normalized_components["groq"]
        normalized["qdrant"] = normalized_components["qdrant"]
        normalized["pipeline"] = normalized_components["pipeline"]
        normalized["raw"] = payload
        return normalized

    @classmethod
    def _first_health_bool(
        cls,
        payload: Dict[str, Any],
        components: Dict[str, Any],
        *keys: str,
    ) -> bool:
        """Return the first health-like boolean found across payload shapes."""
        for key in keys:
            if key in components:
                return cls._coerce_to_bool(components.get(key))
            if key in payload:
                return cls._coerce_to_bool(payload.get(key))
        return False

    @staticmethod
    def _coerce_to_bool(value: Any) -> bool:
        """
        Normalize varied "healthy" indicators (bool, status strings,
        numeric codes) into a plain boolean.

        Args:
            value: Raw value from the health payload.

        Returns:
            True if the value indicates a healthy/online state.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"ok", "true", "healthy", "online", "up"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def check_connection(self) -> APIResponse:
        """
        Perform a startup connectivity check against the backend.

        Intended to be called once when the Streamlit app boots, so the
        UI layer can decide whether to show a "backend offline" warning
        before rendering pages that depend on the API.

        Returns:
            Standard response envelope. On success, ``data`` contains
            the raw health payload. On failure, ``error`` describes why
            the connection check failed.
        """
        logger.info("Running startup connection check against %s", self.base_url)
        result = self.health()
        if result.get("success"):
            logger.info("Startup connection check succeeded.")
        else:
            logger.error(
                "Startup connection check failed: %s", result.get("error")
            )
        return result


# -------------------------------------------------------------------------- #
# Shared client instance
# -------------------------------------------------------------------------- #
# Every Streamlit page should import this single instance rather than
# constructing its own client, so the underlying requests.Session (and
# its connection pool) is reused across the whole app.
api_client = LegalAPIClient()


def check_health(base_url: Optional[str] = None) -> tuple[bool, str]:
    """
    Compatibility helper for pages that test a user-supplied backend URL.

    Returns:
        ``(True, "Connected")`` when the health endpoint responds with a
        usable payload, otherwise ``(False, "<useful error>")``.
    """
    client = api_client if base_url is None else LegalAPIClient(base_url=base_url)
    result = client.health()
    if result.get("success"):
        return True, "Connected"
    return False, str(result.get("error", "Connection Failed"))
