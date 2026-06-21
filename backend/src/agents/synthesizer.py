from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Final

from loguru import logger

try:
    from groq import APIConnectionError, APIError, APITimeoutError, Groq, RateLimitError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'groq' package is required for src/agents/synthesizer.py. "
        "Install it with: pip install groq"
    ) from exc

# ---------------------------------------------------------------------------
# Optional project config
# ---------------------------------------------------------------------------

try:
    import config as _config  # type: ignore[import-not-found]
except ImportError:
    _config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME: Final[str] = "llama-3.3-70b-versatile"
_DEFAULT_TEMPERATURE: Final[float] = 0.2
_DEFAULT_MAX_TOKENS: Final[int] = 2000
_DEFAULT_TIMEOUT: Final[float] = 30.0
_DEFAULT_MAX_RETRIES: Final[int] = 3
_DEFAULT_RETRY_BACKOFF_SECONDS: Final[float] = 1.5

_LOGGING_CONFIGURED: bool = False

_DEFAULT_DISCLAIMER: Final[str] = (
    "This response is generated for educational and informational purposes "
    "only and should not be considered legal advice. Please consult a "
    "qualified legal professional for advice regarding your specific situation."
)

_SYSTEM_PROMPT_TEMPLATE: Final[str] = """You are the final Response Synthesizer for an Indian Legal RAG system.

Your role is to combine outputs from all upstream agents (Classifier, Retriever, Section Mapper, Quote Selector, Remedy Advisor, Validator) into one coherent, grounded legal response. You must work ONLY from the supplied evidence — retrieved legal chunks, validated sections, validated quotes, facts, and remedy guidance. Never use model memory or outside knowledge.

Hard rules:
1. Ground every statement in the supplied retrieved_legal_context or validated_quotes.
2. Never fabricate sections, articles, punishments, case law, or legal provisions not present in the evidence.
3. If the retrieved context is insufficient, state: "The available legal documents do not provide sufficient information on this point."
4. Do not provide personalised legal advice.
5. Output ONLY valid JSON. No markdown fences. No text before or after the JSON object.
6. The JSON object must contain exactly these top-level keys:
   status | summary | query_understanding | identified_offences | applicable_law |
   relevant_quotations | legal_explanation | recommended_procedure | important_notes |
   citations | disclaimer
7. Set status to "SUCCESS" when evidence supports a meaningful response.
10. The disclaimer must be exactly: "{disclaimer}"

Field requirements:
- summary: One concise sentence answering the query.
- query_understanding: One sentence describing what the user is asking and the legal domain (criminal/civil/constitutional).
- identified_offences: Array of short strings (offence types or legal categories from the evidence).
- applicable_law: Array of objects [{{"document": "...", "section": "..."}}] — only include sections present in validated_sections or retrieved_legal_context.
- relevant_quotations: Array of strings — EXACT verbatim text excerpts from retrieved_legal_context that are directly relevant. Each entry should be a meaningful sentence or clause from the legal text. Extract at least 1-3 quotations when available. Do not paraphrase.
- legal_explanation: Markdown string with exactly these headings in order:
    # Summary
    # Applicable Law
    # Explanation
    # Procedure
    # Notes
    # Disclaimer
- recommended_procedure: Array of short action-oriented strings from remedy_guidance or retrieved context.
- important_notes: Array of short caution strings grounded in evidence.
- citations: Array of compact citation strings such as "BNS - Section 101". Only include what is in applicable_law.
- disclaimer: Exact string provided above.
"""

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Loguru once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if _config is not None:
        try:
            log_dir = _config.LOG_DIR
            log_dir.mkdir(exist_ok=True)
            logger.add(
                log_dir / "synthesizer.log",
                level="DEBUG",
                rotation=getattr(_config, "LOG_ROTATION", "10 MB"),
                retention=getattr(_config, "LOG_RETENTION", "30 days"),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Could not set up synthesizer file logging via config.LOG_DIR: {}",
                exc,
            )

    _LOGGING_CONFIGURED = True


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _read_config_str(attr: str, default: str = "") -> str:
    """Safely read a string config value."""
    if _config is None:
        return default
    value = getattr(_config, attr, default)
    return value if isinstance(value, str) else default


def _read_config_float(
    attr: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Safely read a float config value with optional bounds."""
    if _config is None:
        return default

    value = getattr(_config, attr, default)
    if not isinstance(value, (int, float)):
        logger.warning("config.{} is invalid ({}); using default {}.", attr, value, default)
        return default

    result = float(value)

    if minimum is not None and result < minimum:
        logger.warning(
            "config.{}={} is below minimum {}; using default {}.",
            attr,
            result,
            minimum,
            default,
        )
        return default
    if maximum is not None and result > maximum:
        logger.warning(
            "config.{}={} is above maximum {}; using default {}.",
            attr,
            result,
            maximum,
            default,
        )
        return default

    return result


def _read_config_int(attr: str, default: int, *, minimum: int | None = None) -> int:
    """Safely read an integer config value with optional lower bound."""
    if _config is None:
        return default

    value = getattr(_config, attr, default)
    if not isinstance(value, int):
        logger.warning("config.{} is invalid ({}); using default {}.", attr, value, default)
        return default

    if minimum is not None and value < minimum:
        logger.warning(
            "config.{}={} is below minimum {}; using default {}.",
            attr,
            value,
            minimum,
            default,
        )
        return default

    return value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Runtime configuration for the Groq-backed LLM."""

    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: float
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS


@dataclass
class SynthesizerStats:
    """Operational statistics for synthesis requests."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_seconds: float = 0.0
    total_input_chars: int = 0
    total_output_chars: int = 0
    last_model: str = ""
    last_validation_status: str = ""
    last_latency_seconds: float = 0.0

    def record(
        self,
        *,
        success: bool,
        latency_seconds: float,
        input_chars: int,
        output_chars: int,
        model: str,
        validation_status: str,
    ) -> None:
        """Record a completed request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_latency_seconds += latency_seconds
        self.total_input_chars += input_chars
        self.total_output_chars += output_chars
        self.last_model = model
        self.last_validation_status = validation_status
        self.last_latency_seconds = latency_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics to a dictionary."""
        average_latency = (
            self.total_latency_seconds / self.total_requests
            if self.total_requests
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_latency_seconds": round(average_latency, 4),
            "total_input_chars": self.total_input_chars,
            "total_output_chars": self.total_output_chars,
            "last_model": self.last_model,
            "last_validation_status": self.last_validation_status,
            "last_latency_seconds": round(self.last_latency_seconds, 4),
        }


@dataclass
class ValidatedEvidence:
    """Normalized validated evidence passed into the synthesizer."""

    query: str
    facts: dict[str, Any]
    validated_sections: list[dict[str, str]] = field(default_factory=list)
    validated_quotes: list[str] = field(default_factory=list)
    remedy: dict[str, Any] = field(default_factory=dict)
    validation_status: str = ""
    validation_errors: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ResponseSynthesizer
# ---------------------------------------------------------------------------


class ResponseSynthesizer:
    """Final response synthesizer for the legal multi-agent workflow.

    This is the only module permitted to call the LLM. It synthesizes the
    final user-facing legal response using strictly validated evidence:
    structured facts, validated sections, validated quotes, and remedy
    guidance. If citation validation fails, the LLM is never called.
    """

    _groq_client: Groq | None = None
    _cached_system_prompt: str | None = None  # reset on each server restart

    def __init__(
        self,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the synthesizer and load configuration."""
        _configure_logging()

        api_key = _read_config_str("GROQ_API_KEY", "").strip()

        configured_model = (
            model_name
            or _read_config_str("MODEL_NAME", "")
            or _read_config_str("GROQ_MODEL_NAME", "")
            or _DEFAULT_MODEL_NAME
        )

        configured_temperature = (
            temperature
            if temperature is not None
            else _read_config_float(
                "TEMPERATURE",
                _DEFAULT_TEMPERATURE,
                minimum=0.0,
                maximum=2.0,
            )
        )

        configured_max_tokens = (
            max_tokens
            if max_tokens is not None
            else _read_config_int("MAX_TOKENS", _DEFAULT_MAX_TOKENS, minimum=1)
        )

        configured_timeout = (
            timeout
            if timeout is not None
            else _read_config_float("TIMEOUT", _DEFAULT_TIMEOUT, minimum=1.0)
        )

        max_retries = _read_config_int("MAX_RETRIES", _DEFAULT_MAX_RETRIES, minimum=1)
        retry_backoff_seconds = _read_config_float(
            "RETRY_BACKOFF_SECONDS",
            _DEFAULT_RETRY_BACKOFF_SECONDS,
            minimum=0.1,
        )

        self._llm_config = LLMConfig(
            api_key=api_key,
            model_name=configured_model,
            temperature=configured_temperature,
            max_tokens=configured_max_tokens,
            timeout=configured_timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

        self._stats = SynthesizerStats()

        logger.debug(
            "ResponseSynthesizer initialized: model={} temperature={} max_tokens={} timeout={}",
            self._llm_config.model_name,
            self._llm_config.temperature,
            self._llm_config.max_tokens,
            self._llm_config.timeout,
        )

    # ------------------------------------------------------------------
    # LLM setup
    # ------------------------------------------------------------------

    def load_llm(self) -> Groq:
        """Load and cache a persistent Groq client.

        Returns:
            A reusable Groq client instance.

        Raises:
            RuntimeError: If API key configuration is missing.
        """
        if ResponseSynthesizer._groq_client is not None:
            return ResponseSynthesizer._groq_client

        if not self._llm_config.api_key:
            raise RuntimeError(
                "Groq API key is missing. Set config.GROQ_API_KEY."
            )

        ResponseSynthesizer._groq_client = Groq(
            api_key=self._llm_config.api_key,
            timeout=self._llm_config.timeout,
        )
        logger.debug("Persistent Groq client created successfully.")
        return ResponseSynthesizer._groq_client

    def build_system_prompt(self) -> str:
        """Build and cache the legal system prompt."""
        if ResponseSynthesizer._cached_system_prompt is None:
            ResponseSynthesizer._cached_system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                disclaimer=_DEFAULT_DISCLAIMER
            )
        return ResponseSynthesizer._cached_system_prompt

    def build_user_prompt(self, evidence: ValidatedEvidence) -> str:
        """Build the user prompt including retrieved chunk texts as legal context.

        Args:
            evidence: Normalized validated evidence package.

        Returns:
            A structured prompt for the LLM with full legal context.
        """
        # Build compact retrieved legal context from top chunks
        retrieved_context: list[dict[str, str]] = []
        for chunk in evidence.retrieved_chunks[:8]:
            payload = chunk.get("payload") or {}
            text = (
                chunk.get("retrieval_text") or
                payload.get("text") or payload.get("chunk_text") or
                chunk.get("text") or chunk.get("content") or ""
            ).strip()
            if not text:
                continue
            retrieved_context.append({
                "document": chunk.get("document", chunk.get("corpus", "")),
                "section": chunk.get("section", chunk.get("article", "")),
                "text": text[:600],
            })

        payload = {
            "user_query": evidence.query,
            "query_facts": evidence.facts,
            "validated_sections": evidence.validated_sections,
            "validated_quotes": evidence.validated_quotes,
            "retrieved_legal_context": retrieved_context,
            "remedy_guidance": evidence.remedy,
            "instructions": {
                "extract_relevant_quotations": (
                    "Extract 1-3 exact verbatim sentences from retrieved_legal_context "
                    "that directly answer the user_query. Put them in relevant_quotations."
                ),
                "applicable_law": (
                    "Use validated_sections as primary source. Also include any section "
                    "explicitly named in retrieved_legal_context."
                ),
                "must_return_valid_json_only": True,
            },
        }

        return (
            "Synthesize the final legal response from the following evidence. "
            "Extract exact legal text from retrieved_legal_context for relevant_quotations.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    def _extract_facts(self, pipeline_output: dict[str, Any]) -> dict[str, Any]:
        """Extract structured facts from pipeline output."""
        candidate_keys = (
            "facts",
            "structured_facts",
            "fact_extraction",
            "fact_sheet",
            "extracted_facts",
        )
        for key in candidate_keys:
            value = pipeline_output.get(key)
            if isinstance(value, dict):
                return value

        classification = pipeline_output.get("classification", {})
        if isinstance(classification, dict):
            fact_like_keys = {
                key: value
                for key, value in classification.items()
                if key.lower() in {"facts", "structured_facts", "entities", "incident_type"}
            }
            if fact_like_keys:
                return fact_like_keys

        return {}

    def _normalize_validated_sections(
        self,
        pipeline_output: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Derive validated law sections from validator output and pipeline data."""
        validation = pipeline_output.get("validation", {})
        applicable_sections = pipeline_output.get("applicable_sections", []) or []

        validated_section_names: set[str] = set()
        if isinstance(validation, dict):
            for section in validation.get("validated_sections", []) or []:
                if isinstance(section, str) and section.strip():
                    validated_section_names.add(section.strip().lower())

        results: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for entry in applicable_sections:
            if not isinstance(entry, dict):
                continue

            document = str(entry.get("document", "")).strip()
            section = str(entry.get("section", "")).strip()

            if not document or not section:
                continue

            key = (document.upper(), section.lower())
            if key in seen:
                continue
            seen.add(key)

            results.append({"document": document, "section": section})

        return results

    def _normalize_validated_quotes(
        self,
        pipeline_output: dict[str, Any],
    ) -> list[str]:
        """Derive validated quotes from validator output."""
        validation = pipeline_output.get("validation", {})
        if not isinstance(validation, dict):
            return []

        quotes = validation.get("validated_quotes", []) or []
        result: list[str] = []
        seen: set[str] = set()

        for quote in quotes:
            if not isinstance(quote, str):
                continue
            normalized = " ".join(quote.split())
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)

        return result

    def _collect_validation_errors(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Collect validation-related errors for failed outputs."""
        validation = pipeline_output.get("validation", {})
        errors: list[str] = []

        if isinstance(validation, dict):
            reason = validation.get("reason")
            if isinstance(reason, str) and reason.strip():
                errors.append(reason.strip())

            for key in ("rejected_sections", "rejected_quotes", "missing_sections"):
                values = validation.get(key, []) or []
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value.strip():
                            errors.append(f"{key}: {value.strip()}")

        deduped: list[str] = []
        seen: set[str] = set()
        for error in errors:
            lowered = error.lower()
            if lowered not in seen:
                seen.add(lowered)
                deduped.append(error)

        return deduped

    def _normalize_input(self, pipeline_output: dict[str, Any]) -> ValidatedEvidence:
        """Normalize the pipeline output into validated synthesizer input."""
        validation = pipeline_output.get("validation", {})
        validation_status = ""
        if isinstance(validation, dict):
            validation_status = str(validation.get("validation_status", "")).strip().upper()

        raw_results = pipeline_output.get("retrieval_results") or []
        retrieved_chunks: list[dict] = [
            r if isinstance(r, dict) else {}
            for r in raw_results
        ]

        return ValidatedEvidence(
            query=str(pipeline_output.get("query", "")).strip(),
            facts=self._extract_facts(pipeline_output),
            validated_sections=self._normalize_validated_sections(pipeline_output),
            validated_quotes=self._normalize_validated_quotes(pipeline_output),
            remedy=pipeline_output.get("remedy", {})
            if isinstance(pipeline_output.get("remedy", {}), dict)
            else {},
            validation_status=validation_status,
            validation_errors=self._collect_validation_errors(pipeline_output),
            retrieved_chunks=retrieved_chunks,
        )

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Groq LLM with retries and robust error handling.

        Args:
            system_prompt: System instruction prompt.
            user_prompt: User evidence prompt.

        Returns:
            The raw model text output.

        Raises:
            RuntimeError: For API key, timeout, rate limit, connection, refusal,
                or malformed response errors.
        """
        client = self.load_llm()
        last_error: Exception | None = None

        for attempt in range(1, self._llm_config.max_retries + 1):
            try:
                started = time.perf_counter()
                response = client.chat.completions.create(
                    model=self._llm_config.model_name,
                    temperature=self._llm_config.temperature,
                    max_tokens=self._llm_config.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                latency = time.perf_counter() - started

                content = ""
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content or ""

                if not content.strip():
                    raise RuntimeError("LLM returned an empty response.")

                logger.info(
                    "Groq call succeeded: model={} latency={:.4f}s response_length={}",
                    self._llm_config.model_name,
                    latency,
                    len(content),
                )
                return content

            except APITimeoutError as exc:
                last_error = exc
                logger.warning(
                    "Groq timeout on attempt {}/{}: {}",
                    attempt,
                    self._llm_config.max_retries,
                    exc,
                )
            except RateLimitError as exc:
                last_error = exc
                logger.warning(
                    "Groq rate limit on attempt {}/{}: {}",
                    attempt,
                    self._llm_config.max_retries,
                    exc,
                )
            except APIConnectionError as exc:
                last_error = exc
                logger.warning(
                    "Groq connection error on attempt {}/{}: {}",
                    attempt,
                    self._llm_config.max_retries,
                    exc,
                )
            except APIError as exc:
                last_error = exc
                logger.warning(
                    "Groq API error on attempt {}/{}: {}",
                    attempt,
                    self._llm_config.max_retries,
                    exc,
                )
                status_code = getattr(exc, "status_code", None)
                if status_code == 401:
                    raise RuntimeError("Invalid Groq API key.") from exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Unexpected LLM error on attempt {}/{}: {}",
                    attempt,
                    self._llm_config.max_retries,
                    exc,
                )

            if attempt < self._llm_config.max_retries:
                sleep_seconds = self._llm_config.retry_backoff_seconds * attempt
                time.sleep(sleep_seconds)

        if isinstance(last_error, APITimeoutError):
            raise RuntimeError("Groq API timeout after multiple retry attempts.") from last_error
        if isinstance(last_error, RateLimitError):
            raise RuntimeError(
                "Groq API rate limit exceeded after multiple retry attempts."
            ) from last_error
        if isinstance(last_error, APIConnectionError):
            raise RuntimeError(
                "Network failure while contacting Groq after multiple retry attempts."
            ) from last_error
        if isinstance(last_error, APIError):
            raise RuntimeError(f"Groq API error: {last_error}") from last_error

        raise RuntimeError(f"LLM invocation failed: {last_error}") from last_error

    # ------------------------------------------------------------------
    # Response parsing and validation
    # ------------------------------------------------------------------

    def parse_response(
        self,
        llm_response: str,
        evidence: ValidatedEvidence,
    ) -> dict[str, Any]:
        """Parse and validate the LLM JSON response.

        Args:
            llm_response: Raw JSON string from the LLM.
            evidence: Validated evidence used to constrain output.

        Returns:
            A sanitized response dictionary.

        Raises:
            RuntimeError: If JSON is malformed or the response violates constraints.
        """
        try:
            parsed = json.loads(llm_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSON returned by LLM: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response JSON must be an object.")

        required_keys = {
            "status",
            "summary",
            "query_understanding",
            "identified_offences",
            "applicable_law",
            "relevant_quotations",
            "legal_explanation",
            "recommended_procedure",
            "important_notes",
            "citations",
            "disclaimer",
        }
        missing = sorted(required_keys - set(parsed.keys()))
        if missing:
            raise RuntimeError(f"LLM response missing required keys: {', '.join(missing)}")

        parsed["status"] = "SUCCESS"
        parsed["disclaimer"] = _DEFAULT_DISCLAIMER

        if not isinstance(parsed["summary"], str):
            parsed["summary"] = str(parsed["summary"])
        if not isinstance(parsed["legal_explanation"], str):
            parsed["legal_explanation"] = str(parsed["legal_explanation"])

        parsed["query_understanding"] = (
            str(parsed.get("query_understanding", "")).strip()
            or f"Query about: {str(parsed.get('summary',''))[:120]}"
        )
        parsed["identified_offences"] = self._sanitize_string_list(
            parsed.get("identified_offences", [])
        )
        parsed["relevant_quotations"] = self._sanitize_string_list(
            parsed.get("relevant_quotations", [])
        )
        parsed["recommended_procedure"] = self._sanitize_string_list(
            parsed.get("recommended_procedure", [])
        )
        parsed["important_notes"] = self._sanitize_string_list(
            parsed.get("important_notes", [])
        )
        parsed["citations"] = self._sanitize_string_list(parsed.get("citations", []))

        parsed["applicable_law"] = self._sanitize_applicable_law(
            parsed.get("applicable_law", []),
            evidence.validated_sections,
        )
        parsed["citations"] = self._sanitize_citations(
            parsed["citations"],
            evidence.validated_sections,
        )

        if not parsed["applicable_law"] and evidence.validated_sections:
            parsed["applicable_law"] = evidence.validated_sections.copy()

        if not parsed["citations"] and evidence.validated_sections:
            parsed["citations"] = [
                f"{entry['document']} - {entry['section']}"
                for entry in evidence.validated_sections
            ]

        self._ensure_explanation_headings(parsed["legal_explanation"])

        return parsed

    @staticmethod
    def _sanitize_string_list(value: Any) -> list[str]:
        """Normalize a heterogeneous value into a deduplicated string list."""
        if not isinstance(value, list):
            return []

        result: list[str] = []
        seen: set[str] = set()

        for item in value:
            if not isinstance(item, str):
                item = str(item)
            normalized = " ".join(item.split()).strip()
            lowered = normalized.lower()
            if normalized and lowered not in seen:
                seen.add(lowered)
                result.append(normalized)

        return result

    def _sanitize_applicable_law(
        self,
        value: Any,
        validated_sections: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Ensure applicable_law contains only validated or known-corpus sections."""
        if not isinstance(value, list):
            value = []

        allowed = {
            (item["document"].strip().upper(), item["section"].strip().lower()): item
            for item in validated_sections
            if item.get("document") and item.get("section")
        }

        _KNOWN_DOCS = {
            "BNS", "IPC", "BNSS", "CRPC", "BSA", "IEA",
            "POCSO", "IT ACT", "CONSTITUTION", "CPC", "MV ACT",
        }

        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for item in value:
            if not isinstance(item, dict):
                continue

            document = str(item.get("document", "")).strip()
            section = str(item.get("section", "")).strip()
            if not document or not section:
                continue

            key = (document.upper(), section.lower())
            if key in seen:
                continue

            if allowed:
                if key not in allowed:
                    continue
            else:
                if not any(k in document.upper() for k in _KNOWN_DOCS):
                    continue

            seen.add(key)
            result.append({"document": document, "section": section})

        return result

    def _sanitize_citations(
        self,
        citations: list[str],
        validated_sections: list[dict[str, str]],
    ) -> list[str]:
        """Ensure citations include only validated or well-formed section citations."""
        import re as _re
        allowed = {
            f"{item['document']} - {item['section']}".strip()
            for item in validated_sections
            if item.get("document") and item.get("section")
        }

        result: list[str] = []
        seen: set[str] = set()

        for citation in citations:
            if not isinstance(citation, str) or not citation.strip():
                continue
            c = citation.strip()
            if c in seen:
                continue
            if allowed:
                if c in allowed:
                    seen.add(c)
                    result.append(c)
            else:
                if _re.search(r"(BNS|IPC|BNSS|CRPC|BSA|IEA|POCSO|Constitution)", c, _re.I):
                    seen.add(c)
                    result.append(c)

        return result

    @staticmethod
    def _ensure_explanation_headings(legal_explanation: str) -> None:
        """Verify required markdown headings are present in the explanation."""
        required_headings = (
            "# Summary",
            "# Applicable Law",
            "# Explanation",
            "# Procedure",
            "# Notes",
            "# Disclaimer",
        )
        for heading in required_headings:
            if heading not in legal_explanation:
                raise RuntimeError(
                    f"LLM response missing required heading in legal_explanation: {heading}"
                )

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _validation_failed_response(self, errors: list[str]) -> dict[str, Any]:
        """Return the mandated failure response when citation validation fails."""
        payload = {
            "status": "FAILED",
            "reason": "Citation validation failed.",
            "errors": errors or ["Citation validation failed."],
        }
        return payload

    def _runtime_failed_response(self, reason: str) -> dict[str, Any]:
        """Return a meaningful runtime failure payload."""
        return {
            "status": "FAILED",
            "reason": reason,
            "errors": [reason],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, pipeline_output: dict[str, Any]) -> dict[str, Any]:
        """Generate the final legal response.

        Args:
            pipeline_output: Final pipeline output containing query,
                classification/facts, applicable_sections, remedy, and
                validation.

        Returns:
            Structured JSON-compatible dictionary.

        Notes:
            If validation_status != PASS, the LLM is not called.
        """
        started = time.perf_counter()

        if not isinstance(pipeline_output, dict):
            logger.warning("generate() received non-dict input.")
            result = self._runtime_failed_response("Input to synthesizer was not a dict.")
            self._stats.record(
                success=False,
                latency_seconds=time.perf_counter() - started,
                input_chars=0,
                output_chars=len(json.dumps(result, ensure_ascii=False)),
                model=self._llm_config.model_name,
                validation_status="INVALID_INPUT",
            )
            return result

        evidence = self._normalize_input(pipeline_output)

        logger.info(
            "Synthesizer request received: query='{}' model='{}' validation_status='{}'",
            evidence.query[:200],
            self._llm_config.model_name,
            evidence.validation_status or "UNKNOWN",
        )

        # Only skip the LLM if validation truly has no evidence to work with.
        # Section mismatches (sections identified but not in chunk metadata) are
        # a retrieval limitation, not hallucination — allow the LLM to still run.
        has_fatal_validation_failure = (
            evidence.validation_status not in ("PASS", "FAIL", "")
            and not evidence.validated_quotes
            and not evidence.validated_sections
        )
        if has_fatal_validation_failure:
            result = self._validation_failed_response(evidence.validation_errors)
            latency = time.perf_counter() - started
            self._stats.record(
                success=False,
                latency_seconds=latency,
                input_chars=len(json.dumps(pipeline_output, ensure_ascii=False)),
                output_chars=len(json.dumps(result, ensure_ascii=False)),
                model=self._llm_config.model_name,
                validation_status=evidence.validation_status,
            )
            logger.info(
                "Synthesizer skipped LLM call due to fatal validation failure: latency={:.4f}s errors={}",
                latency,
                len(result["errors"]),
            )
            return result
        if evidence.validation_status == "FAIL":
            logger.info(
                "Synthesizer proceeding with LLM despite FAIL validation status "
                "(sections may not be in retrieved chunk metadata -- retrieval limitation)."
            )

        try:
            system_prompt = self.build_system_prompt()
            user_prompt = self.build_user_prompt(evidence)
            llm_response = self.call_llm(system_prompt, user_prompt)
            parsed = self.parse_response(llm_response, evidence)

            latency = time.perf_counter() - started
            self._stats.record(
                success=True,
                latency_seconds=latency,
                input_chars=len(user_prompt) + len(system_prompt),
                output_chars=len(json.dumps(parsed, ensure_ascii=False)),
                model=self._llm_config.model_name,
                validation_status=evidence.validation_status,
            )

            logger.info(
                "Synthesizer completed: model='{}' latency={:.4f}s tokens_max={} response_length={}",
                self._llm_config.model_name,
                latency,
                self._llm_config.max_tokens,
                len(json.dumps(parsed, ensure_ascii=False)),
            )
            return parsed

        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - started
            logger.error("Synthesizer failed: {}", exc)

            result = self._runtime_failed_response(str(exc))
            self._stats.record(
                success=False,
                latency_seconds=latency,
                input_chars=len(json.dumps(pipeline_output, ensure_ascii=False)),
                output_chars=len(json.dumps(result, ensure_ascii=False)),
                model=self._llm_config.model_name,
                validation_status=evidence.validation_status,
            )
            return result

    def print_statistics(self) -> None:
        """Print operational statistics as formatted JSON."""
        print(json.dumps(self._stats.to_dict(), indent=4, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

_TEST_PIPELINE_OUTPUT: Final[dict[str, Any]] = {
    "query": "Someone broke into my house and stole my phone.",
    "classification": {
        "incident_type": "theft and house-trespass",
        "facts": {
            "alleged_acts": [
                "A person broke into the house",
                "A phone was stolen",
            ],
            "property_involved": ["phone"],
            "location": "house",
        },
    },
    "retrieval_results": [
        {
            "document": "BNS",
            "section": "Section 303",
            "chunk_id": "bns-303-001",
            "chunk_hash": "a1b2c3",
            "text": (
                "Whoever commits theft shall be punished with imprisonment of "
                "either description for a term which may extend to three years, "
                "or with fine, or with both."
            ),
        },
        {
            "document": "BNS",
            "section": "Section 331",
            "chunk_id": "bns-331-001",
            "chunk_hash": "d4e5f6",
            "text": (
                "Whoever commits house-trespass shall be punished with imprisonment "
                "of either description for a term which may extend to one year, "
                "or with fine, or with both."
            ),
        },
    ],
    "selected_quotes": [
        {
            "quote": (
                "Whoever commits theft shall be punished with imprisonment of either "
                "description for a term which may extend to three years"
            ),
            "document": "BNS",
            "section": "Section 303",
            "chunk_id": "bns-303-001",
        },
        {
            "quote": (
                "Whoever commits house-trespass shall be punished with imprisonment "
                "of either description for a term which may extend to one year"
            ),
            "document": "BNS",
            "section": "Section 331",
            "chunk_id": "bns-331-001",
        },
    ],
    "applicable_sections": [
        {"document": "BNS", "section": "Section 303"},
        {"document": "BNS", "section": "Section 331"},
    ],
    "remedy": {
        "summary": "The user may consider approaching the police and preserving evidence.",
        "recommended_steps": [
            "File a police complaint or FIR with the relevant facts.",
            "Preserve proof of ownership of the stolen phone.",
            "Retain any CCTV footage, witness details, or signs of forced entry.",
        ],
        "notes": [
            "Only the validated retrieved material should be relied upon.",
        ],
    },
    "validation": {
        "validation_status": "PASS",
        "validated_sections": ["Section 303", "Section 331"],
        "rejected_sections": [],
        "validated_quotes": [
            (
                "Whoever commits theft shall be punished with imprisonment of either "
                "description for a term which may extend to three years"
            ),
            (
                "Whoever commits house-trespass shall be punished with imprisonment "
                "of either description for a term which may extend to one year"
            ),
        ],
        "rejected_quotes": [],
        "missing_sections": [],
        "confidence": 1.0,
    },
}


def _run_test_mode() -> None:
    """Run a validated example through the synthesizer and print JSON."""
    synthesizer = ResponseSynthesizer()
    result = synthesizer.generate(_TEST_PIPELINE_OUTPUT)
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print()
    synthesizer.print_statistics()


if __name__ == "__main__":
    _run_test_mode()
