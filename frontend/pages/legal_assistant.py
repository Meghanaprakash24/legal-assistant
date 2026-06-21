"""
pages/2_⚖️_Legal_Assistant.py
==============================
AI Legal Assistant page for the Indian Legal AI Assistant Streamlit app.

This page renders a professional, Harvey AI / Perplexity-style chat
interface for asking legal questions related to:

- The Constitution of India
- Bharatiya Nyaya Sanhita (BNS)
- Bharatiya Nagarik Suraksha Sanhita (BNSS)
- Bharatiya Sakshya Adhiniyam (BSA)

It is a pure UI module. All backend communication goes through the
shared ``api_client`` instance defined in ``services/api.py``; no URLs
are hardcoded here.

This module only implements the Legal Assistant page. Theme, sidebar,
and the API layer are assumed to already exist elsewhere in the app.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from services.api import api_client

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUGGESTED_QUESTIONS: List[str] = [
    "What is Article 21?",
    "Punishment for theft under BNS",
    "How do I file an FIR?",
    "Consumer rights",
    "Section 303 BNS",
    "What is anticipatory bail?",
]

USER_AVATAR = "🧑‍⚖️"
ASSISTANT_AVATAR = "⚖️"

SESSION_KEY_MESSAGES = "legal_assistant_messages"
SESSION_KEY_PENDING_INPUT = "legal_assistant_pending_input"
SESSION_KEY_INPUT_NONCE = "legal_assistant_input_nonce"
SESSION_KEY_LAST_RESPONSE = "legal_assistant_last_response"
SESSION_KEY_LAST_ERROR = "legal_assistant_last_error"

# --------------------------------------------------------------------------- #
# Custom styling
# --------------------------------------------------------------------------- #
def _inject_custom_css() -> None:
    """
    Inject scoped CSS for chat bubbles, badges, and metric cards so the
    page reads as a dedicated legal-assistant product rather than a
    generic Streamlit chatbot.
    """
    st.markdown(
        """
        <style>
        .legal-chat-container {
            max-height: 65vh;
            overflow-y: auto;
            padding: 0.5rem 0.25rem;
        }
        .legal-msg-row {
            display: flex;
            margin-bottom: 1rem;
        }
        .legal-msg-row.user {
            justify-content: flex-end;
        }
        .legal-msg-row.assistant {
            justify-content: flex-start;
        }
        .legal-bubble {
            max-width: 80%;
            padding: 0.85rem 1.1rem;
            border-radius: 14px;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .legal-bubble.user {
            background-color: #2563eb;
            color: #ffffff;
            border-top-right-radius: 4px;
        }
        .legal-bubble.assistant {
            background-color: #ffffff;
            color: #1f2937;
            border: 1px solid #e5e7eb;
            border-top-left-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }
        .legal-timestamp {
            font-size: 0.72rem;
            opacity: 0.65;
            margin-top: 0.35rem;
        }
        .legal-citation-badge {
            display: inline-block;
            background-color: #eef2ff;
            color: #4338ca;
            border: 1px solid #c7d2fe;
            border-radius: 999px;
            padding: 0.15rem 0.65rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .legal-suggested-chip button {
            border-radius: 999px !important;
        }
        .legal-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: -0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def _init_session_state() -> None:
    """Initialize all session-state keys used by this page, if absent."""
    if SESSION_KEY_MESSAGES not in st.session_state:
        st.session_state[SESSION_KEY_MESSAGES] = []  # type: List[Dict[str, Any]]
    if SESSION_KEY_PENDING_INPUT not in st.session_state:
        st.session_state[SESSION_KEY_PENDING_INPUT] = ""
    if SESSION_KEY_INPUT_NONCE not in st.session_state:
        st.session_state[SESSION_KEY_INPUT_NONCE] = 0
    if SESSION_KEY_LAST_RESPONSE not in st.session_state:
        st.session_state[SESSION_KEY_LAST_RESPONSE] = None
    if SESSION_KEY_LAST_ERROR not in st.session_state:
        st.session_state[SESSION_KEY_LAST_ERROR] = None


def _append_message(role: str, content: Any) -> None:
    """
    Append a message to the conversation history.

    Args:
        role: Either ``"user"`` or ``"assistant"``.
        content: For user messages, a plain string. For assistant
            messages, the parsed backend response dict (or an error
            string).
    """
    st.session_state[SESSION_KEY_MESSAGES].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )


def _clear_chat() -> None:
    """Reset the conversation history and any cached response metadata."""
    st.session_state[SESSION_KEY_MESSAGES] = []
    st.session_state[SESSION_KEY_LAST_RESPONSE] = None
    st.session_state[SESSION_KEY_LAST_ERROR] = None
    st.session_state[SESSION_KEY_INPUT_NONCE] += 1


def _set_pending_input(text: str) -> None:
    """
    Populate the chat input with a suggested question.

    A nonce-based key is used for the text input widget so that we can
    programmatically change its value by re-rendering it under a new
    key (Streamlit text inputs cannot be mutated directly once
    instantiated).

    Args:
        text: The suggested question text.
    """
    st.session_state[SESSION_KEY_PENDING_INPUT] = text
    st.session_state[SESSION_KEY_INPUT_NONCE] += 1


# --------------------------------------------------------------------------- #
# Rendering helpers — conversation
# --------------------------------------------------------------------------- #
def _render_header() -> None:
    """Render the page header and subtitle."""
    st.markdown("## ⚖️ AI Legal Assistant")
    st.markdown(
        """
        <div class="legal-subtitle">
        Ask legal questions related to
        <b>Constitution</b> • <b>Bharatiya Nyaya Sanhita</b> •
        <b>Bharatiya Nagarik Suraksha Sanhita</b> •
        <b>Bharatiya Sakshya Adhiniyam</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def _render_user_message(text: str, timestamp: str) -> None:
    """
    Render a right-aligned, blue user message bubble.

    Args:
        text: The user's message text.
        timestamp: A pre-formatted timestamp string.
    """
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(
            f"""
            <div class="legal-msg-row user">
                <div class="legal-bubble user">
                    {text}
                    <div class="legal-timestamp">{timestamp}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_assistant_message(payload: Any, timestamp: str) -> None:
    """
    Render a left-aligned, white-card assistant message, supporting
    markdown, bullet lists, bold text, tables, and code blocks.

    Args:
        payload: Either a structured backend response dict or a plain
            error string.
        timestamp: A pre-formatted timestamp string.
    """
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown('<div class="legal-bubble assistant">', unsafe_allow_html=True)

        if isinstance(payload, str):
            st.markdown(payload)
        elif isinstance(payload, dict):
            _render_structured_answer(payload)
        else:
            st.markdown(str(payload))

        st.markdown(
            f'<div class="legal-timestamp">{timestamp}</div></div>',
            unsafe_allow_html=True,
        )


def _render_structured_answer(data: Dict[str, Any]) -> None:
    """Render the full structured backend response with all agent outputs."""

    # ── Query Understanding ────────────────────────────────────────────────
    query_understanding = data.get("query_understanding")
    if query_understanding:
        st.caption(f"🔍 **Query Understanding:** {query_understanding}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = data.get("summary")
    if summary:
        st.markdown(f"> {summary}")

    st.divider()

    # ── Offence Classification ────────────────────────────────────────────
    offences = data.get("identified_offences") or []
    if offences:
        st.markdown("**⚖️ Offence Classification**")
        badge_html = " ".join(
            f'<span style="background:#e8f0fe;color:#1a56db;padding:3px 10px;'
            f'border-radius:12px;font-size:0.82em;margin:2px;display:inline-block;">'
            f'{o}</span>'
            for o in offences
        )
        st.markdown(badge_html, unsafe_allow_html=True)

    # ── Applicable Sections ───────────────────────────────────────────────
    applicable_laws = (
        data.get("applicable_law") or data.get("applicable_laws") or []
    )
    if applicable_laws:
        st.markdown("**📋 Applicable Sections**")
        if isinstance(applicable_laws, list):
            for law in applicable_laws:
                if isinstance(law, dict):
                    doc = law.get("document", "")
                    sec = law.get("section", "")
                    st.markdown(
                        f'<span style="background:#f0fdf4;color:#166534;padding:4px 12px;'
                        f'border-radius:8px;font-size:0.85em;margin:3px;display:inline-block;'
                        f'border:1px solid #bbf7d0;">📌 {doc} — {sec}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"- {law}")
        else:
            st.markdown(str(applicable_laws))

    # ── Relevant Legal Quotations ─────────────────────────────────────────
    quotations = data.get("relevant_quotations") or []
    if quotations:
        st.markdown("**📖 Relevant Legal Text**")
        for q in quotations:
            if q.strip():
                st.markdown(
                    f'<blockquote style="border-left:3px solid #6366f1;padding:8px 14px;'
                    f'margin:6px 0;background:#f8f7ff;border-radius:4px;font-style:italic;'
                    f'color:#3730a3;font-size:0.88em;">{q}</blockquote>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Legal Explanation ─────────────────────────────────────────────────
    explanation = data.get("legal_explanation") or data.get("explanation")
    if explanation:
        st.markdown("**📚 Legal Explanation**")
        st.markdown(explanation)

    # ── Procedure ─────────────────────────────────────────────────────────
    procedure = data.get("procedure") or data.get("recommended_procedure")
    if procedure:
        st.markdown("**🧭 Recommended Procedure**")
        _render_list_or_text(procedure)

    # ── Notes ─────────────────────────────────────────────────────────────
    notes = data.get("notes") or data.get("important_notes")
    if notes:
        st.markdown("**📝 Important Notes**")
        _render_list_or_text(notes)

    # ── Citations ─────────────────────────────────────────────────────────
    citations = data.get("citations") or []
    if citations:
        st.markdown("**🔗 Citations**")
        _render_citation_badges(citations)

    # ── Disclaimer ────────────────────────────────────────────────────────
    disclaimer = data.get("disclaimer")
    if disclaimer:
        st.info(disclaimer, icon="⚠️")


def _render_list_or_text(value: Any) -> None:
    """
    Render a value as a markdown bullet list when it is a list, or as
    plain markdown text otherwise.

    Args:
        value: A list of strings or a single string/markdown blob.
    """
    if isinstance(value, list):
        for item in value:
            st.markdown(f"- {item}")
    else:
        st.markdown(str(value))


def _render_citation_badges(citations: List[Any]) -> None:
    """
    Render citations (e.g. "BNS Section 303", "Article 21") as
    clickable badge-style chips.

    Args:
        citations: List of citation strings or dicts containing a
            ``"label"``/``"text"`` field.
    """
    badge_html = ""
    for citation in citations:
        label = (
            citation.get("label") or citation.get("text") or str(citation)
            if isinstance(citation, dict)
            else str(citation)
        )
        badge_html += f'<span class="legal-citation-badge">📌 {label}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)


def _render_conversation() -> None:
    """Render the full conversation history in the left column."""
    st.markdown('<div class="legal-chat-container">', unsafe_allow_html=True)

    for message in st.session_state[SESSION_KEY_MESSAGES]:
        if message["role"] == "user":
            _render_user_message(message["content"], message["timestamp"])
        else:
            _render_assistant_message(message["content"], message["timestamp"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-scroll anchor: a hidden element scrolled into view on each
    # rerun keeps the latest message visible.
    st.markdown(
        """
        <div id="legal-chat-bottom"></div>
        <script>
            var anchor = document.getElementById("legal-chat-bottom");
            if (anchor) { anchor.scrollIntoView({behavior: "smooth"}); }
        </script>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Rendering helpers — right panel
# --------------------------------------------------------------------------- #
def _render_metadata_panel(
    response_data: Optional[Dict[str, Any]],
    backend_online: bool,
) -> None:
    """
    Render the right-hand "Response Information" panel with metric
    cards and expandable sections.

    Args:
        response_data: The most recent successful backend response, or
            None if no successful response exists yet.
        backend_online: Whether the backend is currently reachable.
    """
    st.markdown("### 📊 Response Information")

    if not backend_online:
        st.error("Backend Offline", icon="🔌")

    if response_data is None:
        st.caption("Ask a question to see response details here.")
        return

    confidence = response_data.get("confidence_score") or response_data.get("confidence")
    latency = response_data.get("latency") or response_data.get("response_time")
    status = response_data.get("processing_status") or response_data.get("status", "—")
    retrieved_count = response_data.get("retrieved_documents_count") or response_data.get(
        "retrieved_count"
    )
    validation_status = response_data.get("validation_status") or response_data.get(
        "validation"
    )
    model_name = response_data.get("groq_model") or response_data.get("model", "—")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            "Confidence",
            f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "—",
        )
        st.metric(
            "Latency",
            f"{latency:.2f}s" if isinstance(latency, (int, float)) else "—",
        )
        st.metric("Retrieved Docs", retrieved_count if retrieved_count is not None else "—")
    with col_b:
        st.metric("Status", status if status else "—")
        st.metric("Validation", validation_status if validation_status else "—")
        st.metric("Model", model_name if model_name else "—")

    st.divider()

    with st.expander("📋 Applicable Sections"):
        laws = response_data.get("applicable_law") or response_data.get("applicable_laws")
        if laws and isinstance(laws, list):
            for law in laws:
                if isinstance(law, dict):
                    st.markdown(f"- **{law.get('document','')}** — {law.get('section','')}")
                else:
                    st.markdown(f"- {law}")
        else:
            st.write("No applicable sections returned.")

    with st.expander("📖 Relevant Legal Text"):
        quotes = response_data.get("relevant_quotations") or []
        if quotes:
            for q in quotes:
                st.markdown(f"> *{q}*")
        else:
            st.write("No quotations extracted.")

    with st.expander("🧭 Recommended Procedure"):
        procedure = response_data.get("procedure") or response_data.get(
            "recommended_procedure"
        )
        if procedure and isinstance(procedure, list):
            for step in procedure:
                st.markdown(f"- {step}")
        else:
            st.write(procedure if procedure else "No procedure information returned.")

    with st.expander("📝 Important Notes"):
        notes = response_data.get("notes") or response_data.get("important_notes")
        if notes and isinstance(notes, list):
            for note in notes:
                st.markdown(f"- {note}")
        else:
            st.write(notes if notes else "No additional notes returned.")

    with st.expander("⚠️ Disclaimer"):
        disclaimer = response_data.get("disclaimer")
        st.write(
            disclaimer
            if disclaimer
            else "This response is for informational purposes only and does "
            "not constitute legal advice."
        )


# --------------------------------------------------------------------------- #
# Backend interaction
# --------------------------------------------------------------------------- #
def _submit_query(query: str) -> None:
    """
    Submit a query to the backend chat endpoint, update the conversation
    history, and store the latest response (or error) in session state.

    Displays a "typing" placeholder while waiting for the backend, and
    surfaces backend/network/timeout errors via Streamlit alerts.

    Args:
        query: The user's legal question.
    """
    query = query.strip()
    if not query:
        return

    _append_message("user", query)

    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown("_AI is analysing legal documents..._ ⏳")

    start_time = time.monotonic()
    result = api_client.chat(query)
    elapsed = time.monotonic() - start_time

    placeholder.empty()

    if result.get("success"):
        data = result["data"]
        if isinstance(data, dict):
            data.setdefault("response_time", elapsed)

        # The HTTP call succeeded, but the pipeline itself may have failed.
        # Check the inner response payload for a FAILED status.
        inner = data.get("response", {}) if isinstance(data, dict) else {}
        pipeline_status = inner.get("status", "") if isinstance(inner, dict) else ""
        pipeline_reason = (
            inner.get("reason") or
            (inner.get("errors") or [""])[0]
            if isinstance(inner, dict) else ""
        )

        if pipeline_status == "FAILED" and pipeline_reason:
            error_message = f"Pipeline Error: {pipeline_reason}"
            st.session_state[SESSION_KEY_LAST_ERROR] = error_message
            _append_message("assistant", f"⚠️ {error_message}")
        else:
            # Flatten: prefer data["response"] if it contains the answer fields
            answer = inner if (inner and isinstance(inner, dict) and inner.get("status") != "FAILED") else data
            answer.setdefault("response_time", elapsed)
            st.session_state[SESSION_KEY_LAST_RESPONSE] = answer
            st.session_state[SESSION_KEY_LAST_ERROR] = None
            _append_message("assistant", answer)
    else:
        error_message = _format_error_message(result)
        st.session_state[SESSION_KEY_LAST_ERROR] = error_message
        _append_message("assistant", f"⚠️ {error_message}")


def _format_error_message(result: Dict[str, Any]) -> str:
    """
    Translate a failed API client response into a user-facing error
    message, categorizing common failure modes (offline backend, Groq
    error, timeout, network error).

    Args:
        result: The failed standard response envelope from the API
            client (``success`` is False).

    Returns:
        A human-readable error description.
    """
    raw_error = str(result.get("error", "Unknown error"))
    status_code = result.get("status_code")

    lowered = raw_error.lower()
    if status_code is None and ("timed out" in lowered or "timeout" in lowered):
        return f"Request Timeout: the backend took too long to respond. ({raw_error})"
    if status_code is None and "connection error" in lowered:
        return f"Backend Offline: unable to reach the server. ({raw_error})"
    if status_code and 500 <= status_code < 600:
        return f"Groq Error: the backend reported an internal error. ({raw_error})"
    if status_code is None:
        return f"Network Error: {raw_error}"
    return f"Backend Error ({status_code}): {raw_error}"


# --------------------------------------------------------------------------- #
# Rendering helpers — input area
# --------------------------------------------------------------------------- #
def _render_suggested_questions() -> None:
    """Render six suggested-question chips below the chat input."""
    st.caption("Suggested questions")
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for column, question in zip(columns, SUGGESTED_QUESTIONS):
        with column:
            if st.button(question, key=f"suggested_{question}", use_container_width=True):
                _set_pending_input(question)
                st.rerun()


def _render_chat_input(backend_online: bool) -> None:
    """
    Render the bottom chat input area: text box, Send button, and
    Clear Chat button.

    Args:
        backend_online: Whether the backend is currently reachable. The
            Send button is disabled when False.
    """
    input_key = f"legal_chat_input_{st.session_state[SESSION_KEY_INPUT_NONCE]}"

    query_text = st.text_area(
        label="Describe your legal issue...",
        value=st.session_state[SESSION_KEY_PENDING_INPUT],
        key=input_key,
        placeholder="Describe your legal issue...",
        label_visibility="collapsed",
        height=90,
    )

    send_col, clear_col = st.columns([1, 1])
    with send_col:
        send_clicked = st.button(
            "Send 📨",
            type="primary",
            use_container_width=True,
            disabled=not backend_online,
        )
    with clear_col:
        clear_clicked = st.button("Clear Chat 🧹", use_container_width=True)

    if not backend_online:
        st.warning("Backend is offline — sending is disabled until it reconnects.", icon="🔌")

    if clear_clicked:
        _clear_chat()
        st.rerun()

    if send_clicked and query_text.strip():
        st.session_state[SESSION_KEY_PENDING_INPUT] = ""
        st.session_state[SESSION_KEY_INPUT_NONCE] += 1
        _submit_query(query_text)
        st.rerun()

    _render_suggested_questions()


# --------------------------------------------------------------------------- #
# Page entry point
# --------------------------------------------------------------------------- #
def render_legal_assistant_page() -> None:
    """
    Render the full Legal Assistant page: header, two-column layout
    (conversation + metadata panel), and the bottom chat input area.
    """
    _inject_custom_css()
    _init_session_state()
    _render_header()

    backend_online = api_client.backend_online()

    left_col, right_col = st.columns([7, 3])

    with left_col:
        _render_conversation()

    with right_col:
        _render_metadata_panel(
            st.session_state[SESSION_KEY_LAST_RESPONSE],
            backend_online,
        )

    st.divider()
    _render_chat_input(backend_online)


render_legal_assistant_page()