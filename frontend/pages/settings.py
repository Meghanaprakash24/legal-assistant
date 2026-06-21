"""
Settings — LexAI.
Functional user-configurable settings persisted to session state.
Controls retrieval top-k, display preferences, and LLM parameters
(note: LLM params affect display/context but not the Legal Assistant
chat which uses backend defaults).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

SETTINGS_KEY = "lexai_settings"

DEFAULTS: Dict[str, Any] = {
    "retrieval_top_k": 5,
    "show_citations": True,
    "show_quotations": True,
    "show_procedure": True,
    "show_query_understanding": True,
    "show_offences": True,
    "max_quotations_shown": 3,
    "max_procedure_steps": 10,
    "evidence_page_top_k": 10,
}


def _settings() -> Dict[str, Any]:
    if SETTINGS_KEY not in st.session_state:
        st.session_state[SETTINGS_KEY] = DEFAULTS.copy()
    return st.session_state[SETTINGS_KEY]


def _save(key: str, value: Any) -> None:
    st.session_state[SETTINGS_KEY][key] = value


st.markdown("## ⚙️ Settings")
st.caption(
    "Configure display preferences and retrieval parameters. "
    "Settings are preserved for this browser session."
)

cfg = _settings()

# ── Retrieval ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown("### 🔍 Retrieval")
st.caption("These settings control the live retrieval tool on the Retrieved Evidence page.")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider(
        "Top-K results (Retrieved Evidence page)",
        min_value=1, max_value=20,
        value=cfg.get("evidence_page_top_k", 10),
        help="Number of chunks to retrieve when using the live retrieval tool.",
    )
    _save("evidence_page_top_k", top_k)

with col2:
    st.info(
        "**Note:** The Legal Assistant chat uses the backend's default retrieval settings. "
        "Top-K here only affects the live retrieval tool on the Retrieved Evidence page."
    )

# ── Display preferences ────────────────────────────────────────────────────────

st.divider()
st.markdown("### 🖥️ Display Preferences")
st.caption("Control which sections appear in Legal Assistant responses.")

d1, d2 = st.columns(2)
with d1:
    show_qu = st.toggle(
        "Show Query Understanding",
        value=cfg.get("show_query_understanding", True),
        help="Display the LLM's interpretation of your query.",
    )
    _save("show_query_understanding", show_qu)

    show_offences = st.toggle(
        "Show Identified Offences",
        value=cfg.get("show_offences", True),
        help="Display classified legal categories / offences.",
    )
    _save("show_offences", show_offences)

    show_cit = st.toggle(
        "Show Citations",
        value=cfg.get("show_citations", True),
        help="Display validated citation badges at the bottom of each response.",
    )
    _save("show_citations", show_cit)

with d2:
    show_q = st.toggle(
        "Show Relevant Quotations",
        value=cfg.get("show_quotations", True),
        help="Display verbatim legal text excerpts extracted from the corpus.",
    )
    _save("show_quotations", show_q)

    show_proc = st.toggle(
        "Show Recommended Procedure",
        value=cfg.get("show_procedure", True),
        help="Display the step-by-step legal procedure recommendations.",
    )
    _save("show_procedure", show_proc)

    max_q = st.slider(
        "Max quotations displayed",
        min_value=1, max_value=10,
        value=cfg.get("max_quotations_shown", 3),
    )
    _save("max_quotations_shown", max_q)

# ── Session ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("### 💬 Chat Session")

msgs = st.session_state.get("legal_assistant_messages", [])
user_count = sum(1 for m in msgs if m.get("role") == "user")
st.metric("Queries in current session", user_count)

if st.button("🗑️ Clear Chat History", type="secondary"):
    st.session_state["legal_assistant_messages"] = []
    st.session_state["legal_assistant_last_response"] = None
    st.session_state["legal_assistant_last_error"] = None
    st.session_state["legal_assistant_input_nonce"] = (
        st.session_state.get("legal_assistant_input_nonce", 0) + 1
    )
    st.success("Chat history cleared.")

# ── Export / Import ────────────────────────────────────────────────────────────

st.divider()
st.markdown("### 📤 Export / Import Settings")

col_exp, col_imp = st.columns(2)
with col_exp:
    cfg_json = json.dumps(cfg, indent=2)
    st.download_button(
        "⬇️ Export Settings (JSON)",
        data=cfg_json,
        file_name=f"lexai_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

with col_imp:
    uploaded = st.file_uploader("⬆️ Import Settings (JSON)", type=["json"])
    if uploaded:
        try:
            imported = json.load(uploaded)
            for k, v in imported.items():
                if k in DEFAULTS:
                    _save(k, v)
            st.success("Settings imported successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to import settings: {e}")

# ── Reset ──────────────────────────────────────────────────────────────────────

st.divider()
if st.button("↺ Reset to Defaults", type="secondary"):
    st.session_state[SETTINGS_KEY] = DEFAULTS.copy()
    st.success("Settings reset to defaults.")
    st.rerun()

st.divider()
st.markdown("### 🔧 System Information")
s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Backend API", "FastAPI · port 8000")
with s2:
    st.metric("LLM", "Groq · llama-3.3-70b-versatile")
with s3:
    st.metric("Reranker", "BAAI/bge-reranker-base")
