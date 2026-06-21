"""
Citation Viewer — LexAI.
Displays citations, applicable sections, and verbatim legal quotations
from the current Legal Assistant session. No mock data.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

# ── Session state keys (must match legal_assistant.py) ─────────────────────────
_MESSAGES_KEY = "legal_assistant_messages"
_LAST_RESPONSE_KEY = "legal_assistant_last_response"

st.set_page_config(page_title="Citation Viewer", page_icon="📚", layout="wide")

st.markdown("## 📚 Citation Viewer")
st.caption(
    "Verified citations and verbatim legal text from your Legal Assistant session. "
    "Every citation is grounded in the indexed corpus."
)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def _assistant_payloads() -> List[Dict[str, Any]]:
    messages = st.session_state.get(_MESSAGES_KEY, [])
    return [
        m["content"]
        for m in messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), dict)
    ]


def _user_queries() -> List[str]:
    messages = st.session_state.get(_MESSAGES_KEY, [])
    return [str(m["content"]) for m in messages if m.get("role") == "user"]


# ── No session data guard ────────────────────────────────────────────────────────

payloads = _assistant_payloads()
queries = _user_queries()

if not payloads:
    st.info(
        "No citations yet. Ask a question in the **Legal Assistant** and return here "
        "to inspect the citations used in each response."
    )
    st.stop()

# ── Query selector ───────────────────────────────────────────────────────────────

st.divider()

# Pair queries with responses (same order as messages)
pairs = list(zip(queries, payloads))

selected_idx = st.selectbox(
    "Select query to inspect:",
    range(len(pairs)),
    format_func=lambda i: f"[{i+1}] {pairs[i][0][:80]}",
    index=len(pairs) - 1,
)

query_text, payload = pairs[selected_idx]

st.markdown(f"**Query:** {query_text}")
if payload.get("query_understanding"):
    st.caption(f"🔍 {payload['query_understanding']}")

st.divider()

# ── Applicable Sections ───────────────────────────────────────────────────────

st.markdown("### 📋 Applicable Sections")
laws = payload.get("applicable_law") or []
if laws:
    cols = st.columns(min(len(laws), 4))
    for col, law in zip(cols, laws):
        with col:
            doc = law.get("document", "")
            sec = law.get("section", "")
            color_map = {
                "BNS": "#4f46e5",
                "BNSS": "#0891b2",
                "BSA": "#059669",
                "Constitution": "#dc2626",
                "Constitution of India": "#dc2626",
            }
            color = color_map.get(doc, "#6b7280")
            st.markdown(
                f'<div style="background:{color}10;border:1px solid {color}40;'
                f'border-radius:8px;padding:12px;text-align:center;">'
                f'<div style="font-weight:700;color:{color};font-size:0.85em;">{doc}</div>'
                f'<div style="font-weight:600;font-size:1em;">{sec}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    st.write("")
else:
    st.info("No applicable sections recorded for this query.")

# ── Relevant Legal Quotations ─────────────────────────────────────────────────

st.markdown("### 📖 Verbatim Legal Text")
quotations = payload.get("relevant_quotations") or []
if quotations:
    for q in quotations:
        if q.strip():
            st.markdown(
                f'<blockquote style="border-left:4px solid #6366f1;padding:10px 16px;'
                f'margin:8px 0;background:#f8f7ff;border-radius:4px;'
                f'font-style:italic;color:#3730a3;">{q}</blockquote>',
                unsafe_allow_html=True,
            )
else:
    st.info("No verbatim quotations extracted for this query.")

# ── Citation List ─────────────────────────────────────────────────────────────

st.markdown("### 🔗 Citations")
citations = payload.get("citations") or []
if citations:
    for cit in citations:
        st.markdown(
            f'<span style="background:#e8f0fe;color:#1a56db;padding:4px 12px;'
            f'border-radius:12px;font-size:0.85em;margin:3px;display:inline-block;'
            f'border:1px solid #c7d2fe;">📌 {cit}</span>',
            unsafe_allow_html=True,
        )
    st.write("")
else:
    st.info("No citations for this query.")

# ── Offences ──────────────────────────────────────────────────────────────────

offences = payload.get("identified_offences") or []
if offences:
    st.markdown("### ⚖️ Classified Offences")
    badge_html = " ".join(
        f'<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
        f'border-radius:12px;font-size:0.82em;margin:2px;display:inline-block;'
        f'border:1px solid #fecaca;">{o}</span>'
        for o in offences
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.write("")

# ── Summary ───────────────────────────────────────────────────────────────────

summary = payload.get("summary")
if summary:
    st.divider()
    st.markdown("### 📝 Response Summary")
    st.markdown(f"> {summary}")

# ── Session overview ─────────────────────────────────────────────────────────

st.divider()
st.markdown("### 📊 Session Overview")
o1, o2, o3 = st.columns(3)
with o1:
    st.metric("Total Queries", len(payloads))
with o2:
    total_cit = sum(len(p.get("citations", [])) for p in payloads)
    st.metric("Total Citations", total_cit)
with o3:
    total_q = sum(len(p.get("relevant_quotations", [])) for p in payloads)
    st.metric("Total Quotations", total_q)
