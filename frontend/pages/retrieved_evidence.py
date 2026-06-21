"""
Retrieved Evidence — LexAI.
Shows verbatim legal text, sections, and quotations from session queries.
Also provides a live retrieval tool for direct corpus inspection.
No mock data.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from services.api import api_client

st.set_page_config(page_title="Retrieved Evidence", page_icon="🔎", layout="wide")

_MESSAGES_KEY = "legal_assistant_messages"

st.markdown("## 🔎 Retrieved Evidence")
st.caption(
    "Inspect the verbatim legal text retrieved and validated for each Legal Assistant query. "
    "All content is sourced directly from the indexed corpus."
)


def _pairs() -> List[tuple]:
    msgs = st.session_state.get(_MESSAGES_KEY, [])
    out = []
    i = 0
    while i < len(msgs):
        if msgs[i].get("role") == "user":
            q = str(msgs[i].get("content", ""))
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                p = msgs[i + 1].get("content", {})
                if isinstance(p, dict):
                    out.append((q, p))
                i += 2
            else:
                i += 1
        else:
            i += 1
    return out


pairs = _pairs()

# ── Session Evidence ──────────────────────────────────────────────────────────

st.divider()
st.markdown("### 📄 Session Evidence")

if not pairs:
    st.info("No evidence yet. Ask a question in the **Legal Assistant** and return here to inspect retrieved legal text.")
else:
    selected_idx = st.selectbox(
        "Select query:",
        range(len(pairs)),
        format_func=lambda i: f"[{i+1}] {pairs[i][0][:80]}",
        index=len(pairs) - 1,
    )
    query_text, payload = pairs[selected_idx]

    st.markdown(f"**Query:** `{query_text}`")
    if payload.get("query_understanding"):
        st.caption(f"🔍 {payload['query_understanding']}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### 📋 Validated Sections")
        laws = payload.get("applicable_law") or []
        if laws:
            for law in laws:
                doc = law.get("document", "")
                sec = law.get("section", "")
                color_map = {"BNS": "#4f46e5", "BNSS": "#0891b2", "BSA": "#059669"}
                color = color_map.get(doc, "#dc2626")
                st.markdown(
                    f'<div style="background:{color}10;border:1px solid {color}40;'
                    f'border-radius:8px;padding:10px 14px;margin:4px 0;">'
                    f'<span style="font-weight:700;color:{color};">{doc}</span> — {sec}'
                    f'<span style="float:right;background:#dcfce7;color:#166534;'
                    f'padding:1px 8px;border-radius:9px;font-size:0.75em;">✓ Validated</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No validated sections.")

    with right:
        st.markdown("#### ⚖️ Classified Offences")
        offences = payload.get("identified_offences") or []
        if offences:
            for o in offences:
                st.markdown(f"- {o}")
        else:
            st.caption("No offences classified for this query.")

    st.write("")
    st.markdown("#### 📖 Verbatim Retrieved Text")
    quotations = payload.get("relevant_quotations") or []
    if quotations:
        for i, q in enumerate(quotations, 1):
            with st.container(border=True):
                st.markdown(f"**Excerpt {i}**")
                st.markdown(
                    f'<blockquote style="border-left:3px solid #6366f1;padding:8px 14px;'
                    f'font-style:italic;color:#3730a3;margin:0;">{q}</blockquote>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No verbatim quotations recorded.")

    procedure = payload.get("recommended_procedure") or []
    notes = payload.get("important_notes") or []
    if procedure:
        st.markdown("#### 🧭 Recommended Procedure")
        for step in procedure:
            st.markdown(f"- {step}")
    if notes:
        st.markdown("#### 📝 Important Notes")
        for note in notes:
            st.markdown(f"- {note}")

# ── Live Retrieval Tool ────────────────────────────────────────────────────────

st.divider()
st.markdown("### 🔍 Live Retrieval")
st.caption("Query the retrieval pipeline directly to inspect raw chunks from the indexed legal corpus.")

with st.form("retrieval_form"):
    query = st.text_input("Search query:", placeholder="e.g. punishment for theft BNS Section 303")
    top_k = st.slider("Top-K results:", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Retrieve", type="primary")

if submitted and query.strip():
    with st.spinner("Retrieving from corpus…"):
        result = api_client.retrieve(query.strip(), top_k=top_k)

    if result.get("success"):
        chunks = result.get("data", {}).get("results", [])
        if chunks:
            st.success(f"Retrieved {len(chunks)} chunk(s) for: `{query}`")
            for i, chunk in enumerate(chunks, 1):
                doc = chunk.get("document", "—")
                sec = chunk.get("section", "") or f"Article {chunk.get('article', '')}"
                conf = chunk.get("confidence") or chunk.get("rerank_score") or chunk.get("retrieval_score") or 0
                text = chunk.get("retrieval_text") or (chunk.get("payload") or {}).get("text") or ""
                with st.expander(f"Chunk {i} — {doc} {sec} (confidence: {conf:.3f})"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Document:** {doc}")
                    with c2:
                        st.markdown(f"**Section:** {sec or '—'}")
                    with c3:
                        st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id', '—')}`")
                    if chunk.get("rerank_score") is not None:
                        st.markdown(f"**Reranker Score:** {chunk['rerank_score']:.4f}")
                    if chunk.get("retrieval_score") is not None:
                        st.markdown(f"**Retrieval Score:** {chunk['retrieval_score']:.4f}")
                    if text:
                        st.markdown("**Retrieved Text:**")
                        st.markdown(
                            f'<blockquote style="border-left:3px solid #6366f1;padding:8px 14px;'
                            f'font-style:italic;color:#374151;">{text[:700]}</blockquote>',
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("No chunks returned. Try a more specific query.")
    else:
        st.error(f"Retrieval failed: {result.get('error', 'Unknown error')}")
elif submitted:
    st.warning("Please enter a search query.")
