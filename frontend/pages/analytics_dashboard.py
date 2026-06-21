"""
Analytics Dashboard — LexAI.
Shows real runtime metrics from the backend /statistics endpoint and
per-query data collected in the Legal Assistant session state.
No mock data. "No data available" when nothing has been queried yet.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import streamlit as st

from services.api import api_client

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

# ── Session state keys (must match legal_assistant.py) ─────────────────────────
_MESSAGES_KEY = "legal_assistant_messages"
_LAST_RESPONSE_KEY = "legal_assistant_last_response"


def _messages() -> List[Dict[str, Any]]:
    return st.session_state.get(_MESSAGES_KEY, [])


def _user_messages() -> List[Dict[str, Any]]:
    return [m for m in _messages() if m.get("role") == "user"]


def _assistant_payloads() -> List[Dict[str, Any]]:
    return [
        m["content"]
        for m in _messages()
        if m.get("role") == "assistant" and isinstance(m.get("content"), dict)
    ]


# ── Backend stats ───────────────────────────────────────────────────────────────

def _backend_stats() -> Dict[str, Any]:
    r = api_client.statistics()
    return r.get("data", {}) if r.get("success") else {}


# ── Page ────────────────────────────────────────────────────────────────────────

st.markdown("## 📊 Analytics Dashboard")
st.caption("Live metrics from the backend API and the current session. Refresh the page to update.")

if st.button("🔄 Refresh"):
    st.rerun()

st.divider()

# ── Section 1: Backend lifetime stats ─────────────────────────────────────────

st.markdown("### 🖥️ Backend Statistics")
stats = _backend_stats()

if not stats:
    st.warning("Backend is unreachable — cannot load statistics.")
else:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total API Requests", stats.get("total_requests", "—"))
    with c2:
        st.metric("Successful Requests", stats.get("successful_requests", "—"))
    with c3:
        avg_ms = stats.get("average_latency_ms")
        st.metric("Avg Latency", f"{avg_ms:.0f} ms" if avg_ms else "—")
    with c4:
        uptime = stats.get("uptime_seconds")
        if uptime:
            h, rem = divmod(int(uptime), 3600)
            m, s = divmod(rem, 60)
            st.metric("Uptime", f"{h}h {m}m {s}s")
        else:
            st.metric("Uptime", "—")

    endpoint_counts = stats.get("endpoint_counts", {})
    if endpoint_counts:
        st.markdown("**Endpoint call breakdown:**")
        ep_cols = st.columns(len(endpoint_counts))
        for col, (ep, count) in zip(ep_cols, endpoint_counts.items()):
            with col:
                st.metric(ep, count)

st.divider()

# ── Section 2: Session query stats ────────────────────────────────────────────

st.markdown("### 💬 Current Session")
payloads = _assistant_payloads()
queries = _user_messages()

if not queries:
    st.info("No queries in this session yet. Ask a question in the Legal Assistant to see data here.")
else:
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("Queries in Session", len(queries))
    with q2:
        total_citations = sum(len(p.get("citations", [])) for p in payloads)
        st.metric("Total Citations Generated", total_citations)
    with q3:
        total_quotes = sum(len(p.get("relevant_quotations", [])) for p in payloads)
        st.metric("Legal Quotations Extracted", total_quotes)

    st.divider()

    # ── Document distribution ──────────────────────────────────────────────────

    st.markdown("### 📚 Document Distribution")
    doc_counter: Counter = Counter()
    for p in payloads:
        for law in p.get("applicable_law", []):
            doc = law.get("document", "Unknown")
            doc_counter[doc] += 1

    if doc_counter:
        try:
            import plotly.express as px
            import pandas as pd
            df = pd.DataFrame(
                [(doc, cnt) for doc, cnt in doc_counter.most_common()],
                columns=["Document", "Sections Found"],
            )
            fig = px.bar(
                df, x="Document", y="Sections Found",
                color="Document",
                color_discrete_map={
                    "BNS": "#4f46e5",
                    "BNSS": "#0891b2",
                    "BSA": "#059669",
                    "Constitution": "#dc2626",
                    "Constitution of India": "#dc2626",
                },
                title="Applicable Sections by Document",
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for doc, cnt in doc_counter.most_common():
                st.markdown(f"- **{doc}**: {cnt} sections")
    else:
        st.info("No applicable sections recorded yet.")

    st.divider()

    # ── Query history table ────────────────────────────────────────────────────

    st.markdown("### 📋 Query History")
    rows = []
    msg_list = _messages()
    i = 0
    while i < len(msg_list):
        msg = msg_list[i]
        if msg.get("role") == "user":
            query_text = str(msg.get("content", ""))
            timestamp = msg.get("timestamp", "")
            # Next message should be the assistant response
            if i + 1 < len(msg_list) and msg_list[i + 1].get("role") == "assistant":
                resp = msg_list[i + 1].get("content", {})
                if isinstance(resp, dict):
                    laws = ", ".join(
                        f"{x.get('document')} {x.get('section')}"
                        for x in resp.get("applicable_law", [])
                    ) or "—"
                    cit_count = len(resp.get("citations", []))
                    qu = (resp.get("query_understanding") or "")[:60]
                    rows.append({
                        "Time": timestamp,
                        "Query": query_text[:70],
                        "Applicable Law": laws,
                        "Citations": cit_count,
                        "Query Understanding": qu,
                    })
                i += 2
            else:
                i += 1
        else:
            i += 1

    if rows:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except ImportError:
            for row in rows:
                st.markdown(f"- **{row['Time']}** — {row['Query']} → {row['Applicable Law']}")
    else:
        st.info("No query history yet.")
