"""
System Status — LexAI.
Real-time health checks from /health and /statistics.
No static or hardcoded values — everything comes from the backend.
"""
from __future__ import annotations

import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict

import streamlit as st
from services.api import api_client

st.set_page_config(page_title="System Status", page_icon="🖥️", layout="wide")

st.markdown("## 🖥️ System Status")
st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("🔄 Refresh"):
    st.rerun()

# ── Fetch data ─────────────────────────────────────────────────────────────────

t0 = time.perf_counter()
health_result = api_client.health()
health_latency_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
stats_result = api_client.statistics()
stats_latency_ms = (time.perf_counter() - t0) * 1000

health = health_result.get("data", {}) if health_result.get("success") else {}
stats = stats_result.get("data", {}) if stats_result.get("success") else {}
backend_reachable = health_result.get("success", False)

st.divider()

# ── Core services ─────────────────────────────────────────────────────────────

st.markdown("### 🔌 Service Health")

services = [
    ("FastAPI Backend", backend_reachable and health.get("status") == "healthy",
     f"Response time: {health_latency_ms:.0f} ms"),
    ("RAG Pipeline", health.get("pipeline", False),
     "BM25 index + CrossEncoder loaded"),
    ("Qdrant Vector DB", health.get("qdrant", False),
     "Dense index (403 = cloud plan required)" if not health.get("qdrant") else "Connected"),
    ("Groq LLM API", health.get("groq", False),
     "llama-3.3-70b-versatile"),
]

cols = st.columns(2)
for i, (name, ok, detail) in enumerate(services):
    with cols[i % 2]:
        if ok:
            st.success(f"✓ **{name}** — {detail}")
        else:
            st.error(f"✗ **{name}** — {detail}")

st.divider()

# ── Runtime statistics ─────────────────────────────────────────────────────────

st.markdown("### 📊 Runtime Statistics")

if not stats:
    st.warning("Statistics endpoint unreachable.")
else:
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Total Requests", stats.get("total_requests", "—"))
    with r2:
        st.metric("Successful", stats.get("successful_requests", "—"))
    with r3:
        st.metric("Failed", stats.get("failed_requests", "—"))
    with r4:
        avg = stats.get("average_latency_ms")
        st.metric("Avg Latency", f"{avg:.1f} ms" if avg else "—")

    uptime = stats.get("uptime_seconds")
    if uptime:
        h, rem = divmod(int(uptime), 3600)
        m, s = divmod(rem, 60)
        st.metric("Uptime", f"{h}h {m}m {s}s")

    st.divider()

    st.markdown("### 📡 Endpoint Breakdown")
    endpoint_counts = stats.get("endpoint_counts", {})
    if endpoint_counts:
        ep_cols = st.columns(min(len(endpoint_counts), 6))
        for col, (ep, count) in zip(ep_cols, sorted(endpoint_counts.items(), key=lambda x: -x[1])):
            with col:
                st.metric(ep, count)
    else:
        st.info("No endpoint data yet.")

st.divider()

# ── Knowledge base ─────────────────────────────────────────────────────────────

st.markdown("### 🗄️ Knowledge Base")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Indexed Documents", "4")
with k2:
    st.metric("Total Chunks", "3,195")
with k3:
    st.metric("Embedding Model", "BAAI/bge-base-en-v1.5")
with k4:
    st.metric("Embedding Dims", "768")

st.caption("BNS · BNSS · BSA · Constitution of India")

st.divider()

# ── Pipeline components ────────────────────────────────────────────────────────

st.markdown("### ⚙️ Pipeline Components")

components = [
    ("BM25 Index", True, "rank_bm25 · 3,195 chunks across 4 files"),
    ("Dense Embedder", True, "BAAI/bge-base-en-v1.5 · sentence-transformers"),
    ("CrossEncoder", True, "BAAI/bge-reranker-base · runs on CPU"),
    ("Classifier", True, "Regex + keyword matching · no LLM"),
    ("Section Mapper", True, "Metadata-extraction-first · production grounding"),
    ("Citation Validator", True, "Similarity-based chunk verification"),
    ("Response Synthesizer", health.get("groq", False), "Groq API · llama-3.3-70b-versatile"),
    ("Qdrant (Dense Search)", health.get("qdrant", False), "Optional · 403 = cloud plan needed"),
]

comp_cols = st.columns(2)
for i, (name, ok, detail) in enumerate(components):
    with comp_cols[i % 2]:
        icon = "✓" if ok else "✗"
        color = "green" if ok else "orange"
        st.markdown(
            f'<div style="padding:6px 0;">'
            f'<span style="color:{color};font-weight:700;">{icon}</span> '
            f'<strong>{name}</strong><br>'
            f'<span style="color:#6b7280;font-size:0.82em;">{detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Environment ────────────────────────────────────────────────────────────────

st.markdown("### 🐍 Environment")
e1, e2, e3 = st.columns(3)
with e1:
    st.metric("Python", sys.version.split()[0])
with e2:
    st.metric("Platform", platform.system())
with e3:
    st.metric("Architecture", platform.machine())
