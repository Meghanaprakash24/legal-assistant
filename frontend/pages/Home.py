"""
Home page — LexAI Indian Legal RAG Assistant.
Displays project overview, live backend status, and navigation.
"""
from __future__ import annotations

import streamlit as st
from services.api import api_client

st.set_page_config(page_title="LexAI — Indian Legal Assistant", page_icon="⚖️", layout="wide")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_stats() -> dict:
    r = api_client.statistics()
    return r.get("data", {}) if r.get("success") else {}

def _get_health() -> dict:
    r = api_client.health()
    return r.get("data", {}) if r.get("success") else {}

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("## ⚖️ LexAI — Indian Legal Research Assistant")
st.markdown(
    "A production-grade multi-agent RAG system for researching Indian law — "
    "**Constitution · BNS · BNSS · BSA** — grounded in indexed legal documents."
)
st.write("")

col_chat, col_about, _pad = st.columns([1, 1, 4])
with col_chat:
    if st.button("💬 Open Legal Assistant", type="primary", use_container_width=True):
        st.switch_page("pages/legal_assistant.py")
with col_about:
    if st.button("ℹ️ About this System", use_container_width=True):
        st.switch_page("pages/about.py")

st.divider()

# ── Live backend metrics ───────────────────────────────────────────────────────

stats = _get_stats()
health = _get_health()

st.markdown("### 📊 Runtime Overview")
m1, m2, m3, m4 = st.columns(4)
with m1:
    total = stats.get("total_requests", "—")
    chat_calls = stats.get("endpoint_counts", {}).get("/chat", 0)
    st.metric("Total API Requests", total)
with m2:
    st.metric("Queries Answered", chat_calls)
with m3:
    avg = stats.get("average_latency_ms")
    st.metric("Avg Latency", f"{avg:.0f} ms" if avg else "—")
with m4:
    uptime = stats.get("uptime_seconds")
    if uptime:
        h, m = divmod(int(uptime), 3600)
        m, s = divmod(m, 60)
        st.metric("Uptime", f"{h}h {m}m {s}s")
    else:
        st.metric("Uptime", "—")

st.divider()

# ── Backend status ─────────────────────────────────────────────────────────────

st.markdown("### 🖥️ Backend Health")
services = {
    "FastAPI Backend": health.get("status") == "healthy",
    "RAG Pipeline": health.get("pipeline", False),
    "Qdrant Vector DB": health.get("qdrant", False),
    "Groq LLM": health.get("groq", False),
}
cols = st.columns(len(services))
for col, (name, ok) in zip(cols, services.items()):
    with col:
        if ok:
            st.success(f"✓ {name}")
        else:
            st.error(f"✗ {name}")

st.divider()

# ── Supported documents ────────────────────────────────────────────────────────

st.markdown("### 📚 Indexed Legal Corpus")
docs = [
    ("⚖️", "Constitution of India", "Fundamental rights, directive principles, constitutional structure. Articles 1–395."),
    ("📖", "Bharatiya Nyaya Sanhita (BNS)", "India's principal criminal code — offences, punishments, and criminal liability."),
    ("🔒", "Bharatiya Nagarik Suraksha Sanhita (BNSS)", "Criminal procedure — arrest, investigation, bail, trial, and appeals."),
    ("📜", "Bharatiya Sakshya Adhiniyam (BSA)", "Rules of evidence — admissibility, burden of proof, and witness examination."),
]
doc_cols = st.columns(4)
for col, (icon, title, desc) in zip(doc_cols, docs):
    with col:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)

st.divider()

# ── Multi-agent pipeline ───────────────────────────────────────────────────────

st.markdown("### 🤖 Multi-Agent Workflow")
stages = [
    ("💬", "Classifier", "Extracts incident types, section hints, and keywords from the query."),
    ("🔍", "Hybrid Retriever", "BM25 keyword + dense vector search across all 4 legal documents."),
    ("🎯", "BGE Reranker", "CrossEncoder reranking of candidate chunks by semantic relevance."),
    ("❝", "Quote Selector", "Extracts exact statutory sentences from top-ranked chunks."),
    ("🗺️", "Section Mapper", "Reads document/section metadata directly — no LLM guessing."),
    ("✅", "Validator", "Verifies every cited section exists in the retrieved corpus."),
    ("🧩", "Synthesizer", "Groq LLaMA 3.3 70B generates grounded response from validated evidence."),
]
for i, (icon, name, desc) in enumerate(stages):
    col_num, col_icon, col_text = st.columns([0.4, 0.3, 5])
    with col_num:
        st.markdown(f"**{i+1}.**")
    with col_icon:
        st.markdown(icon)
    with col_text:
        st.markdown(f"**{name}** — {desc}")

st.divider()

# ── Knowledge base stats ───────────────────────────────────────────────────────

st.markdown("### 🗄️ Knowledge Base")
kb1, kb2, kb3 = st.columns(3)
with kb1:
    st.metric("Indexed Documents", "4")
with kb2:
    st.metric("Total Chunks", "3,195")
with kb3:
    st.metric("Embedding Model", "BAAI/bge-base-en-v1.5")

st.caption("BNS + BNSS + BSA + Constitution of India · BM25 index (Qdrant dense index available when connected)")
