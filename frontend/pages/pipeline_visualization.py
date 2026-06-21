"""
Pipeline Visualization — LexAI.
Shows the multi-agent workflow with live backend statistics.
No mock data — static pipeline diagram + real runtime metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from services.api import api_client

st.set_page_config(page_title="Pipeline Visualization", page_icon="🔄", layout="wide")

_MESSAGES_KEY = "legal_assistant_messages"

st.markdown("## 🔄 Pipeline Visualization")
st.caption("The LexAI multi-agent RAG workflow — each stage runs in sequence for every query.")

# ── Live backend stats ─────────────────────────────────────────────────────────

def _stats() -> Dict[str, Any]:
    r = api_client.statistics()
    return r.get("data", {}) if r.get("success") else {}

def _health() -> Dict[str, Any]:
    r = api_client.health()
    return r.get("data", {}) if r.get("success") else {}

if st.button("🔄 Refresh"):
    st.rerun()

stats = _stats()
health = _health()

if stats:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        chat_calls = stats.get("endpoint_counts", {}).get("/chat", 0)
        st.metric("Queries Processed", chat_calls)
    with m2:
        avg = stats.get("average_latency_ms")
        st.metric("Avg End-to-End Latency", f"{avg:.0f} ms" if avg else "—")
    with m3:
        st.metric("Success Rate", f"{stats.get('successful_requests',0)}/{stats.get('total_requests',1)}")
    with m4:
        pipeline_ok = health.get("pipeline", False)
        st.metric("Pipeline Status", "✓ Active" if pipeline_ok else "✗ Degraded")

st.divider()

# ── Pipeline diagram ───────────────────────────────────────────────────────────

st.markdown("### ⚙️ Agent Workflow")

STAGES = [
    {
        "icon": "💬",
        "name": "User Query",
        "role": "Input",
        "description": "Raw natural-language legal question from the user.",
        "detail": "Passed to the Classifier Agent as plain text.",
    },
    {
        "icon": "🧭",
        "name": "Classifier Agent",
        "role": "Information Extraction",
        "description": "Extracts incident types, legal domain, detected section/article numbers, and keywords.",
        "detail": "Pure regex + keyword matching. No LLM. Fast (<5 ms).",
    },
    {
        "icon": "🔍",
        "name": "Hybrid Retriever",
        "role": "Document Retrieval",
        "description": "Runs BM25 keyword search and Qdrant dense vector search in parallel across all 4 legal documents.",
        "detail": "BM25: rank_bm25 index, 3,195 chunks. Dense: BAAI/bge-base-en-v1.5, 768-dim. Fusion: Reciprocal Rank Fusion (RRF). Top-30 candidates.",
    },
    {
        "icon": "🎯",
        "name": "BGE CrossEncoder Reranker",
        "role": "Relevance Scoring",
        "description": "Scores each candidate chunk against the query using BAAI/bge-reranker-base.",
        "detail": "Keeps top-5 chunks by reranker score. Runs on CPU (~1–2 s per query).",
    },
    {
        "icon": "❝",
        "name": "Quote Selector",
        "role": "Sentence Extraction",
        "description": "Splits each chunk's retrieval_text into sentences and selects the most relevant.",
        "detail": "Up to 3 quotes per chunk. Deduplication across chunks.",
    },
    {
        "icon": "🗺️",
        "name": "Section Mapper",
        "role": "Metadata Extraction (Primary)",
        "description": "Reads document, section, and article fields directly from retrieved chunk metadata. No LLM inference.",
        "detail": "PRIMARY: metadata extraction (confidence 0.70–0.98). FALLBACK: SECTION_REGISTRY scoring only when metadata is absent.",
    },
    {
        "icon": "⚖️",
        "name": "Remedy Advisor",
        "role": "Procedure Recommendation",
        "description": "Maps incident types to recommended legal actions, documents to gather, and procedural steps.",
        "detail": "Rule-based lookup. No LLM.",
    },
    {
        "icon": "✅",
        "name": "Citation Validator",
        "role": "Grounding Verification",
        "description": "Verifies every section and quote exists in the retrieved chunk corpus. Rejects fabricated citations.",
        "detail": "Checks section label against chunk metadata. Checks quotes against retrieval_text with similarity scoring.",
    },
    {
        "icon": "🧩",
        "name": "Response Synthesizer",
        "role": "LLM Generation",
        "description": "Sends validated evidence + full retrieved chunk texts to Groq LLaMA 3.3 70B and generates structured JSON.",
        "detail": "Extracts verbatim quotations. Produces: summary, query_understanding, applicable_law, relevant_quotations, legal_explanation, recommended_procedure, citations.",
    },
    {
        "icon": "📄",
        "name": "Final Response",
        "role": "Output",
        "description": "Structured JSON response returned to the frontend.",
        "detail": "All sections are grounded in retrieved chunks. No hallucinated citations.",
    },
]

for i, stage in enumerate(STAGES):
    is_llm = stage["name"] == "Response Synthesizer"
    border_color = "#6366f1" if is_llm else "#e5e7eb"

    cols = st.columns([0.05, 0.05, 0.35, 3])
    with cols[0]:
        st.markdown(f"**{i+1 if i < len(STAGES)-1 else '✓'}**")
    with cols[1]:
        st.markdown(f"## {stage['icon']}")
    with cols[2]:
        label_color = "#6366f1" if is_llm else "#374151"
        st.markdown(
            f'<div style="font-weight:700;color:{label_color};">{stage["name"]}</div>'
            f'<div style="font-size:0.75em;color:#6b7280;">{stage["role"]}</div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(f"{stage['description']}")
        st.caption(stage["detail"])

    if i < len(STAGES) - 1:
        st.markdown(
            '<div style="margin-left:3rem;color:#9ca3af;font-size:1.2em;">↓</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Session last query analysis ────────────────────────────────────────────────

st.markdown("### 📊 Last Query Analysis")

msgs = st.session_state.get(_MESSAGES_KEY, [])
last_assistant = None
last_user = None
for m in reversed(msgs):
    if m.get("role") == "assistant" and isinstance(m.get("content"), dict) and last_assistant is None:
        last_assistant = m["content"]
    if m.get("role") == "user" and last_user is None:
        last_user = str(m.get("content", ""))
    if last_assistant and last_user:
        break

if last_assistant:
    st.markdown(f"**Query:** `{last_user}`")
    a1, a2, a3 = st.columns(3)
    with a1:
        laws = last_assistant.get("applicable_law") or []
        st.metric("Sections Identified", len(laws))
    with a2:
        quotes = last_assistant.get("relevant_quotations") or []
        st.metric("Quotations Extracted", len(quotes))
    with a3:
        citations = last_assistant.get("citations") or []
        st.metric("Citations Generated", len(citations))

    if laws:
        st.markdown("**Applicable Sections:**")
        for law in laws:
            st.markdown(
                f'<span style="background:#e8f0fe;color:#1a56db;padding:3px 10px;'
                f'border-radius:12px;font-size:0.85em;margin:3px;display:inline-block;">'
                f'📌 {law.get("document")} — {law.get("section")}</span>',
                unsafe_allow_html=True,
            )
else:
    st.info("No queries processed in this session yet. Run a query in the Legal Assistant to see per-query pipeline analysis.")
