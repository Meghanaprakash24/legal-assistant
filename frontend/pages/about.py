"""
About page — LexAI Indian Legal RAG Assistant.
Concise technical reference: overview, architecture, stack, workflow, retrieval, knowledge base.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="About — LexAI", page_icon="ℹ️", layout="wide")

st.markdown("## ℹ️ About LexAI")
st.caption(
    "A production-grade Indian Legal Research Assistant built on Hybrid RAG, "
    "multi-agent orchestration, and grounded citation validation."
)

# ── 1. Project Overview ────────────────────────────────────────────────────────

st.divider()
st.markdown("### 🎯 Project Overview")
st.markdown(
    """
LexAI is an AI-powered legal research assistant that answers questions about Indian law
by retrieving and grounding every response in indexed statutory documents.

**Design principles:**
- Every section, article, and citation is extracted from retrieved chunk metadata — never inferred by an LLM.
- The Validator agent rejects any section not found in the retrieved corpus.
- If the corpus does not contain an answer, the system says so explicitly instead of hallucinating.
- All four primary Indian criminal/constitutional law texts are indexed and searchable together.
"""
)

# ── 2. Architecture Summary ────────────────────────────────────────────────────

st.divider()
st.markdown("### 🏛️ Architecture Summary")
st.markdown(
    """
The system is a **FastAPI backend** (port 8000) with a **Streamlit frontend** (port 5000).

The backend runs a **LangGraph multi-agent workflow** for every query.
Each node is a specialized agent; the graph is compiled once and reused across requests.

```
User Query → FastAPI /chat
           → LangGraph workflow
               [Classifier] → [Retriever] → [Quote Selector]
               → [Section Mapper] → [Remedy Advisor]
               → [Validator] → [Synthesizer]
           → Structured JSON response
```

The retriever uses a **BM25 index** (3,195 chunks) and, when Qdrant is reachable,
an additional **dense vector index** (BAAI/bge-base-en-v1.5 embeddings, 768 dims).
Results are merged and reranked by the **BGE CrossEncoder**.
"""
)

# ── 3. Technology Stack ────────────────────────────────────────────────────────

st.divider()
st.markdown("### 🛠️ Technology Stack")

stack = [
    ("🚀", "Backend API", "FastAPI + Uvicorn (Python 3.11)"),
    ("🖥️", "Frontend UI", "Streamlit"),
    ("🕸️", "Orchestration", "LangGraph (stateful multi-agent workflow)"),
    ("🧬", "Embedding Model", "BAAI/bge-base-en-v1.5 (768-dim dense vectors)"),
    ("🎯", "Reranker", "BAAI/bge-reranker-base (CrossEncoder)"),
    ("🔍", "Keyword Search", "BM25 (rank_bm25) — 3,195 legal chunks"),
    ("🗄️", "Vector Database", "Qdrant (dense search, optional)"),
    ("🤖", "LLM", "Groq API — llama-3.3-70b-versatile"),
]
col_a, col_b = st.columns(2)
for i, (icon, role, tool) in enumerate(stack):
    col = col_a if i % 2 == 0 else col_b
    with col:
        st.markdown(f"**{icon} {role}:** {tool}")

# ── 4. Multi-Agent Workflow ────────────────────────────────────────────────────

st.divider()
st.markdown("### 🤖 Multi-Agent Workflow")

agents = [
    ("Classifier", "Extracts `incident_type`, `keywords`, detected section/article numbers, and legal domain from the raw user query. No LLM — pure regex and keyword matching."),
    ("Hybrid Retriever", "Runs BM25 keyword search and dense vector search in parallel across all 4 documents. Merges results with Reciprocal Rank Fusion (RRF)."),
    ("BGE Reranker", "CrossEncoder scores each candidate chunk against the query. Returns the top-K highest-scoring chunks."),
    ("Quote Selector", "Splits each chunk's `retrieval_text` into sentences and selects the most relevant sentences as verbatim quotes."),
    ("Section Mapper", "**PRIMARY**: reads `document`, `section`, `article` metadata directly from retrieved chunk dicts — no LLM inference. **FALLBACK**: SECTION_REGISTRY if metadata is absent."),
    ("Remedy Advisor", "Maps incident types to recommended legal actions, documents, and procedural steps."),
    ("Citation Validator", "Verifies every section and quote exists in the retrieved chunk corpus. Rejects fabricated citations."),
    ("Response Synthesizer", "Sends validated evidence + full retrieved chunk texts to Groq LLaMA 3.3 70B. Extracts verbatim quotations and produces structured JSON output."),
]
for name, desc in agents:
    with st.expander(f"**{name}**"):
        st.markdown(desc)

# ── 5. Retrieval Pipeline ──────────────────────────────────────────────────────

st.divider()
st.markdown("### 🔍 Retrieval Pipeline")
st.markdown(
    """
| Step | Component | Detail |
|------|-----------|--------|
| 1 | BM25 Search | Top-30 keyword candidates from `rank_bm25` index |
| 2 | Dense Search | Top-30 semantic candidates from Qdrant (when available) |
| 3 | Fusion | Reciprocal Rank Fusion merges BM25 + dense lists |
| 4 | Reranking | BGE CrossEncoder scores all candidates; top-5 kept |
| 5 | Metadata extraction | `document`, `section`, `article` read from chunk dicts |
| 6 | Validation | Validator confirms every cited section is in retrieved chunks |
"""
)

# ── 6. Legal Knowledge Base ────────────────────────────────────────────────────

st.divider()
st.markdown("### 📚 Legal Knowledge Base")

kb_docs = [
    ("⚖️", "Constitution of India", "Articles 1–395 + Schedules", "Fundamental rights, directive principles, government structure"),
    ("📖", "Bharatiya Nyaya Sanhita (BNS)", "Sections 1–358", "Offences and punishments — replaces IPC"),
    ("🔒", "Bharatiya Nagarik Suraksha Sanhita (BNSS)", "Sections 1–531", "Criminal procedure — replaces CrPC"),
    ("📜", "Bharatiya Sakshya Adhiniyam (BSA)", "Sections 1–170", "Evidence law — replaces Indian Evidence Act"),
]
for icon, name, coverage, desc in kb_docs:
    col_icon, col_text = st.columns([0.3, 5])
    with col_icon:
        st.markdown(f"## {icon}")
    with col_text:
        st.markdown(f"**{name}**")
        st.caption(f"{coverage} · {desc}")
    st.write("")

st.caption("Total indexed chunks: 3,195 · Embedding model: BAAI/bge-base-en-v1.5 · Index: BM25 + Qdrant")
