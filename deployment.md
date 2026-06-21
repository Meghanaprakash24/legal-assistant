# Render Deployment Guide

Complete instructions for deploying the Indian Legal RAG System to [Render](https://render.com) as two separate web services.

---

## Architecture Overview

```
┌────────────────────────┐         ┌────────────────────────┐
│   Streamlit Frontend   │────────▶│   FastAPI Backend      │
│   (Render Web Service) │  HTTP   │   (Render Web Service) │
│                        │         │                        │
│   Port: $PORT          │         │   Port: $PORT          │
│   Env: API_URL         │         │   Env: GROQ_API_KEY    │
│                        │         │        QDRANT_URL       │
│                        │         │        QDRANT_API_KEY   │
└────────────────────────┘         └───────────┬────────────┘
                                               │
                                   ┌───────────▼────────────┐
                                   │   Qdrant Cloud         │
                                   │   (External Service)   │
                                   │                        │
                                   │   Collection:          │
                                   │   legal_documents      │
                                   └────────────────────────┘
```

---

## Prerequisites

Before deploying, ensure you have:

1. **Qdrant Cloud account** with a cluster and the `legal_documents` collection already populated with vectors (see [First-Time Indexing](#first-time-indexing) if not).
2. **Groq API key** from [console.groq.com](https://console.groq.com).
3. **Render account** at [render.com](https://render.com).
4. **Repository pushed to GitHub/GitLab** (Render deploys from Git).

---

## Environment Variables

### Required for Backend

| Variable | Description | Example |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM inference | `gsk_...` |
| `QDRANT_URL` | Qdrant Cloud cluster URL | `https://abc123.aws.cloud.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant Cloud API key | `eyJhbG...` |
| `COLLECTION_NAME` | Qdrant collection name | `legal_documents` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `https://legal-rag-frontend.onrender.com` |

### Required for Frontend

| Variable | Description | Example |
|----------|-------------|---------|
| `API_URL` | Backend service URL | `https://legal-rag-backend.onrender.com` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model to use |
| `TEMPERATURE` | `0.2` | LLM temperature |
| `MAX_TOKENS` | `1200` | LLM max tokens |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | CrossEncoder model |
| `RERANKER_TOP_K` | `5` | Results after reranking |

See [`.env.example`](.env.example) for the full list.

---

## Step-by-Step Render Setup

### Step 1: Deploy the Backend

1. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect your GitHub/GitLab repository.
3. Configure the service:

   | Setting | Value |
   |---------|-------|
   | **Name** | `legal-rag-backend` |
   | **Region** | Oregon (US West) or nearest |
   | **Branch** | `main` |
   | **Runtime** | Python |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
   | **Plan** | Starter or higher (512MB+ RAM recommended) |
   | **Health Check Path** | `/health` |

4. Add environment variables in the **Environment** tab:
   - `GROQ_API_KEY` = your Groq API key
   - `QDRANT_URL` = your Qdrant Cloud URL
   - `QDRANT_API_KEY` = your Qdrant API key
   - `COLLECTION_NAME` = `legal_documents`
   - `CORS_ORIGINS` = *(leave empty for now, set after frontend deploy)*
   - `LOG_LEVEL` = `INFO`

5. Click **Create Web Service** and wait for the first deploy to complete.

6. Note the service URL (e.g., `https://legal-rag-backend.onrender.com`).

7. Verify: visit `https://legal-rag-backend.onrender.com/health` — you should see:
   ```json
   {"status": "healthy", "pipeline": true, "qdrant": true, "groq": true}
   ```

### Step 2: Deploy the Frontend

1. Go to Render Dashboard → **New** → **Web Service**.
2. Connect the **same repository**.
3. Configure the service:

   | Setting | Value |
   |---------|-------|
   | **Name** | `legal-rag-frontend` |
   | **Region** | Same as backend |
   | **Branch** | `main` |
   | **Runtime** | Python |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false` |
   | **Plan** | Starter or higher |

4. Add environment variables:
   - `API_URL` = `https://legal-rag-backend.onrender.com` *(your backend URL from Step 1)*

5. Click **Create Web Service**.

### Step 3: Configure CORS

After the frontend is deployed, go back to the **backend** service:

1. Open **Environment** settings.
2. Set `CORS_ORIGINS` to your frontend URL:
   ```
   https://legal-rag-frontend.onrender.com
   ```
   For multiple origins, comma-separate them:
   ```
   https://legal-rag-frontend.onrender.com,http://localhost:8501
   ```
3. Save — Render will auto-redeploy the backend.

---

## First-Time Indexing

If the Qdrant collection does not exist yet or is empty, you must run the indexing pipeline **before deployment**. This is a one-time offline operation:

```bash
# 1. Set environment variables
export QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io
export QDRANT_API_KEY=your_key_here

# 2. Ensure data/chunked/ contains the chunk JSON files
ls data/chunked/
# Expected: BNS_chunks.json  BNSS_chunks.json  BSA_chunks.json  Constitution_chunks.json

# 3. Generate embeddings (writes to data/embeddings/)
python -m backend.src.embedder

# 4. Index into Qdrant (uploads vectors)
python -m backend.src.indexer

# 5. Create payload indexes for metadata filtering
python create_indexes.py
```

> **Note**: The `legal_documents` collection should already contain vectors if you've previously indexed. Check via the Qdrant Cloud dashboard or the `/health` endpoint after backend deployment.

### Qdrant Collection Details

| Property | Value |
|----------|-------|
| Collection Name | `legal_documents` |
| Embedding Model | `BAAI/bge-base-en-v1.5` |
| Embedding Dimension | 768 |
| Distance Metric | Cosine |
| Payload Indexes | `document`, `section`, `article`, `chunk_type`, `parent_chunk_id` |
| Documents Indexed | BNS, BNSS, BSA, Constitution of India |

---

## Local Development

For local development, create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Then run both services:

```bash
# Terminal 1: Backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend
cd frontend && streamlit run app.py --server.port 8501
```

---

## Troubleshooting

### Backend fails to start

- **`GROQ_API_KEY is not set`**: Set the `GROQ_API_KEY` environment variable in Render.
- **`QDRANT_URL is not set`**: Set the `QDRANT_URL` environment variable.
- **`Pipeline initialization failed`**: Check that `data/chunked/` contains the chunk JSON files and that the Qdrant collection exists with vectors.
- **Out of memory**: The backend loads ML models (~1-2GB). Upgrade to a Starter plan or higher.

### Frontend cannot reach backend

- **Connection refused**: Verify `API_URL` is set to the correct backend URL.
- **CORS errors**: Ensure `CORS_ORIGINS` on the backend includes the frontend's full origin URL (including `https://`).
- **Mixed content**: Both services must use HTTPS (Render provides this by default).

### Health endpoint shows `degraded`

- **`pipeline: false`**: Check backend startup logs for BM25 index build errors. Ensure `data/chunked/` files are present.
- **`qdrant: false`**: Verify `QDRANT_URL` and `QDRANT_API_KEY`. Check Qdrant Cloud cluster status.
- **`groq: false`**: Verify `GROQ_API_KEY` is set and valid.

---

## Files Modified for Deployment

| File | Changes |
|------|---------|
| `config.py` | CORS from env, PORT from env, data dirs relative to BASE_DIR, missing indexer constants, production LOG_LEVEL |
| `app.py` | `__main__` uses config values, startup validation, safe log dir creation |
| `frontend/config.py` | `API_URL` env var support |
| `frontend/components/sidebar.py` | Removed hardcoded localhost fallback |
| `render-backend.yaml` | **NEW** — Render backend Blueprint |
| `render-frontend.yaml` | **NEW** — Render frontend Blueprint |
| `.env.example` | **NEW** — Environment variable template |
| `deployment.md` | **NEW** — This file |
| `.gitignore` | Added `data/`, `*.log` |

> **No business logic, retrieval pipeline, prompt logic, or reranking behavior was modified.**
