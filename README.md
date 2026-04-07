# RAG Lab

An interactive, local-first laboratory for exploring RAG pipeline decisions visually. Pick a chunking strategy, embedding model, or retrieval algorithm and immediately see how each choice affects the output — no cloud dependencies, no abstractions hiding the mechanics.

## Pipeline stages

| Stage | Status | What you can do |
|-------|--------|-----------------|
| **Chunking** | Done | Compare 5 strategies (recursive, fixed, paragraph, sentence, semantic), tune parameters, inspect per-chunk quality scores, and see topic-similarity between sentences on a live chart |
| **Embedding** | Planned | Convert chunks to dense vectors using local sentence-transformers models, visualise the vector space as a 2D scatter plot (PCA / UMAP), swap models and watch the space change |
| **Indexing** | Planned | Index the embeddings with HNSW and DiskANN, compare recall vs latency tradeoffs interactively |
| **Retrieval** | Planned | Issue queries and see which chunks are retrieved, explore GraphRAG's multi-hop graph traversal against flat vector search |

## Stack

- **Backend** — FastAPI, Python, LangChain text splitters, TF-IDF cosine similarity (chunking), sentence-transformers + scikit-learn (embedding, upcoming)
- **Frontend** — Next.js, React, TypeScript, Tailwind CSS v4, App Router

## Running locally

```bash
# Backend — hard-restart required after any main.py change
cd backend
venv/Scripts/uvicorn main:app --port 8000

# Frontend
cd frontend
npm run dev   # → http://localhost:3000
```

If port 8000 is stuck:
```bash
powershell -Command "Get-Process python* | Stop-Process -Force"
```
