# RAG Lab

An interactive, local-first laboratory for exploring RAG pipeline decisions visually. Pick a chunking strategy, embedding model, or retrieval algorithm and immediately see how each choice affects the output — no cloud dependencies, no abstractions hiding the mechanics.

## Pipeline stages

| Stage | Status | What you can do |
|-------|--------|-----------------|
| **Chunking** | Done | Compare 5 strategies (recursive, fixed, paragraph, sentence, semantic), tune parameters, inspect per-chunk quality scores, and see topic-similarity between sentences on a live chart |
| **Embedding** | Done | Embed chunks with 4 local models (MiniLM, BGE-Small, MPNet, Nomic), visualise the vector space with PCA / UMAP / PaCMAP, inspect the cosine similarity heatmap with chunking-issue detection, and explore raw vector values |
| **Retrieval** | Planned | Type a query, embed it in the same space, see which chunks are nearest — then add HNSW indexing to show how approximate search scales to millions of vectors |
| **Reranking** | Planned | Cross-encoder reranking on top of vector retrieval for higher accuracy |

## Stack

- **Backend** — FastAPI, Python, LangChain text splitters, sentence-transformers, scikit-learn, umap-learn, pacmap
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

> First run of the embedding stage will download the selected model (~90–400 MB). Subsequent runs use the cached version.
