# RAG Lab

An interactive, local-first laboratory for exploring RAG pipeline decisions visually. Pick a chunking strategy, embedding model, or retrieval algorithm and immediately see how each choice affects the output — no cloud dependencies, no abstractions hiding the mechanics.

## Pipeline stages

| Stage | Status | What you can do |
|-------|--------|-----------------|
| **Chunking** | ✅ Done | Compare 5 strategies (recursive, fixed, paragraph, sentence, semantic), tune parameters, inspect per-chunk quality scores, and see topic-similarity between sentences on a live chart. Semantic chunking uses real MiniLM embeddings — threshold changes produce visibly different chunk boundaries. |
| **Embedding** | ✅ Done | Embed chunks with 4 local models (MiniLM, BGE-Small, MPNet, Nomic), visualise the vector space with PCA / UMAP / PaCMAP, inspect the cosine similarity heatmap with chunking-issue detection, and explore raw vector values |
| **Indexing** | ✅ Done | Build Flat / HNSW / IVF / MRL indexes over your embedded vectors. Visualise the real HNSW multilayer graph with layer-by-layer traversal, IVF cluster boundaries with nprobe selection, and MRL dimension-truncation recall table. Orange/green colour coding distinguishes index-rebuild vs query-only parameters. See `docs/indexing101.md` for concept notes. |
| **Retrieval** | 🔜 Next | Type a query, embed it, and watch it find the nearest chunks — compare dense (cosine), sparse (BM25), and hybrid (Reciprocal Rank Fusion) side-by-side; experiment with HyDE and query decomposition |
| **Reranking** | Planned | Cross-encoder reranking to re-score the top-k results more accurately; visualise rank shifts and explore ColBERT-style late interaction scoring |
| **Generation** | Planned | Assemble retrieved chunks into a prompt, call a real LLM, and see faithfulness highlighting that shows which parts of the answer are grounded vs. likely hallucinated |

### State-of-the-art concepts woven in throughout

| Concept | Stage |
|---------|-------|
| Contextual chunking (Anthropic) — prepend per-chunk summaries before embedding | Chunking |
| Late chunking — embed the full document first, then pool chunk token windows | Chunking / Embedding |
| HyDE — generate a hypothetical answer and embed *that* as the query | Retrieval |
| Query decomposition — split a complex question into focused sub-queries | Retrieval |
| ColBERT late interaction — token-level MaxSim scoring between query and chunk | Reranking |
| GraphRAG — knowledge-graph traversal as an alternative to pure vector search | Retrieval / Generation |

## Reference notes

- [`docs/indexing101.md`](docs/indexing101.md) — FAISS, HNSW, IVF, Flat explained with build vs query parameter breakdown
- [`docs/retrieval101.md`](docs/retrieval101.md) — Dense, BM25, RRF hybrid, cross-encoder re-ranking, ColBERT late interaction

## Stack

- **Backend** — FastAPI, Python, LangChain text splitters, sentence-transformers, scikit-learn, FAISS, umap-learn, pacmap
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

> First run downloads models: embedding stage (~90–400 MB depending on model), semantic chunking stage (MiniLM, ~90 MB). Subsequent runs use the cached version.
