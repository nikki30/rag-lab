# RAG Lab

An interactive, local-first laboratory for exploring RAG pipeline decisions visually. Pick a chunking strategy, embedding model, or retrieval algorithm and immediately see how each choice affects the output — no cloud dependencies, no abstractions hiding the mechanics.

## Pipeline stages

| Stage | Status | What you can do |
|-------|--------|-----------------|
| **Chunking** | ✅ Done | Compare 5 strategies (recursive, fixed, paragraph, sentence, semantic), tune parameters, inspect per-chunk quality scores, and see topic-similarity between sentences on a live chart. Semantic chunking uses real MiniLM embeddings — threshold changes produce visibly different chunk boundaries. |
| **Embedding** | ✅ Done | Embed chunks with 4 local models (MiniLM, BGE-Small, MPNet, Nomic), visualise the vector space with PCA / UMAP / PaCMAP, inspect the cosine similarity heatmap with chunking-issue detection, and explore raw vector values |
| **Indexing** | ✅ Done | Build Flat / HNSW / IVF / MRL indexes over your embedded vectors. Visualise the real HNSW multilayer graph with layer-by-layer traversal, IVF cluster boundaries with nprobe selection, and MRL dimension-truncation recall table. Orange/green colour coding distinguishes index-rebuild vs query-only parameters. See `docs/indexing101.md` for concept notes. |
| **Retrieval** | ✅ Done | Type a query, embed it, and watch it find the nearest chunks — compare dense (cosine), sparse (BM25), and hybrid (RRF) side-by-side. Experiment with HyDE (hypothetical document embedding) and query decomposition. See `docs/retrieval101.md`. |
| **Reranking** | ✅ Done | Cross-encoder re-scores the top-k retrieval candidates with full query-chunk interaction. ColBERT late interaction shows token-level MaxSim heatmaps. Rank shift table compares all four strategies side-by-side. |
| **Generation** | ✅ Done | Select an LLM (Ollama local, Groq free tier, OpenAI, Anthropic), choose a compaction strategy (raw / contextual / LLMLingua), chunk ordering, and context strategy — then see the assembled prompt breakdown and grounding highlighting on the answer. |
| **Evaluation** | ✅ Done | Score the full pipeline end-to-end with [RAGAS](https://docs.ragas.io)-style metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Noise Sensitivity. Radar chart, sentence-level grounding breakdown, chunk relevance table, and optional ground truth recall. |
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

- [`docs/chunking101.md`](docs/chunking101.md) — 5 strategies compared, quality scores, hub chunks, σ, contextual chunking, late chunking
- [`docs/embedding101.md`](docs/embedding101.md) — vectors, cosine similarity, MRL, PCA vs UMAP vs PaCMAP, heatmap interpretation
- [`docs/indexing101.md`](docs/indexing101.md) — FAISS, HNSW, IVF, Flat explained with build vs query parameter breakdown
- [`docs/retrieval101.md`](docs/retrieval101.md) — Dense, BM25, RRF hybrid, cross-encoder re-ranking, ColBERT late interaction
- [`docs/reranking101.md`](docs/reranking101.md) — Cross-encoder vs ColBERT, rank shift analysis, MonoT5/DuoT5, Cohere Rerank, Lost-in-the-Middle, Microsoft Semantic Ranking
- [`docs/generate101.md`](docs/generate101.md) — token budget problem, compaction algorithms, chunk ordering, context strategies
- [`docs/eval101.md`](docs/eval101.md) — RAGAS metrics, LLM-as-judge, Microsoft Azure AI Eval, Anthropic model-graded evaluation, TruLens RAG Triad

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

### Generation stage — LLM setup

The Generation stage supports four providers. The easiest to get started with is **Ollama** (runs entirely locally, no API key, no cost):

**Ollama (recommended for first run)**
1. Download and install from [ollama.com](https://ollama.com)
2. Pull the Llama 3.2 model:
```bash
ollama pull llama3.2
```
3. Ollama starts automatically — no separate server command needed. The model runs on your machine (~2 GB download).

**Groq (free cloud tier)**
1. Sign up at [console.groq.com](https://console.groq.com) — free tier, no credit card
2. Create an API key and set it as `GROQ_API_KEY` in `backend/.env`

**OpenAI / Anthropic**
Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `backend/.env`.
