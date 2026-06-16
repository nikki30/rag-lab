<div align="center">

# 🔬 RAG Lab

### Learn RAG by watching it work.

**An interactive, local-first laboratory for every decision in a Retrieval-Augmented Generation pipeline — chunking, embedding, indexing, retrieval, reranking, generation, and evaluation. No abstractions, no cloud dependencies, every algorithm visualised in real time.**

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)
![No API Key](https://img.shields.io/badge/API%20Key-None%20Required-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

<br>

![RAG Lab demo](docs/assets/RAGLABDemo.gif)

<br>

**[Quick start](#-quick-start)** · **[Pipeline stages](#-pipeline-stages)** · **[Concept guides](#-concept-guides)** · **[SOTA concepts](#-state-of-the-art-concepts-woven-in)**

</div>

<br>

> **Why this exists.** Most RAG tutorials show you how to build a pipeline. RAG Lab shows you *why* each decision matters — with the real HNSW graph being traversed, the cosine similarity heatmap you can hover, the reranking shifts laid out side-by-side, and a final radar chart scoring the whole pipeline. Built for learning the field deeply, then teaching it.

<br>

## ⚡ Quick start

**Prerequisites:** Python 3.10+, Node.js 18+. No API keys. No cloud accounts. Runs entirely on your machine.

```bash
git clone https://github.com/nikki30/rag-lab.git
cd rag-lab
```

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

```bash
# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Then open **http://localhost:3000** and work down the page — chunk → embed → index → retrieve → generate → evaluate.

> **First run** downloads embedding models (~90–400 MB depending on model chosen). Subsequent runs use the local cache.
>
> **Generation stage** works out of the box with a built-in mock LLM — no Ollama, no API key, no cloud service required. See [how the mock LLM works](#mock-llm).

<br>

## 🧬 Pipeline stages

| Stage | Status | What you can do |
|-------|--------|-----------------|
| **Chunking** | ✅ Done | Compare 5 strategies (recursive, fixed, paragraph, sentence, semantic), tune parameters, inspect per-chunk quality scores, watch topic-similarity between sentences on a live chart. Semantic chunking uses real MiniLM embeddings — threshold changes produce visibly different chunk boundaries. |
| **Embedding** | ✅ Done | Embed chunks with 4 local models (MiniLM, BGE-Small, MPNet, Nomic). Visualise the vector space with PCA / UMAP / PaCMAP, inspect the cosine similarity heatmap with chunking-issue detection, and explore raw vector values per chunk. |
| **Indexing** | ✅ Done | Build Flat / HNSW / IVF / MRL indexes. Visualise the real HNSW multilayer graph with layer-by-layer traversal, IVF cluster boundaries with `nprobe` selection, and the MRL dimension-truncation recall table. Orange/green colour coding distinguishes rebuild vs query-only parameters. |
| **Retrieval** | ✅ Done | Type a query, embed it, watch it find the nearest chunks. Compare dense (cosine), sparse (BM25), and hybrid (Reciprocal Rank Fusion) side-by-side. |
| **Reranking** | ✅ Done | Cross-encoder re-scores the top-k candidates with full query-chunk interaction. ColBERT late interaction shows token-level MaxSim heatmaps. Rank shift table compares all four strategies in one view. |
| **Generation** | ✅ Done | Choose from four mock LLMs (GPT-4o-mini, Claude Haiku, Llama 3.3 70B, Local Llama). Apply contextual compaction, control chunk ordering (relevance vs sandwich). See the assembled prompt broken down by section, real token cost, and per-sentence grounding highlighting on the answer. |
| **Evaluation** | ✅ Done | Score the full pipeline end-to-end with [RAGAS](https://docs.ragas.io)-style metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall, Noise Sensitivity. Radar chart, sentence-level grounding breakdown, per-rank chunk relevance table, optional ground truth coverage check. |

<br>

## 🤖 Mock LLM

The Generation stage uses a fully local mock LLM — no API calls, no tokens, no cost.

**How it works:** The mock scores every sentence in the retrieved chunks by query-term overlap (ignoring stop words), selects the top 3, and assembles a coherent answer. It's deterministic — the same query produces the same answer every time.

**What's still real:** Grounding scores, token counting, context window limits, cost estimates, compaction algorithms, and chunk ordering all behave exactly as they would with a real model. The pipeline mechanics you're learning are accurate.

**Why mock?** So anyone can run the full pipeline immediately — no account, no credit card, no rate limits. When you understand how each stage works, you can swap in a real model with one line.

<br>

## 🧠 State-of-the-art concepts woven in

| Concept | Where to see it |
|---------|----------------|
| Late chunking (Jina, 2024) — embed the full document first, then pool chunk token windows | Chunking — *[docs/chunking101.md](docs/chunking101.md)* |
| Matryoshka Representation Learning (MRL) — truncate 768d vectors gracefully to 64d | Indexing — *active in MRL index tab* |
| HNSW (Malkov & Yashunin) — hierarchical navigable small-world graph | Indexing — *active, with traversal viz* |
| Reciprocal Rank Fusion — merge dense + sparse retrieval | Retrieval — *active* |
| Cross-encoder reranking (ms-marco-MiniLM) | Reranking — *active* |
| ColBERT late interaction — token-level MaxSim scoring | Reranking — *active, with heatmap viz* |
| Lost-in-the-Middle (Liu et al., 2023) — recency / primacy bias in long contexts | Generation — *active, drives chunk-order options* |
| Contextual compaction — query-conditioned chunk filtering | Generation — *active* |
| RAGAS evaluation (Faithfulness, Relevancy, Precision, Recall) | Evaluation — *active* |

<br>

## 📚 Concept guides

Each stage has a long-form concept guide explaining the algorithms and intuitions, with references to the foundational papers and how Microsoft / Anthropic / Google deploy them in production.

- [`docs/chunking101.md`](docs/chunking101.md) — 5 strategies compared, quality scores, hub chunks, σ, contextual + late chunking
- [`docs/embedding101.md`](docs/embedding101.md) — vectors, cosine similarity, MRL, PCA vs UMAP vs PaCMAP, heatmap interpretation
- [`docs/indexing101.md`](docs/indexing101.md) — FAISS, HNSW, IVF, Flat — build vs query parameter breakdown
- [`docs/retrieval101.md`](docs/retrieval101.md) — Dense, BM25, RRF hybrid, cross-encoder re-ranking, ColBERT late interaction
- [`docs/reranking101.md`](docs/reranking101.md) — Cross-encoder vs ColBERT, rank shift analysis, MonoT5/DuoT5, Cohere Rerank, Lost-in-the-Middle, Microsoft Semantic Ranking
- [`docs/generate101.md`](docs/generate101.md) — token budget problem, compaction algorithms, chunk ordering, context strategies
- [`docs/eval101.md`](docs/eval101.md) — RAGAS metrics, LLM-as-judge, Microsoft Azure AI Eval, Anthropic model-graded evaluation, TruLens RAG Triad

<br>

## 🛠️ Stack

- **Backend** — FastAPI · Python · LangChain text splitters · sentence-transformers · scikit-learn · FAISS · umap-learn · pacmap
- **Frontend** — Next.js 16 · React 19 · TypeScript · Tailwind CSS v4 · App Router

<br>

### Troubleshooting

If port 8000 is stuck:
```bash
# Windows
powershell -Command "Get-Process python* | Stop-Process -Force"

# macOS / Linux
pkill -f uvicorn
```

The backend requires a hard restart after any `main.py` change (FastAPI's `--reload` is off by default for stability).
