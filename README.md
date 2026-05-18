<div align="center">

# 🔬 RAG Lab

### Learn RAG by watching it work.

**An interactive, local-first laboratory for every decision in a Retrieval-Augmented Generation pipeline — chunking, embedding, indexing, retrieval, reranking, generation, and evaluation. No abstractions, no cloud dependencies, every algorithm visualised in real time.**

<br>

![RAG Lab demo](docs/assets/RAGLABDemo.gif)

<br>

**[Quick start](#-quick-start)** · **[Pipeline stages](#-pipeline-stages)** · **[Roadmap](#-roadmap)** · **[Concept guides](#-concept-guides)** · **[SOTA concepts](#-state-of-the-art-concepts-woven-in)**

</div>

<br>

> **Why this exists.** Most RAG tutorials show you how to build a pipeline. RAG Lab shows you *why* each decision matters — with the real HNSW graph being traversed, the cosine similarity heatmap you can hover, the reranking shifts laid out side-by-side, and a final radar chart scoring the whole pipeline. Built for learning the field deeply, then teaching it.

<br>

## ⚡ Quick start

```bash
# Backend
cd backend
venv/Scripts/uvicorn main:app --port 8000

# Frontend (new terminal)
cd frontend
npm run dev
```

Then open **http://localhost:3000** and work down the page — chunk → embed → index → retrieve → generate → evaluate.

> First run downloads embedding models (~90–400 MB depending on model). Subsequent runs use the cache.
> For the Generation stage, install [Ollama](https://ollama.com) and run `ollama pull llama3.2` — runs locally, no API key, no cost. [Full LLM setup ↓](#full-llm-setup)

<br>

## 🧬 Pipeline stages

| Stage | Status | What you can do |
|-------|--------|-----------------|
| **Chunking** | ✅ Done | Compare 5 strategies (recursive, fixed, paragraph, sentence, semantic), tune parameters, inspect per-chunk quality scores, watch topic-similarity between sentences on a live chart. Semantic chunking uses real MiniLM embeddings — threshold changes produce visibly different chunk boundaries. |
| **Embedding** | ✅ Done | Embed chunks with 4 local models (MiniLM, BGE-Small, MPNet, Nomic). Visualise the vector space with PCA / UMAP / PaCMAP, inspect the cosine similarity heatmap with chunking-issue detection, and explore raw vector values per chunk. |
| **Indexing** | ✅ Done | Build Flat / HNSW / IVF / MRL indexes. Visualise the real HNSW multilayer graph with layer-by-layer traversal, IVF cluster boundaries with `nprobe` selection, and the MRL dimension-truncation recall table. Orange/green colour coding distinguishes rebuild vs query-only parameters. |
| **Retrieval** | ✅ Done | Type a query, embed it, watch it find the nearest chunks. Compare dense (cosine), sparse (BM25), and hybrid (Reciprocal Rank Fusion) side-by-side. |
| **Reranking** | ✅ Done | Cross-encoder re-scores the top-k candidates with full query-chunk interaction. ColBERT late interaction shows token-level MaxSim heatmaps. Rank shift table compares all four strategies in one view. |
| **Generation** | ✅ Done | Pick an LLM (Ollama local, Groq free tier, OpenAI, Anthropic). Apply contextual compaction. Control chunk ordering (relevance vs sandwich). See the assembled prompt broken down by section, real token cost, and per-sentence grounding highlighting on the answer. |
| **Evaluation** | ✅ Done | Score the full pipeline end-to-end with [RAGAS](https://docs.ragas.io)-style metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall, Noise Sensitivity. Radar chart, sentence-level grounding breakdown, per-rank chunk relevance table, optional ground truth coverage check. |

<br>

## 🗺️ Roadmap

Where this is headed next. The goal is to teach the modern RAG field deeply by building it visually — each item below explains a SOTA technique you'll be able to *see* working, not just read about.

### 🎯 Next up

**1. HyDE — Hypothetical Document Embedding** — Instead of embedding the query, ask an LLM to generate a hypothetical *answer* to it first, then embed *that* as the search vector. Outperforms direct query embedding because question phrasing rarely matches document phrasing — but a hypothetical answer does. (Gao et al., 2022.)

**2. Contextual chunking (Anthropic, 2024)** — Prepend an LLM-generated context summary to each chunk before embedding ("This chunk is from a 2023 annual report describing Q3 revenue"). Anthropic's paper showed dramatic retrieval improvements. Will demonstrate the lift visibly through Stage 6 eval scores.

**3. LLM-as-judge evaluation toggle** — Switch Stage 6 from cosine-based metrics to a real LLM judging the answers (Claude / GPT-4 / local Llama via Ollama). Head-to-head comparison shows the gap between fast embedding metrics and high-fidelity LLM scoring — the same trade-off Microsoft Azure AI Eval and Anthropic's evaluation systems navigate in production.

### 🚀 The bigger features

**Pipeline comparison mode** — Run the same query through two different pipeline configurations side-by-side. See two radar charts diff'd. Instantly shows that pipeline decisions actually matter.

**GraphRAG (Microsoft Research, 2024)** — Build an interactive knowledge graph from your documents using LLM-extracted entities and relationships, then traverse it with global search (community summarisation) vs local search (subgraph traversal). The biggest 2024 advance in RAG, and no visual learning tool for it exists anywhere on the web.

**ColPali / multi-modal RAG (2024)** — Move RAG beyond text. ColPali uses late interaction on rendered document *images*, retrieving by visual layout instead of OCR'd text. Will enable RAG over real PDFs, tables, charts, and slide decks — closer to production reality.

**PDF / document upload — production realism** — Move beyond pasted text. Native upload for PDFs, web pages, and structured documents with proper parsing (tables, headings, layout). Lets you run the whole lab over your own corpus — turning RAG Lab from "learn RAG" into "learn RAG with my documents."

### 🧪 Polish + smaller features

- **"Why did it fail?" diagnostic** — When faithfulness drops below 0.5, trace the failure back through every pipeline stage to identify root cause
- **BGE asymmetric prefix demo** — Show `query:` vs `passage:` prefix effect on retrieval quality
- **Long-context vs RAG tradeoff** — With Gemini 2M and Claude 200k, show same query as full-doc-in-context vs RAG, compare cost + quality
- **Pipeline scorecard export** — Export the radar chart and metric summary as a shareable image, suitable for blog posts, presentations, or social sharing

### ⚠️ Described but not yet wired up

A few advanced techniques are referenced in tooltips and concept docs for educational context, but the implementations are not yet complete. These appear greyed out in the UI.

- **Stage 5 — Compaction:** LLMLingua, LLMLingua-2 (Microsoft), RECOMP (Google)
- **Stage 5 — Context strategies:** Map-Reduce, Refine, Map-Rerank (only Stuffing is currently active)
- **Stage 1 — Late chunking** (Jina, 2024) — described in `docs/chunking101.md` but not yet a selectable strategy

<br>

## 🧠 State-of-the-art concepts woven in

| Concept | Where to see it |
|---------|----------------|
| Contextual chunking (Anthropic, 2024) — prepend per-chunk summaries before embedding | Chunking — *roadmap* |
| Late chunking (Jina, 2024) — embed the full document first, then pool chunk token windows | Chunking / Embedding — *docs/chunking101.md* |
| Matryoshka Representation Learning (MRL) — truncate 768d vectors gracefully to 64d | Indexing — *active in MRL index tab* |
| HNSW (Malkov & Yashunin) — hierarchical navigable small-world graph | Indexing — *active, with traversal viz* |
| HyDE (Gao et al., 2022) — generate a hypothetical answer, embed *that* as the query | Retrieval — *roadmap* |
| Query decomposition — split complex questions into sub-queries | Retrieval — *roadmap* |
| Reciprocal Rank Fusion — merge dense + sparse retrieval | Retrieval — *active* |
| Cross-encoder reranking (ms-marco-MiniLM) | Reranking — *active* |
| ColBERT late interaction — token-level MaxSim scoring | Reranking — *active, with heatmap viz* |
| Lost-in-the-Middle (Liu et al., 2023) — recency / primacy bias in long contexts | Generation — *active, drives chunk-order options* |
| Contextual compaction — query-conditioned chunk filtering | Generation — *active* |
| RAGAS evaluation (Faithfulness, Relevancy, Precision, Recall) | Evaluation — *active* |
| LLM-as-judge (Microsoft G-EVAL, Anthropic model-graded eval) | Evaluation — *roadmap* |
| GraphRAG (Microsoft Research, 2024) — knowledge graph traversal | Retrieval / Generation — *roadmap* |
| ColPali (2024) — late interaction on document images | Retrieval — *roadmap* |

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

## 📦 Full LLM setup

The Generation stage supports four providers. Easiest by far is **Ollama** (local, no key, no cost).

**Ollama** *(recommended for first run)*
1. Install from [ollama.com](https://ollama.com)
2. Pull the Llama 3.2 model: `ollama pull llama3.2` *(~2 GB download)*
3. Ollama starts automatically — no separate server command needed

**Groq** *(free cloud tier, no credit card)*
1. Sign up at [console.groq.com](https://console.groq.com)
2. Set `GROQ_API_KEY` in `backend/.env`

**OpenAI / Anthropic**
Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `backend/.env`.

<br>

### Troubleshooting

If port 8000 is stuck:
```bash
powershell -Command "Get-Process python* | Stop-Process -Force"
```

The backend requires a hard restart after any `main.py` change (FastAPI's `--reload` is off by default for stability).
