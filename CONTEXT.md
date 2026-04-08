# RAG Lab — Project Context

## What this is
Full-stack interactive RAG pipeline explorer. Pick chunking strategies, embedding models, retrieval algorithms — see the effect visually. Stages: Chunking → Embedding → Retrieval (HNSW) → Reranking.

## Stack
- **Backend:** FastAPI, Python, LangChain text splitters, sentence-transformers, scikit-learn, numpy
- **Frontend:** Next.js 16.2.2, React 19.2.4, Tailwind CSS v4, TypeScript, App Router
- **App dir:** `frontend/app/` — no `src/` folder

## Running locally
```bash
# Backend — hard-restart required after any main.py change
cd backend && venv/Scripts/uvicorn main:app --port 8000

# Frontend
cd frontend && npm run dev   # → http://localhost:3000

# Kill stuck backend
powershell -Command "Get-Process python* | Stop-Process -Force"
```

---

## Backend — `backend/main.py`

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/api/chunk` | Chunk text with one strategy |
| POST | `/api/compare` | Run all 5 strategies, return stats + preview |
| POST | `/api/embed` | Embed chunks, return vectors + 2D coords + similarity matrix |

### `/api/chunk`
Request: `{ text, chunk_size, chunk_overlap, strategy, breakpoint_threshold }`
Response: `{ chunks[{page_content, metadata.start_index, scores{size_score, boundary_score, quality}}], total, stats{avg_size, min_size, max_size, std_dev, avg_quality}, similarity_scores }`
`similarity_scores` only populated for semantic strategy (one per sentence boundary).

### `/api/embed`
Request: `{ chunks: string[], model: "minilm|bge-small|mpnet|nomic", reduction: "pca|umap|pacmap" }`
Response: `{ model, dimensions, coords_2d: [[x,y],...], similarity_matrix: [[...]], vectors: [[...]] }`
- Models cached after first load. First run downloads model (~90–400MB).
- Nomic requires `trust_remote_code=True`, `einops` package, and `search_document:` prefix on chunks.
- PaCMAP requires float32 cast before fitting.
- Vectors normalised → similarity matrix = `vectors @ vectors.T` (cosine = dot product when |v|=1).

### Chunking strategies
| Strategy | Splits on | Key trait |
|----------|-----------|-----------|
| `recursive` | `\n\n → \n → space → char` (fallback chain) | Industry default |
| `fixed` | Every N chars, no separator | Cuts mid-word — shows why naive chunking fails |
| `paragraph` | `\n\n`, merges short paragraphs | Content-aware, high σ |
| `sentence` | Regex `.!?`, groups to chunk_size | Clean boundaries |
| `semantic` | TF-IDF cosine < threshold between adjacent sentences | Only strategy that reads meaning |

### Quality scores
- `size_score`: `min(len, chunk_size) / max(len, chunk_size)`
- `boundary_score`: +0.5 starts capital, +0.5 ends `.!?`
- `quality`: average of both
- `chunk_overlap` must be < `chunk_size` (semantic exempt)

### Dependencies installed
```
fastapi, uvicorn, langchain-text-splitters, pydantic
scikit-learn, numpy, sentence-transformers==5.3.0
umap-learn==0.5.11, pacmap==0.9.1, einops==0.8.2
```

---

## Frontend — `frontend/app/page.tsx`

Single `'use client'` page (~1000 lines). All UI in one file.

### State
```ts
// Stage 1 — Chunking
text, chunkSize, chunkOverlap, strategy, breakpointThreshold
chunks, stats, similarityScores, compareData
loading, compareLoading, error, activeChunk, view

// Stage 2 — Embedding
embedModel, reduction, embedResult, embedLoading, embedError, embedStale
hoveredEmbedChunk, selectedEmbedChunk, embedSectionRef
```

### Key behaviours
- Strategy card click calls `handleChunk(strategyId)` directly — avoids stale React state
- Threshold slider calls `resetChunksOnly()` — keeps similarity scores alive for live chart recolouring
- `embedStale` flag: set when chunks change after embedding — shows amber "Re-embed" CTA
- After embed: auto-scrolls to stage 2 via `embedSectionRef`
- Model card click re-embeds immediately if results already exist

### Key components
| Component | Purpose |
|-----------|---------|
| `InfoTooltip` | `position:fixed` on hover — escapes overflow clipping |
| `QualityBadge` | Color-coded quality %, built-in tooltip |
| `StatPill` | Stat + tooltip pill |
| `SliderControl` | Slider + number input + optional tooltip |
| `EmbedScatterPlot` | SVG scatter, click to select, hover to preview |
| `EmbedHeatmap` | N×N cosine grid, amber ⚠ adjacent-issue detection, green ✓ diagonal-dominant |
| `VectorInspector` | Sparkline of first 64 raw vector values |

### Heatmap pattern detection
- **Adjacent issue** (amber ⚠): `matrix[i][i+1] >= 0.75` — likely chunking boundary mid-thought
- **Diagonal dominant** (green ✓): avg off-diagonal < 0.50 — healthy distinct chunks

### Important bugs already fixed
- Strategy card stale state: override param passed directly to `handleChunk`
- Threshold slider: calls `resetChunksOnly()` not `resetResults()` — preserves scores
- `InfoTooltip`: `position:fixed` + `onMouseEnter` coordinates — avoids scroll container clipping

---

## What's been learned

### Chunking
- Chunking is the most impactful RAG decision — a fact split across a boundary is unrecoverable
- Recursive is the default, semantic is the most intelligent, fixed exists to show what not to do
- σ (std dev): high σ = unpredictable embedding density = unreliable retrieval
- Hub chunks: one chunk similar to many others across topics → retrieval drift
- Fixes for drift: MMR, cross-encoder reranking, parent retrieval
- Masquerading: adjacent red cells in heatmap often = chunking problem, not embedding problem

### Embedding
- Vectors encode meaning as direction in N-dimensional space. Length is irrelevant.
- Dimension counts are powers of 2 for GPU memory alignment (384=3×128, 768=6×128)
- Same dims ≠ same quality: BGE-Small beats MiniLM at 384d because it was trained with contrastive learning specifically for retrieval; MiniLM used knowledge distillation for general similarity
- Nomic supports Matryoshka (MRL) — vector can be truncated from 768d to 64d gracefully
- Cosine similarity: measures angle, not distance. Divides out magnitude — a short and long chunk about the same topic both score 1.0. This is why cosine was chosen over Euclidean or raw dot product.
- Normalisation: set |v|=1 for all vectors → cosine becomes dot product → full similarity matrix = one matrix multiply (`vectors @ vectors.T`)
- Curse of dimensionality: in high dims all points equidistant. Embedding training explicitly forces similar text to cluster.
- PCA = true linear projection, fast, misses curved structure. UMAP = learned layout, great clusters, forces boundary chunks into one island. PaCMAP = learned layout balancing near/mid/far pairs — best for boundary chunks.
- "Projection" is only technically correct for PCA. UMAP and PaCMAP are learned layouts — calling them projections is imprecise. UI labels them "2D Layout".
- Heatmap is ground truth (384D, no projection). Scatter plots are shadows for pattern recognition.
- Production retrieval: cosine = what similarity is. HNSW = how to find similar chunks fast without checking everything. Bi-encoder fast first pass → cross-encoder reranking for accuracy. BM25 hybrid beats dense alone.

---

## Next stage — Retrieval (Stage 3)

### Plan
- Query box: user types a question → embedded in same space → nearest chunks highlighted on scatter + heatmap
- HNSW index: replace brute-force cosine with approximate nearest neighbour graph — show speed vs accuracy tradeoff
- Reranking: cross-encoder re-scores top-k candidates with query for true relevance

### Deferred teaching moments (bring up at right time)
- **Model mismatch demo:** index with MiniLM, query with BGE-Small → show broken retrieval. User explicitly wants to recreate this.
- **BGE asymmetric prefixes:** `query:` vs `passage:` — remind when building query embedding in retrieval stage
- **Frozen model / domain fine-tuning:** embedding models don't know your domain vocabulary — raise when retrieval fails on specialist text
