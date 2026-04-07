# RAG Lab — Project Context

## What this is
A full-stack interactive RAG pipeline explorer. The goal is an end-to-end visual tool where you can pick chunking algorithms, embedding models, and retrieval strategies (HNSW, DiskANN, GraphRAG) and see the effect of each decision visually. Currently at the chunking stage.

## Stack
- **Backend:** FastAPI, Python, LangChain text splitters, pure-Python TF-IDF (no sklearn needed yet)
- **Frontend:** Next.js 16.2.2, React 19.2.4, Tailwind CSS v4, TypeScript, App Router
- **Important:** App dir is `frontend/app/` — there is NO `src/` folder

## Running locally
```bash
# Backend — must hard-restart after any main.py change (--reload is unreliable)
cd backend
venv/Scripts/uvicorn main:app --port 8000

# Frontend
cd frontend
npm run dev   # → http://localhost:3000
```

If port 8000 is stuck (stale process), kill all Python processes:
```bash
powershell -Command "Get-Process python* | Stop-Process -Force"
```

---

## Backend — `backend/main.py`

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/api/chunk` | Chunk text with one strategy |
| POST | `/api/compare` | Run all 5 strategies, return stats + preview chunks |

### `/api/chunk` request
```json
{
  "text": "string",
  "chunk_size": 200,
  "chunk_overlap": 50,
  "strategy": "recursive | fixed | paragraph | sentence | semantic",
  "breakpoint_threshold": 0.4
}
```

### `/api/chunk` response
```json
{
  "chunks": [
    {
      "page_content": "string",
      "metadata": { "start_index": 0 },
      "scores": {
        "size_score": 0.87,
        "boundary_score": 1.0,
        "quality": 0.93
      }
    }
  ],
  "total": 7,
  "stats": {
    "avg_size": 192,
    "min_size": 45,
    "max_size": 200,
    "std_dev": 38,
    "avg_quality": 0.81
  },
  "similarity_scores": [0.41, 0.12, 0.67, ...]
}
```
`similarity_scores` is only populated for the `semantic` strategy. It has one value per sentence boundary (len = sentences - 1).

### `/api/compare` response
```json
{
  "recursive": {
    "total": 7, "avg_size": 192, "min_size": 45, "max_size": 200,
    "std_dev": 18, "avg_quality": 0.81, "sizes": [...], "preview_chunks": ["chunk1 text", "chunk2 text"]
  },
  "fixed": { ... },
  ...
}
```

### 5 chunking strategies
| Strategy | How it works | Key characteristic |
|----------|-------------|-------------------|
| `recursive` | Tries `\n\n → \n → space → char` in order, only falls back when chunk still too big | Industry default, respects natural boundaries |
| `fixed` | Hard cuts every N chars, `separator=""` | Cuts mid-word, shows why naive chunking fails |
| `paragraph` | Splits on `\n\n`, merges short paragraphs up to chunk_size | Content-aware but wildly uneven sizes |
| `sentence` | Regex sentence split, groups sentences up to chunk_size | Clean boundaries, variable sizes |
| `semantic` | TF-IDF cosine similarity between adjacent sentences, cuts where similarity < threshold | Only strategy that sees topic structure |

### Per-chunk quality scores
- `size_score`: `min(len, chunk_size) / max(len, chunk_size)` — 1.0 = perfect size fit
- `boundary_score`: 0.5 if starts with capital + 0.5 if ends with `.!?`
- `quality`: average of the two

### Validation
- `chunk_overlap` must be < `chunk_size` (enforced with `@model_validator`, semantic strategy exempt)

### Dependencies installed
```
fastapi, uvicorn, langchain-text-splitters, pydantic
scikit-learn, numpy  ← installed, not yet used (reserved for embedding stage)
```

---

## Frontend — `frontend/app/page.tsx`

Single `'use client'` page (~500 lines). All UI in one file.

### State
```ts
text            // source document
chunkSize       // default 200
chunkOverlap    // default 50
strategy        // 'recursive' | 'fixed' | 'paragraph' | 'sentence' | 'semantic'
breakpointThreshold  // 0.4, only used by semantic
chunks          // Chunk[] from last /api/chunk call
stats           // Stats from last call
similarityScores // number[] | null, only for semantic
compareData     // CompareResponse | null
view            // 'highlight' | 'cards' | 'compare'
```

### UI layout
1. **Strategy cards** — 5 buttons, clicking one sets strategy AND immediately re-chunks (calls `handleChunk(strategyId)` to avoid stale state)
2. **Source textarea** + **sliders** — chunk_size/overlap for non-semantic; breakpoint_threshold for semantic
3. **Chunk It!** + **Compare All Strategies** buttons
4. **Semantic similarity chart** — appears only for semantic strategy:
   - Bars normalized to actual min/max score range (not fixed 0–1), so differences are visible
   - Dashed threshold line moves live as slider is dragged (threshold change does NOT clear scores, only chunks)
   - Blue = above threshold (same chunk), orange ✂ = below threshold (cut)
   - Hover each bar for exact score
5. **Stats bar** — chunk count, avg size, σ, min–max, quality %
6. **View toggle** — Highlight / Cards / Compare

### Views
- **Highlight:** original text color-coded by chunk (10 colors cycling), hover a chunk card to dim all others
- **Cards:** full chunk text + quality badge with sz/bd scores
- **Compare:** stats table with σ and quality winners highlighted green, size distribution histograms (with min/max labels), first-chunk preview grid for each strategy

### Key components
| Component | Purpose |
|-----------|---------|
| `InfoTooltip` | `position: fixed` on hover — escapes overflow clipping from scroll containers |
| `QualityBadge` | Color-coded quality %, has built-in InfoTooltip explaining sz/bd |
| `StatPill` | Stat label + InfoTooltip in one |
| `CompareHeader` | Table `<th>` + InfoTooltip |
| `SizeDistribution` | Mini histogram with min/max char labels |
| `SliderControl` | Slider + number input + optional tooltip prop |

### Important bugs already fixed
- `handleChunk(overrideStrategy?)` takes an override param — clicking a strategy card passes the new strategy directly because React state hasn't updated yet at click time
- Threshold slider calls `resetChunksOnly()` not `resetResults()` — keeps similarity scores alive so chart recolours live
- `InfoTooltip` uses `position: fixed` + `onMouseEnter` coordinates — avoids being clipped by `overflow-y-auto` scroll containers

---

## What's been learned / taught so far
- Why chunking strategy is the most impactful RAG parameter
- Recursive: separator hierarchy (paragraph → line → word → char)
- Fixed: why naive character chunking fails (mid-word cuts)
- Paragraph: content-aware but uneven sizes
- Sentence: clean boundaries, variable size tradeoff
- Semantic: TF-IDF cosine similarity to find topic transitions
- σ (std dev): why consistent chunk sizes matter for embedding quality
- Quality score: size efficiency + boundary cleanliness
- Compare view: how to read σ, quality, distribution histogram side-by-side
- Similarity chart: bars = sentence-to-sentence topical relatedness, threshold line = cut sensitivity

---

## Next stage — Embedding
Take the chunks and convert them to dense vectors.

**Plan:**
- Add `POST /api/embed` endpoint
- Use `sentence-transformers` (local, free) — model `all-MiniLM-L6-v2`
  - Also upgrades the semantic chunker from TF-IDF to real embeddings
- Frontend: show chunks as 2D scatter plot (PCA or UMAP dimensionality reduction)
- Allow picking different embedding models and seeing how the vector space changes
- sklearn + numpy are already installed and ready
