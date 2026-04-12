# Indexing 101

Notes from building Stage 3 of RAG Lab. The goal of indexing is to organise your embedded vectors so that at query time you can find the nearest ones without comparing against every single vector.

---

## The core idea

After embedding, each chunk is a vector (e.g. 384d or 768d). Finding the most similar chunk to a query = finding the nearest vector. The naive way: compute cosine similarity between the query and every vector. That's **flat search** — correct, but slow at scale.

**Indexing** builds a data structure over those vectors so you can skip most comparisons and still find the right answer most of the time.

---

## FAISS

**Facebook AI Similarity Search** (Meta Research). A library that builds and searches vector indexes. It provides the data structures (HNSW graph, IVF inverted lists) that make approximate nearest-neighbour search fast.

Without FAISS you are always doing flat search. FAISS gives you the non-flat options.

Key point: FAISS handles both **building** the index (the expensive one-time setup) and **searching** it (the fast per-query step).

---

## The three index types

### Flat — brute force

- No data structure. No skipping. Pure cosine similarity against every vector.
- Doesn't need FAISS — we just use numpy dot products.
- **100% recall by definition** — this is the ground truth all other indexes are measured against.
- Cost: O(n × d) per query. Fine for hundreds of chunks; unusable at millions.

### HNSW — Hierarchical Navigable Small World (FAISS `IndexHNSWFlat`)

HNSW is a **multilayer graph**. Each chunk is a node. Edges connect nearby chunks. The graph is what lets you skip cosine comparisons — instead of comparing the query to every chunk, you navigate the graph to the right neighbourhood.

The similarity metric is still **cosine throughout**. HNSW just determines which chunks you ever bother comparing against.

**Layers:**
- **Layer 0 (precise):** all chunks, each connected to 2×M neighbours. Final precision lives here.
- **Higher layers (highway):** a random ~1/M subset of chunks with long-range connections. Used for fast coarse navigation — you skip large distances quickly before zooming in.

Search starts at the top layer, descends greedily (each step moves to the most similar connected neighbour), and the winner of each layer becomes the entry point for the layer below.

**Build parameters (require index rebuild):**
- `M` — edges per chunk. More edges → richer graph → better recall, more memory.
- `ef_construction` — beam width during build. Controls how carefully HNSW picks each node's M neighbours. Higher → better graph quality, slower build. Zero effect on query speed.

**Query parameter (just re-run search):**
- `ef_search` — beam width at query time. How many candidates layer 0 considers before returning top-k. Higher → better recall, slower query.

**Entry point:** the last chunk inserted — effectively arbitrary. In a well-connected graph this barely matters; the highway layers navigate away from it fast.

### IVF — Inverted File Index (FAISS `IndexIVFFlat`)

IVF is a set of **clusters**. At build time, k-means partitions your vectors into groups. At query time, only the nearest few clusters are searched — everything else is skipped.

The similarity metric is still **cosine throughout** — used twice:
1. Cosine between query and centroids → pick the nprobe nearest clusters.
2. Cosine between query and every vector inside those clusters → return top-k.

**Centroids are not data points.** They are computed averages — the mathematical mean of all vectors assigned to a cluster. They float in space between the actual chunks; they don't correspond to any real document.

**How k-means++ finds centroids:**
1. Pick one chunk at random as the first centroid seed.
2. Each subsequent seed is chosen with probability proportional to its distance from the nearest existing centroid (so seeds spread out).
3. Assign every vector to its nearest centroid.
4. Move each centroid to the mean of its members.
5. Repeat steps 3–4 until stable.

The final centroids (× markers on the scatter plot) are the result of this iterative process.

**Build parameter (requires index rebuild):**
- `n_clusters` — how many k-means groups to create. Rule of thumb: √n for n vectors. More clusters → finer partitions → better recall at low nprobe.

**Query parameter (just re-run search):**
- `nprobe` — how many clusters to search per query. nprobe=1 → fastest, may miss true neighbours. nprobe=n_clusters → 100% recall, same as flat.

**Why a visually distant cluster can still be searched:** IVF cluster selection happens in the original high-dimensional space (384d / 768d). The 2D scatter plot is a PCA/UMAP projection that distorts distances. Two points that look far apart in 2D may be genuinely close in 768d space.

---

## Side-by-side summary

| | Flat | HNSW | IVF |
|---|---|---|---|
| Data structure | None | Multilayer graph | K-means clusters |
| FAISS used? | No (numpy) | Yes (`IndexHNSWFlat`) | Yes (`IndexIVFFlat`) in production; sklearn in this demo |
| Skips comparisons? | No | Yes — graph navigation limits which chunks you visit | Yes — only enters nprobe clusters |
| Similarity metric | Cosine | Cosine | Cosine (×2: centroid selection + within-cluster search) |
| Recall | 100% always | Near-100% (tunable via ef_search) | Depends on nprobe |
| Build params | — | M, ef_construction | n_clusters |
| Query params | — | ef_search | nprobe |

---

## Build vs query parameters

**Build parameters** shape the index data structure. Changing them requires rebuilding the index from scratch.

**Query parameters** only affect how a single search runs. Change them and re-run the query — no rebuild needed.

| Parameter | Type | Affects |
|---|---|---|
| M | Build | HNSW graph density |
| ef_construction | Build | HNSW graph quality |
| n_clusters | Build | IVF partition granularity |
| ef_search | Query | HNSW recall vs speed |
| nprobe | Query | IVF recall vs speed |
