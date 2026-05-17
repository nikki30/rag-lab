# Embedding 101

Notes from building Stage 2 of RAG Lab. The goal of embedding is to convert each text chunk into a vector — a point in high-dimensional space — such that chunks with similar meaning end up near each other.

---

## The core idea

An embedding model reads a chunk of text and outputs a list of numbers: `[0.12, -0.87, 0.34, ...]`. This is the vector. Every chunk becomes a point in the same N-dimensional space.

The key property: **similar meaning = similar direction**. A chunk about "fever treatment" and a chunk about "pyrexia management" end up pointing roughly the same direction in 384-dimensional space, even though they share zero words. This is what makes semantic retrieval possible.

---

## What the dimensions mean

Nobody knows what individual dimensions mean. Dimension 47 is not "topic: medicine". The model learned a continuous distributed representation during training — meaning is spread across all dimensions simultaneously.

**Dimension counts are powers of 2** for GPU memory alignment: 384 = 3×128, 768 = 6×128. It's an engineering constraint, not a semantic one.

**Same dims ≠ same quality.** BGE-Small and MiniLM are both 384d but BGE-Small beats MiniLM for retrieval. BGE-Small was trained with contrastive learning specifically for information retrieval — pushed similar passages together, dissimilar ones apart. MiniLM used knowledge distillation for general semantic similarity, which is a different task.

---

## The four models in Stage 2

| Model | Dims | Strength |
|-------|------|----------|
| MiniLM | 384 | Fast, general-purpose, good baseline |
| BGE-Small | 384 | Better retrieval than MiniLM at same size |
| MPNet | 768 | High quality, trained on diverse tasks |
| Nomic | 768 | Supports MRL — can truncate to 64d gracefully |

---

## Cosine similarity

Cosine measures the **angle** between two vectors, not their distance. Two vectors pointing the same direction score 1.0 regardless of their lengths.

This is the right metric for text embeddings because:
- A short and a long chunk about the same topic should score 1.0 — they should
- Raw distance (Euclidean) would penalise the longer chunk for being further from the origin
- Cosine divides out magnitude, leaving only direction

**When vectors are unit-normalised** (|v| = 1), cosine similarity = dot product. `sentence-transformers` always normalises output vectors. So the full similarity matrix becomes one matrix multiply: `vectors @ vectors.T`. This is why normalisation matters — it turns an O(n²·d) loop into a single BLAS operation.

---

## Matryoshka Representation Learning (MRL)

Normal models: truncating a 768d vector to 64d destroys its structure — it was trained as a whole.

Nomic was trained with MRL. Each prefix of the vector (first 64d, first 128d, first 256d...) is itself a valid embedding. You can truncate from 768d to 64d and still get useful retrieval.

Why it matters in production:
- Store 64d vectors instead of 768d → 12× storage reduction
- Query with 64d first, re-score top candidates with 768d → speed/accuracy tradeoff
- Graceful degradation — you decide the storage budget at query time, not train time

---

## 2D layouts: PCA, UMAP, PaCMAP

Vectors are 384d or 768d — you can't plot them directly. These methods project to 2D for visualisation.

**PCA** — true linear projection. Finds the two directions of maximum variance and projects onto them. Fast, deterministic, mathematically exact. Downside: real data lives on curved manifolds; a linear projection misses that structure. Boundary chunks (semantically in-between) look isolated even when they're not.

**UMAP** — learned nonlinear layout. Optimises to preserve local neighbourhood structure. Great for showing tight clusters. Downside: aggressively snaps points into islands — boundary chunks that sit between topics get pushed to the edges or into a separate visual cluster that doesn't reflect their true position.

**PaCMAP** — learned layout that balances near, mid, and far pairs simultaneously. Handles boundary chunks better because it explicitly preserves mid-range relationships, not just nearest neighbours.

All three are visualisations only. The heatmap (computed in full 384d/768d space) is ground truth. "Projection" is only technically correct for PCA — UMAP and PaCMAP are **learned layouts**.

---

## The heatmap — ground truth

The N×N cosine similarity matrix computed in the original high-dimensional space. No projection distortion.

**Adjacent issues (amber ⚠):** `matrix[i][i+1] >= 0.75` — chunk i and chunk i+1 are very similar, which usually means a chunking boundary cut one thought into two. This is a chunking diagnosis, not an embedding diagnosis.

**Diagonal dominant (green ✓):** avg off-diagonal < 0.50 — chunks are distinct from each other. Healthy sign for retrieval — a query can find a specific chunk without being confused by similar neighbours.

---

## Curse of dimensionality

In very high dimensions, all points become roughly equidistant. Random vectors in 768d space have cosine similarities clustered tightly around 0. If your embedding model just produced random numbers, nothing would cluster — everything would score ~0 against everything else.

Embedding training explicitly fights this: contrastive loss pushes similar texts together and dissimilar texts apart during training. The result is a space where meaningful clusters exist despite the high dimensions.

---

## Side-by-side summary

| | PCA | UMAP | PaCMAP |
|---|---|---|---|
| Type | Linear projection | Nonlinear learned layout | Nonlinear learned layout |
| Preserves | Global variance directions | Local neighbourhood | Near + mid + far pairs |
| Boundary chunks | Isolated | Forced to island edges | Handled well |
| Speed | Fast | Medium | Medium |
| Deterministic? | Yes | No (random seed) | No (random seed) |

---

## Production note

Embedding is a one-time cost — you embed at index time, not at query time (except for the query itself). Storage is the constraint at scale, which is why MRL and quantisation (converting float32 → int8) matter. A 768d float32 vector = 3KB. One million chunks = 3GB. MRL at 128d = 500MB.
