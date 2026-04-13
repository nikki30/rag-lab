# Retrieval 101

Notes from building Stage 4 of RAG Lab. The goal of retrieval is to take a user query and find the chunks most likely to answer it — from the same vector space you built during embedding.

---

## The core idea

After indexing, you have N chunk vectors. A user types a query. You embed the query with the same model, then find the nearest chunk vectors.

"Nearest" sounds simple but depends heavily on the strategy. This stage compares four, each with different strengths.

---

## Dense retrieval

Embed the query → compare against every chunk vector → return top-k by cosine similarity.

**Dense** means the retrieval works from **dense vectors** — vectors where every dimension carries information. This is what embeddings are.

**Does dense mean cosine?** No. Dense describes the vector type, not the distance function. You could use cosine, dot product, or L2. For text embeddings, cosine and dot product are equivalent when vectors are unit-normalised (which sentence-transformers always does). Cosine is just the most common choice.

Strength: finds semantically similar chunks even when no words overlap ("what causes fever" → "pyrexia treatment").
Weakness: struggles with exact keywords, rare terms, proper names.

---

## Sparse retrieval — BM25

No embeddings. No neural network. Scores chunks purely by keyword frequency.

BM25 (Best Match 25) weights term frequency by:
1. How often the term appears in this chunk (TF)
2. How rare the term is across all chunks (IDF — rare terms score higher)
3. A length normalisation so long chunks don't dominate

Strength: exact keyword matching — names, codes, rare terms.
Weakness: misses synonyms and paraphrases entirely ("pyrexia" won't match "fever").

---

## Hybrid — Reciprocal Rank Fusion (RRF)

Merge the dense and sparse ranked lists into one.

### The formula

```
score(chunk) = 1/(60 + dense_rank) + 1/(60 + sparse_rank)
```

Chunks that rank high in both lists get the highest combined score.

### Why reciprocal?

There's an **inverse relationship** between rank and value — rank 1 should score more than rank 2, which should score more than rank 3. Reciprocal (`1/rank`) captures this naturally.

### Why 60?

The 60 flattens the curve so rank 1 vs rank 2 (or rank 49 vs rank 50) don't differ wildly. Without it, `1/rank` gives rank 1 → 1.0 and rank 2 → 0.5 — a cliff. With 60: rank 1 → 0.0164, rank 2 → 0.0161 — nearly equal. The constant controls how steeply the curve drops off. The original 2009 paper found 60 robust across many datasets; values between 10–100 all behave similarly.

**TL;DR: reciprocal = inverse relationship between rank and value. 60 = don't overweight the exact rank position, especially at the top.**

### Why not just average the ranks?

Two problems:

**Missing chunks.** Most chunks appear in only one list. You'd need to invent a penalty rank for the other list — that arbitrary number dominates the average. RRF treats "not in this list" as +0 contribution. No invented penalty.

**Equal spacing is wrong.** Averaging treats rank 1→2 the same as rank 49→50. But the top-1 vs top-2 distinction is far more meaningful. Reciprocal rank compresses the bottom naturally — `1/110` and `1/111` are barely different — while keeping the top distinct enough to matter.

### Why throw away the raw scores?

The actual cosine value (e.g. 0.87) and BM25 score (e.g. 12.3) live on completely different scales — you can't add or average them directly. RRF avoids the problem entirely by discarding values and only using rank positions.

---

## Cross-encoder re-ranking

Takes the top-20 candidates from hybrid and re-scores each one with a **cross-encoder**.

Standard dense retrieval encodes query and chunk **separately**, then computes cosine between two fixed vectors. The query and chunk never see each other during encoding — all the interaction happens in a single dot product at the end.

A cross-encoder takes the query and chunk **together** as one input: `[query] [SEP] [chunk]`. The model reads both simultaneously and produces a single relevance score. It can model the relationship between them directly — which words in the query are answered by which words in the chunk.

Much more accurate. The tradeoff: you can't pre-compute cross-encoder scores the way you pre-compute embeddings. Every query requires a fresh forward pass for each candidate. So you run it on a small candidate set (top-20) only, after cheaper methods have narrowed things down.

---

## ColBERT — late interaction

Standard dense retrieval: one vector per chunk. One vector per query. One dot product. Done.

ColBERT: one **token-level** vector per chunk token. One token-level vector per query token.

At score time, for each query token, find the chunk token most similar to it — this is **MaxSim**. Sum the MaxSim scores across all query tokens → the ColBERT score.

```
ColBERT score = Σ  max  sim(q_token_i, c_token_j)
               i    j
```

This is "late interaction" — query and chunk don't interact at encode time (chunk vectors are pre-computed), only at score time (MaxSim). More expressive than a single dot product because individual token matches contribute independently.

The heatmap in Stage 4 shows the full `(query_tokens × chunk_tokens)` similarity matrix. Gold cells are the MaxSim winners per query token — the ones driving the score.

---

## Side-by-side summary

| | Dense | Sparse (BM25) | Hybrid (RRF) | Re-ranked | ColBERT |
|---|---|---|---|---|---|
| Uses embeddings? | Yes | No | Both | Needs candidates first | Yes (token-level) |
| Semantic matching? | ✓ | ✗ | ✓ | ✓✓ | ✓✓ |
| Exact keyword matching? | ✗ | ✓ | ✓ | ✓✓ | ✓ |
| Pre-computable? | Yes | Yes (index) | Yes | No | Partial (chunk side) |
| Speed at query time | Fast | Fast | Fast | Slow (N forward passes) | Medium |
| Production role | Baseline | Complement | Standard | Final re-score on top-K | Research / late-stage |

---

## The standard production stack

```
Query
  → embed
  → dense top-20  ┐
  → BM25 top-20   ┤ → RRF merge → top-20 candidates → cross-encoder re-rank → top-5
```

Each stage progressively narrows and refines. The expensive steps (cross-encoder) only run on the small set the cheap steps already filtered.
