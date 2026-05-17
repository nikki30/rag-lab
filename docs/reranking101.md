# Reranking 101

Notes from building Stage 5 of RAG Lab. Reranking is the step between retrieval and generation — take the top-k candidates the retriever found, and re-score them with a better but slower model before passing them to the LLM.

---

## Why reranking exists

Retrieval is a recall problem: find all the probably-relevant chunks fast. The retriever scores with a single dot product between two pre-computed vectors — cheap, but the query and chunk never see each other during encoding.

Reranking is a precision problem: from the candidates, find the truly relevant ones. A reranker takes query and chunk together and models their interaction directly — much more accurate, but too slow to run on your full index.

The classic production stack:
```
Query
  → embed
  → dense top-50  ┐
  → BM25 top-50   ┤ → RRF merge → top-20 → reranker → top-5 → LLM
```

Retrieval provides breadth. Reranking provides accuracy. Never skip retrieval to run the reranker directly — it scales O(N) forward passes, not O(log N).

---

## Cross-encoder

Standard retrieval encodes query and chunk **separately**, then computes cosine at the end. The two vectors are fixed before they ever "see" each other.

A cross-encoder takes query and chunk **together**:

```
input = "[query] [SEP] [chunk]"
output = single relevance score (0–1)
```

The model reads both simultaneously. It can attend from query tokens to chunk tokens and back — which words in the chunk answer which words in the query. This is far more expressive than a dot product.

**The cost:** you can't pre-compute cross-encoder scores. Every query requires a fresh forward pass for each candidate. With 20 candidates, that's 20 forward passes per query. Fine for a re-rank step on a small candidate set; not fine as a first-stage retriever over millions of chunks.

### The model

The standard open-source cross-encoder for English passage ranking is `cross-encoder/ms-marco-MiniLM-L-6-v2` — fine-tuned on the MS MARCO passage ranking dataset (530k query/passage pairs from Bing search logs). Downloads ~80MB on first run and is fast enough to run on CPU.

Alternatives: `cross-encoder/ms-marco-TinyBERT-L-2-v2` (faster, slightly weaker), `BAAI/bge-reranker-v2-m3` (multilingual, stronger), Cohere Rerank API (cloud, strongest off-the-shelf).

---

## ColBERT — late interaction

Cross-encoder: one score per (query, chunk) pair. ColBERT: one vector per **token** in the query and chunk.

```
chunk tokens:  [c1, c2, c3, ..., cn]  → n vectors
query tokens:  [q1, q2, q3, ..., qm]  → m vectors
```

At score time, for each query token, find the single most-similar chunk token (**MaxSim**). Sum the MaxSim scores:

```
ColBERT score = Σᵢ  maxⱼ  sim(qᵢ, cⱼ)
```

This is **late interaction** — chunk vectors are pre-computed (like standard dense retrieval), but the scoring step is richer than a single dot product. Each query token contributes independently based on which chunk token best matches it.

### Why the heatmap matters

The Lab shows a `(query_tokens × chunk_tokens)` similarity matrix. Gold cells are the MaxSim winners — the tokens actually driving the score. This reveals what the model "matched":

- A query token that lights up on a meaningful chunk word → strong signal match
- A query token whose best match is a stopword (`the`, `of`) → weak query token that adds noise
- Sparse gold cells across the heatmap → low ColBERT score, low relevance

### ColBERT in production

Pre-computing token-level vectors for every chunk requires ~10× more storage than standard dense retrieval (one vector per token vs one per chunk). Stanford's ColBERT v2 uses aggressive compression (residual quantisation) to bring this down.

In practice, ColBERT is used as a late-stage reranker, not a first-stage retriever, for exactly this storage reason. RAGatouille is the easiest Python library for wrapping ColBERT in a RAG pipeline.

---

## Rank shift analysis

The value of reranking is only visible in the rank changes. A chunk that the retriever found at position 8 may jump to position 1 after the cross-encoder re-scores it — because the retriever found it semantically nearby but the cross-encoder found it specifically answers the query.

The Rank Shift table in Stage 5 shows:

- **↑ green**: cross-encoder promoted this chunk (the retriever undersold it)
- **↓ red**: cross-encoder demoted this chunk (the retriever oversold it)
- **= grey**: no rank change

Large up-shifts on relevant chunks = your retriever is finding the right content but not ranking it well — the cross-encoder is doing significant work. If there are no rank shifts at all, either your retriever is already perfect or your cross-encoder and retriever are scoring the same way.

---

## Mono vs duo architectures

**MonoT5 (Google, 2020):** Uses T5 as a cross-encoder. Takes `"Query: ... Document: ..."` as input, generates "true" or "false" as output, uses the token logit ratio as the relevance score. Effectively a generation-based reranker. Strong but slow.

**DuoT5:** Pairwise — given two documents, outputs which one is more relevant to the query. More accurate than pointwise scoring (MonoT5) because it directly compares candidates. Much slower: O(n²) pairs instead of O(n) pointwise scores.

**Production reality:** MonoT5/DuoT5 are research benchmarks. In practice, MiniLM-based cross-encoders dominate because of the speed/accuracy tradeoff. DuoT5 is overkill unless you're reranking ≤10 candidates in a latency-tolerant pipeline.

---

## SOTA and production notes

**Cohere Rerank:** The most-used cloud reranker. Supports multilingual, handles 512-token chunks, returns a relevance score per document. Widely used in enterprise RAG because it's a single API call with no model hosting. Not SOTA but reliable and easy to swap in.

**BGE-Reranker-v2 (BAAI, 2024):** Strong open-source cross-encoder family. The `-m3` variant handles multilingual inputs. Outperforms MiniLM cross-encoders on BEIR benchmarks while staying fast.

**RankLLaMA / RankZephyr (2023):** Fine-tune a large language model (LLaMA, Zephyr) on ranking tasks using listwise training — the model takes a list of candidates and orders them directly, not one at a time. Listwise training avoids the position bias that pointwise rerankers inherit from pairwise training data. Still research-stage for most deployments.

**Microsoft Azure AI Search:** Semantic Ranking (powered by Bing) is a cross-encoder reranker built into Azure AI Search. Enabled with `queryType=semantic`. Scores documents using a Bing-trained model. The default ADA embedding retrieval + Semantic Ranking combination is Microsoft's recommended production stack for Azure RAG.

**The Lost-in-the-Middle problem:** Even after reranking, chunk order sent to the LLM matters. Research (Liu et al., 2023) showed LLMs perform worse when the most relevant context is in the middle of a long prompt — they attend more strongly to the start and end. After reranking, place the highest-scoring chunks at positions 1 and N (first and last), not in the middle.

---

## Side-by-side comparison

| | Dense | Cross-encoder | ColBERT |
|---|---|---|---|
| Encoding | Separate query + chunk | Joint query + chunk | Separate, token-level |
| Pre-computable? | Yes (chunk side) | No | Yes (chunk side) |
| Score expressiveness | Single dot product | Full attention across both | MaxSim per query token |
| Speed at query time | Fast | Slow (N forward passes) | Medium |
| Storage overhead | 1× | 0 (no pre-compute) | ~10× |
| Production role | First-stage retrieval | Final re-score on top-K | Late-stage re-score |
