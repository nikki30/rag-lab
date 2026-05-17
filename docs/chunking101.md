# Chunking 101

Notes from building Stage 1 of RAG Lab. The goal of chunking is to split a document into pieces small enough to embed individually — while keeping each piece coherent enough to be useful when retrieved.

---

## Why chunking is the most impactful RAG decision

A fact split across a chunk boundary is unrecoverable. The retrieval stage can only return chunks that exist — it cannot stitch together a sentence that was cut in half. Every downstream quality problem (bad embeddings, missed retrieval, hallucinated answers) is easier to fix than a bad split.

This is the stage where "garbage in, garbage out" is most literal.

---

## The five strategies

### Fixed — naive baseline

Split every N characters. No separator logic. Cuts mid-word, mid-sentence, mid-thought.

Exists to show why naive chunking fails. High σ (standard deviation) because some chunks land at sentence endings by luck and others cut mid-word. Not used in production.

### Recursive — industry default

LangChain's `RecursiveCharacterTextSplitter`. Tries a cascade of separators in order: `\n\n → \n → space → char`. Uses the first one that produces chunks under the size limit.

Falls back down the chain gracefully — a paragraph separator first, then line break, then space, then character. The result: chunks that respect natural language structure without requiring it.

Default choice for most RAG systems. Predictable, robust, no ML required.

### Paragraph — content-aware

Splits on `\n\n`. Merges short paragraphs until the chunk_size budget is reached.

Produces semantically complete units — each chunk is a full thought. High σ because paragraph lengths vary widely (a two-sentence paragraph vs. a ten-sentence one). The chunk size parameter becomes a soft target, not a hard limit.

### Sentence — clean boundaries

Splits on `.!?` using regex, then groups sentences until chunk_size is hit.

Every chunk starts and ends at a sentence boundary — no cut-off mid-thought. More uniform sizes than paragraph chunking. Slight weakness: misses paragraph context (a sentence in paragraph 3 has no idea what paragraph 1 established).

### Semantic — the only strategy that reads meaning

Uses real MiniLM embeddings. Embeds adjacent sentences and computes cosine similarity between them. When similarity drops below a threshold, it treats that as a topic boundary and splits there.

The only strategy that actually measures whether two adjacent sentences belong together. The threshold slider controls sensitivity — higher threshold = more aggressive splitting = smaller, more focused chunks.

Downside: requires an embedding model at chunking time, so it's slower than the others.

---

## Quality scores

Each chunk gets three scores:

**size_score** — how close the chunk is to the target chunk_size:
```
min(len, chunk_size) / max(len, chunk_size)
```
A chunk that exactly hits chunk_size scores 1.0. A 50-char chunk with a 500 target scores 0.1.

**boundary_score** — whether the chunk starts and ends cleanly:
- +0.5 if it starts with a capital letter
- +0.5 if it ends with `.`, `!`, or `?`

**quality** — average of both.

These scores are heuristics, not ground truth. A 0.3 quality score is a flag to investigate, not a guarantee of bad retrieval.

---

## Standard deviation (σ) — why it matters

High σ = chunks of wildly varying lengths → wildly varying embedding densities → unreliable retrieval.

A 50-token chunk and a 500-token chunk about the same topic will have different vector norms even after normalisation, because one chunk has more specific detail packed in. Nearest-neighbour search treats them the same — but retrieval quality is worse when sizes are inconsistent.

Rule of thumb: lower σ = more predictable retrieval behaviour.

---

## Hub chunks — retrieval drift

A hub chunk is one that is semantically similar to many other chunks across different topics. When retrieved, it "drifts" — it scores high for queries it can't actually answer, crowding out more relevant chunks.

Hub chunks usually arise from:
- Poor boundaries (a chunk spanning two topics)
- High-frequency boilerplate ("In this section, we discuss...")
- Very short chunks that embed as generic

Fixes: MMR (Maximal Marginal Relevance) to penalise redundancy, cross-encoder reranking, or parent retrieval.

---

## Masquerading

Adjacent red cells in the embedding heatmap (high cosine similarity between neighbouring chunks) often look like an embedding problem — "why are these two chunks so similar?" — but the real cause is a chunking boundary that split one thought into two pieces.

Both pieces ended up with nearly identical vectors because they're actually the same sentence. The heatmap pattern is a diagnostic for chunking quality, not just embedding quality.

---

## Side-by-side summary

| | Fixed | Recursive | Paragraph | Sentence | Semantic |
|---|---|---|---|---|---|
| Splits on | Every N chars | `\n\n → \n → space → char` | `\n\n`, merges short | `.!?` regex | Cosine similarity drop |
| ML required? | No | No | No | No | Yes (MiniLM) |
| Respects sentences? | No | Usually | Yes | Yes | Yes |
| Size consistency (σ) | High | Medium | High | Medium | Variable |
| Production use? | No | Default | Sometimes | Sometimes | When quality matters |

---

## The SOA moves beyond simple splitting

**Contextual chunking (Anthropic):** prepend a per-chunk summary before embedding so the chunk vector carries document context, not just local text. Fixes the problem where a chunk says "it was first described in 1847" with no referent.

**Late chunking:** embed the full document first (preserving cross-chunk context in the token stream), then pool the token embeddings into chunk-sized windows. Chunks get context-aware vectors without an LLM call per chunk.
