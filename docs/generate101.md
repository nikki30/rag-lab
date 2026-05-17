# Generate 101

Notes from building Stage 5 of RAG Lab. The goal of this stage is to take the re-ranked chunks from retrieval and turn them into an answer — but before you send anything to the LLM, there are three independent decisions you make that significantly affect quality, cost, and latency.

The three decisions: **how to compress the chunks** (compaction), **in what order to present them** (chunk ordering), and **how to split the work across LLM calls** (context strategy).

---

## The token budget problem

Every LLM has a context window — a hard limit on how many tokens it can process at once. Tokens = money. A RAG prompt with 5 chunks easily hits 1000–2000 input tokens. At scale:

- 1000 queries/day × 1500 tokens × $0.15/1M = $0.22/day on GPT-4o-mini
- Same on GPT-4o = $3.75/day
- Same on Claude Sonnet = $4.50/day

Context window budget = system prompt tokens + chunk tokens + query tokens + output reserve.

The chunk tokens dominate. Retrieved chunks are noisy — a 500-token chunk might contain 400 tokens of setup, transitions, and padding around the 100 tokens you actually need. Compaction is the engineering lever that reduces this.

---

## Part 1: Compaction — reducing tokens before sending

### The query-independence spectrum

Compaction methods sit on a spectrum of how much they use the query when deciding what to cut:

```
Query-blind                                               Query-conditioned
    |                                                             |
   Raw → LLMLingua (2023) → Contextual → LLMLingua-2 (2024) → RECOMP
```

Earlier = cheaper, faster, less accurate compression.
Later = more expensive, more accurate compression.

### Raw — no compaction

The baseline. Every retrieved token is sent verbatim. Transparent but wastes the context budget on noise. Use this to establish a quality baseline, then compare against compacted variants.

### Contextual — sentence-level cosine filtering

Split each chunk into sentences. Embed each sentence. Compute cosine similarity between each sentence and the query. Drop sentences below a threshold (0.2 in this lab).

**Query-dependent**: "important" is measured relative to what you asked. The same chunk produces different output for different queries.

Preserves complete sentences — no partial-sentence artifacts. Cheaper than the LLMLingua approaches (no extra model beyond the embedding model you already have). LangChain's `ContextualCompressionRetriever` uses this exact pattern.

---

### LLMLingua — token-level perplexity scoring (Microsoft Research, 2023)

Rather than filtering whole sentences, LLMLingua operates at the **token level** — it can remove individual tokens mid-sentence.

**How perplexity works:**

A small compressor LLM (e.g. LLaMA-7B) reads the chunk. At each position `i`, it computes the conditional probability of that token given everything that came before it:

```
P(ti | t1, t2, ... t_{i-1})
```

This probability is converted to a perplexity score using negative log-likelihood:

```
score(ti) = -log P(ti | t1...t_{i-1})
```

- **High score** = the model was surprised by this token = it carries specific information → **keep**
- **Low score** = the model expected this token = predictable filler → **drop**

**Example:**

In the sentence *"As we noted earlier, Q3 revenue rose 12%"*:
- `"As"`, `"we"`, `"noted"`, `"earlier"` → high probability, low perplexity score → **dropped**
- `"Q3"`, `"revenue"`, `"12%"` → low probability (specific, surprising) → **kept**

Result: `"Q3 revenue rose 12%"` — 50% fewer tokens, same information.

**The query-blindness problem:**

Perplexity measures *general* information density. The word `"shareholders"` gets the same perplexity score whether your query is about ownership structure or product revenue. LLMLingua has no idea what you asked — it scores by surprise, not by relevance.

Achieves 3–5× compression. Requires a GPU-hosted compressor LLM, making deployment heavier than contextual.

---

### LLMLingua-2 — query-conditioned classifier (Microsoft Research, 2024)

Fixes the query-blindness of v1 by replacing perplexity scoring with a trained binary classifier.

**Training:**

GPT-4 was given (query, chunk) pairs and asked to label every token `keep` or `drop` — knowing the query when deciding. This means `"shareholders"` gets `keep=true` if the query is about ownership, `keep=false` if about revenue.

A BERT-class model (~125M parameters) was then fine-tuned on these (token, label) pairs.

**Inference:**

```
input to classifier:  [query tokens] + [chunk tokens]
output:               keep/drop probability per chunk token only
```

The query is **never compressed** — it's the context that recalibrates what "important" means for the chunk tokens. Tokens below a threshold are dropped before the prompt is assembled.

**Key improvements over v1:**

| | LLMLingua (2023) | LLMLingua-2 (2024) |
|---|---|---|
| Method | Perplexity via generative LLM | Trained binary classifier |
| Query-aware | No | Yes |
| Hardware | GPU required | CPU, ~20ms |
| Instruction preservation | Degrades | Preserved |
| Production use | Research | Microsoft Copilot |

**Used in Microsoft Copilot's "context budget management" layer.** It sits between retrieval and the OpenAI API call — compressing chunks on every query, adding ~20ms while saving hundreds of milliseconds and significant cost on the model side.

---

### RECOMP — abstractive rewriting (Google Research, 2023)

Instead of filtering tokens or sentences, a small model **rewrites** each chunk into a compact summary shaped by the query.

Input: `(query, chunk)` → Output: a short, dense paragraph that answers "what from this chunk is relevant to the query?"

Most query-conditioned of all approaches: the compression isn't filtered text, it's newly generated text. Produces fluent, coherent output rather than choppy fragments.

**Trade-off:** requires one generation call per chunk (slowest approach), but produces the highest quality compression. Best when the LLM's answer quality matters more than latency.

---

## Part 2: Chunk ordering — Lost-in-the-Middle

### The primacy/recency bias

Liu et al. (2023) showed that LLMs recall information near the **start** and **end** of their context window most reliably. Evidence placed in the **middle** of a long context is statistically underweighted — accuracy drops ~20% for evidence at position 5 vs position 1 in a 10-chunk context. This effect worsens as context grows.

This is called **Lost-in-the-Middle**. It's not a bug that will get patched — it's a property of how attention works at long contexts.

### Three orderings

**Relevance descending (default):**
Most relevant chunk first, least relevant last. Intuitive, but as you add more chunks your best evidence drifts toward the middle. At 10+ chunks, this is actively harmful.

**Relevance ascending (recency bias):**
Least relevant first, most relevant last. Exploits the recency effect — the model finishes reading your best evidence immediately before generating. Counter-intuitive but empirically stronger on several benchmarks, especially with 3–5 chunks.

**Sandwich (recommended):**
Most relevant at position 1, second-most relevant at the last position, everything else buried in the middle.

```
[most relevant chunk]
[mediocre chunks...]
[second-most relevant chunk]
[query]
```

Exploits both primacy and recency simultaneously. Your two highest-quality chunks frame the context from both ends. The middle is filler the model is already statistically likely to underweight.

Used by Perplexity and Bing in production. The recommended default in the Liu et al. 2023 paper.

---

## Part 3: Context strategy — what to do when chunks don't fit

### Stuffing — one call (default)

All retrieved chunks are concatenated into a single prompt. One LLM call. Fast, simple, the default for most production RAG.

The LLM sees everything simultaneously — no information lost across calls. Breaks only when total tokens (system prompt + chunks + query + output reserve) exceed the context window.

For most use cases with ≤20 average-length chunks, this is the right choice.

### Map-Reduce — N+1 calls

**Map:** run the query against each chunk independently in a separate LLM call, getting a partial answer per chunk.

**Reduce:** send all partial answers to a final LLM call to synthesise the full answer.

N+1 calls total. The map phase is parallelisable. Handles arbitrarily large corpora — no window size limit because you never send everything at once.

Used by Amazon Bedrock, LangChain, and LlamaIndex for long-document Q&A.

**When to use:** when your retrieved context is too large for stuffing, or your document corpus is very long (legal, finance, research).

### Refine — N sequential calls

Call 1: generate an initial answer from chunk 1 alone.
Call 2: "here is the current answer — refine it using chunk 2."
Call 3, 4, ..., N: repeat.

Each call sees the current best answer plus one more chunk. N calls total, **not parallelisable** — each depends on the previous output.

Most coherent synthesis: evidence is integrated progressively rather than all at once. Highest latency of all strategies. Best when sequential order matters (reasoning through a document in order).

### Map-Rerank — N parallel calls, pick best

Each chunk generates its own candidate answer with a confidence score. The highest-confidence answer wins. No synthesis step.

N calls, parallelisable. Best for factoid QA where you expect the answer to live entirely within one chunk ("what is the CEO's name?"). Wastes calls if the answer requires combining evidence across chunks.

---

## The query-independence chain

Looking across the full RAG pipeline:

| Stage | Query-dependent? | Why |
|---|---|---|
| Chunking | No | Done offline, at ingest time |
| Embedding | No | Chunk vectors pre-computed |
| Indexing | No | Index built once, reused for all queries |
| Retrieval (dense/sparse/hybrid) | Yes | Query embedded, matched against index |
| Re-ranking (cross-encoder, ColBERT) | Yes | Query and chunk read together |
| Compaction: Raw | No | No inspection of content |
| Compaction: LLMLingua (2023) | No | Perplexity is query-blind |
| Compaction: Contextual | Yes | Sentence similarity to query |
| Compaction: LLMLingua-2 (2024) | Yes | Classifier conditioned on query |
| Compaction: RECOMP | Yes | Rewrites guided by query |
| Chunk ordering | Partial | Uses retrieval scores from the query |
| Generation | Yes | Query is part of the prompt |

The query-independence of early stages is by design: embedding and indexing happen offline so query time only needs fast lookups. Making them query-dependent would require re-processing the corpus for every query — infeasible at scale.

---

## The production stack (full pipeline)

```
Document corpus
  → chunk → embed → index                     [offline, query-independent]

User query
  → embed
  → dense top-20  ┐
  → BM25 top-20   ┘ → RRF merge → top-20
  → cross-encoder re-rank → top-5
  → LLMLingua-2 compaction (conditioned on query)
  → sandwich ordering
  → stuffing into prompt
  → LLM call
  → answer with grounding scores
```

Each step progressively narrows context and improves quality. Expensive steps (cross-encoder, LLMLingua-2) only run on the small set cheaper steps already filtered.
