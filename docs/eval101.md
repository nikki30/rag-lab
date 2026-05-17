# Evaluation 101

Notes from building Stage 6 of RAG Lab. The goal of evaluation is to score the full pipeline — not just "does it return an answer?" but "is the answer faithful to the retrieved context, is the context actually relevant, and did we retrieve everything needed?"

---

## Why evaluation is the hardest RAG stage

You can visually inspect chunking. You can plot embeddings. You can watch rank shifts. But whether an answer is *good* requires a definition of "good" — and that definition differs by metric, by use case, and by who is doing the judging.

The field has converged on two approaches:
- **Embedding-based metrics**: fast, no API cost, good for catching obvious failures
- **LLM-as-judge**: slower, costs tokens, much closer to human judgment

This lab implements embedding-based versions of the core RAGAS metrics. The notes below explain what RAGAS actually does, then how Microsoft, Anthropic, and others go further.

---

## The core metrics

### Faithfulness

**Question:** Is the answer grounded in the retrieved context?

A faithful answer only makes claims that are supported by the retrieved chunks. An unfaithful answer mixes in facts from the model's training weights — which may be outdated, wrong, or simply not from your documents.

**RAGAS approach:** Use an LLM to extract all atomic claims from the answer ("the revenue was $12M", "the CEO is Alice"), then for each claim, ask the LLM whether the context entails it. Score = claims entailed / total claims.

**Embedding-based approximation (this lab):** For each answer sentence, compute cosine similarity against all retrieved chunks. Score = sentences with max_sim ≥ 0.35 / total sentences. Faster but misses paraphrased faithfulness failures.

**Microsoft Azure AI Evaluation:** Calls this "Groundedness". Uses GPT-4 to evaluate each claim on a 1–5 scale. Their prompt: *"Rate how well the answer is supported by the given context, where 1 = completely unsupported and 5 = all claims directly supported."* Used in Azure AI Studio, GitHub Copilot evaluation pipelines, and Bing's quality monitoring.

**Anthropic approach:** Uses Claude as the judge. Their internal evaluation systems ask the model to identify which sentences are not supported by the context and return a structured JSON verdict. They have published that Claude tends to be more conservative (higher precision on unsupported claim detection) than GPT-4, making it better for catching subtle hallucinations.

---

### Answer Relevancy

**Question:** Does the answer actually address the query?

A high faithfulness score doesn't guarantee a useful answer. The model might faithfully quote the retrieved chunks while completely ignoring the question ("What is the capital of France?" → "Paris has a population of 2.1 million." — faithful but not relevant).

**RAGAS approach:** Generate N questions from the answer using an LLM, embed each generated question, average their cosine similarity to the original query. If the answer talks about the right things, questions derived from it should be similar to the original question.

**Embedding-based approximation (this lab):** cosine(query_embedding, answer_embedding). Simpler but doesn't catch answers that are topically nearby but structurally evasive.

**Microsoft:** Calls this "Relevance". GPT-4 evaluates whether the answer addresses the question on a 1–5 scale. Separately from faithfulness — an answer can be fully grounded (faithful) but still irrelevant if the retrieved chunks were about the wrong topic.

---

### Context Precision

**Question:** Of the chunks we retrieved, how many were actually relevant — and were the relevant ones ranked first?

A retriever that returns 10 chunks with 3 relevant ones buried at positions 7, 8, 9 scores low on context precision even if the relevant chunks exist. Rank matters because:
1. Compaction often drops lower-ranked chunks
2. LLMs pay more attention to earlier context (Lost-in-the-Middle)
3. More irrelevant chunks = more noise = higher hallucination risk

**RAGAS formula (weighted precision@k):**
```
CP = Σk [relevant_k × precision@k] / total_relevant
```
Where `relevant_k` = 1 if chunk at rank k is relevant, 0 otherwise.
`precision@k` = (number of relevant chunks in top-k) / k.

This rewards having relevant chunks ranked earlier. A retriever that returns 3 relevant chunks at ranks 1, 2, 3 scores higher than one that returns the same chunks at ranks 1, 5, 10.

**This lab:** Relevance defined as cosine(chunk, query) ≥ 0.30. No LLM required.

---

### Context Recall

**Question:** Did we retrieve everything needed to answer correctly?

Requires a ground truth answer. For each sentence in the ground truth, check whether any retrieved chunk supports it. Score = fraction of ground truth sentences that are covered by the context.

Low context recall means the retriever missed key information — your chunking strategy may have split the answer across boundaries, your embedding model may have failed to map the query to the right region, or the relevant content simply wasn't retrieved within the top-k limit.

**RAGAS:** LLM-based — asks the model whether each GT sentence can be attributed to the retrieved context.

**This lab:** Cosine-based approximation of the same check.

---

### Noise Sensitivity

**Question:** What fraction of retrieved chunks actually contributed to the answer?

Not a standard RAGAS metric but highly diagnostic. If you retrieved 10 chunks and only 2 were cited by grounded answer sentences, 8 were noise — they took up context window space, potentially confused the model, and cost tokens.

**TruLens** calls the inverse of this "context utilisation". High noise means your retriever is returning too many irrelevant results or your top-k is set too high.

Reducing noise is one of the highest-ROI production optimisations: fewer chunks → smaller prompt → lower cost, lower latency, lower hallucination rate.

---

## The RAG Triad (TruLens)

TruLens (by TruEra, now part of Snowflake) popularised three metrics as the minimum bar for trustworthy RAG:

```
Answer Relevance   — does the answer address the query?
Context Relevance  — does the context address the query?
Groundedness       — is the answer supported by the context?
```

All three must pass. A RAG system can fail in three distinct ways:
- Retrieved irrelevant context → low Context Relevance
- Generated an answer unrelated to the question → low Answer Relevance
- Generated claims not in the context → low Groundedness

The triad maps directly to RAGAS: Context Relevance ≈ Context Precision, Groundedness = Faithfulness, Answer Relevance ≈ Answer Relevancy.

---

## LLM-as-judge — the state of the art

Embedding-based metrics are fast and free but have a fundamental limit: they measure semantic similarity, not logical entailment or factual correctness.

Consider: "The company was founded in 1999" (from context) vs "The company was founded in 1998" (from answer). Cosine similarity between these sentences is very high — they're almost identical. But the answer is wrong. Embedding-based faithfulness scores this as grounded. An LLM judge catches it.

**G-EVAL (Microsoft Research, 2023):** Prompts GPT-4 with chain-of-thought reasoning to evaluate coherence, consistency, fluency, and relevance on a 1–5 scale. Uses form-filling format — the LLM fills in an evaluation rubric, which then gets weighted-averaged. First paper to show LLM judges correlate with human judgements at >90% on NLG tasks.

**Prometheus (Kaist AI, 2023):** Open-source fine-tuned judge (Llama-based). Trained on evaluation feedback from GPT-4. Can run locally, unlike GPT-4 judges. Important for privacy-sensitive enterprise deployments.

**ARES (Stanford + Microsoft, 2023):** Automated RAG Evaluation System. Trains small task-specific LM classifiers (distilbert-class) for each of context relevance, answer faithfulness, and answer relevance. 10–100× faster than LLM-as-judge at inference. Microsoft uses this pattern in Azure AI Evaluation for high-volume pipelines.

**The key production trade-off:**

| Method | Cost | Speed | Human alignment |
|--------|------|-------|-----------------|
| Embedding cosine | ~0 | Very fast | Medium |
| Small LM classifier (ARES) | Low | Fast | High |
| LLM-as-judge (GPT-4/Claude) | High | Slow | Very high |
| Human evaluation | Highest | Slowest | Ground truth |

In production: use embedding metrics for continuous monitoring at scale, LLM-as-judge for periodic deep evaluation and catching regressions, human evaluation for establishing benchmarks.

---

## What Microsoft ships in production

**Azure AI Evaluation SDK** (formerly PromptFlow Evaluation):
- Built-in evaluators: Groundedness, Relevance, Coherence, Fluency, Violence, Sexual, Self-harm, Hate/Unfairness
- Groundedness and Relevance use GPT-4 as judge with specific rubric prompts
- Output is a 1–5 float with a reasoning trace (chain of thought)
- Integrates with Azure ML experiments — evaluation scores tracked alongside model versions
- GitHub Actions integration: CI/CD evaluation pipelines that fail on quality regression

**Key insight from Microsoft:** They separate "safety evaluators" (violence, self-harm, hate) from "quality evaluators" (groundedness, relevance, coherence). Different thresholds, different response policies. Bing and Copilot run safety evaluators on every response; quality evaluators run in batch on sampled traffic.

---

## What Anthropic does

Anthropic uses Claude as its own judge — both for Constitutional AI training and for production quality evaluation.

**Model-graded evaluation:** A Claude instance reads (query, context, answer) and produces a structured JSON verdict: `{"faithful": true/false, "reasoning": "...", "unsupported_claims": [...]}`. The reasoning trace is used to diagnose failure modes, not just to produce a score.

**Key insight from Anthropic:** Claude is more precise on faithfulness than GPT-4 — it tends to flag claims as unsupported more conservatively, which matters more for enterprise deployments where false confidence is dangerous. They published this in their model card evaluations for Claude 3.

**Anthropic's recommended evaluation stack for production RAG:**
1. Answer quality: Claude-as-judge for faithfulness + relevance
2. Citation accuracy: structured extraction of claims + per-claim verification
3. Abstention quality: does the model say "I don't know" when context is insufficient rather than hallucinating?

The abstention metric is particularly important and often missed by standard RAGAS evaluations.

---

## Beyond RAGAS — what the field is moving toward

**FActScoring (MIT/University of Washington, 2023):** Breaks evaluation to the atomic fact level. Generates all atomic facts from the answer, verifies each fact against a knowledge source. More granular than sentence-level but expensive.

**RAGAS v2 (2024):** Moved most metrics to LLM-as-judge. Added "Agent Goal Accuracy" for agentic RAG. Added multi-turn conversation evaluation. The embedding-based metrics from v1 are still available but no longer the default.

**HELMET (Princeton, 2024):** Long-context evaluation benchmark. Tests whether models can retrieve and use specific information buried in 128k token contexts. Relevant for RAG with very large context windows.

**Context length scaling:** As models move to 200k+ context windows (Gemini 1.5, Claude 3.7, GPT-4.1), the "stuffing vs MAP-reduce" tradeoff changes. Evaluation must account for whether the model *can* attend to information at position 150,000.
