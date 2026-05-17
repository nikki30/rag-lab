'use client'

import React, { useState, useMemo, useRef, useEffect } from 'react'

const SAMPLE_TEXT = `Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of large language models with external knowledge retrieval. Instead of relying solely on the knowledge encoded in a model's parameters during training, RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context when generating a response.

The retrieval step typically involves converting documents and queries into dense vector embeddings, then using approximate nearest neighbor search to find the most semantically similar passages. Common algorithms for this include HNSW (Hierarchical Navigable Small World) and DiskANN, which trade off between recall accuracy and query latency.

Once relevant chunks are retrieved, they are injected into the prompt as additional context. This grounds the language model's response in factual, up-to-date information without requiring expensive fine-tuning. The quality of the final answer is therefore heavily dependent on both the chunking strategy used to split documents and the embedding model used to represent them.

GraphRAG extends this paradigm by building a knowledge graph over the document corpus. Entities and relationships are extracted, then community detection algorithms cluster related concepts. At query time, the graph structure allows for multi-hop reasoning that flat vector search cannot easily achieve.`

type Strategy = 'recursive' | 'fixed' | 'paragraph' | 'sentence' | 'semantic'
type MainView = 'highlight' | 'cards' | 'compare'

const STRATEGIES: { id: Strategy; label: string; description: string }[] = [
  { id: 'recursive', label: 'Recursive', description: 'Tries paragraph → line → word → char breaks. Respects natural boundaries. Industry default.' },
  { id: 'fixed',     label: 'Fixed',     description: 'Hard cuts every N characters. Ignores words and sentences entirely.' },
  { id: 'paragraph', label: 'Paragraph', description: 'Splits only on blank lines. Content-aware but wildly uneven sizes.' },
  { id: 'sentence',  label: 'Sentence',  description: 'Groups full sentences up to chunk_size. Preserves meaning, variable sizes.' },
  { id: 'semantic',  label: 'Semantic',  description: 'Splits where topic similarity drops. Threshold controls sensitivity.' },
]

const COLORS = [
  { bg: 'bg-violet-500/25', text: 'text-violet-300', border: 'border-violet-500/40' },
  { bg: 'bg-sky-500/25',    text: 'text-sky-300',    border: 'border-sky-500/40'    },
  { bg: 'bg-emerald-500/25',text: 'text-emerald-300',border: 'border-emerald-500/40'},
  { bg: 'bg-amber-500/25',  text: 'text-amber-300',  border: 'border-amber-500/40'  },
  { bg: 'bg-rose-500/25',   text: 'text-rose-300',   border: 'border-rose-500/40'   },
  { bg: 'bg-pink-500/25',   text: 'text-pink-300',   border: 'border-pink-500/40'   },
  { bg: 'bg-cyan-500/25',   text: 'text-cyan-300',   border: 'border-cyan-500/40'   },
  { bg: 'bg-lime-500/25',   text: 'text-lime-300',   border: 'border-lime-500/40'   },
  { bg: 'bg-orange-500/25', text: 'text-orange-300', border: 'border-orange-500/40' },
  { bg: 'bg-teal-500/25',   text: 'text-teal-300',   border: 'border-teal-500/40'   },
]

const STRATEGY_COLORS: Record<Strategy, string> = {
  recursive: 'text-violet-400', fixed: 'text-rose-400', paragraph: 'text-amber-400',
  sentence: 'text-sky-400', semantic: 'text-emerald-400',
}
const STRATEGY_BAR: Record<Strategy, string> = {
  recursive: 'bg-violet-500', fixed: 'bg-rose-500', paragraph: 'bg-amber-500',
  sentence: 'bg-sky-500', semantic: 'bg-emerald-500',
}

interface ChunkScores { size_score: number; boundary_score: number; quality: number }
interface Chunk { page_content: string; metadata: { start_index: number }; scores?: ChunkScores }
interface Stats { avg_size: number; min_size: number; max_size: number; std_dev: number; avg_quality: number }
interface ChunkResponse { chunks: Chunk[]; total: number; stats: Stats; similarity_scores: number[] | null }
interface CompareResult { total: number; avg_size: number; min_size: number; max_size: number; std_dev: number; avg_quality: number; sizes: number[]; preview_chunks: string[]; error?: string }
type CompareResponse = Record<Strategy, CompareResult>

type EmbedModelId = 'minilm' | 'bge-small' | 'mpnet' | 'nomic'
type Reduction = 'pca' | 'umap' | 'pacmap'
interface EmbedModel { id: EmbedModelId; label: string; dims: number; speed: string; description: string; tagline: string }
interface EmbedResult { model: EmbedModelId; dimensions: number; coords_2d: [number, number][]; similarity_matrix: number[][]; vectors: number[][] }

// ── Stage 3 types ─────────────────────────────────────────────────────────────
type IndexTab = 'flat' | 'hnsw' | 'ivf' | 'mrl'
interface IndexResult { idx: number; sim: number; text: string }
interface HNSWLayer { level: number; nodes: number[]; edges: [number, number][] }
interface HNSWMeta { layers: HNSWLayer[]; node_levels: number[]; max_level: number; entry_point: number; M: number }
interface IVFData { cluster_assignments: number[]; n_clusters: number; centroids_2d: [number, number][] }
interface BuildIndexResponse { num_vectors: number; dimensions: number; hnsw: HNSWMeta; ivf: IVFData }
interface TraversalStep { layer: number; visited: number[]; best: number }
interface MRLData { results_by_dims: Record<string, { idx: number; sim: number }[]>; recall: Record<string, number> }
interface QueryIndexResponse {
  query_2d: [number, number]
  flat_results: IndexResult[]
  hnsw_results: IndexResult[]
  hnsw_recall: number
  hnsw_traversal: TraversalStep[]
  ivf_results: IndexResult[]
  ivf_recall: number
  ivf_searched_clusters: number[]
  mrl: MRLData | null
}

// ── Stage 4: Retrieval types ──────────────────────────────────────────────────
interface RetrieveResult { idx: number; score: number; text: string }
interface RankShift {
  idx: number; text: string
  dense_rank: number | null; sparse_rank: number | null
  hybrid_rank: number | null; reranked_rank: number | null
  colbert_rank: number | null
}
interface ColBERTData {
  query_tokens: string[]; chunk_tokens: string[]
  sim_matrix: number[][]; chunk_idx: number
  scores: Record<string, number>
}
interface RetrieveResponse {
  dense: RetrieveResult[]; sparse: RetrieveResult[]
  hybrid: RetrieveResult[]; reranked: RetrieveResult[]
  colbert: ColBERTData | null; rank_shifts: RankShift[]
}

// ── Stage 5: Generate types ───────────────────────────────────────────────────
type LLMProvider = 'openai' | 'anthropic' | 'groq' | 'ollama'
type CompactionAlgo = 'raw' | 'contextual' | 'llmlingua' | 'llmlingua2' | 'recomp'
type ChunkOrderMode = 'relevance_desc' | 'relevance_asc' | 'sandwich'
type ContextStrategy = 'stuffing' | 'map_reduce' | 'refine' | 'map_rerank'

interface LLMModel {
  id: string; label: string; provider: LLMProvider
  contextWindow: number; inputPricePerM: number; outputPricePerM: number
  description: string; tagline: string
}

interface PromptSection {
  label: string; text: string; tokens: number
  role: 'system' | 'chunk' | 'query'
  chunk_idx: number | null; original_tokens: number | null
}

interface GroundingSentence {
  sentence: string; max_similarity: number; grounded: boolean
}

interface GenerateResult {
  answer: string
  sections: PromptSection[]
  grounding: GroundingSentence[]
  total_input_tokens: number; total_output_tokens: number
  cost_usd: number; model: string; context_window: number
  compaction_stats: { original_tokens: number; compressed_tokens: number; ratio: number }
}

const LLM_MODELS: LLMModel[] = [
  { id: 'llama3.2', label: 'Llama 3.2 (Ollama)', provider: 'ollama', contextWindow: 128000,
    inputPricePerM: 0, outputPricePerM: 0, tagline: 'Free · local · no key needed',
    description: 'Runs entirely on your machine via Ollama — no API key, no cost, no data leaving your computer.\n\nInstall: ollama.com → then run: ollama pull llama3.2\n\nBecause it runs locally, latency depends on your hardware. A modern laptop CPU gives usable results; a GPU makes it fast.' },
  { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B (Groq)', provider: 'groq', contextWindow: 128000,
    inputPricePerM: 0, outputPricePerM: 0, tagline: 'Free · cloud · Groq API key',
    description: 'Meta\'s Llama 3.3 70B running on Groq\'s LPU (Language Processing Unit) hardware — purpose-built silicon for transformer inference, not a GPU.\n\nFree tier at console.groq.com. Near-instant responses (~300 tokens/sec). The best free cloud option for RAG — strong reasoning, 128k context, no billing required.' },
]

const COMPACTION_ALGOS: { id: CompactionAlgo; label: string; tagline: string; available: boolean; description: string }[] = [
  { id: 'raw', label: 'Raw', tagline: 'No compaction', available: true,
    description: 'No compaction — every token retrieved is sent verbatim to the LLM.\n\nThis is the baseline. Transparent but wastes context budget on irrelevant sentences and connective tissue. 100% of retrieval noise reaches the model.\n\nQuery-independent: the chunks are not inspected before sending.' },
  { id: 'contextual', label: 'Contextual', tagline: 'Sentence-level · query-aware', available: true,
    description: 'Each sentence in each chunk is embedded and compared to the query (cosine similarity). Sentences below threshold 0.2 are dropped — only query-relevant sentences survive.\n\nQuery-dependent: "relevant" is measured relative to what you asked. Same chunk, different query → different sentences kept.\n\nLangChain\'s ContextualCompressionRetriever uses this exact pattern. Preserves complete sentences — no partial sentence artifacts.' },
  { id: 'llmlingua', label: 'LLMLingua', tagline: 'Microsoft 2023 · token-level · query-blind', available: false,
    description: 'Microsoft Research (2023). Token-level compression using perplexity scoring.\n\nHow perplexity works: a small compressor LLM (e.g. LLaMA-7B) reads the chunk and at each position computes P(token | all preceding tokens) — the probability it assigned to that token given context. This is measured as negative log-likelihood: −log P(ti | t1…ti−1).\n\nHigh score = the model was surprised = this token carries information → keep.\nLow score = the model expected this = predictable filler → drop.\n\nExample: in "As we noted earlier, Q3 revenue rose 12%", the tokens "Q3", "revenue", "12%" score high (specific, surprising). "As", "we", "noted", "earlier" score low (predictable connective tissue) and get dropped.\n\nQuery-blind: perplexity measures general information density, not relevance to your query. "shareholders" gets the same score whether you asked about ownership or revenue.\n\nAchieves 3–5× compression. Requires a GPU-hosted compressor LLM.\n\n⚠ Greyed out — not yet wired up in this lab. Would require: pip install llmlingua + a hosted compressor model.' },
  { id: 'llmlingua2', label: 'LLMLingua-2', tagline: 'Microsoft 2024 · query-conditioned · used in Copilot', available: false,
    description: 'Microsoft Research (2024). Fixes LLMLingua\'s query-blindness by replacing perplexity scoring with a trained binary classifier.\n\nTraining: GPT-4 read (query, chunk) pairs and labelled each token keep or drop — knowing the query when deciding. This created supervision data where "shareholders" is labelled keep=true if the query is about ownership, keep=false if about revenue.\n\nA BERT-class model (~125M params) was then fine-tuned on these labels.\n\nAt inference: classifier receives [query tokens] + [chunk tokens] and outputs a keep/drop score per chunk token. The query is never compressed — it\'s the context that shifts what "important" means. The chunk tokens below threshold are dropped before the prompt is assembled.\n\nKey improvements over v1:\n• Query-conditioned — "important" is relative to what you asked\n• ~10× faster — runs on CPU in ~20ms (no GPU needed)\n• Preserves instruction tokens better\n• Used in Microsoft Copilot\'s context budget management layer\n\n⚠ Greyed out — not yet wired up in this lab. Would require: pip install llmlingua.' },
  { id: 'recomp', label: 'RECOMP', tagline: 'Google 2023 · abstractive · most query-aware', available: false,
    description: 'Google Research (2023). Instead of filtering tokens, a small model rewrites each chunk into a compact, query-focused summary.\n\nMost query-conditioned of all three: the rewrite is shaped entirely by the query — it doesn\'t preserve the original wording, it generates new text that answers "what from this chunk is relevant to the query?"\n\nProduces fluent, dense output rather than choppy filtered text. Slowest (one generation call per chunk) but highest quality. Best when coherent, readable context matters more than raw faithfulness to the source.' },
]

const CHUNK_ORDERS: { id: ChunkOrderMode; label: string; description: string }[] = [
  { id: 'relevance_desc', label: 'Relevance ↓ (default)',
    description: 'Most relevant chunk first, least relevant last. The intuitive default.\n\nThe problem: as you add more chunks, your best evidence drifts toward the middle — where LLMs pay least attention. Liu et al. (2023) showed ~20% accuracy drop for evidence at position 5 vs position 1 in a 10-chunk context. This effect gets worse as context grows.' },
  { id: 'relevance_asc', label: 'Relevance ↑ (recency bias)',
    description: 'Least relevant first, most relevant last. Exploits the recency effect — LLMs recall information near the end of their context most reliably, because it\'s closest to where generation starts.\n\nCounter-intuitive but empirically stronger on several benchmarks, especially with 3–5 chunks. The model finishes reading your best evidence right before it generates.' },
  { id: 'sandwich', label: 'Sandwich (recommended)',
    description: 'Most relevant chunk at position 1, second-most relevant at the end, the rest buried in the middle.\n\nExploits both primacy and recency bias simultaneously — your two highest-quality chunks frame the context window from both ends. Everything in the middle is filler the model is statistically likely to underweight anyway.\n\nDirectly addresses Lost-in-the-Middle (Liu et al. 2023). Used by Perplexity and Bing in production. Recommended default for production RAG.' },
]

const CONTEXT_STRATEGIES: { id: ContextStrategy; label: string; tagline: string; available: boolean; description: string; calls: string }[] = [
  { id: 'stuffing', label: 'Stuffing', tagline: 'All chunks · one call', available: true, calls: '1 LLM call',
    description: 'All retrieved chunks are concatenated into a single prompt and sent in one LLM call. Simple, fast, the standard approach for most production RAG.\n\nThe LLM sees everything at once — no information is lost across calls. Breaks only when the total tokens (system prompt + chunks + query + output reserve) exceed the model\'s context window.\n\nFor most use cases with ≤20 chunks this is the right choice.' },
  { id: 'map_reduce', label: 'Map-Reduce', tagline: 'Summarise → combine', available: false, calls: 'N+1 LLM calls',
    description: 'Map: each chunk gets its own LLM call — "given this chunk, what is relevant to the query?" Reduce: all partial answers are combined in a final LLM call to produce the full answer.\n\nN+1 LLM calls total, but the map phase is parallelisable.\n\nHandles arbitrarily large corpora — no window size limit. Amazon Bedrock, LangChain, and LlamaIndex expose this as a first-class option for long-document Q&A. Use when stuffing hits the window limit.' },
  { id: 'refine', label: 'Refine', tagline: 'Iterative refinement', available: false, calls: 'N LLM calls',
    description: 'Call 1: generate an initial answer from chunk 1 alone. Call 2: "here is the current answer — refine it using chunk 2." Repeat sequentially through all N chunks.\n\nN LLM calls, not parallelisable — each call depends on the previous answer.\n\nMost coherent synthesis: each step builds on the last, so the final answer integrates evidence progressively rather than trying to combine it all at once. Best when sequential order matters — e.g., reasoning through a document in order. Highest latency of all strategies.' },
  { id: 'map_rerank', label: 'Map-Rerank', tagline: 'Answer per chunk · pick best', available: false, calls: 'N LLM calls',
    description: 'Each chunk generates its own candidate answer with a confidence score. The highest-confidence answer wins — no synthesis step.\n\nN LLM calls, parallelisable.\n\nBest for factoid QA where you expect the answer to live entirely within one chunk ("what is the CEO\'s name?"). Wastes calls if the answer requires combining evidence across multiple chunks — the synthesis step never happens.' },
]

// ── Stage 6: Evaluate types ───────────────────────────────────────────────────
interface SentenceScore { sentence: string; max_similarity: number; grounded: boolean; best_chunk_idx: number }
interface ChunkRelevance { chunk_idx: number; preview: string; similarity: number; relevant: boolean; precision_at_k: number }
interface GtSentenceScore { sentence: string; max_similarity: number; supported: boolean; best_chunk_idx: number }
interface EvalResult {
  faithfulness: number; answer_relevancy: number
  context_precision: number; context_recall: number | null
  noise_sensitivity: number
  sentence_scores: SentenceScore[]; chunk_relevance: ChunkRelevance[]; gt_sentence_scores: GtSentenceScore[]
  n_grounded: number; n_sentences: number; n_relevant_chunks: number; n_chunks: number; n_contributing_chunks: number
}

const EMBED_MODELS: EmbedModel[] = [
  { id: 'minilm',    label: 'MiniLM-L6',       dims: 384, speed: 'Fast',    tagline: 'Industry default',       description: 'The workhorse of local RAG. Small, fast, surprisingly good. 384 numbers per chunk. Best starting point.' },
  { id: 'bge-small', label: 'BGE-Small',        dims: 384, speed: 'Fast',    tagline: 'State-of-the-art small', description: 'BAAI\'s model beats MiniLM on retrieval benchmarks at the same size. Same speed, better recall. Current SOTA at 384d.' },
  { id: 'mpnet',     label: 'MPNet-Base',       dims: 768, speed: 'Medium',  tagline: 'More accurate, 2× size', description: 'Doubles the vector size (768d) for higher accuracy. Takes ~2× longer to embed. Good when recall matters more than speed.' },
  { id: 'nomic',     label: 'Nomic-Embed-1.5',  dims: 768, speed: 'Medium',  tagline: 'Best open-source 2024',  description: 'Top open-source general embedding model. Supports Matryoshka — you can shrink vectors from 768d down to 64d with graceful accuracy loss. Cutting edge.' },
]

// ── Tooltip ───────────────────────────────────────────────────────────────────
// Uses position:fixed so it escapes any parent overflow:auto clipping context.
function InfoTooltip({ title, body, pos = 'top' }: {
  title: string; body: string; pos?: 'top' | 'top-left' | 'top-right'
}) {
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null)

  function handleEnter(e: React.MouseEvent) {
    const r = e.currentTarget.getBoundingClientRect()
    setAnchor({ x: r.left + r.width / 2, y: r.top })
  }

  const TIP_W = 384 // matches w-96
  const tooltipStyle: React.CSSProperties = anchor ? (() => {
    const rawLeft = pos === 'top-left'  ? anchor.x + 8 - TIP_W :
                    pos === 'top-right' ? anchor.x - 8          :
                    anchor.x - TIP_W / 2
    const clampedLeft = Math.min(Math.max(rawLeft, 8), window.innerWidth - TIP_W - 8)
    return { position: 'fixed', top: anchor.y - 8, left: clampedLeft, transform: 'translateY(-100%)', zIndex: 9999 }
  })() : {}

  return (
    <div className="relative inline-flex shrink-0"
      onMouseEnter={handleEnter}
      onMouseLeave={() => setAnchor(null)}
    >
      <span className="w-4 h-4 rounded-full border border-zinc-700 text-zinc-600 hover:text-zinc-300 hover:border-zinc-500 text-[10px] flex items-center justify-center transition-colors cursor-help select-none">?</span>
      {anchor && (
        <div style={tooltipStyle} className="w-96 rounded-xl bg-zinc-800 border border-zinc-700 p-3.5 pointer-events-none shadow-2xl">
          <p className="font-semibold text-zinc-100 mb-1.5 text-xs">{title}</p>
          <p className="text-zinc-400 leading-relaxed text-xs whitespace-pre-line">{body}</p>
        </div>
      )}
    </div>
  )
}

// ── Quality badge ─────────────────────────────────────────────────────────────
function QualityBadge({ scores }: { scores?: ChunkScores }) {
  if (!scores) return null
  const q = scores.quality
  const color = q >= 0.75 ? 'text-emerald-400 border-emerald-500/30' : q >= 0.5 ? 'text-amber-400 border-amber-500/30' : 'text-rose-400 border-rose-500/30'
  return (
    <div className={`flex items-center gap-1 rounded-md border px-1.5 py-0.5 ${color}`}>
      <span className="text-xs font-mono font-semibold">{Math.round(q * 100)}%</span>
      <InfoTooltip
        pos="top-left"
        title="Chunk quality breakdown"
        body={`sz (size score): ${Math.round(scores.size_score * 100)}%\nHow close this chunk is to your target chunk_size. 100% = perfect fit. Drops if the chunk is much shorter (wasted budget) or longer (exceeded limit).\n\nbd (boundary score): ${Math.round(scores.boundary_score * 100)}%\nDoes this chunk start with a capital letter (+50%) and end with . ! ? (+50%)? A clean boundary means the chunk is a complete thought — not cut mid-word or mid-sentence.\n\nFixed strategy typically scores 0% on bd. Sentence and Recursive score highest.`}
      />
    </div>
  )
}

// ── Highlight segmenter ───────────────────────────────────────────────────────
function buildHighlightSegments(text: string, chunks: Chunk[]) {
  if (!chunks.length) return [{ text, chunkIndex: -1 }]
  const charMap = new Int16Array(text.length).fill(-1)
  for (let ci = 0; ci < chunks.length; ci++) {
    const content = chunks[ci].page_content
    const hint = chunks[ci].metadata.start_index
    const idx = text.indexOf(content, Math.max(0, hint - 20))
    if (idx === -1) continue
    for (let j = idx; j < idx + content.length && j < text.length; j++) charMap[j] = ci
  }
  const segs: { text: string; chunkIndex: number }[] = []
  let i = 0
  while (i < text.length) {
    const ci = charMap[i]; let j = i + 1
    while (j < text.length && charMap[j] === ci) j++
    segs.push({ text: text.slice(i, j), chunkIndex: ci }); i = j
  }
  return segs
}

// ── Size distribution histogram ───────────────────────────────────────────────
function SizeDistribution({ sizes, color }: { sizes: number[]; color: string }) {
  if (!sizes.length) return null
  const min = Math.min(...sizes), max = Math.max(...sizes)
  const BINS = 8
  const binSize = Math.max(1, Math.ceil((max - min + 1) / BINS))
  const counts = Array(BINS).fill(0)
  sizes.forEach(s => { const b = Math.min(BINS - 1, Math.floor((s - min) / binSize)); counts[b]++ })
  const maxCount = Math.max(...counts, 1)
  return (
    <div className="space-y-1">
      <div className="flex items-end gap-0.5 h-8">
        {counts.map((c, i) => (
          <div key={i} className={`flex-1 rounded-sm ${color} opacity-80`}
            style={{ height: `${(c / maxCount) * 100}%`, minHeight: c > 0 ? 2 : 0 }} />
        ))}
      </div>
      <div className="flex justify-between text-zinc-600" style={{ fontSize: '9px' }}>
        <span>{min}</span>
        <span>{max}ch</span>
      </div>
    </div>
  )
}

// ── Slider ────────────────────────────────────────────────────────────────────
function SliderControl({ label, value, min, max, step, onChange, hint, error, decimals = 0, tooltip }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; hint: string; error?: string; decimals?: number
  tooltip?: { title: string; body: string }
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <label className="text-sm font-medium text-zinc-400">{label}</label>
          {tooltip && <InfoTooltip title={tooltip.title} body={tooltip.body} />}
        </div>
        <input type="number" value={decimals ? value.toFixed(decimals) : value} min={min} max={max} step={step}
          onChange={(e) => onChange(Number(e.target.value))}
          className={`w-20 rounded-md bg-zinc-800 border px-2 py-1 text-sm text-zinc-100 text-right focus:outline-none focus:ring-1 focus:ring-violet-600 ${error ? 'border-red-500' : 'border-zinc-700'}`} />
      </div>
      <input type="range" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-violet-600 cursor-pointer" />
      <p className={`text-xs ${error ? 'text-red-400' : 'text-zinc-600'}`}>{error ?? hint}</p>
    </div>
  )
}

// ── ColBERT heatmap ───────────────────────────────────────────────────────────
function ColBERTHeatmap({ data, topChunk }: { data: ColBERTData; topChunk: RetrieveResult | undefined }) {
  const { query_tokens, chunk_tokens, sim_matrix } = data

  // Find the MaxSim winner per query token (max value index per row)
  const maxSimCols: number[] = sim_matrix.map(row => {
    let best = 0
    for (let c = 1; c < row.length; c++) if (row[c] > row[best]) best = c
    return best
  })

  const colbertScore = sim_matrix.reduce((sum, row, r) => sum + (row[maxSimCols[r]] ?? 0), 0)

  // Color scale: 0 → zinc-900, 1 → amber-400
  function cellColor(v: number): string {
    const t = Math.max(0, Math.min(1, v))
    // interpolate zinc-900 (#18181b) → amber-400 (#fbbf24)
    const r = Math.round(24 + t * (251 - 24))
    const g = Math.round(24 + t * (191 - 24))
    const b = Math.round(27 + t * (36 - 27))
    return `rgb(${r},${g},${b})`
  }

  const CELL = 36
  const ROW_LABEL_W = 120
  const COL_LABEL_H = 80
  const svgW = ROW_LABEL_W + chunk_tokens.length * CELL
  const svgH = COL_LABEL_H + query_tokens.length * CELL + 32

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5 space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-amber-400">ColBERT Late Interaction</span>
        <InfoTooltip
          title="ColBERT late interaction"
          body={"Each query token gets its own embedding. Each chunk token gets its own embedding.\n\nFor every query token, find the chunk token most similar to it (MaxSim — the gold-outlined cell in each row). Sum those MaxSim scores → the ColBERT score.\n\nThis is more expressive than a single dot product because individual token matches can outweigh unrelated tokens."}
        />
      </div>

      <div className="overflow-x-auto">
        <svg width={svgW} height={svgH} style={{ display: 'block' }}>
          {/* Rotated column labels (chunk tokens) */}
          {chunk_tokens.map((tok, c) => (
            <text
              key={c}
              x={ROW_LABEL_W + c * CELL + CELL / 2}
              y={COL_LABEL_H - 6}
              textAnchor="start"
              transform={`rotate(-45, ${ROW_LABEL_W + c * CELL + CELL / 2}, ${COL_LABEL_H - 6})`}
              fontSize={10}
              fill="#a1a1aa"
            >{tok}</text>
          ))}

          {/* Row labels (query tokens) + cells */}
          {query_tokens.map((qtok, r) => (
            <g key={r}>
              {/* Query token label */}
              <text
                x={ROW_LABEL_W - 8}
                y={COL_LABEL_H + r * CELL + CELL / 2 + 4}
                textAnchor="end"
                fontSize={11}
                fontWeight={600}
                fill="#e4e4e7"
              >{qtok}</text>

              {/* Cells */}
              {chunk_tokens.map((_, c) => {
                const v = sim_matrix[r]?.[c] ?? 0
                const isMax = maxSimCols[r] === c
                return (
                  <g key={c}>
                    <rect
                      x={ROW_LABEL_W + c * CELL + 1}
                      y={COL_LABEL_H + r * CELL + 1}
                      width={CELL - 2}
                      height={CELL - 2}
                      rx={3}
                      fill={cellColor(v)}
                    />
                    {/* Gold MaxSim outline */}
                    {isMax && (
                      <rect
                        x={ROW_LABEL_W + c * CELL + 1}
                        y={COL_LABEL_H + r * CELL + 1}
                        width={CELL - 2}
                        height={CELL - 2}
                        rx={3}
                        fill="none"
                        stroke="#fbbf24"
                        strokeWidth={2}
                      />
                    )}
                    {/* Value label */}
                    <text
                      x={ROW_LABEL_W + c * CELL + CELL / 2}
                      y={COL_LABEL_H + r * CELL + CELL / 2 + 4}
                      textAnchor="middle"
                      fontSize={9}
                      fill={v > 0.5 ? '#18181b' : '#71717a'}
                    >{v.toFixed(2)}</text>
                  </g>
                )
              })}

              {/* MaxSim score label at right */}
              <text
                x={ROW_LABEL_W + chunk_tokens.length * CELL + 6}
                y={COL_LABEL_H + r * CELL + CELL / 2 + 4}
                fontSize={10}
                fill="#fbbf24"
              >+{(sim_matrix[r]?.[maxSimCols[r]] ?? 0).toFixed(3)}</text>
            </g>
          ))}

          {/* ColBERT total score */}
          <text
            x={ROW_LABEL_W + chunk_tokens.length * CELL + 6}
            y={COL_LABEL_H + query_tokens.length * CELL + 20}
            fontSize={11}
            fontWeight={700}
            fill="#fbbf24"
          >= {colbertScore.toFixed(3)}</text>
        </svg>
      </div>

      <div className="flex items-center gap-3 text-xs text-zinc-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm border-2 border-amber-400" style={{ background: 'transparent' }} />
          MaxSim per query token
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgb(251,191,36)' }} />
          High similarity
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgb(24,24,27)' }} />
          Low similarity
        </span>
      </div>

      {topChunk && (
        <div className="rounded-lg bg-zinc-800/60 border border-amber-500/20 px-4 py-3 text-xs text-zinc-400">
          <span className="text-amber-400 font-semibold mr-2">Top re-ranked chunk:</span>
          {topChunk.text.slice(0, 200)}{topChunk.text.length > 200 ? '…' : ''}
        </div>
      )}
    </div>
  )
}

// ── Context Window Tank ───────────────────────────────────────────────────────
const TANK_CHUNK_COLORS = ['#8b5cf6', '#0ea5e9', '#10b981', '#f59e0b', '#f43f5e', '#ec4899', '#06b6d4', '#a3e635', '#fb923c', '#2dd4bf']

function ContextWindowTank({ sections, contextWindow }: { sections: PromptSection[]; contextWindow: number }) {
  const OUTPUT_RESERVE = 1000
  const usedTokens = sections.reduce((s, sec) => s + sec.tokens, 0) + OUTPUT_RESERVE
  const usedPct = Math.min(100, (usedTokens / contextWindow) * 100)
  const overflowing = usedPct > 90

  // Assign colours per-section upfront
  let chunkIdx = 0
  const withColors = sections.map(sec => {
    const color = sec.role === 'system' ? '#52525b'
      : sec.role === 'query' ? '#f97316'
      : TANK_CHUNK_COLORS[chunkIdx++ % TANK_CHUNK_COLORS.length]
    return { ...sec, color }
  })

  // Render order in tank: bottom = system, then chunks, then query, then output reserve at top
  // We render from bottom by using flex-col-reverse on the tank container
  const tankSections = [
    ...withColors,
    { label: 'Output reserve', tokens: OUTPUT_RESERVE, role: 'output', color: '#ef444480', chunk_idx: null, original_tokens: null, text: '' },
  ]

  return (
    <div className="flex gap-5 items-start">
      {/* Tank bar */}
      <div className="flex flex-col items-center gap-1.5 shrink-0">
        <span className="text-[10px] text-zinc-500 font-mono">{(contextWindow / 1000).toFixed(0)}k</span>
        <div
          className={`w-10 rounded-lg border overflow-hidden flex flex-col-reverse ${overflowing ? 'border-red-500/60' : 'border-zinc-700'}`}
          style={{ height: 240, backgroundColor: '#18181b' }}
        >
          {tankSections.map((sec, i) => {
            const heightPct = Math.min((sec.tokens / contextWindow) * 100, 100)
            if (heightPct < 0.1) return null
            return (
              <div key={i} title={`${sec.label}: ${sec.tokens.toLocaleString()} tokens`}
                className="w-full shrink-0 transition-all duration-500"
                style={{ height: `${heightPct}%`, backgroundColor: sec.color, opacity: sec.role === 'output' ? 0.4 : 0.75 }}
              />
            )
          })}
        </div>
        <span className={`text-[10px] font-mono font-semibold ${overflowing ? 'text-red-400' : 'text-zinc-500'}`}>
          {Math.round(usedPct)}%
        </span>
      </div>

      {/* Legend — reversed so top of tank = top of list */}
      <div className="flex-1 min-w-0">
        <div className="flex flex-col-reverse gap-1">
          {tankSections.map((sec, i) => (
            <div key={i} className="flex items-center gap-2 min-w-0">
              <div className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: sec.color, opacity: sec.role === 'output' ? 0.5 : 0.8 }} />
              <span className="text-xs text-zinc-400 flex-1 truncate min-w-0">{sec.label}</span>
              <div className="flex items-center gap-1.5 shrink-0">
                {sec.original_tokens && sec.original_tokens !== sec.tokens && (
                  <span className="text-[10px] text-zinc-600 line-through font-mono">{sec.original_tokens.toLocaleString()}</span>
                )}
                <span className="text-xs font-mono text-zinc-500">{sec.tokens.toLocaleString()}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Usage bar */}
        <div className="mt-3 space-y-1">
          <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${overflowing ? 'bg-red-500' : usedPct > 70 ? 'bg-amber-500' : 'bg-emerald-500'}`}
              style={{ width: `${usedPct}%` }} />
          </div>
          <div className="flex justify-between text-[10px] text-zinc-600">
            <span>{usedTokens.toLocaleString()} tokens used</span>
            <span>{(contextWindow - usedTokens).toLocaleString()} remaining</span>
          </div>
          {overflowing && (
            <p className="text-[10px] text-red-400">Context nearly full — consider reducing top-k or switching to a model with a larger window</p>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [text, setText] = useState(SAMPLE_TEXT)
  const [chunkSize, setChunkSize] = useState(200)
  const [chunkOverlap, setChunkOverlap] = useState(50)
  const [strategy, setStrategy] = useState<Strategy>('recursive')
  const [breakpointThreshold, setBreakpointThreshold] = useState(0.4)

  const [chunks, setChunks] = useState<Chunk[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [similarityScores, setSimilarityScores] = useState<number[] | null>(null)
  const [compareData, setCompareData] = useState<CompareResponse | null>(null)

  const [loading, setLoading] = useState(false)
  const [compareLoading, setCompareLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeChunk, setActiveChunk] = useState<number | null>(null)
  const [view, setView] = useState<MainView>('highlight')

  // ── Embedding stage ──
  const [embedModel, setEmbedModel] = useState<EmbedModelId>('minilm')

  const [reduction, setReduction] = useState<Reduction>('pca')
  const [embedResult, setEmbedResult] = useState<EmbedResult | null>(null)
  const [embedLoading, setEmbedLoading] = useState(false)
  const [embedError, setEmbedError] = useState<string | null>(null)
  const [embedStale, setEmbedStale] = useState(false)   // chunks changed since last embed
  const [hoveredEmbedChunk, setHoveredEmbedChunk] = useState<number | null>(null)
  const [selectedEmbedChunk, setSelectedEmbedChunk] = useState<number | null>(null)
  const embedSectionRef = useRef<HTMLDivElement>(null)

  // ── Indexing stage ──
  const [indexData, setIndexData] = useState<BuildIndexResponse | null>(null)
  const [indexLoading, setIndexLoading] = useState(false)
  const [indexError, setIndexError] = useState<string | null>(null)
  const [indexTab, setIndexTab] = useState<IndexTab>('flat')
  const [indexM, setIndexM] = useState(16)
  const [indexEf, setIndexEf] = useState(100)
  const [indexNClusters, setIndexNClusters] = useState(4)
  const [queryText, setQueryText] = useState('')
  const [queryResults, setQueryResults] = useState<QueryIndexResponse | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [nprobe, setNprobe] = useState(2)
  const [hnswEfSearch, setHnswEfSearch] = useState(50)
  const [activeLayer, setActiveLayer] = useState(0)
  const [mrlDims, setMrlDims] = useState(768)
  const [traversalStep, setTraversalStep] = useState(-1)
  const indexSectionRef = useRef<HTMLDivElement>(null)

  // ── Generate stage ──
  const [genModel, setGenModel] = useState<LLMModel>(LLM_MODELS[0])
  const [genApiKey, setGenApiKey] = useState('')
  const [genCompaction, setGenCompaction] = useState<CompactionAlgo>('raw')
  const [genChunkOrder, setGenChunkOrder] = useState<ChunkOrderMode>('relevance_desc')
  const [genContextStrategy, setGenContextStrategy] = useState<ContextStrategy>('stuffing')
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null)
  const [generateLoading, setGenerateLoading] = useState(false)
  const [generateError, setGenerateError] = useState<string | null>(null)
  const [apiKeyMissing, setApiKeyMissing] = useState(false)
  const generateSectionRef = useRef<HTMLDivElement>(null)

  // ── Evaluate stage ──
  const [evalGroundTruth, setEvalGroundTruth] = useState('')
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null)
  const [evalLoading, setEvalLoading] = useState(false)
  const [evalError, setEvalError] = useState<string | null>(null)
  const evalSectionRef = useRef<HTMLDivElement>(null)

  // ── Retrieval stage ──
  const [retrieveQuery, setRetrieveQuery] = useState('')
  // Pre-fill Stage 4 query from Stage 3 when it first becomes non-empty
  useEffect(() => {
    if (queryText && !retrieveQuery) setRetrieveQuery(queryText)
  }, [queryText])
  const [retrieveLoading, setRetrieveLoading] = useState(false)
  const [retrieveResult, setRetrieveResult] = useState<RetrieveResponse | null>(null)
  const [retrieveError, setRetrieveError] = useState<string | null>(null)
  const [retrieveTab, setRetrieveTab] = useState<'results' | 'shifts' | 'reranking'>('results')
  const retrieveSectionRef = useRef<HTMLDivElement>(null)

  const overlapError = strategy !== 'semantic' && chunkOverlap >= chunkSize
    ? 'Overlap must be less than chunk size' : null

  function resetResults() { setChunks([]); setStats(null); setSimilarityScores(null); setCompareData(null); setEmbedResult(null); setEmbedStale(false) }
  // When only the threshold changes, scores stay — only the coloring updates live
  function resetChunksOnly() { setChunks([]); setStats(null); setCompareData(null); setEmbedStale(true) }

  async function handleEmbed(overrideModel?: EmbedModelId, overrideReduction?: Reduction) {
    if (!chunks.length) return
    const model = overrideModel ?? embedModel
    const red = overrideReduction ?? reduction
    setEmbedLoading(true); setEmbedError(null); setEmbedStale(false)
    setSelectedEmbedChunk(null); setHoveredEmbedChunk(null)
    try {
      const res = await fetch('http://localhost:8000/api/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunks: chunks.map(c => c.page_content), model, reduction: red }),
      })
      if (!res.ok) { const b = await res.json().catch(() => ({})); const msg = typeof b?.detail === 'string' ? b.detail : JSON.stringify(b?.detail) ?? `Error ${res.status}`; throw new Error(msg) }
      setEmbedResult(await res.json())
      setTimeout(() => embedSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) { setEmbedError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setEmbedLoading(false) }
  }

  async function handleChunk(overrideStrategy?: Strategy) {
    const activeStrategy = overrideStrategy ?? strategy
    if (activeStrategy !== 'semantic' && chunkOverlap >= chunkSize) return
    setLoading(true); setError(null); resetResults()
    setEmbedStale(true)
    try {
      const res = await fetch('http://localhost:8000/api/chunk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, chunk_size: chunkSize, chunk_overlap: chunkOverlap, strategy: activeStrategy, breakpoint_threshold: breakpointThreshold }),
      })
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b?.detail?.[0]?.msg ?? `Error ${res.status}`) }
      const data: ChunkResponse = await res.json()
      setChunks(data.chunks); setStats(data.stats); setSimilarityScores(data.similarity_scores)
    } catch (e) { setError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setLoading(false) }
  }

  async function handleBuildIndex() {
    setIndexLoading(true); setIndexError(null); setQueryResults(null); setTraversalStep(-1)
    try {
      const res = await fetch('http://localhost:8000/api/build-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ M: indexM, ef_construction: indexEf, n_clusters: indexNClusters }),
      })
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b?.detail ?? `Error ${res.status}`) }
      const data: BuildIndexResponse = await res.json()
      setIndexData(data)
      setActiveLayer(0)
      setIndexTab('flat')
      setTimeout(() => indexSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) { setIndexError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setIndexLoading(false) }
  }

  async function handleQueryIndex() {
    if (!queryText.trim()) return
    setQueryLoading(true); setQueryError(null); setTraversalStep(-1)
    try {
      const res = await fetch('http://localhost:8000/api/query-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, k: 5, ef: hnswEfSearch, nprobe }),
      })
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b?.detail ?? `Error ${res.status}`) }
      setQueryResults(await res.json())
    } catch (e) { setQueryError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setQueryLoading(false) }
  }

  async function handleRetrieve() {
    if (!retrieveQuery.trim()) return
    setRetrieveLoading(true); setRetrieveError(null)
    try {
      const res = await fetch('http://localhost:8000/api/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: retrieveQuery, k: 5, top_k_rerank: 20 }),
      })
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b?.detail ?? `Error ${res.status}`) }
      const data: RetrieveResponse = await res.json()
      setRetrieveResult(data)
      setRetrieveTab('results')
      setTimeout(() => retrieveSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) { setRetrieveError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setRetrieveLoading(false) }
  }

  async function handleGenerate() {
    if (!retrieveResult?.reranked.length) return
    // Always scroll to stage 5 first so the user sees any errors
    generateSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    if (genModel.provider !== 'ollama' && !genApiKey.trim()) { setApiKeyMissing(true); return }
    setApiKeyMissing(false)
    setGenerateLoading(true); setGenerateError(null); setGenerateResult(null)
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30000)
      const res = await fetch('http://localhost:8000/api/generate', {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: retrieveQuery,
          chunks: retrieveResult.reranked,
          model: genModel.id,
          api_key: genApiKey,
          compaction: genCompaction,
          chunk_order: genChunkOrder,
          context_strategy: genContextStrategy,
          temperature: 0.1,
        }),
      })
      clearTimeout(timeout)
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b?.detail ?? `Error ${res.status}`) }
      const data: GenerateResult = await res.json()
      setGenerateResult(data)
      setTimeout(() => generateSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) {
      const msg = e instanceof Error ? (e.name === 'AbortError' ? 'Request timed out after 30s — the LLM call took too long' : e.message) : 'Unknown error'
      setGenerateError(msg)
    }
    finally { setGenerateLoading(false) }
  }

  async function handleEvaluate() {
    if (!generateResult || !retrieveResult) return
    evalSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setEvalLoading(true); setEvalError(null); setEvalResult(null)
    try {
      const res = await fetch('http://localhost:8000/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: retrieveQuery,
          answer: generateResult.answer,
          chunks: retrieveResult.reranked.map(r => r.text),
          ground_truth: evalGroundTruth.trim() || null,
          embed_model: embedModel,
        }),
      })
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b?.detail ?? `Error ${res.status}`) }
      const data: EvalResult = await res.json()
      setEvalResult(data)
      setTimeout(() => evalSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) { setEvalError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setEvalLoading(false) }
  }

  async function handleCompare() {
    setCompareLoading(true); setError(null); setCompareData(null)
    try {
      const res = await fetch('http://localhost:8000/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, chunk_size: chunkSize, chunk_overlap: Math.min(chunkOverlap, chunkSize - 1), strategy, breakpoint_threshold: breakpointThreshold }),
      })
      if (!res.ok) throw new Error(`Error ${res.status}`)
      setCompareData(await res.json())
    } catch (e) { setError(e instanceof Error ? e.message : 'Unknown error') }
    finally { setCompareLoading(false) }
  }

  const segments = useMemo(() => buildHighlightSegments(text, chunks), [text, chunks])

  // Maps each similarity score bar index → which chunk it belongs to.
  // Blue bars: the chunk containing this sentence boundary.
  // Orange cut bars: the new chunk starting after this cut (increment first).
  const barToChunk = useMemo(() => {
    if (!similarityScores) return [] as number[]
    let cuts = 0
    return similarityScores.map(score => {
      if (score < breakpointThreshold) cuts++
      return cuts
    })
  }, [similarityScores, breakpointThreshold])

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-violet-600 flex items-center justify-center text-xs font-bold">R</div>
          <h1 className="text-lg font-semibold tracking-tight">RAG Lab</h1>
          <span className="ml-2 text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded-full">chunk → embed → index</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

        {/* ── Stage 1 header ── */}
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-violet-600 flex items-center justify-center text-xs font-bold">1</div>
          <h2 className="text-lg font-semibold tracking-tight">Chunking</h2>
          <div className="flex items-center gap-1.5 ml-1">
            <InfoTooltip
              title="Why chunking is stage 1"
              body={"Before a document can be searched, it must be split into pieces. Each piece gets turned into a vector (embedding) in stage 2 and stored in an index.\n\nChunking is the most impactful decision in the entire RAG pipeline — more than the embedding model or retrieval algorithm. If a key fact spans a chunk boundary, it might never be retrieved.\n\nThe strategy you pick here directly shapes what the embedding space looks like in stage 2."}
              pos="top-right"
            />
          </div>
        </div>

        {/* ── Strategy picker ── */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Strategy</p>
            <InfoTooltip
              title="What is a chunking strategy?"
              body={"Before embedding, your document must be split into pieces. The strategy controls where those splits happen.\n\nThis matters enormously: if a key fact spans a chunk boundary, it might never be retrieved. Try all five strategies on the same text to see how different the splits are — that difference directly affects retrieval quality downstream."}
              pos="top-right"
            />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {STRATEGIES.map((s) => (
              <button key={s.id} onClick={() => { setStrategy(s.id); handleChunk(s.id) }}
                className={`rounded-xl border px-3 py-3 text-left transition-all ${strategy === s.id ? 'border-violet-500 bg-violet-500/10 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600'}`}>
                <p className="text-sm font-semibold mb-1">{s.label}</p>
                <p className="text-xs leading-relaxed opacity-70">{s.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* ── Text + controls ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-2">
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider">Source Text</label>
            <textarea value={text} onChange={(e) => { setText(e.target.value); resetResults() }} rows={10}
              className="w-full rounded-xl bg-zinc-900 border border-zinc-700 px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 resize-y focus:outline-none focus:ring-2 focus:ring-violet-600 transition" />
            <p className="text-xs text-zinc-600">{text.length.toLocaleString()} characters</p>
          </div>

          <div className="space-y-4">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Parameters</p>
            {strategy === 'semantic' ? (
              <SliderControl label="Breakpoint Threshold" value={breakpointThreshold} min={0.05} max={0.95} step={0.05}
                onChange={(v) => { setBreakpointThreshold(v); resetChunksOnly() }}
                hint="Split when similarity drops below this" decimals={2}
                tooltip={{
                  title: 'Breakpoint Threshold',
                  body: 'When two adjacent sentences have embedding cosine similarity below this threshold, the algorithm decides the topic changed and starts a new chunk.\n\nThe similarity is computed using real MiniLM sentence embeddings — not word overlap. Sentences on the same topic typically score 0.5–0.9. A topic shift drops toward 0.2–0.4.\n\nLow threshold (0.2) → only splits at strong topic shifts → fewer, bigger chunks.\nHigh threshold (0.7) → splits at any slight change → many small chunks.\n\nWatch the bar chart: red bars are where splits happen. Set the threshold then click Chunk Text to apply.',
                }}
              />
            ) : (
              <>
                <SliderControl label="Chunk Size" value={chunkSize} min={50} max={2000} step={50}
                  onChange={(v) => { setChunkSize(v); resetResults() }} hint="Max characters per chunk"
                  tooltip={{
                    title: 'Chunk Size',
                    body: 'Maximum characters allowed per chunk.\n\nLarger → more context per retrieved piece, but noisier (irrelevant sentences dilute the signal).\nSmaller → precise retrieval, but a chunk may lack enough context to be meaningful on its own.\n\nMost production RAG systems use 200–500 chars. Start at 200 and tune based on your document structure.',
                  }}
                />
                <SliderControl label="Chunk Overlap" value={chunkOverlap} min={0} max={500} step={10}
                  onChange={(v) => { setChunkOverlap(v); resetResults() }} hint="Shared chars between chunks"
                  error={overlapError ?? undefined}
                  tooltip={{
                    title: 'Chunk Overlap',
                    body: 'Characters copied from the END of one chunk to the START of the next.\n\nIf a key sentence falls on a boundary, overlap ensures it appears in both chunks — so retrieval never completely misses it.\n\nToo high → redundant storage and duplicate embeddings.\nToo low → boundary blind spot.\n\nGood rule of thumb: 10–15% of chunk_size.',
                  }}
                />
              </>
            )}
            <button onClick={() => handleChunk()} disabled={loading || !text.trim() || !!overlapError}
              className="w-full px-6 py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-semibold transition-colors">
              {loading ? 'Chunking…' : 'Chunk It!'}
            </button>
            <button onClick={() => { setView('compare'); handleCompare() }} disabled={compareLoading || !text.trim()}
              className="w-full px-6 py-2 rounded-lg border border-zinc-700 hover:border-zinc-500 text-sm text-zinc-400 hover:text-zinc-200 transition-colors disabled:opacity-40">
              {compareLoading ? 'Comparing…' : 'Compare All Strategies'}
            </button>
          </div>
        </div>

        {error && <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">{error}</div>}

        {/* ── Semantic similarity chart ── */}
        {strategy === 'semantic' && similarityScores && similarityScores.length > 0 && (() => {
          const CHART_H = 80
          const minScore = Math.min(...similarityScores)
          const maxScore = Math.max(...similarityScores)
          const scoreRange = maxScore - minScore || 0.01
          // Normalize bars to fill the full chart height so differences are visible
          const barH = (s: number) => Math.round(8 + ((s - minScore) / scoreRange) * (CHART_H - 8))
          // Where the threshold line sits in px from the bottom
          const threshLinePx = Math.max(0, Math.min(CHART_H, Math.round(8 + ((breakpointThreshold - minScore) / scoreRange) * (CHART_H - 8))))
          const cuts = similarityScores.filter(s => s < breakpointThreshold).length

          return (
            <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Topic similarity between sentences</p>
                  <InfoTooltip
                    title="How to read this chart"
                    body={"Each bar = one boundary between two adjacent sentences.\nBar height = how topically similar those sentences are.\n\nTall bar (blue) → same topic → sentences stay in the same chunk.\nShort bar (orange) ✂ → topic shifted → chunk boundary placed here.\n\nThe dashed line is your threshold. Drag the slider and watch it move — bars below the line turn orange and become cuts.\n\nWhy this matters: every other strategy ignores topic structure. This one only cuts where the writing itself signals a topic change."}
                    pos="top-right"
                  />
                </div>
                <span className="text-xs text-zinc-600">
                  {cuts} cut{cuts !== 1 ? 's' : ''} → <span className="text-zinc-400">{cuts + 1} chunks</span>
                </span>
              </div>

              {/* Chart */}
              <div className="relative flex items-end gap-1" style={{ height: CHART_H }}>
                {/* Dashed threshold line */}
                <div
                  className="absolute left-0 right-0 border-t border-dashed border-violet-500/70 pointer-events-none transition-all duration-150 z-10"
                  style={{ bottom: threshLinePx }}
                >
                  <span className="absolute -top-4 right-0 text-violet-400 font-mono text-[10px]">{breakpointThreshold}</span>
                </div>

                {/* Bars */}
                {similarityScores.map((score, i) => {
                  const isSplit = score < breakpointThreshold
                  const chunkIdx = barToChunk[i]
                  const isHighlighted = activeChunk === chunkIdx
                  const c = COLORS[chunkIdx % COLORS.length]
                  return (
                    <div key={i} className="flex-1 relative group/bar flex flex-col justify-end cursor-pointer" style={{ height: CHART_H }}
                      onMouseEnter={() => setActiveChunk(chunkIdx)}
                      onMouseLeave={() => setActiveChunk(null)}
                    >
                      {isSplit && (
                        <div className="absolute top-0 left-1/2 -translate-x-1/2 text-orange-400 text-xs leading-none">✂</div>
                      )}
                      <div
                        className={`w-full rounded-t transition-all duration-150 ${isSplit ? 'bg-orange-500/80' : 'bg-blue-400/70'} ${activeChunk === null ? '' : isHighlighted ? 'ring-1 ring-white/40 brightness-125' : 'opacity-25'}`}
                        style={{ height: `${barH(score)}px` }}
                      />
                      {/* Hover tooltip */}
                      <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1 text-xs whitespace-nowrap invisible group-hover/bar:visible z-20 pointer-events-none shadow-lg">
                        <span className={`font-mono font-semibold ${c.text}`}>#{chunkIdx + 1}</span>
                        <span className="font-mono text-zinc-200 ml-1.5">{score.toFixed(3)}</span>
                        <span className={`ml-1.5 ${isSplit ? 'text-orange-400' : 'text-blue-400'}`}>
                          {isSplit ? '✂ cut' : '↔ same chunk'}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Scale labels */}
              <div className="flex justify-between text-zinc-600" style={{ fontSize: '10px' }}>
                <span>lowest similarity in doc ({minScore.toFixed(2)})</span>
                <span>highest ({maxScore.toFixed(2)})</span>
              </div>

              {/* Legend */}
              <div className="flex items-center gap-4 text-xs text-zinc-500 pt-1 border-t border-zinc-800">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-2.5 rounded-sm bg-blue-400/70 inline-block" /> same topic — stay together
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-2.5 rounded-sm bg-orange-500/80 inline-block" /> topic shift — chunk boundary ✂
                </span>
                <span className="flex items-center gap-1.5 ml-auto text-zinc-600">
                  <span className="w-4 border-t border-dashed border-violet-500/70 inline-block" /> threshold (drag slider to move)
                </span>
              </div>
            </div>
          )
        })()}

        {/* ── Results ── */}
        {(chunks.length > 0 || compareData) && (
          <section className="space-y-4">

            {/* Stats bar */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2 flex-wrap">
                {stats && (
                  <>
                    <StatPill label={`${chunks.length} chunks`}
                      tooltip={{ title: 'Total chunks', body: 'How many pieces your document was split into.\n\nMore chunks → finer retrieval granularity, but a larger index to search.\nFewer chunks → each piece is richer in context but less precise.\n\nNeither is universally better — it depends on your query length and document structure.' }} />
                    <StatPill label={`avg ${stats.avg_size} chars`}
                      tooltip={{ title: 'Average chunk size', body: `Mean character count across all chunks. Ideally close to your chunk_size (${chunkSize}).\n\nIf avg << chunk_size, the splitter is oversplitting (e.g. paragraph strategy with short paragraphs). If avg ≈ chunk_size, it's using the size budget efficiently.` }} />
                    <StatPill label={`σ ${stats.std_dev}`}
                      tooltip={{ title: 'σ — Standard deviation of chunk sizes', body: 'How inconsistent your chunk sizes are.\n\nσ = 0 → every chunk is exactly the same size.\nσ = 150 → sizes vary wildly — some chunks are one sentence, others are three paragraphs.\n\nIn RAG, high σ hurts because small chunks embed differently than large ones, making retrieval quality unpredictable. Lower σ = more consistent embeddings = more reliable retrieval.' }} />
                    <StatPill label={`${stats.min_size}–${stats.max_size}`}
                      tooltip={{ title: 'Smallest – Largest chunk', body: 'Watch the outliers.\n\nA minimum of ~5–20 chars usually means a stray sentence fragment became its own chunk — it will embed poorly because there\'s not enough context for the model to understand what it\'s about.\n\nA maximum much larger than chunk_size means the splitter couldn\'t find a valid split point and gave up.' }} />
                    <StatPill label={`quality ${Math.round(stats.avg_quality * 100)}%`}
                      tooltip={{ title: 'Average chunk quality score', body: 'Average of two sub-scores across all chunks:\n\n• Size score (50%): how close each chunk is to your target chunk_size. 100% = perfect fit.\n• Boundary score (50%): does the chunk start with a capital letter (clean start) AND end with .!? (clean end)?\n\nFixed strategy scores ~0% on boundary because it cuts mid-word. Sentence and Recursive score highest because they respect sentence boundaries.' }} />
                  </>
                )}
              </div>
              <div className="flex rounded-lg overflow-hidden border border-zinc-700 text-xs">
                {(['highlight', 'cards', 'compare'] as MainView[]).map((v) => (
                  <button key={v} onClick={() => { setView(v); if (v === 'compare' && !compareData) handleCompare() }}
                    className={`px-3 py-1.5 capitalize transition-colors ${view === v ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
                    {v}
                  </button>
                ))}
              </div>
            </div>

            {/* ── HIGHLIGHT VIEW ── */}
            {view === 'highlight' && chunks.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="lg:col-span-2 rounded-xl bg-zinc-900 border border-zinc-800 p-5">
                  <p className="text-xs text-zinc-500 mb-3 font-medium uppercase tracking-wider">Hover a card → dims all other chunks</p>
                  <p className="text-sm leading-7 whitespace-pre-wrap font-mono">
                    {segments.map((seg, i) => {
                      if (seg.chunkIndex === -1) return <span key={i} className="text-zinc-600">{seg.text}</span>
                      const c = COLORS[seg.chunkIndex % COLORS.length]
                      const isActive = activeChunk === null || activeChunk === seg.chunkIndex
                      return <span key={i} className={`rounded px-0.5 transition-all duration-150 ${c.bg} ${c.text} ${isActive ? 'opacity-100' : 'opacity-15'}`}>{seg.text}</span>
                    })}
                  </p>
                </div>
                <div className="space-y-2 overflow-y-auto max-h-[540px] pr-1">
                  {chunks.map((chunk, i) => {
                    const c = COLORS[i % COLORS.length]
                    return (
                      <div key={i} onMouseEnter={() => setActiveChunk(i)} onMouseLeave={() => setActiveChunk(null)}
                        className={`rounded-lg border ${c.border} ${c.bg} p-3 cursor-default transition-all`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-xs font-mono font-bold ${c.text}`}>#{i + 1}</span>
                          <div className="flex items-center gap-1.5">
                            <QualityBadge scores={chunk.scores} />
                            <span className="text-xs text-zinc-500">{chunk.page_content.length}ch</span>
                          </div>
                        </div>
                        <p className={`text-xs leading-relaxed ${c.text} opacity-80 line-clamp-2`}>{chunk.page_content}</p>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* ── CARDS VIEW ── */}
            {view === 'cards' && chunks.length > 0 && (
              <div className="grid grid-cols-1 gap-3">
                {chunks.map((chunk, i) => {
                  const c = COLORS[i % COLORS.length]
                  return (
                    <div key={i} className={`rounded-xl bg-zinc-900 border ${c.border} p-4`}>
                      <div className="flex items-center justify-between mb-3">
                        <span className={`text-xs font-mono font-semibold ${c.text}`}>#{i + 1}</span>
                        <div className="flex items-center gap-3">
                          <QualityBadge scores={chunk.scores} />
                          <div className="flex gap-3 text-xs text-zinc-600">
                            {chunk.scores && <span>sz {Math.round(chunk.scores.size_score * 100)}%</span>}
                            {chunk.scores && <span>bd {Math.round(chunk.scores.boundary_score * 100)}%</span>}
                            <span>offset {chunk.metadata.start_index}</span>
                            <span>{chunk.page_content.length} chars</span>
                          </div>
                        </div>
                      </div>
                      <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">{chunk.page_content}</p>
                    </div>
                  )
                })}
              </div>
            )}

            {/* ── COMPARE VIEW ── */}
            {view === 'compare' && (
              compareLoading ? (
                <div className="text-sm text-zinc-500 py-8 text-center">Running all strategies…</div>
              ) : compareData ? (
                <div className="space-y-4">
                  <div className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-zinc-800 text-xs text-zinc-500 uppercase tracking-wider">
                          <th className="text-left px-4 py-3">Strategy</th>
                          <CompareHeader label="Chunks"
                            tooltip={{ title: 'Total chunks', body: 'More = finer granularity but larger index.\nFewer = richer context per piece but less precise retrieval.' }} />
                          <CompareHeader label="Avg size"
                            tooltip={{ title: 'Average chunk size', body: `Ideally close to your chunk_size (${chunkSize}). A large gap means the strategy is over- or under-splitting.` }} />
                          <CompareHeader label="Min–Max"
                            tooltip={{ title: 'Smallest – Largest chunk', body: 'Outliers. A very small minimum (<20 chars) embeds poorly. A very large maximum means the splitter couldn\'t find a boundary and gave up.' }} />
                          <CompareHeader label="σ std dev"
                            tooltip={{ title: 'σ — Standard deviation', body: 'Lower is better. Measures how inconsistent chunk sizes are.\n\nHigh σ = some chunks are one word, others are a full paragraph. This causes wildly different embedding densities and makes retrieval unreliable.\n\nGreen ↓ = lowest σ = winner on consistency.' }} />
                          <CompareHeader label="Quality"
                            tooltip={{ title: 'Average quality score', body: 'Combined size + boundary score across all chunks.\n\nHigh quality = chunks are close to the size target AND start/end at sentence boundaries.\n\nGreen ↑ = highest quality = winner.' }} />
                          <CompareHeader label="Distribution"
                            tooltip={{ title: 'Chunk size distribution', body: 'Mini histogram: left = small chunks, right = large.\n\nA narrow spike near the middle = consistent sizes = good.\nA flat spread = wildly varying sizes = bad for embedding.\n\nYou want a tight cluster near your chunk_size target.' }} />
                        </tr>
                      </thead>
                      <tbody>
                        {(Object.entries(compareData) as [Strategy, CompareResult][]).map(([strat, result]) => {
                          if (result.error) return (
                            <tr key={strat} className="border-b border-zinc-800/50">
                              <td className={`px-4 py-3 font-semibold capitalize ${STRATEGY_COLORS[strat]}`}>{strat}</td>
                              <td colSpan={6} className="px-4 py-3 text-xs text-red-400">{result.error}</td>
                            </tr>
                          )
                          const valid = Object.values(compareData).filter(r => !r.error)
                          const bestSigma = Math.min(...valid.map(r => r.std_dev))
                          const bestQuality = Math.max(...valid.map(r => r.avg_quality))
                          return (
                            <tr key={strat} className={`border-b border-zinc-800/50 hover:bg-zinc-800/30 ${strat === strategy ? 'bg-zinc-800/20' : ''}`}>
                              <td className={`px-4 py-3 font-semibold capitalize ${STRATEGY_COLORS[strat]}`}>
                                {strat}{strat === strategy && <span className="ml-2 text-xs text-zinc-600">(active)</span>}
                              </td>
                              <td className="px-4 py-3 text-right text-zinc-300">{result.total}</td>
                              <td className="px-4 py-3 text-right text-zinc-300">{result.avg_size}</td>
                              <td className="px-4 py-3 text-right text-zinc-500">{result.min_size}–{result.max_size}</td>
                              <td className={`px-4 py-3 text-right font-mono ${result.std_dev === bestSigma ? 'text-emerald-400 font-bold' : 'text-zinc-300'}`}>
                                {result.std_dev}{result.std_dev === bestSigma && ' ↓'}
                              </td>
                              <td className={`px-4 py-3 text-right font-mono ${result.avg_quality === bestQuality ? 'text-emerald-400 font-bold' : 'text-zinc-300'}`}>
                                {Math.round(result.avg_quality * 100)}%{result.avg_quality === bestQuality && ' ↑'}
                              </td>
                              <td className="px-4 py-3 w-28">
                                <SizeDistribution sizes={result.sizes} color={STRATEGY_BAR[strat]} />
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* First chunk preview */}
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <p className="text-xs text-zinc-500 uppercase tracking-wider font-medium">First chunk — same text, different cut point</p>
                      <InfoTooltip
                        title="Why look at the first chunk?"
                        body={"This is the most direct way to see how strategies differ. All five receive the exact same text and the same chunk_size.\n\nLook at where each one ends — Recursive finds the nearest sentence boundary, Fixed cuts mid-word, Paragraph grabs the whole first paragraph, Sentence ends on a clean period, Semantic ends when the topic shifts.\n\nThe end-point of chunk #1 is the start-point of chunk #2, so this snowballs through the whole document."}
                        pos="top-right"
                      />
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {(Object.entries(compareData) as [Strategy, CompareResult][])
                        .filter(([, r]) => !r.error && r.preview_chunks.length > 0)
                        .map(([strat, result]) => (
                          <div key={strat} className="rounded-xl bg-zinc-900 border border-zinc-800 p-4">
                            <div className="flex items-center justify-between mb-2">
                              <span className={`text-xs font-semibold capitalize ${STRATEGY_COLORS[strat]}`}>{strat}</span>
                              <span className="text-xs text-zinc-600">{result.preview_chunks[0].length} chars</span>
                            </div>
                            <p className="text-xs text-zinc-400 leading-relaxed line-clamp-6 font-mono">{result.preview_chunks[0]}</p>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              ) : null
            )}
          </section>
        )}

        {/* ── Embed CTA — appears once chunks exist ── */}
        {chunks.length > 0 && (
          <div className="flex items-center gap-4 rounded-xl border border-zinc-800 bg-zinc-900/60 px-5 py-4">
            <div className="flex-1">
              <p className="text-sm font-semibold text-zinc-200">Ready to embed</p>
              <p className="text-xs text-zinc-500 mt-0.5">
                {embedStale && embedResult ? 'Chunks changed — re-embed to update the vector space.' : `Convert your ${chunks.length} chunks into vectors so they can be searched semantically.`}
              </p>
            </div>
            <button
              onClick={() => handleEmbed()}
              disabled={embedLoading}
              className={`px-5 py-2 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${embedStale && embedResult ? 'bg-amber-600 hover:bg-amber-500 text-white' : 'bg-emerald-600 hover:bg-emerald-500 text-white'}`}
            >
              {embedLoading ? 'Embedding…' : embedStale && embedResult ? 'Re-embed Chunks' : 'Embed Chunks →'}
            </button>
          </div>
        )}
      </main>

      {/* ══════════════════════════════════════════════════════════════════════
          STAGE 2 — EMBEDDING
      ══════════════════════════════════════════════════════════════════════ */}
      {(embedResult || embedLoading || embedError) && (
        <div ref={embedSectionRef} className="border-t-2 border-zinc-800 mt-2">
          <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

            {/* Stage header */}
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-md bg-emerald-600 flex items-center justify-center text-xs font-bold">2</div>
              <h2 className="text-lg font-semibold tracking-tight">Embedding</h2>
              <div className="flex items-center gap-1.5 ml-1">
                <InfoTooltip
                  title="What is embedding?"
                  body={"Embedding turns text into numbers — specifically a list of hundreds of numbers called a vector.\n\nEach vector is a point in high-dimensional space. Chunks about similar topics end up near each other. Chunks about different topics are far apart.\n\nThis is what makes semantic search possible: instead of matching keywords, the retriever finds the chunks whose vectors are closest to the query vector.\n\nThe embedding model is what does this conversion. Different models produce different spaces — some are better at retrieval, some are more compact, some are more accurate."}
                  pos="top-right"
                />
              </div>
              <span className="ml-auto text-xs text-zinc-600 font-mono">{chunks.length} chunks from stage 1</span>
            </div>

            {/* Model picker */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Embedding Model</p>
                <InfoTooltip
                  title="Which model should I pick?"
                  body={"Each model converts your text into vectors differently. They've been trained on different data with different objectives.\n\nStart with MiniLM — it's the fastest and a good baseline. Then try BGE-Small to see if retrieval improves. MPNet and Nomic are heavier but more accurate.\n\nWatch the scatter plot change between models — same chunks, different geometry. That geometry is what your retriever will search."}
                  pos="top-right"
                />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {EMBED_MODELS.map(m => (
                  <button key={m.id}
                    onClick={() => { setEmbedModel(m.id); if (embedResult) handleEmbed(m.id, reduction) }}
                    className={`rounded-xl border px-3 py-3 text-left transition-all ${embedModel === m.id ? 'border-emerald-500 bg-emerald-500/10 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600'}`}>
                    <div className="flex items-start justify-between mb-1">
                      <p className="text-sm font-semibold">{m.label}</p>
                      <InfoTooltip title={m.label} body={m.description} pos="top-left" />
                    </div>
                    <p className="text-xs text-zinc-500 mb-1">{m.tagline}</p>
                    <div className="flex gap-2 text-xs">
                      <span className="text-zinc-600">{m.dims}d</span>
                      <span className={`${m.speed === 'Fast' ? 'text-emerald-500' : 'text-amber-500'}`}>{m.speed}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {embedError && <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">{embedError}</div>}

            {embedLoading && (
              <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-10 text-center space-y-2">
                <p className="text-sm text-zinc-400">Embedding {chunks.length} chunks with {EMBED_MODELS.find(m => m.id === embedModel)?.label}…</p>
                <p className="text-xs text-zinc-600">First run downloads the model (~90–400 MB). Subsequent runs are instant.</p>
              </div>
            )}

            {embedResult && !embedLoading && (
              <div className={`flex items-center gap-4 rounded-xl border px-5 py-4 ${embedStale ? 'border-amber-700/40 bg-amber-950/20' : 'border-zinc-800 bg-zinc-900/60'}`}>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-zinc-200">
                    {embedStale ? 'Re-embed before indexing' : 'Ready to index'}
                  </p>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    {embedStale
                      ? 'Your chunks changed — re-embed first so the index reflects the current vectors.'
                      : indexData
                        ? 'Index is built. Adjust parameters and use the Rebuild Index button in Stage 3 below.'
                        : `Build a searchable index from your ${embedResult.coords_2d.length} vectors so you can run queries against them.`}
                  </p>
                </div>
                {indexError && <p className="text-xs text-red-400 max-w-xs">{indexError}</p>}
                <button onClick={handleBuildIndex} disabled={indexLoading || embedStale}
                  className="px-5 py-2 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed bg-amber-600 hover:bg-amber-500 text-white whitespace-nowrap">
                  {indexLoading ? 'Building…' : indexData ? 'Rebuild Index' : 'Build Index →'}
                </button>
              </div>
            )}

            {embedResult && !embedLoading && (
              <div className="space-y-6">

                {/* Stats bar */}
                <div className="flex items-center gap-3 flex-wrap">
                  <StatPill label={`${embedResult.dimensions} dimensions`}
                    tooltip={{ title: 'Vector dimensions', body: 'Each chunk is represented by this many numbers. More dimensions = the model can capture finer distinctions between concepts, but uses more memory and is slower to search.\n\nMiniLM and BGE-Small use 384. MPNet and Nomic use 768. In practice the difference is small for a single document.' }} />
                  <StatPill label={`${embedResult.coords_2d.length} vectors`}
                    tooltip={{ title: 'Vectors produced', body: 'One vector per chunk. These are the points you see on the scatter plot below.' }} />
                  <div className="ml-auto flex items-center gap-2">
                    <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">2D Layout</span>
                    <div className="flex rounded-lg border border-zinc-700 overflow-hidden text-xs">
                    {(['pca', 'umap', 'pacmap'] as Reduction[]).map(r => (
                      <button key={r} onClick={() => { setReduction(r); handleEmbed(embedModel, r) }}
                        className={`px-3 py-1.5 uppercase font-mono transition-colors ${reduction === r ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
                        {r}
                      </button>
                    ))}
                    </div>
                    <InfoTooltip
                      title="PCA vs UMAP vs PaCMAP — 2D Layout algorithms"
                      body={"All three are dimensionality reduction techniques — they take your 384D vectors and find a 2D arrangement that preserves as much structure as possible. Only PCA is a true geometric projection (a shadow). UMAP and PaCMAP are learned layouts — they run an optimisation to find the best 2D positions, so there's no direct geometric shadow being cast.\n\nPCA — only true projection. Fast, deterministic. Honest about global spread but misses curved cluster structure.\n\nUMAP — learned layout. Great at revealing tight clusters. But 'contention points' (chunks mid-way between two topics) get forced into one island arbitrarily. Inter-cluster distances are meaningless.\n\nPaCMAP — learned layout designed to fix both. Balances near, mid-range, and far pairs. Boundary points stay mid-range rather than being forced into an island.\n\nTip: use PaCMAP as your default, PCA to sanity-check global spread, UMAP if you want to see tight cluster structure."}
                      pos="top-left"
                    />
                  </div>
                </div>

                {/* Scatter plot */}
                <EmbedScatterPlot
                  coords={embedResult.coords_2d}
                  chunks={chunks}
                  hovered={hoveredEmbedChunk}
                  selected={selectedEmbedChunk}
                  onHover={setHoveredEmbedChunk}
                  onSelect={setSelectedEmbedChunk}
                  reduction={reduction}
                />

                {/* Similarity heatmap */}
                <EmbedHeatmap
                  matrix={embedResult.similarity_matrix}
                  chunks={chunks}
                  hovered={hoveredEmbedChunk}
                  selected={selectedEmbedChunk}
                  onHover={setHoveredEmbedChunk}
                  onSelect={setSelectedEmbedChunk}
                />

                {/* Vector inspector */}
                {selectedEmbedChunk !== null && (
                  <VectorInspector
                    vector={embedResult.vectors[selectedEmbedChunk]}
                    chunkIndex={selectedEmbedChunk}
                    chunk={chunks[selectedEmbedChunk]}
                  />
                )}
              </div>
            )}
          </main>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          STAGE 3 — INDEXING
      ══════════════════════════════════════════════════════════════════════ */}
      {indexData && (
        <div ref={indexSectionRef} className="border-t-2 border-zinc-800 mt-2">
          <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

            {/* Stage header */}
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-md bg-amber-600 flex items-center justify-center text-xs font-bold">3</div>
              <h2 className="text-lg font-semibold tracking-tight">Indexing</h2>
              <div className="flex items-center gap-1.5 ml-1">
                <InfoTooltip
                  title="What is an index?"
                  body={"After embedding, you have a list of vectors — but searching them by computing similarity to every single one is O(n). For millions of documents, that's too slow.\n\nAn index is a data structure built from those vectors that makes search fast. Different index types make different tradeoffs between speed, recall, and memory.\n\nThis stage lets you build and query three types — Flat (exact), HNSW (graph-based approximate), and IVF (cluster-based approximate) — and see exactly what each one does differently."}
                  pos="top-right"
                />
              </div>
              <span className="ml-auto text-xs text-zinc-600 font-mono">{indexData.num_vectors} vectors · {indexData.dimensions}d</span>
            </div>

            {/* Query input */}
            <div className="flex gap-3">
              <input
                value={queryText}
                onChange={e => setQueryText(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleQueryIndex()}
                placeholder="Type a query — e.g. 'How does retrieval work?'"
                className="flex-1 rounded-xl bg-zinc-900 border border-zinc-700 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
              <button onClick={handleQueryIndex} disabled={queryLoading || !queryText.trim()}
                className="px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-sm font-semibold transition-colors whitespace-nowrap">
                {queryLoading ? 'Searching…' : 'Search →'}
              </button>
            </div>
            {queryError && <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">{queryError}</div>}

            {/* Build parameters — always visible, above tabs */}
            <div className="rounded-xl bg-zinc-900 border border-orange-500/30 p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-1 h-4 rounded-full bg-orange-500/60" />
                  <p className="text-xs font-medium text-orange-400 uppercase tracking-wider">Index build parameters</p>
                  <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold bg-orange-500/10 text-orange-400/80 border border-orange-500/20">change → click Rebuild Index</span>
                  <InfoTooltip
                    title="Build vs query parameters"
                    body={"These are build-time parameters — they shape how the index itself is constructed from your vectors. Changing any of them requires clicking Rebuild Index.\n\n• M — how many edges per chunk in the HNSW graph\n• ef_construction — how carefully HNSW picks those edges\n• n_clusters — how many IVF partitions to create\n\nQuery-time parameters (ef_search in HNSW tab, nprobe in IVF tab) only affect how each search runs. You can change those and just re-run the query — no rebuild needed."}
                    pos="top-right"
                  />
                </div>
                <button onClick={handleBuildIndex} disabled={indexLoading}
                  className="px-4 py-1.5 rounded-lg bg-orange-600 hover:bg-orange-500 disabled:opacity-40 text-xs font-semibold transition-colors whitespace-nowrap text-white">
                  {indexLoading ? 'Building…' : 'Rebuild Index'}
                </button>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <SliderControl label="M — edges per chunk" value={indexM} min={4} max={32} step={2}
                  onChange={v => setIndexM(v)} hint="→ click Rebuild Index to apply"
                  tooltip={{ title: 'M — connections per chunk (HNSW only)', body: 'M is an HNSW-specific parameter. IVF does not use it at all.\n\nHNSW stores explicit edges: "chunk 7 is connected to chunks 2, 14, 9...". M controls how many edges each chunk gets. FAISS physically stores these edges — the graph is the index.\n\nIVF has no edges. Vectors are dropped into buckets (clusters). Neighbourhood is implicit — everything in the same bucket is roughly nearby — but nothing is connected. n_clusters is IVF\'s equivalent of M.\n\nFor HNSW:\nM=4 → sparse graph, fast build, lower recall.\nM=16 → richer graph, better recall.\nM=32 → very dense, excellent recall, more memory.\n\nYou can see the edges on the HNSW scatter plot. Try reducing M to 4 and rebuilding — the graph visibly thins out.' }} />
                <SliderControl label="ef_construction" value={indexEf} min={50} max={400} step={50}
                  onChange={v => setIndexEf(v)} hint="→ click Rebuild Index to apply"
                  tooltip={{ title: 'ef_construction — how carefully HNSW finds neighbours at build time', body: 'When a new chunk is added to the graph, HNSW must find its M nearest neighbours to connect to. It does this by running a greedy beam search — exploring candidate nodes and keeping a list of the best ones seen so far.\n\nef_construction is the size of that candidate list.\n\nLow ef (e.g. 50): the beam is narrow. It explores fewer candidates → finds approximate neighbours quickly → done fast, but some nodes end up connected to sub-optimal neighbours.\n\nHigh ef (e.g. 400): wider beam → explores more of the graph → finds truer nearest neighbours for each node → better-quality graph, but every single node insertion takes longer.\n\nThe impact: a graph built with ef=400 will have more accurate edges than ef=50, so searches at query time achieve higher recall even with the same ef_search.\n\nThis parameter has zero effect on query speed — only on build quality.' }} />
                <SliderControl label="n_clusters" value={indexNClusters} min={2} max={Math.min(10, indexData.num_vectors)} step={1}
                  onChange={v => setIndexNClusters(v)} hint="→ click Rebuild Index to apply"
                  tooltip={{ title: 'n_clusters — IVF partitions (IVF only, no M)', body: 'n_clusters is IVF\'s equivalent of M in HNSW — but the mechanism is completely different.\n\nHNSW stores explicit edges between chunks (M edges per chunk). The graph is the index. M directly controls how connected each node is.\n\nIVF has no edges at all. Vectors are dropped into buckets via k-means. "Neighbourhood" is implicit — everything in the same bucket is roughly nearby — but nothing is connected to anything. n_clusters controls how fine the buckets are, not any edge structure.\n\nMore clusters → smaller, tighter buckets → higher precision at low nprobe.\nFewer clusters → broader buckets → safer but less selective.\n\nRule of thumb: √(number of vectors). For our demo size, 3–6 is plenty.' }} />
              </div>
            </div>

            {/* Tab bar */}
            <div className="flex items-center gap-1 rounded-xl bg-zinc-900 border border-zinc-800 p-1 w-fit">
              {(['flat', 'hnsw', 'ivf'] as IndexTab[]).map(tab => (
                <button key={tab} onClick={() => setIndexTab(tab)}
                  className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors uppercase tracking-wide ${indexTab === tab ? 'bg-amber-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}>
                  {tab}
                </button>
              ))}
              {embedModel === 'nomic' ? (
                <button onClick={() => setIndexTab('mrl')}
                  className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors uppercase tracking-wide ${indexTab === 'mrl' ? 'bg-amber-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}>
                  MRL ✦
                </button>
              ) : (
                <div className="relative group">
                  <button disabled className="px-4 py-1.5 rounded-lg text-sm font-semibold uppercase tracking-wide text-zinc-700 cursor-not-allowed">
                    MRL ✦
                  </button>
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg bg-zinc-800 border border-zinc-700 p-2.5 text-xs text-zinc-400 hidden group-hover:block pointer-events-none z-50 whitespace-normal">
                    MRL requires the <span className="text-amber-400 font-semibold">Nomic</span> embedding model — it's the only model trained with Matryoshka dimensions. Switch to Nomic in Stage 2 and re-embed to unlock this tab.
                  </div>
                </div>
              )}
            </div>

            {/* ── FLAT TAB ── */}
            {indexTab === 'flat' && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Flat — exact brute-force search</p>
                  <InfoTooltip
                    title="Flat / brute-force search"
                    body={"The simplest possible approach: compute cosine similarity between the query and every single vector, sort, return top-k. No data structure, no skipping — pure math.\n\nBecause it skips nothing, it doesn't need FAISS or any special index. We just use numpy dot products directly.\n\nThis is the ground truth — 100% recall by definition. Every other index type is measured against it.\n\nThe cost scales as (number of vectors) × (dimensions per vector) per query. For 1M vectors at 768 dimensions that's 768 million multiply-adds — roughly 100–500ms on a laptop CPU. Fine here with a handful of chunks; unusable at production scale.\n\nHNSW and IVF exist entirely to avoid this full scan."}
                    pos="top-right"
                  />
                  <div className="ml-auto flex items-center gap-2 text-xs text-zinc-600">
                    <span className="text-emerald-400 font-semibold">100% recall</span>
                    <span>·</span>
                    <span>O(vectors × dims) per query</span>
                  </div>
                </div>
                {queryResults ? (
                  <IndexScatterPlot
                    coords={embedResult!.coords_2d}
                    chunks={chunks}
                    queryPoint={queryResults.query_2d}
                    highlightedIndices={queryResults.flat_results.map(r => r.idx)}
                    clusterAssignments={null}
                    centroids2d={null}
                    searchedClusters={null}
                    hnswMeta={null}
                    activeLayer={0}
                    traversal={null}
                    traversalStep={-1}
                    tab="flat"
                  />
                ) : (
                  <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-10 text-center text-sm text-zinc-600">Enter a query above to see results plotted here</div>
                )}
                {queryResults && <IndexResultsList results={queryResults.flat_results} label="Flat results — ranked by cosine similarity" color="amber" />}
              </div>
            )}

            {/* ── HNSW TAB ── */}
            {indexTab === 'hnsw' && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">HNSW — hierarchical navigable small world <span className="text-zinc-600 normal-case font-normal">(Facebook AI Similarity Search)</span></p>
                  <InfoTooltip
                    title="How HNSW works (FAISS IndexHNSWFlat)"
                    body={"FAISS (Facebook AI Similarity Search — Meta Research) is the library that builds and searches the index. Without FAISS, every query would fall back to flat search.\n\nHNSW is the data structure FAISS builds: a multilayer graph where each chunk connects to M neighbours. The graph is what lets FAISS skip most cosine comparisons — instead of comparing the query to every chunk, it navigates the graph layer by layer to reach the right neighbourhood.\n\nThe similarity metric is still cosine throughout — HNSW just determines which chunks you ever bother comparing against.\n\nUpper layers (highway): sparse, random subset of chunks — used for fast long-range navigation.\nLayer 0 (precise): all chunks — used for the final local search.\n\nResult: near-perfect recall while computing cosine similarity against only a tiny fraction of all vectors."}
                    pos="top-right"
                  />
                  {queryResults && (
                    <div className={`ml-auto flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${queryResults.hnsw_recall === 1 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                      {Math.round(queryResults.hnsw_recall * 100)}% recall vs flat
                      <InfoTooltip title="HNSW recall" body={"How many of the true top-5 (from flat search) HNSW also found.\n\n100% = perfect — HNSW found exactly the same results as brute force.\n<100% = HNSW missed some true nearest neighbours — the cost of approximation.\n\nIncrease ef (search beam width) to improve recall at the cost of speed."} pos="top-left" />
                    </div>
                  )}
                </div>

                {/* Layer selector — only if HNSW built successfully */}
                {'layers' in indexData.hnsw ? (
                <div className="flex items-center gap-3">
                  <span className="text-xs text-zinc-500">Layer:</span>
                  <div className="flex rounded-lg border border-zinc-700 overflow-hidden text-xs">
                    {[...(indexData.hnsw as HNSWMeta).layers].reverse().map(l => (
                      <button key={l.level} onClick={() => setActiveLayer(l.level)}
                        className={`px-3 py-1.5 transition-colors ${activeLayer === l.level ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
                        {l.level === 0 ? '0 — all chunks (precise)' : `${l.level} — highway (coarse)`}
                        <span className="ml-1.5 text-zinc-600">{l.nodes.length} chunks</span>
                      </button>
                    ))}
                  </div>
                  <InfoTooltip
                    title="HNSW layers"
                    body={"HNSW search starts at the highest layer and descends to layer 0.\n\nHigher layers (highway): only a random subset of chunks exist here — roughly 1/M of all chunks. They have long-range connections spanning the space, so the search can skip large distances quickly.\n\nLayer 0 (precise): every chunk exists here with 2×M connections each. This is where the final nearest-neighbour search happens.\n\nThe chunk count shown on each button is how many chunks exist at that layer. The top-5 results shown on the scatter plot may include chunks from any layer — that's why you might see 5 highlighted but only 3 chunks at layer 1."}
                  />
                  <div className="ml-auto flex items-center gap-3 text-xs text-zinc-600">
                    <span>M = {(indexData.hnsw as HNSWMeta).M}</span>
                    <span>·</span>
                    <span className="flex items-center gap-1">
                      entry = chunk #{(indexData.hnsw as HNSWMeta).entry_point + 1}
                      <InfoTooltip title="Entry point" body={"The entry point is the last chunk inserted into the index — effectively arbitrary from the data's perspective. Every search starts here regardless of the query.\n\nIn a well-connected graph this doesn't matter much — the highway layers navigate away from it quickly. But in a poorly-connected graph (low M, low ef_construction), a bad entry point can hurt recall."} />
                    </span>
                  </div>
                </div>
                ) : (
                  <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">
                    HNSW failed to build: {(indexData.hnsw as {error: string}).error} — try rebuilding.
                  </div>
                )}

                {'layers' in indexData.hnsw && (
                <IndexScatterPlot
                  coords={embedResult!.coords_2d}
                  chunks={chunks}
                  queryPoint={queryResults?.query_2d ?? null}
                  highlightedIndices={queryResults?.hnsw_results.map(r => r.idx) ?? []}
                  clusterAssignments={null}
                  centroids2d={null}
                  searchedClusters={null}
                  hnswMeta={indexData.hnsw as HNSWMeta}
                  activeLayer={activeLayer}
                  traversal={queryResults?.hnsw_traversal ?? null}
                  traversalStep={traversalStep}
                  tab="hnsw"
                />
                )}

                {/* Traversal stepper */}
                {queryResults?.hnsw_traversal && queryResults.hnsw_traversal.length > 0 && (
                  <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Query traversal</p>
                      <InfoTooltip
                        title="How to read the traversal"
                        body={"Each row = one layer of the HNSW graph. The search starts at the entry point on the highest layer and descends.\n\nVisited nodes = every node the greedy search touched at that layer.\nBest = the node it landed on before dropping to the next layer.\n\nNotice that upper layers visit very few nodes (fast coarse navigation) while layer 0 visits more (precise local search)."}
                        pos="top-right"
                      />
                    </div>
                    <div className="space-y-0">
                      {[...queryResults.hnsw_traversal].reverse().map((step, i, arr) => {
                        const isLast = i === arr.length - 1
                        const nextStep = arr[i + 1]
                        return (
                          <div key={step.layer}>
                            <div className="flex items-start gap-3 text-xs py-2">
                              <div className="w-32 shrink-0">
                                <span className="text-zinc-400 font-semibold">Layer {step.layer}</span>
                                <span className="ml-1.5 text-zinc-600">{step.layer === 0 ? '(precise)' : '(highway)'}</span>
                              </div>
                              <div className="flex flex-wrap gap-1 flex-1">
                                {step.visited.map(v => (
                                  <span key={v} className={`rounded px-1.5 py-0.5 font-mono ${v === step.best ? 'bg-amber-500/30 text-amber-300 font-bold ring-1 ring-amber-500/50' : 'bg-zinc-800 text-zinc-400'}`}>
                                    #{v + 1}
                                  </span>
                                ))}
                              </div>
                              <span className="text-zinc-600 shrink-0">{step.visited.length} visited</span>
                            </div>
                            {!isLast && (
                              <div className="flex items-center gap-2 text-xs pl-32 py-1 text-zinc-600 border-l border-zinc-800 ml-3">
                                <span className="text-amber-500/70">↓</span>
                                <span>landed on chunk #{step.best + 1} — use as entry point for layer {nextStep.layer}</span>
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                    <p className="text-xs text-zinc-600 pt-1 border-t border-zinc-800">Gold chip = best chunk at that layer · each layer's winner becomes the starting point for the layer below</p>
                  </div>
                )}

                {/* Query-time param only */}
                <div className="rounded-xl bg-zinc-900 border border-emerald-500/25 p-4 space-y-3 max-w-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-4 rounded-full bg-emerald-500/60" />
                    <p className="text-xs font-medium text-emerald-400 uppercase tracking-wider">Query parameter</p>
                    <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold bg-emerald-500/10 text-emerald-400/80 border border-emerald-500/20">change → re-run Search</span>
                  </div>
                  <SliderControl label="ef_search — beam width" value={hnswEfSearch} min={10} max={200} step={10}
                    onChange={v => setHnswEfSearch(v)} hint="→ re-run Search to apply"
                    tooltip={{ title: 'ef_search — query-time beam width', body: 'How many candidates the search considers at layer 0 before returning top-k results.\n\nHigher ef → better recall (finds true nearest neighbours more reliably), slower query.\nLower ef → faster query, may miss some true nearest neighbours.\n\nWhen ef ≈ k, recall can drop. When ef >> k, recall ≈ 100%.\n\nThis is a query-only parameter — changing it does not require rebuilding the index.' }} />
                </div>

                {queryResults && <IndexResultsList results={queryResults.hnsw_results} label="HNSW results" color="amber" />}
              </div>
            )}

            {/* ── IVF TAB ── */}
            {indexTab === 'ivf' && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">IVF — inverted file index <span className="text-zinc-600 normal-case font-normal">(Facebook AI Similarity Search)</span></p>
                  <InfoTooltip
                    title="How IVF works (FAISS IndexIVFFlat)"
                    body={"FAISS (Facebook AI Similarity Search — Meta Research) builds and searches the IVF index in production. In this demo we use sklearn for the clustering step, but the concept is identical.\n\nIVF is the data structure: a set of clusters that lets FAISS skip cosine comparisons against most vectors. The similarity metric is still cosine throughout — IVF just controls which vectors you ever compare against.\n\n① BUILD — k-means clusters your vectors into n_clusters groups. Each cluster gets a centroid — a computed average position, NOT one of your original vectors. The × markers you see are these averages.\n\nHow centroids are found (k-means++):\n• Pick a first centroid from your data points at random\n• Each subsequent seed is chosen with probability proportional to its distance from the nearest existing centroid\n• Assign every vector to its nearest centroid → move each centroid to the mean of its members → repeat until stable\n\n② QUERY — use cosine to find the nprobe nearest centroids, then use cosine to search only the vectors inside those clusters. Everything else is skipped.\n\nTrade-off: nprobe=1 → fastest, may miss true neighbours. nprobe=n_clusters → 100% recall, same as flat."}
                    pos="top-right"
                  />
                  {queryResults && (
                    <div className={`ml-auto flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${queryResults.ivf_recall === 1 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                      {Math.round(queryResults.ivf_recall * 100)}% recall
                      <InfoTooltip title="IVF recall" body={"How many of the true top-5 (flat) IVF found by only searching nprobe clusters.\n\nIncrease nprobe to search more clusters → higher recall but slower.\nnprobe = n_clusters → same as flat search (100% recall)."} pos="top-left" />
                    </div>
                  )}
                </div>

                <IndexScatterPlot
                  coords={embedResult!.coords_2d}
                  chunks={chunks}
                  queryPoint={queryResults?.query_2d ?? null}
                  highlightedIndices={queryResults?.ivf_results.map(r => r.idx) ?? []}
                  clusterAssignments={indexData.ivf.cluster_assignments}
                  centroids2d={indexData.ivf.centroids_2d}
                  searchedClusters={queryResults?.ivf_searched_clusters ?? null}
                  hnswMeta={null}
                  activeLayer={0}
                  traversal={null}
                  traversalStep={-1}
                  tab="ivf"
                />

                {/* Query-time param — n_clusters is in the shared build params panel above */}
                <div className="rounded-xl bg-zinc-900 border border-emerald-500/25 p-4 space-y-3 max-w-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-4 rounded-full bg-emerald-500/60" />
                    <p className="text-xs font-medium text-emerald-400 uppercase tracking-wider">Query parameter</p>
                    <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold bg-emerald-500/10 text-emerald-400/80 border border-emerald-500/20">change → re-run Search</span>
                  </div>
                  <SliderControl label="nprobe — clusters to search" value={nprobe} min={1} max={indexData.ivf.n_clusters} step={1}
                    onChange={v => setNprobe(v)} hint="→ re-run Search to apply"
                    tooltip={{ title: 'nprobe — query-only, no rebuild needed', body: `How many of the ${indexData.ivf.n_clusters} clusters to search per query.\n\nnprobe=1 → fastest, lowest recall — only the single nearest cluster is searched.\nnprobe=${indexData.ivf.n_clusters} → same recall as flat search, 100%.\n\nProduction IVF indexes have thousands of clusters with nprobe ≈ 20 — searching only ~2% of the index.\n\nThis is a query-only parameter. The clusters themselves don't change — you're just deciding how many to look at each time you search.` }} />
                </div>

                {queryResults && (
                  <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Cluster search summary</p>
                      <InfoTooltip
                        title="Why did a visually distant cluster get searched?"
                        body={"IVF cluster selection is based on distance in the original high-dimensional vector space (e.g. 384d or 768d), not on the 2D scatter plot you see.\n\nThe 2D layout is a PCA/UMAP projection — it compresses hundreds of dimensions into two, which inevitably distorts distances. Two points that look far apart in 2D may actually be close in 768d space, and vice versa.\n\nSo if a searched cluster looks distant on screen, it means it was genuinely among the nprobe nearest centroids in full-dimensional space — the 2D view is just misleading you."}
                        pos="top-right"
                      />
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {Array.from({ length: indexData.ivf.n_clusters }, (_, i) => {
                        const searched = queryResults.ivf_searched_clusters.includes(i)
                        const count = indexData.ivf.cluster_assignments.filter(a => a === i).length
                        return (
                          <div key={i} className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs ${searched ? 'border-amber-500/50 bg-amber-500/10 text-amber-300' : 'border-zinc-700 bg-zinc-900 text-zinc-600'}`}>
                            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: IVF_CLUSTER_COLORS[i % IVF_CLUSTER_COLORS.length] }} />
                            <span className="font-semibold">Cluster {i}</span>
                            <span>{count} chunk{count !== 1 ? 's' : ''}</span>
                            {searched && <span className="text-amber-500">✓ searched</span>}
                          </div>
                        )
                      })}
                    </div>
                    <p className="text-xs text-zinc-600">Searched {queryResults.ivf_searched_clusters.length} of {indexData.ivf.n_clusters} clusters ({Math.round(queryResults.ivf_searched_clusters.length / indexData.ivf.n_clusters * 100)}% of index)</p>
                  </div>
                )}

                {queryResults && <IndexResultsList results={queryResults.ivf_results} label="IVF results" color="amber" />}
              </div>
            )}

            {/* ── MRL TAB (Nomic only) ── */}
            {indexTab === 'mrl' && embedModel === 'nomic' && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">MRL — Matryoshka representation learning</p>
                  <span className="text-[10px] text-zinc-600 font-normal normal-case">Nomic only</span>
                  <InfoTooltip
                    title="What is MRL — and why Nomic only?"
                    body={"MRL is a training-time property, not something you can apply to any model.\n\nA normal model (MiniLM, BGE, MPNet) produces a vector where all dimensions work together as a unit. Truncating it to fewer dimensions produces garbage — those first N numbers were never trained to be independently meaningful.\n\nNomic Embed 1.5 was trained with Matryoshka Representation Learning, which forces the model to pack the most important information into the first dimensions, then progressively add detail. So the first 64 dimensions are themselves a valid (though less precise) representation. Truncating to 64d, 128d, 256d gives graceful accuracy loss instead of garbage.\n\nThe pipeline rule: embedding and retrieval must use the same dimensionality. You can't embed chunks at 768d and search at 64d unless you truncate both sides consistently — stored chunk vectors AND the query vector — to the same d. That's exactly what this tab demonstrates.\n\nWhy it matters in production: smaller vectors = faster index search, less RAM, lower storage cost. A 64d Nomic vector is 12× smaller than 768d with surprisingly good recall."}
                    pos="top-right"
                  />
                </div>

                {/* Dim selector */}
                <div className="flex items-center gap-3">
                  <span className="text-xs text-zinc-500">Dimensions:</span>
                  <div className="flex rounded-lg border border-zinc-700 overflow-hidden text-xs">
                    {[768, 512, 256, 128, 64].map(d => (
                      <button key={d} onClick={() => setMrlDims(d)}
                        className={`px-3 py-1.5 font-mono transition-colors ${mrlDims === d ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
                        {d}d
                      </button>
                    ))}
                  </div>
                </div>

                {/* Recall table */}
                {queryResults?.mrl && (
                  <div className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-zinc-800 text-xs text-zinc-500 uppercase tracking-wider">
                          <th className="text-left px-4 py-3">Dimensions</th>
                          <th className="text-right px-4 py-3">Memory vs 768d</th>
                          <th className="text-right px-4 py-3">
                            <div className="flex items-center justify-end gap-1">Recall@5 <InfoTooltip title="Recall@5" body={"How many of the true top-5 results (at full 768d) are also in the top-5 at this truncated dimension.\n\n1.0 = identical results.\n0.8 = 4 of 5 correct.\n\nMRL ensures graceful degradation — even at 64d, results are often still useful."} pos="top-left" /></div>
                          </th>
                          <th className="text-right px-4 py-3">Top result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[768, 512, 256, 128, 64].map(d => {
                          const recall = queryResults.mrl!.recall[String(d)] ?? 0
                          const topResult = queryResults.mrl!.results_by_dims[String(d)]?.[0]
                          const isActive = mrlDims === d
                          return (
                            <tr key={d} onClick={() => setMrlDims(d)}
                              className={`border-b border-zinc-800/50 cursor-pointer transition-colors ${isActive ? 'bg-zinc-800/50' : 'hover:bg-zinc-800/20'}`}>
                              <td className="px-4 py-3 font-mono text-zinc-300">{d}d {d === 768 && <span className="text-xs text-zinc-600 ml-1">full</span>}</td>
                              <td className="px-4 py-3 text-right text-zinc-500 font-mono text-xs">{Math.round(768 / d)}×</td>
                              <td className="px-4 py-3 text-right">
                                <span className={`font-mono font-semibold ${recall === 1 ? 'text-emerald-400' : recall >= 0.8 ? 'text-amber-400' : 'text-rose-400'}`}>
                                  {Math.round(recall * 100)}%
                                </span>
                                <div className="inline-block ml-2 w-16 h-1.5 rounded-full bg-zinc-800 align-middle">
                                  <div className={`h-full rounded-full ${recall === 1 ? 'bg-emerald-500' : recall >= 0.8 ? 'bg-amber-500' : 'bg-rose-500'}`} style={{ width: `${recall * 100}%` }} />
                                </div>
                              </td>
                              <td className="px-4 py-3 text-right text-xs text-zinc-500">
                                {topResult ? `#${topResult.idx + 1} (${topResult.sim.toFixed(3)})` : '—'}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {queryResults?.mrl && (
                  <IndexResultsList
                    results={queryResults.mrl.results_by_dims[String(mrlDims)]?.map(r => ({ idx: r.idx, sim: r.sim, text: chunks[r.idx]?.page_content ?? '' })) ?? []}
                    label={`Results at ${mrlDims}d (recall ${Math.round((queryResults.mrl.recall[String(mrlDims)] ?? 0) * 100)}% vs 768d)`}
                    color="amber"
                  />
                )}

                {!queryResults && <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-10 text-center text-sm text-zinc-600">Enter a query to see how recall changes as you truncate dimensions</div>}
              </div>
            )}

          </main>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          STAGE 4 — RETRIEVAL
      ══════════════════════════════════════════════════════════════════════ */}
      {embedResult && (
        <div ref={retrieveSectionRef} className="border-t-2 border-zinc-800 mt-2">
          <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

            {/* Stage header */}
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-md bg-violet-600 flex items-center justify-center text-xs font-bold">4</div>
              <h2 className="text-lg font-semibold tracking-tight">Retrieval</h2>
              <InfoTooltip
                title="What is retrieval?"
                body={"Retrieval is the step where your query meets your chunks.\n\nThe query is embedded with the same model used for chunks, then the index is searched for the most similar vectors. But 'most similar' depends on which strategy you use:\n\n• Dense — cosine similarity on embeddings. Great for semantic matches ('what causes fever' → finds 'pyrexia treatment' even with no shared words).\n• Sparse (BM25) — keyword frequency scoring, no embeddings. Great for exact matches: names, codes, rare terms.\n• Hybrid (RRF) — merge both ranked lists. Best of both worlds in practice.\n• Re-ranked — take the top-20 from hybrid, run a cross-encoder that reads both query and chunk together. Much more accurate but slower.\n\nThe improvement from left to right on this page is the standard production RAG stack."}
                pos="top-right"
              />
              <span className="ml-auto text-xs text-zinc-600 font-mono">{embedResult.coords_2d.length} chunks</span>
            </div>

            {/* Query input */}
            <div className="flex gap-3">
              <input
                value={retrieveQuery}
                onChange={e => setRetrieveQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleRetrieve()}
                placeholder="Type a query to compare all four retrieval strategies…"
                className="flex-1 rounded-xl bg-zinc-900 border border-zinc-700 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-violet-600"
              />
              <button onClick={handleRetrieve} disabled={retrieveLoading || !retrieveQuery.trim()}
                className="px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-sm font-semibold transition-colors whitespace-nowrap">
                {retrieveLoading ? 'Retrieving…' : 'Retrieve →'}
              </button>
            </div>
            {retrieveError && <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">{retrieveError}</div>}
            {retrieveLoading && (
              <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-10 text-center space-y-2">
                <p className="text-sm text-zinc-400">Running all four strategies…</p>
                <p className="text-xs text-zinc-600">First run downloads the cross-encoder model (~80 MB). Subsequent runs are fast.</p>
              </div>
            )}

            {!retrieveResult && !retrieveLoading && (
              <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-10 text-center text-sm text-zinc-600">
                Enter a query to compare Dense · BM25 · Hybrid · Re-ranked side by side
              </div>
            )}

            {retrieveResult && (
              <div className="space-y-6">

                {/* Pipeline flow banner */}
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-3.5 flex items-center gap-2 text-xs flex-wrap">
                  <span className="text-zinc-500 font-medium uppercase tracking-wider">Pipeline</span>
                  <span className="text-zinc-700 mx-1">—</span>
                  <span className="px-2 py-0.5 rounded-md bg-sky-500/10 border border-sky-500/20 text-sky-400 font-semibold">Dense</span>
                  <span className="text-zinc-700">+</span>
                  <span className="px-2 py-0.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-semibold">Sparse</span>
                  <span className="text-zinc-600 text-[10px] italic">find candidates independently</span>
                  <span className="text-zinc-700 mx-1">→</span>
                  <span className="px-2 py-0.5 rounded-md bg-violet-500/10 border border-violet-500/20 text-violet-400 font-semibold">Hybrid RRF</span>
                  <span className="text-zinc-600 text-[10px] italic">merge both ranked lists</span>
                  <span className="text-zinc-700 mx-1">→</span>
                  <span className="px-2 py-0.5 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-400 font-semibold">Re-rank</span>
                  <span className="text-zinc-600 text-[10px] italic">score top-20 more carefully</span>
                  <InfoTooltip
                    title="Two-phase pipeline"
                    body={"Retrieval and re-ranking are different jobs.\n\nPHASE 1 — RETRIEVAL: find candidate chunks fast.\n• Dense: embed query → cosine similarity against all chunk vectors. Finds semantic matches ('fever' → 'pyrexia') even with no shared words.\n• Sparse (BM25): no embeddings, pure keyword frequency. Finds exact term matches that dense might miss.\n• Hybrid (RRF): merge both ranked lists. Chunks that rank high in both get boosted. This is the production standard.\n\nPHASE 2 — RE-RANKING: take the top-20 candidates from hybrid and score them more accurately.\n• Cross-encoder: reads the query and chunk *together* in one forward pass. Far more accurate than cosine between two separate vectors, but too slow to run on the whole corpus — only used on the small candidate set.\n\nThe output score from cross-encoder is not cosine — it's a learned relevance logit from a model trained on (query, relevant chunk) / (query, irrelevant chunk) pairs."}
                    pos="top-left"
                  />
                </div>

                {/* Tab bar */}
                <div className="flex items-center gap-1 rounded-xl bg-zinc-900 border border-zinc-800 p-1 w-fit">
                  {([
                    ['results',   'Results'],
                    ['shifts',    'Rank Shifts'],
                    ['reranking', 'Re-ranking ✦'],
                  ] as [typeof retrieveTab, string][]).map(([id, label]) => (
                    <button key={id} onClick={() => setRetrieveTab(id)}
                      className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors ${retrieveTab === id ? 'bg-violet-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}>
                      {label}
                    </button>
                  ))}
                </div>

                {/* ── RESULTS TAB ── */}
                {retrieveTab === 'results' && (
                  <div className="space-y-3">
                    {/* Phase 1 label */}
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-600">Phase 1 — Retrieval</span>
                      <div className="flex-1 h-px bg-zinc-800" />
                      <span className="text-[10px] text-zinc-700">find candidates from the full corpus</span>
                    </div>

                    <div className="grid grid-cols-3 gap-3">
                      {([
                        { key: 'dense',  label: 'Dense',  tagline: 'semantic vector search', sublabel: 'cosine similarity', color: 'sky',     results: retrieveResult.dense,  scoreLabel: 'cos',
                          tip: 'RETRIEVAL — searches the whole corpus.\n\nEmbeds the query with the same model used for chunks, then computes cosine similarity against every chunk vector. Finds semantically similar chunks even when no words overlap ("fever" → "pyrexia").\n\n"Dense" means the vectors are dense — every dimension carries information. Cosine is just the most common distance function; for unit-normalised vectors (which sentence-transformers always produces) cosine and dot product give identical results.\n\nStrength: semantic matching, paraphrase handling.\nWeakness: struggles with exact keywords, rare terms, proper names.' },
                        { key: 'sparse', label: 'Sparse', tagline: 'exact keyword matching',  sublabel: 'BM25',              color: 'emerald', results: retrieveResult.sparse, scoreLabel: 'BM25',
                          tip: 'RETRIEVAL — searches the whole corpus.\n\nBM25 — no embeddings, no neural network. Scores chunks by how often your query terms appear, weighted by how rare they are globally (rare terms score higher).\n\nStrength: exact keyword matching — names, codes, rare terms, anything that needs to match literally.\nWeakness: misses synonyms and paraphrases entirely. "Fever" will not match "pyrexia".' },
                        { key: 'hybrid', label: 'Hybrid', tagline: 'best of both via RRF',    sublabel: 'RRF fusion',        color: 'violet',  results: retrieveResult.hybrid, scoreLabel: 'RRF',
                          tip: 'RETRIEVAL — merges dense and sparse results.\n\nReciprocal Rank Fusion: each chunk scores 1/(60 + rank) from the dense list and 1/(60 + rank) from the sparse list. Chunks that rank high in both get the highest combined score.\n\nThe raw cosine and BM25 values are thrown away — only rank position matters. This sidesteps the problem that cosine (0–1) and BM25 (unbounded) live on incompatible scales.\n\nWhy reciprocal? Inverse relationship: rank 1 should score more than rank 2, which more than rank 3. Why 60? Flattens the curve so rank 1 vs 2 don\'t differ wildly — 10–100 all behave similarly.\n\nThis is the production standard — outperforms either strategy alone on most queries.' },
                      ] as const).map(({ key, label, tagline, sublabel, color, results, scoreLabel, tip }) => {
                        const ring: Record<string, string> = { sky: 'border-sky-500/30', emerald: 'border-emerald-500/30', violet: 'border-violet-500/30' }
                        const hdr: Record<string, string>  = { sky: 'text-sky-400', emerald: 'text-emerald-400', violet: 'text-violet-400' }
                        const dot: Record<string, string>  = { sky: 'bg-sky-500/60', emerald: 'bg-emerald-500/60', violet: 'bg-violet-500/60' }
                        return (
                          <div key={key} className={`rounded-xl bg-zinc-900 border ${ring[color]} p-3 space-y-2`}>
                            <div className="flex items-start gap-1.5">
                              <div className={`w-2 h-2 rounded-full mt-1 shrink-0 ${dot[color]}`} />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-1">
                                  <span className={`text-xs font-bold uppercase tracking-wide ${hdr[color]}`}>{label}</span>
                                  <InfoTooltip title={label} body={tip} />
                                </div>
                                <p className="text-[10px] text-zinc-600 mt-0.5">{tagline}</p>
                              </div>
                              <span className="text-[10px] text-zinc-700 font-mono shrink-0">{sublabel}</span>
                            </div>
                            {results.map((r, rank) => (
                              <div key={r.idx} className="rounded-lg bg-zinc-800/60 px-2.5 py-2 space-y-1">
                                <div className="flex items-center gap-1.5">
                                  <span className={`text-[10px] font-bold font-mono ${hdr[color]}`}>#{rank + 1}</span>
                                  <span className="text-[10px] text-zinc-600 ml-auto font-mono">{scoreLabel} {r.score.toFixed(3)}</span>
                                </div>
                                <p className="text-xs text-zinc-400 leading-relaxed line-clamp-3">{r.text}</p>
                              </div>
                            ))}
                          </div>
                        )
                      })}
                    </div>

                    {/* Phase 2 label */}
                    <div className="flex items-center gap-2 pt-2">
                      <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-600">Phase 2 — Re-ranking</span>
                      <div className="flex-1 h-px bg-zinc-800" />
                      <span className="text-[10px] text-zinc-700">score top-20 hybrid candidates more accurately</span>
                    </div>

                    <div className="grid grid-cols-1 gap-3">
                      {([
                        { key: 'reranked', label: 'Re-ranked', tagline: 'cross-encoder reads query + chunk together', sublabel: 'cross-encoder', color: 'amber', results: retrieveResult.reranked, scoreLabel: 'CE',
                          tip: 'RE-RANKING — runs on the top-20 hybrid candidates only, not the whole corpus.\n\nDense retrieval encodes query and chunk separately, then compares two fixed vectors with cosine. The query and chunk never see each other during encoding.\n\nA cross-encoder takes them together as one input:\n[query] [SEP] [chunk] → full transformer forward pass → one relevance score\n\n[SEP] is a special separator token that tells the model "query ends here, chunk begins here." Every query token attends to every chunk token through the full attention mechanism — it can find that "caused" in the chunk directly answers "what causes" in the query.\n\nThe output score is not cosine — it\'s a learned relevance number from a model trained on (query, relevant chunk) vs (query, irrelevant chunk) pairs. Only use it for ranking, not as an absolute value.\n\nThe tradeoff: you cannot pre-compute cross-encoder scores. Every query requires a fresh forward pass per candidate. This is why it only runs on a small candidate set.' },
                      ] as const).map(({ key, label, tagline, sublabel, color, results, scoreLabel, tip }) => (
                        <div key={key} className="rounded-xl bg-zinc-900 border border-amber-500/30 p-3 space-y-2">
                          <div className="flex items-start gap-1.5">
                            <div className="w-2 h-2 rounded-full mt-1 shrink-0 bg-amber-500/60" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1">
                                <span className="text-xs font-bold uppercase tracking-wide text-amber-400">{label}</span>
                                <InfoTooltip title={label} body={tip} />
                              </div>
                              <p className="text-[10px] text-zinc-600 mt-0.5">{tagline}</p>
                            </div>
                            <span className="text-[10px] text-zinc-700 font-mono shrink-0">{sublabel}</span>
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            {results.map((r, rank) => (
                              <div key={r.idx} className="rounded-lg bg-zinc-800/60 px-2.5 py-2 space-y-1">
                                <div className="flex items-center gap-1.5">
                                  <span className="text-[10px] font-bold font-mono text-amber-400">#{rank + 1}</span>
                                  <span className="text-[10px] text-zinc-600 ml-auto font-mono">{scoreLabel} {r.score.toFixed(3)}</span>
                                </div>
                                <p className="text-xs text-zinc-400 leading-relaxed line-clamp-3">{r.text}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── RANK SHIFTS TAB ── */}
                {retrieveTab === 'shifts' && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Rank shifts — same chunks, four strategies</p>
                      <InfoTooltip
                        title="How to read rank shifts"
                        body={"Each row is a chunk that appeared in at least one strategy's top-5.\n\nThe columns show its rank in each strategy (1 = best). A dash means it didn't make the top-5 for that strategy.\n\nLook for big jumps: a chunk ranked #4 in dense that jumps to #1 in re-ranked means the cross-encoder found a strong query-chunk relationship that cosine similarity alone missed.\n\nHybrid often bridges the gap between dense (semantic) and sparse (keyword), and re-ranking makes the final call."}
                        pos="top-right"
                      />
                    </div>
                    <div className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
                      <table className="w-full text-xs">
                        <thead>
                          {/* Phase group headers */}
                          <tr className="border-b border-zinc-800/50">
                            <th className="px-4 py-2" />
                            <th colSpan={3} className="text-center px-3 py-2 text-zinc-600 text-[10px] uppercase tracking-widest font-normal border-l border-zinc-800">
                              Phase 1 — Retrieval
                            </th>
                            <th colSpan={2} className="text-center px-3 py-2 text-amber-600/70 text-[10px] uppercase tracking-widest font-normal border-l border-zinc-800">
                              Phase 2 — Re-ranking
                            </th>
                          </tr>
                          {/* Column headers */}
                          <tr className="border-b border-zinc-800 text-zinc-500 uppercase tracking-wider">
                            <th className="text-left px-4 py-2.5">Chunk</th>
                            <th className="text-center px-3 py-2.5 text-sky-500/70 border-l border-zinc-800">Dense</th>
                            <th className="text-center px-3 py-2.5 text-emerald-500/70">Sparse</th>
                            <th className="text-center px-3 py-2.5 text-violet-500/70">Hybrid</th>
                            <th className="text-center px-3 py-2.5 text-amber-500/70 border-l border-zinc-800">Cross-enc</th>
                            <th className="text-center px-3 py-2.5 text-amber-300/70">ColBERT</th>
                          </tr>
                        </thead>
                        <tbody>
                          {retrieveResult.rank_shifts.map((s) => {
                            const rankCell = (r: number | null, col: string, borderLeft = false) => {
                              if (r === null) return <td key={col} className={`text-center px-3 py-2.5 text-zinc-700 ${borderLeft ? 'border-l border-zinc-800' : ''}`}>—</td>
                              const bg = r === 1 ? 'bg-amber-500/20 text-amber-300 font-bold' : r <= 3 ? 'text-zinc-300' : 'text-zinc-500'
                              return <td key={col} className={`text-center px-3 py-2.5 font-mono ${bg} ${borderLeft ? 'border-l border-zinc-800' : ''}`}>#{r}</td>
                            }
                            return (
                              <tr key={s.idx} className="border-b border-zinc-800/50 hover:bg-zinc-800/20 transition-colors">
                                <td className="px-4 py-2.5 text-zinc-400 max-w-xs">
                                  <span className="text-zinc-600 font-mono mr-2">#{s.idx + 1}</span>
                                  <span className="line-clamp-1">{s.text}</span>
                                </td>
                                {rankCell(s.dense_rank,     'dense',     true)}
                                {rankCell(s.sparse_rank,    'sparse'        )}
                                {rankCell(s.hybrid_rank,    'hybrid'        )}
                                {rankCell(s.reranked_rank,  'reranked',  true)}
                                {rankCell(s.colbert_rank,   'colbert'       )}
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* ── RE-RANKING TAB ── */}
                {retrieveTab === 'reranking' && (
                  <div className="space-y-6">

                    {/* Cross-encoder section */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-amber-500/60" />
                        <p className="text-xs font-bold uppercase tracking-wide text-amber-400">Cross-encoder scoring</p>
                        <InfoTooltip
                          title="How the cross-encoder re-ranks"
                          body={"After hybrid retrieval finds the top-20 candidates, each one is passed to the cross-encoder with the query.\n\nDense retrieval encoded query and chunk separately, then compared two fixed vectors. The query and chunk never saw each other.\n\nThe cross-encoder takes them together:\n[query] [SEP] [chunk] → full transformer forward pass → one relevance score\n\n[SEP] is a separator token. The model reads both sides with full attention — every query token can attend to every chunk token. It can find that 'caused by' in the chunk directly answers 'what causes' in the query.\n\nThe output score is a learned relevance logit — not cosine. Only use it for ranking, not as an absolute value.\n\nLook for rank shifts: a chunk that hybrid ranked #4 but cross-encoder moved to #1 means cosine similarity alone underestimated how well it answers the query."}
                          pos="top-right"
                        />
                        <span className="text-[10px] text-zinc-600 ml-2">scores top-20 hybrid candidates — shows top {retrieveResult.reranked.length}</span>
                      </div>
                      <div className="rounded-xl bg-zinc-900 border border-amber-500/20 overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-zinc-800/50">
                              <th className="px-4 py-1.5" />
                              <th className="text-center px-3 py-1.5 text-zinc-600 text-[10px] uppercase tracking-widest font-normal border-l border-zinc-800" colSpan={1}>Retrieval</th>
                              <th className="px-1" />
                              <th colSpan={2} className="text-center px-3 py-1.5 text-amber-600/70 text-[10px] uppercase tracking-widest font-normal border-l border-zinc-800">Re-ranking</th>
                              <th className="px-4 py-1.5" />
                            </tr>
                            <tr className="border-b border-zinc-800 text-zinc-500 uppercase tracking-wider">
                              <th className="text-left px-4 py-2.5">Chunk</th>
                              <th className="text-center px-3 py-2.5 text-violet-500/70 border-l border-zinc-800">Hybrid</th>
                              <th className="text-center px-2 py-2.5">→</th>
                              <th className="text-center px-3 py-2.5 text-amber-500/70 border-l border-zinc-800">Cross-enc</th>
                              <th className="text-center px-3 py-2.5 text-amber-300/70">ColBERT</th>
                              <th className="text-right px-4 py-2.5 text-zinc-600">CE score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {retrieveResult.reranked.map((r, newRank) => {
                              const shift = retrieveResult.rank_shifts.find(s => s.idx === r.idx)
                              const hybridRank = shift?.hybrid_rank ?? null
                              const colbertRank = shift?.colbert_rank ?? null
                              const ceMoved = hybridRank !== null ? hybridRank - (newRank + 1) : null
                              const cbMoved = hybridRank !== null && colbertRank !== null ? hybridRank - colbertRank : null
                              const shiftBadge = (moved: number | null) => {
                                if (moved === null) return <span className="text-zinc-600">—</span>
                                if (moved > 0)  return <span className="text-emerald-400">↑{moved}</span>
                                if (moved < 0)  return <span className="text-rose-400">↓{Math.abs(moved)}</span>
                                return <span className="text-zinc-600">=</span>
                              }
                              return (
                                <tr key={r.idx} className="border-b border-zinc-800/50 hover:bg-zinc-800/20 transition-colors">
                                  <td className="px-4 py-2.5 text-zinc-400 max-w-xs">
                                    <span className="text-zinc-600 font-mono mr-2">#{r.idx + 1}</span>
                                    <span className="line-clamp-1">{r.text}</span>
                                  </td>
                                  <td className="text-center px-3 py-2.5 font-mono text-violet-400 border-l border-zinc-800">
                                    {hybridRank !== null ? `#${hybridRank}` : '—'}
                                  </td>
                                  <td className="text-center px-2 py-2.5 text-zinc-700">→</td>
                                  <td className="text-center px-3 py-2.5 border-l border-zinc-800">
                                    <span className="font-mono font-bold text-amber-400">#{newRank + 1}</span>
                                    <span className="ml-1.5 text-[10px]">{shiftBadge(ceMoved)}</span>
                                  </td>
                                  <td className="text-center px-3 py-2.5">
                                    {colbertRank !== null
                                      ? <><span className="font-mono font-bold text-amber-300">#{colbertRank}</span>
                                          <span className="ml-1.5 text-[10px]">{shiftBadge(cbMoved)}</span></>
                                      : <span className="text-zinc-600">—</span>}
                                  </td>
                                  <td className="text-right px-4 py-2.5 font-mono text-zinc-500">
                                    {r.score.toFixed(3)}
                                  </td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* ColBERT section */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-amber-300/60" />
                        <p className="text-xs font-bold uppercase tracking-wide text-amber-300">ColBERT late interaction</p>
                        <InfoTooltip
                          title="ColBERT — an alternative re-ranking approach"
                          body={"Both cross-encoder and ColBERT are re-ranking methods — they score a small candidate set more carefully than dense retrieval. Different mechanisms, same job.\n\nCross-encoder reads query + chunk together in one pass. Nothing pre-computable.\n\nColBERT keeps one vector per chunk token (pre-computed). At query time, for each query token, find the chunk token most similar to it (MaxSim). Sum those per-token scores → ColBERT score.\n\nWhy 'late interaction'? Query and chunk are encoded separately. The interaction — matching query tokens to chunk tokens — happens late, at score time only.\n\nThe heatmap shows the (query tokens × chunk tokens) similarity matrix for the top-ranked chunk. Gold outline on each row = the MaxSim winner for that query token, the score that gets added to the total. The numbers on the right sum to the final ColBERT score.\n\nStorage cost: ~100 tokens × 128 dims per chunk vs 384 dims for dense — roughly 33× more. This is ColBERT's main production drawback."}
                          pos="top-right"
                        />
                        <span className="text-[10px] text-zinc-600 ml-2">shown for top re-ranked chunk only</span>
                      </div>
                      {retrieveResult.colbert ? (
                        <ColBERTHeatmap data={retrieveResult.colbert} topChunk={retrieveResult.reranked[0]} />
                      ) : (
                        <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-10 text-center text-sm text-zinc-600">
                          ColBERT token embeddings are not available for this model configuration.
                        </div>
                      )}
                    </div>

                  </div>
                )}
              </div>
            )}

            {/* ── Generate CTA — appears once retrieval result exists ── */}
            {retrieveResult && (
              <div className="flex items-center gap-4 rounded-xl border border-zinc-800 bg-zinc-900/60 px-5 py-4">
                <div className="flex-1">
                  <p className="text-sm font-semibold text-zinc-200">Ready to generate</p>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    {retrieveResult.reranked.length} re-ranked chunks ready — send them to an LLM and see the answer grounded in your retrieved context.
                  </p>
                </div>
                <button
                  onClick={handleGenerate}
                  disabled={generateLoading}
                  className="px-5 py-2 rounded-lg text-sm font-semibold transition-colors bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white whitespace-nowrap"
                >
                  {generateLoading ? 'Generating…' : 'Generate Answer →'}
                </button>
              </div>
            )}

          </main>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          STAGE 5 — GENERATE
      ══════════════════════════════════════════════════════════════════════ */}
      {retrieveResult && (
        <div ref={generateSectionRef} className="border-t-2 border-zinc-800 mt-2">
          <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

            {/* Stage header */}
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-md bg-violet-600 flex items-center justify-center text-xs font-bold">5</div>
              <h2 className="text-lg font-semibold tracking-tight">Generate</h2>
              <div className="flex items-center gap-1.5 ml-1">
                <InfoTooltip
                  title="What happens in the generate stage?"
                  body={"The retrieved and re-ranked chunks get assembled into a prompt and sent to a large language model.\n\nThis is the G in RAG. Everything before this was retrieval — finding the right context. Generation is where that context is used to produce an answer.\n\nThe key variables:\n• Which model you use (context window, cost, reasoning quality)\n• How you compact the chunks before sending (raw vs compressed)\n• The order you place chunks in the prompt (Lost-in-the-Middle effect)\n\nAfter the answer comes back, each sentence is scored for grounding — how well it's supported by the retrieved context vs. generated from the model's training weights."}
                  pos="top-right"
                />
              </div>
              <span className="ml-auto text-xs text-zinc-600 font-mono">{retrieveResult.reranked.length} chunks from stage 4</span>
            </div>

            {/* ── Model picker ── */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">LLM Model</p>
                <InfoTooltip
                  title="How the model choice affects RAG"
                  body={"The model determines three things:\n\n1. Context window — how many tokens you can send. Gemini and Claude offer 200k; OpenAI caps at 128k. This directly limits how many chunks (and how long) you can include.\n\n2. Cost — billed per token in and out. A RAG prompt with 5 chunks easily hits 1000 input tokens. At scale, model choice dominates your bill.\n\n3. Reasoning quality — some models better at faithfully synthesising across multiple chunks, or at flagging when the context is insufficient.\n\nFor most RAG workloads: start with GPT-4o-mini or Haiku (cheap, fast). Upgrade to GPT-4o or Sonnet only if answer quality suffers."}
                  pos="top-right"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {LLM_MODELS.map(m => (
                  <button key={m.id} onClick={() => setGenModel(m)}
                    className={`rounded-xl border px-3 py-3 text-left transition-all ${genModel.id === m.id ? 'border-violet-500 bg-violet-500/10 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600'}`}>
                    <div className="flex items-start justify-between mb-1">
                      <p className="text-sm font-semibold leading-tight">{m.label}</p>
                      <InfoTooltip title={m.label} body={m.description} pos="top-left" />
                    </div>
                    <p className="text-xs text-zinc-500 mb-1.5">{m.tagline}</p>
                    <div className="flex gap-2 text-xs">
                      <span className="text-zinc-600">{(m.contextWindow / 1000).toFixed(0)}k ctx</span>
                      <span className="text-zinc-600">${m.inputPricePerM}/M in</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* ── API Key ── */}
            <div className="space-y-2">
              {genModel.provider === 'ollama' ? (
                <div className="flex items-center gap-3 rounded-xl border border-emerald-800/40 bg-emerald-950/20 px-4 py-3">
                  <span className="text-emerald-400 text-sm">✓</span>
                  <div>
                    <p className="text-sm text-emerald-300 font-medium">No API key needed</p>
                    <p className="text-xs text-zinc-500 mt-0.5">Ollama runs locally — make sure it's running and the model is pulled (<span className="font-mono">ollama pull {genModel.id}</span>)</p>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                      {genModel.provider === 'openai' ? 'OpenAI' : genModel.provider === 'groq' ? 'Groq' : 'Anthropic'} API Key
                    </p>
                    <InfoTooltip
                      title="API key"
                      body={`Your key is used for exactly one API call and never stored or logged.\n\n${genModel.provider === 'openai' ? 'Get an OpenAI key at platform.openai.com → API keys.' : genModel.provider === 'groq' ? 'Get a free Groq key at console.groq.com → API Keys. Groq has a generous free tier with no billing required.' : 'Get an Anthropic key at console.anthropic.com → API keys.'}`}
                    />
                  </div>
                  <input
                    type="password"
                    value={genApiKey}
                    onChange={e => { setGenApiKey(e.target.value); if (e.target.value.trim()) setApiKeyMissing(false) }}
                    placeholder={genModel.provider === 'groq' ? 'gsk_... (Groq API key — free at console.groq.com)' : `sk-... (${genModel.provider === 'openai' ? 'OpenAI' : 'Anthropic'} API key)`}
                    className={`w-full max-w-md rounded-xl bg-zinc-900 border px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 ${apiKeyMissing ? 'border-red-500 ring-2 ring-red-500/40 focus:ring-red-500' : 'border-zinc-700 focus:ring-violet-600'}`}
                  />
                  {apiKeyMissing && <p className="text-xs text-red-400 mt-1">API key is required to generate an answer</p>}
                </>
              )}
            </div>

            {/* ── Compaction ── */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Compaction</p>
                <InfoTooltip
                  title="Why compact before sending?"
                  body={"Retrieved chunks are noisy — a 500-token chunk might contain 400 tokens of setup, transitions, and padding around the 100 tokens you actually need.\n\nCompaction removes the irrelevant parts before sending to the LLM. The context window tank updates to show the before/after size.\n\nQuery-independence spectrum:\n• Raw — no inspection, fully query-blind\n• LLMLingua (2023) — drops low-perplexity tokens, still query-blind\n• Contextual — sentence-level cosine similarity to the query, query-aware\n• LLMLingua-2 (2024) — token-level classifier conditioned on the query, used in Microsoft Copilot\n• RECOMP — rewrites each chunk as a query-focused summary, most query-aware\n\nQuery-aware compaction generally produces better answers but costs more compute."}
                  pos="top-right"
                />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {COMPACTION_ALGOS.map(algo => (
                  <button key={algo.id}
                    onClick={() => algo.available && setGenCompaction(algo.id)}
                    disabled={!algo.available}
                    className={`rounded-xl border px-3 py-3 text-left transition-all relative ${
                      !algo.available ? 'opacity-40 cursor-not-allowed border-zinc-800 bg-zinc-900' :
                      genCompaction === algo.id ? 'border-violet-500 bg-violet-500/10 text-zinc-100' :
                      'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600'
                    }`}>
                    <div className="flex items-start justify-between mb-1">
                      <p className="text-sm font-semibold leading-tight">{algo.label}</p>
                      <InfoTooltip title={algo.label} body={algo.description} pos="top-left" />
                    </div>
                    <p className="text-xs text-zinc-500">{algo.tagline}</p>
                    {!algo.available && (
                      <span className="absolute top-1.5 right-1.5 text-[9px] text-zinc-600 bg-zinc-800 rounded px-1 py-0.5">soon</span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* ── Chunk ordering ── */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Chunk Ordering</p>
                <InfoTooltip
                  title="Lost-in-the-Middle (Liu et al. 2023)"
                  body={"LLMs have a primacy and recency bias — they recall information near the start and end of their context window most reliably. Content in the middle is statistically underweighted.\n\nThis is a direct RAG problem: if your most relevant chunk sits at position 5 of 10, the model produces a lower-quality answer than if the same chunk was at position 1 or 10.\n\nOrdering strategies let you exploit these biases rather than fight them:\n• Relevance ↓ — naive but common\n• Relevance ↑ — exploits recency, counter-intuitive but works\n• Sandwich — exploits both ends simultaneously, recommended for production\n\nSandwich ordering is used by Perplexity and Bing. The paper showed ~20% accuracy drop for evidence buried in the middle of long contexts."}
                  pos="top-right"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {CHUNK_ORDERS.map(ord => (
                  <button key={ord.id} onClick={() => setGenChunkOrder(ord.id)}
                    className={`rounded-xl border px-3 py-3 text-left transition-all ${genChunkOrder === ord.id ? 'border-violet-500 bg-violet-500/10 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600'}`}>
                    <div className="flex items-start justify-between mb-1">
                      <p className="text-sm font-semibold leading-tight">{ord.label}</p>
                      <InfoTooltip title={ord.label} body={ord.description} pos="top-left" />
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* ── Context strategy ── */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Context Strategy</p>
                <InfoTooltip
                  title="How chunks are assembled and sent to the LLM"
                  body={"Stuffing is the default — send everything in one call. It works until your context window fills up.\n\nThe other strategies handle the overflow problem differently:\n\n• Map-Reduce: summarise each chunk independently, then combine — N+1 calls total\n• Refine: build the answer iteratively, one chunk at a time — N calls\n• Map-Rerank: generate a candidate answer per chunk, score each, pick the best — N calls\n\nEach strategy trades latency and cost (more LLM calls) for the ability to handle more context than any single window allows. At production scale, LangChain and LlamaIndex both expose these as first-class options."}
                  pos="top-right"
                />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {CONTEXT_STRATEGIES.map(strat => (
                  <button key={strat.id}
                    onClick={() => strat.available && setGenContextStrategy(strat.id)}
                    disabled={!strat.available}
                    className={`rounded-xl border px-3 py-3 text-left transition-all relative ${
                      !strat.available ? 'opacity-40 cursor-not-allowed border-zinc-800 bg-zinc-900' :
                      genContextStrategy === strat.id ? 'border-violet-500 bg-violet-500/10 text-zinc-100' :
                      'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600'
                    }`}>
                    <div className="flex items-start justify-between mb-1">
                      <p className="text-sm font-semibold leading-tight">{strat.label}</p>
                      <InfoTooltip title={strat.label} body={strat.description} pos="top-left" />
                    </div>
                    <p className="text-xs text-zinc-500 mb-1">{strat.tagline}</p>
                    <span className={`text-[10px] font-mono ${strat.available ? 'text-violet-400/70' : 'text-zinc-600'}`}>{strat.calls}</span>
                    {!strat.available && (
                      <span className="absolute top-1.5 right-1.5 text-[9px] text-zinc-600 bg-zinc-800 rounded px-1 py-0.5">soon</span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* ── Context window preview ── */}
            <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5 space-y-4">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Context Window</p>
                <InfoTooltip
                  title="The context window tax"
                  body={"Every token in this prompt costs money and occupies space. The tank shows what's actually being sent to the model.\n\nSystem prompt: your instructions to the model (fixed overhead).\nChunks: the retrieved context (the RAG payload).\nQuery: the user's question.\nOutput reserve: tokens the model needs to generate the answer — this headroom must exist or the answer gets cut off.\n\nWatch the tank fill and shrink as you change compaction algorithms. Switching from Raw to Contextual should visibly shrink the chunk sections."}
                  pos="top-right"
                />
                <span className="ml-auto text-xs text-zinc-600 font-mono">{genModel.label} · {(genModel.contextWindow / 1000).toFixed(0)}k token budget</span>
              </div>

              {/* Live preview using estimated token counts */}
              {(() => {
                const previewSections: PromptSection[] = [
                  { label: 'System prompt', text: '', tokens: 42, role: 'system', chunk_idx: null, original_tokens: null },
                  ...retrieveResult.reranked.map((r, i) => ({
                    label: `Chunk #${r.idx + 1}`,
                    text: r.text,
                    tokens: Math.max(1, Math.round(r.text.length / 4)),
                    role: 'chunk' as const,
                    chunk_idx: r.idx,
                    original_tokens: null,
                  })),
                  { label: 'User query', text: retrieveQuery, tokens: Math.max(1, Math.round(retrieveQuery.length / 4)), role: 'query', chunk_idx: null, original_tokens: null },
                ]
                const displaySections = generateResult ? generateResult.sections : previewSections
                return <ContextWindowTank sections={displaySections} contextWindow={genModel.contextWindow} />
              })()}

              {generateResult && genCompaction !== 'raw' && (
                <div className="flex items-center gap-3 rounded-lg bg-zinc-800/50 px-3 py-2 text-xs">
                  <span className="text-zinc-400">Compaction ({genCompaction}):</span>
                  <span className="font-mono text-zinc-500 line-through">{generateResult.compaction_stats.original_tokens.toLocaleString()} tokens</span>
                  <span className="text-zinc-600">→</span>
                  <span className="font-mono text-emerald-400">{generateResult.compaction_stats.compressed_tokens.toLocaleString()} tokens</span>
                  <span className="text-zinc-500">({Math.round((1 - generateResult.compaction_stats.ratio) * 100)}% reduction)</span>
                </div>
              )}
            </div>

            {generateLoading && (
              <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-8 text-center text-sm text-zinc-500">
                Calling {genModel.label}…
              </div>
            )}

            {generateError && !generateLoading && (
              <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">
                {generateError}
              </div>
            )}

            {/* ── Answer + grounding ── */}
            {generateResult && !generateLoading && (
              <div className="space-y-4">

                {/* Answer with grounding overlay */}
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Answer</p>
                      <InfoTooltip
                        title="Hallucination grounding overlay"
                        body={"Each sentence in the answer is scored against the retrieved chunks using cosine similarity.\n\nGreen highlight = sentence is well-supported by the context (similarity ≥ 0.35). The model is drawing from your retrieved documents.\n\nYellow highlight = sentence is borderline — partial grounding. May mix context with model knowledge.\n\nNo highlight = sentence similarity is low — the model may be generating from its training weights rather than your documents. This is where hallucination risk is highest.\n\nThis is the trust signal enterprise RAG systems (Glean, Cohere, Vectara) show before returning answers to users. Most RAG demos skip it."}
                        pos="top-right"
                      />
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="flex items-center gap-1.5">
                        <span className="w-3 h-2.5 rounded-sm bg-emerald-500/40 border border-emerald-500/30 inline-block" />
                        grounded
                        <InfoTooltip title="Grounded" body={"Cosine similarity ≥ 0.35 between this sentence and the closest retrieved chunk.\n\nThe model is drawing from your documents — this sentence can be traced back to retrieved context.\n\nNote: grounded ≠ factually correct. If your chunks contained wrong information, a grounded sentence faithfully repeats it. Grounding tells you the source, not the truth."} pos="top-right" />
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="w-3 h-2.5 rounded-sm bg-amber-500/30 border border-amber-500/20 inline-block" />
                        borderline
                        <InfoTooltip title="Borderline" body={"Cosine similarity between 0.2 and 0.35.\n\nWeak overlap with retrieved context. The model may be paraphrasing loosely, mixing context with prior training knowledge, or the relevant chunk used different vocabulary.\n\nTreat with mild scepticism — partially supported but not strongly traceable to a specific chunk."} pos="top-right" />
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="w-3 h-2.5 rounded-sm bg-zinc-700/50 border border-zinc-700 inline-block" />
                        ungrounded
                        <InfoTooltip title="Ungrounded" body={"Cosine similarity below 0.2 — little to no overlap with any retrieved chunk.\n\nThe model is likely generating from its training weights rather than your documents. This is where hallucination risk is highest.\n\nNot always wrong: transitional phrases like 'Based on the information provided' score low because they\'re not semantically tied to any chunk. Context matters — ungrounded sentences in the middle of otherwise grounded answers are more suspicious than standalone connective phrases."} pos="top-right" />
                      </span>
                      <button onClick={handleGenerate} disabled={generateLoading}
                        className="ml-2 px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-zinc-400 hover:text-zinc-200 transition-colors text-xs">
                        ↺ Re-run
                      </button>
                    </div>
                  </div>

                  <p className="text-sm text-zinc-200 leading-relaxed">
                    {generateResult.grounding.map((g, i) => {
                      const bg = g.grounded ? 'bg-emerald-500/25 border-b border-emerald-500/40'
                        : g.max_similarity >= 0.2 ? 'bg-amber-500/20 border-b border-amber-500/30'
                        : ''
                      return (
                        <span key={i} title={`similarity: ${g.max_similarity.toFixed(3)}`}
                          className={`rounded-sm px-0.5 ${bg} transition-colors`}>
                          {g.sentence}{i < generateResult.grounding.length - 1 ? ' ' : ''}
                        </span>
                      )
                    })}
                  </p>

                  {/* Grounding summary */}
                  {(() => {
                    const total = generateResult.grounding.length
                    const grounded = generateResult.grounding.filter(g => g.grounded).length
                    const pct = total > 0 ? Math.round((grounded / total) * 100) : 0
                    return (
                      <div className="flex items-center gap-3 pt-2 border-t border-zinc-800">
                        <div className="flex-1 h-1.5 rounded-full bg-zinc-800 overflow-hidden">
                          <div className="h-full rounded-full bg-emerald-500/70 transition-all" style={{ width: `${pct}%` }} />
                        </div>
                        <span className={`text-xs font-mono font-semibold ${pct >= 75 ? 'text-emerald-400' : pct >= 50 ? 'text-amber-400' : 'text-rose-400'}`}>
                          {pct}% grounded
                        </span>
                        <span className="text-xs text-zinc-600">({grounded}/{total} sentences)</span>
                      </div>
                    )
                  })()}
                </div>

                {/* Cost + token breakdown */}
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-4">
                  <div className="flex items-center gap-2 mb-3">
                    <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Usage</p>
                    <InfoTooltip
                      title="Token cost breakdown"
                      body={"RAG is token-expensive. A typical retrieval prompt with 5 chunks easily sends 800–2000 input tokens per query.\n\nAt scale:\n• 1000 queries/day × 1000 tokens × $0.15/1M = $0.15/day on GPT-4o-mini\n• Same on GPT-4o = $2.50/day\n\nCompaction is the lever that reduces input token cost. The context window tank shows you exactly what you're paying for."}
                    />
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: 'Input tokens', value: generateResult.total_input_tokens.toLocaleString(), color: 'text-sky-400' },
                      { label: 'Output tokens', value: generateResult.total_output_tokens.toLocaleString(), color: 'text-violet-400' },
                      { label: 'Total tokens', value: (generateResult.total_input_tokens + generateResult.total_output_tokens).toLocaleString(), color: 'text-zinc-300' },
                      { label: 'Est. cost', value: generateResult.cost_usd < 0.0001 ? '<$0.0001' : `$${generateResult.cost_usd.toFixed(4)}`, color: 'text-emerald-400' },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="rounded-lg bg-zinc-800/50 px-3 py-2.5">
                        <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">{label}</p>
                        <p className={`text-lg font-mono font-bold ${color}`}>{value}</p>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            )}

            {/* ── Evaluate CTA — appears once answer exists ── */}
            {generateResult && !generateLoading && (
              <div className="flex items-center gap-4 rounded-xl border border-zinc-800 bg-zinc-900/60 px-5 py-4">
                <div className="flex-1">
                  <p className="text-sm font-semibold text-zinc-200">Ready to evaluate</p>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    Score the full pipeline with RAGAS-style metrics — faithfulness, answer relevancy, context precision, and more.
                  </p>
                </div>
                <button
                  onClick={handleEvaluate}
                  disabled={evalLoading}
                  className="px-5 py-2 rounded-lg text-sm font-semibold transition-colors bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white whitespace-nowrap"
                >
                  {evalLoading ? 'Evaluating…' : 'Evaluate Pipeline →'}
                </button>
              </div>
            )}

          </main>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          STAGE 6 — EVALUATE
      ══════════════════════════════════════════════════════════════════════ */}
      {generateResult && (
        <div ref={evalSectionRef} className="border-t-2 border-zinc-800 mt-2">
          <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

            {/* Stage header */}
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-md bg-emerald-600 flex items-center justify-center text-xs font-bold">6</div>
              <h2 className="text-lg font-semibold tracking-tight">Evaluate</h2>
              <div className="flex items-center gap-1.5 ml-1">
                <InfoTooltip
                  title="What the evaluation stage measures"
                  body={"This stage scores the full pipeline end-to-end using RAGAS-style metrics.\n\nFaithfulness — are the answer's claims actually supported by the retrieved context?\nAnswer Relevancy — does the answer address the original query?\nContext Precision — were the right chunks retrieved, and were they ranked first?\nContext Recall — was everything needed to answer correctly actually retrieved? (needs ground truth)\nNoise Sensitivity — what fraction of retrieved chunks actually contributed to the answer?\n\nAll metrics are computed with cosine similarity over embeddings — the same model used in Stage 2. The embedding-based approach is fast and free. Microsoft and Anthropic use LLM-as-judge (GPT-4 / Claude) for production evaluation, which is more accurate but costs tokens. See docs/eval101.md for the full comparison."}
                  pos="top-right"
                />
              </div>
              <span className="ml-auto text-xs text-zinc-600 font-mono">RAGAS-style · embedding-based</span>
            </div>

            {/* ── Optional ground truth ── */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Ground Truth Answer</p>
                <span className="text-[10px] text-zinc-600 border border-zinc-800 rounded px-1.5 py-0.5">optional</span>
                <InfoTooltip
                  title="Why ground truth unlocks Context Recall"
                  body={"Without ground truth, only three metrics are computable: Faithfulness, Answer Relevancy, and Context Precision. All three can be evaluated purely from the (query, answer, retrieved chunks) triple.\n\nContext Recall requires knowing what the correct answer should contain. It checks: 'of all the things the correct answer mentions, did we retrieve the context that would allow us to say them?'\n\nA low Context Recall score means your retriever missed key information — either the top-k was too small, the relevant chunks were ranked too low, or the embedding model failed to map the query to the right region.\n\nYou don't need a perfect answer — a rough summary of what a correct answer should cover is enough."}
                  pos="top-right"
                />
              </div>
              <textarea
                value={evalGroundTruth}
                onChange={e => setEvalGroundTruth(e.target.value)}
                placeholder="Optionally paste a reference answer here to unlock Context Recall scoring…"
                rows={2}
                className="w-full rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-200 placeholder-zinc-600 resize-none focus:outline-none focus:border-zinc-600"
              />
            </div>

            {/* ── Run button ── */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleEvaluate}
                disabled={evalLoading}
                className="px-6 py-2.5 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white transition-colors"
              >
                {evalLoading ? 'Evaluating…' : 'Run Evaluation'}
              </button>
              {evalError && <p className="text-sm text-red-400">{evalError}</p>}
            </div>

            {evalResult && !evalLoading && (
              <div className="space-y-6">

                {/* ── Metric score cards ── */}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                  {[
                    {
                      label: 'Faithfulness',
                      value: evalResult.faithfulness,
                      sub: `${evalResult.n_grounded}/${evalResult.n_sentences} sentences grounded`,
                      tooltipTitle: 'Faithfulness — are claims supported by context?',
                      tooltipBody: "Am I only saying things the documents actually back up?\n\n🎣 You caught 10 fish — but are they from the lake you gave the net, or did some sneak in from somewhere else?\n\nScore 1.0 = every sentence traces back to a retrieved chunk. Score 0.0 = the model made it all up from training memory.\n\nRAGAS: extracts atomic claims, verifies each is entailed by context using an LLM.\nThis lab: cosine similarity per sentence vs all chunks (threshold 0.35).\nMicrosoft Azure AI: 'Groundedness' — GPT-4 rates 1–5.\nAnthropic: Claude identifies unsupported claims and returns a JSON verdict with reasoning.\n\nLow faithfulness = hallucination. The most important metric for enterprise RAG.",
                      color: evalResult.faithfulness >= 0.7 ? 'emerald' : evalResult.faithfulness >= 0.4 ? 'amber' : 'red',
                    },
                    {
                      label: 'Answer Relevancy',
                      value: evalResult.answer_relevancy,
                      sub: 'cosine(query, answer)',
                      tooltipTitle: 'Answer Relevancy — does the answer address the query?',
                      tooltipBody: "Am I actually answering the question that was asked?\n\n🎣 You caught the right fish — but delivered them to the wrong person.\n\nYour answer can be 100% grounded in the documents and still miss the point. Example: you ask 'what is the capital of France?' and the model faithfully quotes 'Paris has a population of 2.1 million.' Faithful — yes. Relevant — no.\n\nHow it's measured: embed the query and the answer as vectors, then take the cosine similarity between them. If the answer talks about the same thing as the question, they'll point in the same direction in embedding space.\n\nRAGAS goes further: it generates a few questions from the answer using an LLM, then checks if those reverse-engineered questions look like your original query. The idea is — if someone read your answer and tried to guess what question prompted it, would they arrive at your question?\n\nLow relevancy = the model answered a different question than you asked.",
                      color: evalResult.answer_relevancy >= 0.7 ? 'emerald' : evalResult.answer_relevancy >= 0.4 ? 'amber' : 'red',
                    },
                    {
                      label: 'Context Precision',
                      value: evalResult.context_precision,
                      sub: `${evalResult.n_relevant_chunks}/${evalResult.n_chunks} chunks relevant`,
                      tooltipTitle: 'Context Precision — did we retrieve the right chunks, ranked first?',
                      tooltipBody: "Of the chunks I retrieved, how many were actually useful — and were the best ones first?\n\n🎣 Of the fish in your net, what fraction were the ones you wanted — and were the best ones at the top, not buried at the bottom?\n\nPrecision = quality of what you retrieved. A retriever returning 10 chunks where only 2 are relevant has low precision — 8 chunks are noise eating your context budget and confusing the model.\n\nThe 'ranked first' part matters: relevant chunks buried at rank 8 are less useful than ones at rank 1, because compaction and LLM attention both favour early context.\n\nFormula: RAGAS weighted precision@k — relevant chunks ranked higher contribute more to the score.\n\nFix low precision: reduce top-k, improve your embedding model, or add a reranker (Stage 4).",
                      color: evalResult.context_precision >= 0.7 ? 'emerald' : evalResult.context_precision >= 0.4 ? 'amber' : 'red',
                    },
                    {
                      label: 'Context Recall',
                      value: evalResult.context_recall,
                      sub: evalResult.context_recall === null ? 'needs ground truth' : `${evalResult.gt_sentence_scores.filter(s => s.supported).length}/${evalResult.gt_sentence_scores.length} GT sentences covered`,
                      tooltipTitle: 'Context Recall — did we retrieve everything needed?',
                      tooltipBody: "Did I miss anything the correct answer needed?\n\n🎣 Of all the fish you wanted from the lake, what fraction did you actually catch? You might have only caught the easy ones and missed half the important ones.\n\nRecall = coverage. Precision asks 'were the chunks I got relevant?' Recall asks 'did I miss anything?'\n\nLow recall = the retriever never fetched the right chunk — so the model couldn't possibly answer correctly. Not a hallucination problem, a retrieval gap.\n\nFix: increase top-k, re-check chunking boundaries (a fact split across two chunks won't match any query well), or try a different embedding model.\n\nRequires a reference answer — paste one in the ground truth field above to unlock this metric.",
                      color: evalResult.context_recall === null ? 'zinc' : evalResult.context_recall >= 0.7 ? 'emerald' : evalResult.context_recall >= 0.4 ? 'amber' : 'red',
                    },
                    {
                      label: 'Noise Sensitivity',
                      value: evalResult.noise_sensitivity,
                      sub: `${evalResult.n_contributing_chunks}/${evalResult.n_chunks} chunks used`,
                      tooltipTitle: 'Noise Sensitivity — how much of the context was actually used?',
                      tooltipBody: "How much of what I retrieved actually got used?\n\n🎣 You pulled up 10 fish, but only 2 made it to the dinner plate. The other 8 were bycatch — took up space in the boat and got thrown back.\n\nIf you retrieved 10 chunks and only 2 were cited by grounded sentences, 80% of your context budget was wasted. Those unused chunks cost tokens, can confuse the model, and increase hallucination risk.\n\nTruLens calls this 'context utilisation'. Fix: reduce top-k, use a cross-encoder reranker (Stage 4), or apply compaction (Stage 5) to strip irrelevant sentences before sending.\n\nNot a standard RAGAS metric but one of the most actionable in production.",
                      color: evalResult.noise_sensitivity >= 0.6 ? 'emerald' : evalResult.noise_sensitivity >= 0.3 ? 'amber' : 'red',
                    },
                  ].map(({ label, value, sub, tooltipTitle, tooltipBody, color }) => {
                    const pct = value === null ? null : Math.round((value as number) * 100)
                    const colorMap: Record<string, { ring: string; text: string; bar: string; bg: string }> = {
                      emerald: { ring: 'border-emerald-500/40', text: 'text-emerald-400', bar: 'bg-emerald-500', bg: 'bg-emerald-500/10' },
                      amber:   { ring: 'border-amber-500/40',   text: 'text-amber-400',   bar: 'bg-amber-500',   bg: 'bg-amber-500/10'   },
                      red:     { ring: 'border-red-500/40',     text: 'text-red-400',     bar: 'bg-red-500',     bg: 'bg-red-500/10'     },
                      zinc:    { ring: 'border-zinc-700',       text: 'text-zinc-500',    bar: 'bg-zinc-700',    bg: 'bg-zinc-800/50'    },
                    }
                    const c = colorMap[color]
                    return (
                      <div key={label} className={`rounded-xl border ${c.ring} ${c.bg} px-4 py-4 space-y-2`}>
                        <div className="flex items-start justify-between gap-1">
                          <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider leading-tight">{label}</p>
                          <InfoTooltip title={tooltipTitle} body={tooltipBody} pos="top-left" />
                        </div>
                        <p className={`text-2xl font-mono font-bold ${c.text}`}>
                          {pct === null ? '—' : `${pct}%`}
                        </p>
                        <div className="w-full h-1 rounded-full bg-zinc-800">
                          {pct !== null && <div className={`h-1 rounded-full ${c.bar}`} style={{ width: `${pct}%` }} />}
                        </div>
                        <p className="text-[10px] text-zinc-600 leading-tight">{sub}</p>
                      </div>
                    )
                  })}
                </div>

                {/* ── Radar chart ── */}
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-5">
                  <div className="flex items-center gap-2 mb-4">
                    <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Pipeline health — radar</p>
                    <InfoTooltip
                      title="How to read the radar chart"
                      body={"Each axis represents one evaluation metric. The outer edge = 1.0 (perfect). The shaded polygon shows your pipeline's current scores.\n\nA healthy RAG pipeline has a large, roughly symmetric polygon. Dents in specific axes diagnose specific problems:\n\n• Dent in Faithfulness → hallucination problem — model is generating beyond the retrieved context\n• Dent in Answer Relevancy → off-topic retrieval — wrong chunks are being retrieved\n• Dent in Context Precision → retrieval ranking problem — relevant chunks buried too low\n• Dent in Context Recall → coverage problem — retriever missed key information (check top-k)\n• Dent in Noise Sensitivity → too many irrelevant chunks — reduce top-k or improve reranker"}
                      pos="top-right"
                    />
                  </div>
                  <EvalRadarChart result={evalResult} />
                </div>

                {/* ── Sentence-level faithfulness breakdown ── */}
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Faithfulness — sentence breakdown</p>
                    <InfoTooltip
                      title="How each answer sentence scored"
                      body={"Each sentence in the generated answer is scored against all retrieved chunks. Max cosine similarity across all chunks determines whether the sentence is considered grounded (≥ 0.35) or ungrounded.\n\nGreen = grounded (the model is drawing from your documents)\nRed = ungrounded (the model may be generating from training weights)\n\nThis is the same calculation used by the Stage 5 grounding overlay. Here you see the raw similarity scores, not just the highlight colours.\n\nMicrosoft Azure AI: uses GPT-4 to decide whether each claim is entailed by context — more accurate than cosine for paraphrased facts.\nAnthropic: Claude returns a JSON list of unsupported claims with reasoning traces."}
                      pos="top-right"
                    />
                  </div>
                  <div className="space-y-2">
                    {evalResult.sentence_scores.map((s, i) => (
                      <div key={i} className={`rounded-lg border px-4 py-3 ${s.grounded ? 'border-emerald-800/40 bg-emerald-950/20' : 'border-red-800/40 bg-red-950/20'}`}>
                        <div className="flex items-start justify-between gap-3">
                          <p className="text-sm text-zinc-300 leading-relaxed flex-1">{s.sentence}</p>
                          <div className="shrink-0 text-right">
                            <p className={`text-sm font-mono font-bold ${s.grounded ? 'text-emerald-400' : 'text-red-400'}`}>
                              {Math.round(s.max_similarity * 100)}%
                            </p>
                            <p className="text-[10px] text-zinc-600">sim to chunk #{s.best_chunk_idx + 1}</p>
                          </div>
                        </div>
                        <p className={`text-[10px] mt-1 font-medium ${s.grounded ? 'text-emerald-600' : 'text-red-600'}`}>
                          {s.grounded ? '✓ grounded' : '✗ ungrounded — hallucination risk'}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ── Chunk relevance breakdown (context precision) ── */}
                <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Context Precision — chunk relevance by rank</p>
                    <InfoTooltip
                      title="How each retrieved chunk contributed to context precision"
                      body={"For each retrieved chunk (in rank order), this shows: how similar it is to your query (cosine), whether it's considered relevant (≥ 0.30), and the running precision@k.\n\nPrecision@k = (# relevant chunks seen so far) / k\n\nThe context precision score is the weighted average of precision@k values for ranks where the chunk was relevant. This means a relevant chunk at rank 1 contributes more than an equally relevant chunk at rank 5.\n\nA good reranker should push relevant chunks to ranks 1, 2, 3 — which is exactly what the cross-encoder in Stage 4 does. If context precision is low even after reranking, the issue is in the retrieval candidates, not the reranking."}
                      pos="top-right"
                    />
                  </div>
                  <div className="space-y-2">
                    {evalResult.chunk_relevance.map((c, i) => (
                      <div key={i} className={`rounded-lg border px-4 py-3 ${c.relevant ? 'border-emerald-800/40 bg-emerald-950/10' : 'border-zinc-800 bg-zinc-900'}`}>
                        <div className="flex items-start gap-3">
                          <div className="shrink-0 w-6 h-6 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-mono text-zinc-500 mt-0.5">
                            {i + 1}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-xs text-zinc-400 leading-relaxed truncate">{c.preview}</p>
                          </div>
                          <div className="shrink-0 text-right space-y-0.5">
                            <p className={`text-sm font-mono font-bold ${c.relevant ? 'text-emerald-400' : 'text-zinc-600'}`}>
                              {Math.round(c.similarity * 100)}%
                            </p>
                            <p className="text-[10px] text-zinc-600">p@{i + 1}: {Math.round(c.precision_at_k * 100)}%</p>
                          </div>
                        </div>
                        <p className={`text-[10px] mt-1 font-medium ml-9 ${c.relevant ? 'text-emerald-600' : 'text-zinc-700'}`}>
                          {c.relevant ? '✓ relevant' : '✗ not relevant — noise'}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ── Context Recall breakdown (only if GT provided) ── */}
                {evalResult.gt_sentence_scores.length > 0 && (
                  <div className="rounded-xl bg-zinc-900 border border-zinc-800 px-5 py-5 space-y-3">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Context Recall — ground truth coverage</p>
                      <InfoTooltip
                        title="Which ground truth sentences are covered by retrieved context?"
                        body={"For each sentence in your reference answer, this checks whether any retrieved chunk contains supporting information.\n\nGreen = this part of the correct answer is supported by your retrieved context — the model had what it needed.\nRed = this part of the correct answer is NOT in your retrieved context — the retriever missed it.\n\nRed rows are the most actionable: they point directly to information that exists in your documents but wasn't retrieved. Fix options:\n• Increase top-k to retrieve more candidates\n• Check if the relevant chunk was split across boundaries (chunking problem)\n• Try a different embedding model that better maps this query to the right document region\n• Add HyDE — generate a hypothetical answer first, use that as the query vector"}
                        pos="top-right"
                      />
                    </div>
                    <div className="space-y-2">
                      {evalResult.gt_sentence_scores.map((s, i) => (
                        <div key={i} className={`rounded-lg border px-4 py-3 ${s.supported ? 'border-emerald-800/40 bg-emerald-950/20' : 'border-red-800/40 bg-red-950/20'}`}>
                          <div className="flex items-start justify-between gap-3">
                            <p className="text-sm text-zinc-300 leading-relaxed flex-1">{s.sentence}</p>
                            <div className="shrink-0 text-right">
                              <p className={`text-sm font-mono font-bold ${s.supported ? 'text-emerald-400' : 'text-red-400'}`}>
                                {Math.round(s.max_similarity * 100)}%
                              </p>
                              <p className="text-[10px] text-zinc-600">best chunk #{s.best_chunk_idx + 1}</p>
                            </div>
                          </div>
                          <p className={`text-[10px] mt-1 font-medium ${s.supported ? 'text-emerald-600' : 'text-red-600'}`}>
                            {s.supported ? '✓ covered by retrieved context' : '✗ missing from retrieved context — retrieval gap'}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              </div>
            )}

          </main>
        </div>
      )}

    </div>
  )
}

// ── Evaluation Radar Chart ────────────────────────────────────────────────────
function EvalRadarChart({ result }: { result: EvalResult }) {
  const metrics = [
    { label: 'Faithfulness',      value: result.faithfulness,      color: '#10b981' },
    { label: 'Ans. Relevancy',    value: result.answer_relevancy,   color: '#8b5cf6' },
    { label: 'Ctx Precision',     value: result.context_precision,  color: '#3b82f6' },
    { label: 'Noise Sensitivity', value: result.noise_sensitivity,  color: '#f59e0b' },
    { label: 'Ctx Recall',        value: result.context_recall ?? 0, color: '#ec4899' },
  ]
  const n = metrics.length
  const cx = 180, cy = 160, R = 110, PAD = 60

  function pt(i: number, r: number) {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }
  }

  const gridLevels = [0.25, 0.5, 0.75, 1.0]
  const polyPoints = metrics.map((m, i) => pt(i, m.value * R))
  const polyStr = polyPoints.map(p => `${p.x},${p.y}`).join(' ')

  return (
    <svg width={cx * 2 + PAD} height={cy * 2 + 20} className="mx-auto block">
      {/* Grid rings */}
      {gridLevels.map(lvl => {
        const pts = metrics.map((_, i) => pt(i, lvl * R))
        return <polygon key={lvl} points={pts.map(p => `${p.x},${p.y}`).join(' ')}
          fill="none" stroke="#27272a" strokeWidth={1} />
      })}
      {/* Axis spokes */}
      {metrics.map((_, i) => {
        const { x, y } = pt(i, R)
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#3f3f46" strokeWidth={1} />
      })}
      {/* Score polygon */}
      <polygon points={polyStr} fill="#10b98130" stroke="#10b981" strokeWidth={1.5} />
      {/* Score dots */}
      {polyPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={4} fill={metrics[i].color} />
      ))}
      {/* Labels */}
      {metrics.map((m, i) => {
        const { x, y } = pt(i, R + 22)
        const anchor = x < cx - 5 ? 'end' : x > cx + 5 ? 'start' : 'middle'
        const isNull = i === 4 && result.context_recall === null
        return (
          <g key={i}>
            <text x={x} y={y - 4} textAnchor={anchor} fontSize={10} fill={isNull ? '#52525b' : '#a1a1aa'} fontFamily="ui-monospace,monospace">
              {m.label}
            </text>
            <text x={x} y={y + 9} textAnchor={anchor} fontSize={11} fill={isNull ? '#52525b' : metrics[i].color} fontFamily="ui-monospace,monospace" fontWeight="700">
              {isNull ? '—' : `${Math.round(m.value * 100)}%`}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
function EmbedScatterPlot({ coords, chunks, hovered, selected, onHover, onSelect, reduction }: {
  coords: [number, number][]; chunks: Chunk[]
  hovered: number | null; selected: number | null
  onHover: (i: number | null) => void; onSelect: (i: number | null) => void
  reduction: Reduction
}) {
  const W = 600, H = 340, PAD = 32
  const xs = coords.map(c => c[0]), ys = coords.map(c => c[1])
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1

  function px(x: number) { return PAD + ((x - xMin) / xRange) * (W - PAD * 2) }
  function py(y: number) { return H - PAD - ((y - yMin) / yRange) * (H - PAD * 2) }

  const activeIdx = selected ?? hovered

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Vector space — {reduction.toUpperCase()} layout</p>
        <InfoTooltip
          title="How to read the scatter plot"
          body={"Each dot = one chunk. The position comes from squashing your high-dimensional vectors down to 2D.\n\nDots that are close together = chunks with similar meaning. Dots far apart = different topics.\n\nThis is what your retriever sees when searching. A query vector would appear somewhere on this plot — the retriever picks the nearest dots.\n\nClick a dot to inspect its raw vector below. Hover to see the chunk text."}
          pos="top-right"
        />
      </div>
      <div className="w-full overflow-x-auto">
        <svg width={W} height={H} className="block mx-auto">
          {/* Grid lines */}
          {[0.25, 0.5, 0.75].map(t => (
            <g key={t} className="text-zinc-800">
              <line x1={PAD} x2={W - PAD} y1={PAD + t * (H - PAD * 2)} y2={PAD + t * (H - PAD * 2)} stroke="currentColor" strokeDasharray="4,4" />
              <line x1={PAD + t * (W - PAD * 2)} x2={PAD + t * (W - PAD * 2)} y1={PAD} y2={H - PAD} stroke="currentColor" strokeDasharray="4,4" />
            </g>
          ))}
          {/* Lines from hovered/selected to its neighbours */}
          {activeIdx !== null && coords.map((_, j) => {
            if (j === activeIdx) return null
            return (
              <line key={j}
                x1={px(coords[activeIdx][0])} y1={py(coords[activeIdx][1])}
                x2={px(coords[j][0])} y2={py(coords[j][1])}
                stroke="#52525b" strokeWidth={0.5} opacity={0.4}
              />
            )
          })}
          {/* Dots */}
          {coords.map((c, i) => {
            const col = COLORS[i % COLORS.length]
            const isActive = activeIdx === null || activeIdx === i
            const r = activeIdx === i ? 9 : 6
            const colMap: Record<string, string> = {
              'text-violet-300': '#c4b5fd', 'text-sky-300': '#7dd3fc',
              'text-emerald-300': '#6ee7b7', 'text-amber-300': '#fcd34d',
              'text-rose-300': '#fda4af', 'text-pink-300': '#f9a8d4',
              'text-cyan-300': '#67e8f9', 'text-lime-300': '#bef264',
              'text-orange-300': '#fdba74', 'text-teal-300': '#5eead4',
            }
            const fill = colMap[col.text] ?? '#a1a1aa'
            return (
              <g key={i} className="cursor-pointer"
                onMouseEnter={() => onHover(i)} onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(selected === i ? null : i)}
              >
                <circle cx={px(c[0])} cy={py(c[1])} r={r + 6} fill="transparent" />
                <circle cx={px(c[0])} cy={py(c[1])} r={r}
                  fill={fill} fillOpacity={isActive ? 0.9 : 0.2}
                  stroke={fill} strokeWidth={activeIdx === i ? 2 : 1}
                />
                <text x={px(c[0])} y={py(c[1]) - r - 3} textAnchor="middle"
                  fontSize={9} fill={fill} fillOpacity={isActive ? 1 : 0.3}
                  className="select-none pointer-events-none font-mono">
                  #{i + 1}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      {activeIdx !== null && (
        <div className="rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-xs text-zinc-300 leading-relaxed">
          <span className={`font-mono font-bold ${COLORS[activeIdx % COLORS.length].text} mr-2`}>#{activeIdx + 1}</span>
          {chunks[activeIdx]?.page_content.slice(0, 160)}{(chunks[activeIdx]?.page_content.length ?? 0) > 160 ? '…' : ''}
        </div>
      )}
      <p className="text-xs text-zinc-600">Click a dot to inspect its raw vector · hover to preview chunk</p>
    </div>
  )
}

// ── Embed heatmap ─────────────────────────────────────────────────────────────
const ADJACENT_THRESHOLD = 0.75  // off-diagonal similarity this high = likely chunking issue
const DIAGONAL_DOMINANT_THRESHOLD = 0.5  // avg off-diagonal below this = healthy distinct chunks

function EmbedHeatmap({ matrix, chunks, hovered, selected, onHover, onSelect }: {
  matrix: number[][]; chunks: Chunk[]
  hovered: number | null; selected: number | null
  onHover: (i: number | null) => void; onSelect: (i: number | null) => void
}) {
  const n = matrix.length
  const cellSize = Math.max(8, Math.min(36, Math.floor(480 / n)))
  const showLabels = cellSize >= 18

  // Detect adjacent-chunk chunking issues: matrix[i][i+1] above threshold
  const adjacentIssues = new Set<number>() // i means the pair (i, i+1) is flagged
  for (let i = 0; i < n - 1; i++) {
    if (matrix[i][i + 1] >= ADJACENT_THRESHOLD) adjacentIssues.add(i)
  }

  // Detect diagonal-dominant (healthy) pattern:
  // avg off-diagonal similarity is low, meaning chunks are distinct
  let offDiagSum = 0, offDiagCount = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j) { offDiagSum += matrix[i][j]; offDiagCount++ }
  const avgOffDiag = offDiagCount > 0 ? offDiagSum / offDiagCount : 0
  const isDiagonalDominant = avgOffDiag < DIAGONAL_DOMINANT_THRESHOLD && n >= 3

  function simColor(v: number) {
    const t = Math.max(0, Math.min(1, v))
    if (t < 0.4) return `rgba(63,131,248,${0.15 + t * 0.4})`
    if (t < 0.7) return `rgba(251,191,36,${0.3 + (t - 0.4) * 0.9})`
    return `rgba(239,68,68,${0.5 + (t - 0.7) * 1.5})`
  }

  const activeRow = selected ?? hovered

  // Which rows are part of a flagged adjacent pair
  const flaggedRows = new Set<number>()
  adjacentIssues.forEach(i => { flaggedRows.add(i); flaggedRows.add(i + 1) })

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Cosine similarity heatmap</p>
        <InfoTooltip
          title="How to read the heatmap"
          body={"Each cell = cosine similarity between two chunks. Cosine measures the angle between vectors — not their length. This is deliberate: a short chunk and a long chunk about the same topic point in the same direction in 384D space, so they score 1.0 regardless of size. Direction = meaning. Length = irrelevant.\n\nRed/orange = similar direction = same topic.\nBlue = different direction = different topic.\n\nThe diagonal is always 1.0 (a chunk compared to itself).\n\nOff-diagonal red = near-duplicate chunks — same meaning, wasted embedding budget.\nMostly blue row = that chunk is topically isolated — easy to retrieve precisely."}
          pos="top-right"
        />
      </div>

      {/* Pattern banners */}
      {adjacentIssues.size > 0 && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2">
          <span className="text-amber-400 text-sm mt-0.5">⚠</span>
          <div className="text-xs text-amber-300 space-y-0.5">
            <p className="font-semibold">Chunking issue detected</p>
            <p className="text-amber-400/80">
              {[...adjacentIssues].map(i => `#${i + 1}↔#${i + 2}`).join(', ')} {adjacentIssues.size === 1 ? 'are' : 'pairs are'} very similar ({'>'}
              {Math.round(ADJACENT_THRESHOLD * 100)}%). Adjacent chunks this close usually means the chunker split mid-thought. Consider semantic chunking or a larger chunk_size.
            </p>
          </div>
          <InfoTooltip
            title="Why adjacent similarity flags a chunking problem"
            body={"If chunk #5 and chunk #6 are both very similar to each other AND they are next to each other in the document, it strongly suggests the chunker placed a boundary in the middle of a continuous idea — splitting one thought across two chunks.\n\nThis hurts retrieval because: (1) neither chunk is a complete thought on its own, so they embed weakly. (2) one becomes a 'hub' — similar to its neighbour and potentially to unrelated chunks nearby.\n\nFix: try semantic chunking, which explicitly avoids splitting where sentences are highly similar. Or increase chunk_size so the two pieces stay together."}
            pos="top-left"
          />
        </div>
      )}

      {isDiagonalDominant && adjacentIssues.size === 0 && (
        <div className="flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2">
          <span className="text-emerald-400 text-sm mt-0.5">✓</span>
          <div className="text-xs text-emerald-300 space-y-0.5">
            <p className="font-semibold">Healthy chunk separation</p>
            <p className="text-emerald-400/80">
              Diagonal is red, off-diagonal is mostly blue (avg {avgOffDiag.toFixed(2)}). Each chunk is distinct from the others — good for precise retrieval.
            </p>
          </div>
          <InfoTooltip
            title="What diagonal-dominant means"
            body={"The diagonal of the heatmap represents each chunk's similarity to itself — always 1.0, always red.\n\nWhen everything else is blue, it means each chunk is topically distinct from every other chunk. This is the ideal pattern: the retriever can pinpoint exactly which chunk answers a given query, without confusion from near-duplicates.\n\nIt doesn't mean your document has no coherence — it means the chunking preserved meaningful boundaries between ideas."}
            pos="top-left"
          />
        </div>
      )}

      <div className="overflow-auto flex justify-center">
        <div style={{ position: 'relative' }}>
          <div style={{ display: 'grid', gridTemplateColumns: `${showLabels ? 20 : 0}px repeat(${n}, ${cellSize}px)`, gap: 1 }}>
            {/* Column headers */}
            {showLabels && <div />}
            {Array.from({ length: n }, (_, i) => (
              <div key={i} style={{ width: cellSize, fontSize: 9 }}
                className={`text-center overflow-hidden font-mono ${flaggedRows.has(i) ? 'text-amber-400' : 'text-zinc-600'}`}>
                {showLabels ? i + 1 : ''}
              </div>
            ))}
            {/* Rows */}
            {matrix.map((row, i) => (
              <React.Fragment key={i}>
                {showLabels && (
                  <div style={{ fontSize: 9, lineHeight: `${cellSize}px` }}
                    className={`text-right pr-1 font-mono ${flaggedRows.has(i) ? 'text-amber-400' : 'text-zinc-600'}`}>
                    {i + 1}
                  </div>
                )}
                {row.map((val, j) => {
                  // Is this cell part of an adjacent-issue pair?
                  const isAdjacentIssue = (i === j + 1 || i + 1 === j) && adjacentIssues.has(Math.min(i, j))
                  return (
                    <div key={j}
                      style={{
                        width: cellSize, height: cellSize,
                        background: simColor(val),
                        opacity: activeRow === null ? 1 : (activeRow === i || activeRow === j) ? 1 : 0.25,
                        outline: isAdjacentIssue
                          ? '2px solid rgba(251,191,36,0.9)'
                          : activeRow === i && activeRow === j ? '2px solid white' : 'none',
                        transition: 'opacity 0.15s',
                        zIndex: isAdjacentIssue ? 1 : 0,
                        position: 'relative',
                      }}
                      className="rounded-sm cursor-pointer"
                      onMouseEnter={() => onHover(i)}
                      onMouseLeave={() => onHover(null)}
                      onClick={() => onSelect(selected === i ? null : i)}
                      title={`#${i + 1} vs #${j + 1}: ${val.toFixed(3)}${isAdjacentIssue ? ' ⚠ adjacent chunks too similar' : ''}`}
                    />
                  )
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-zinc-500 pt-1 border-t border-zinc-800">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'rgba(63,131,248,0.3)' }} /> dissimilar
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'rgba(251,191,36,0.7)' }} /> similar
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'rgba(239,68,68,0.9)' }} /> very similar
        </span>
        {adjacentIssues.size > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm inline-block border-2 border-amber-400/90" style={{ background: 'rgba(239,68,68,0.9)' }} /> chunking issue
          </span>
        )}
        <span className="ml-auto text-zinc-600">Click a row to lock · hover to explore</span>
      </div>

      {activeRow !== null && (
        <div className="rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-xs space-y-1">
          <p className={`font-mono font-bold ${COLORS[activeRow % COLORS.length].text}`}>Chunk #{activeRow + 1} similarity to all others</p>
          <div className="flex flex-wrap gap-2 mt-1">
            {matrix[activeRow].map((val, j) => j !== activeRow && (
              <span key={j} className="flex items-center gap-1">
                <span className={`font-mono ${COLORS[j % COLORS.length].text}`}>#{j + 1}</span>
                <span className={`${adjacentIssues.has(Math.min(activeRow, j)) && Math.abs(activeRow - j) === 1 ? 'text-amber-400 font-semibold' : 'text-zinc-400'}`}>{val.toFixed(3)}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Vector inspector ──────────────────────────────────────────────────────────
function VectorInspector({ vector, chunkIndex, chunk }: {
  vector: number[]; chunkIndex: number; chunk: Chunk
}) {
  const SHOW = 64  // show first 64 dimensions — enough to see the pattern
  const slice = vector.slice(0, SHOW)
  const maxAbs = Math.max(...slice.map(Math.abs), 0.001)

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
          Vector inspector — chunk #{chunkIndex + 1}
        </p>
        <InfoTooltip
          title="What am I looking at?"
          body={`This is the actual embedding vector for chunk #${chunkIndex + 1} — the list of numbers the model produced.\n\nEach bar = one number (one "dimension"). Positive = bar goes right, negative = bar goes left. The pattern of ups and downs encodes the meaning of the chunk.\n\nNo individual number means anything on its own — it's the whole pattern that matters. Two chunks with similar meaning have similar patterns, which is why they end up close on the scatter plot.\n\nShowing the first ${SHOW} of ${vector.length} dimensions.`}
          pos="top-right"
        />
        <span className="ml-auto text-xs text-zinc-600 font-mono">showing {SHOW} of {vector.length} dims</span>
      </div>
      <p className="text-xs text-zinc-500 leading-relaxed line-clamp-2 font-mono">{chunk.page_content.slice(0, 200)}{chunk.page_content.length > 200 ? '…' : ''}</p>
      {/* Sparkline */}
      <div className="flex items-center gap-px" style={{ height: 48 }}>
        {slice.map((v, i) => {
          const h = Math.round(Math.abs(v) / maxAbs * 20)
          const pos = v >= 0
          return (
            <div key={i} className="flex-1 flex flex-col items-center justify-center" style={{ height: 48 }}>
              {pos
                ? <><div className="w-full" style={{ height: 24 }} /><div className={`w-full rounded-t ${COLORS[chunkIndex % COLORS.length].bg}`} style={{ height: h, minHeight: 1 }} /></>
                : <><div className={`w-full rounded-b ${COLORS[chunkIndex % COLORS.length].bg} opacity-60`} style={{ height: h, minHeight: 1 }} /><div className="w-full" style={{ height: 24 }} /></>}
            </div>
          )
        })}
      </div>
      <div className="flex justify-between text-zinc-600" style={{ fontSize: 9 }}>
        <span>dim 1</span><span>dim {SHOW}</span>
      </div>
    </div>
  )
}

// ── Stat pill with tooltip ────────────────────────────────────────────────────
function StatPill({ label, tooltip }: { label: string; tooltip: { title: string; body: string } }) {
  return (
    <div className="flex items-center gap-1 bg-zinc-800 rounded-full pl-2.5 pr-1.5 py-0.5">
      <span className="text-xs text-zinc-400">{label}</span>
      <InfoTooltip title={tooltip.title} body={tooltip.body} />
    </div>
  )
}

// ── Compare table header cell ─────────────────────────────────────────────────
function CompareHeader({ label, tooltip }: { label: string; tooltip: { title: string; body: string } }) {
  return (
    <th className="text-right px-4 py-3">
      <div className="flex items-center justify-end gap-1">
        <span>{label}</span>
        <InfoTooltip title={tooltip.title} body={tooltip.body} pos="top-left" />
      </div>
    </th>
  )
}

// ── Convex hull helpers ───────────────────────────────────────────────────────
function convexHull(pts: [number, number][]): [number, number][] {
  if (pts.length < 2) return pts
  if (pts.length === 2) return pts
  let left = 0
  for (let i = 1; i < pts.length; i++) if (pts[i][0] < pts[left][0]) left = i
  const hull: [number, number][] = []
  let cur = left
  do {
    hull.push(pts[cur])
    let nxt = (cur + 1) % pts.length
    for (let i = 0; i < pts.length; i++) {
      const cross = (pts[nxt][0] - pts[cur][0]) * (pts[i][1] - pts[cur][1])
                  - (pts[nxt][1] - pts[cur][1]) * (pts[i][0] - pts[cur][0])
      if (cross < 0) nxt = i
    }
    cur = nxt
  } while (cur !== left && hull.length <= pts.length)
  return hull
}

function padHull(hull: [number, number][], pad: number): [number, number][] {
  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length
  return hull.map(([x, y]) => {
    const dx = x - cx, dy = y - cy
    const len = Math.sqrt(dx * dx + dy * dy) || 1
    return [x + (dx / len) * pad, y + (dy / len) * pad] as [number, number]
  })
}

function smoothPath(pts: [number, number][]): string {
  if (pts.length < 2) return ''
  if (pts.length === 2) return `M ${pts[0][0]},${pts[0][1]} L ${pts[1][0]},${pts[1][1]}`
  const n = pts.length
  let d = `M ${pts[0][0]},${pts[0][1]}`
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n]
    const cp1x = p1[0] + (p2[0] - p0[0]) / 6
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6
    d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`
  }
  return d + ' Z'
}

// ── Stage 3: Index scatter plot ───────────────────────────────────────────────
const IVF_CLUSTER_COLORS = [
  '#f59e0b', '#3b82f6', '#10b981', '#f43f5e', '#8b5cf6',
  '#06b6d4', '#84cc16', '#fb923c', '#a78bfa', '#34d399',
]

function IndexScatterPlot({ coords, chunks, queryPoint, highlightedIndices, clusterAssignments, centroids2d, searchedClusters, hnswMeta, activeLayer, traversal, traversalStep, tab }: {
  coords: [number, number][]; chunks: Chunk[]
  queryPoint: [number, number] | null
  highlightedIndices: number[]
  clusterAssignments: number[] | null
  centroids2d: [number, number][] | null
  searchedClusters: number[] | null
  hnswMeta: HNSWMeta | null
  activeLayer: number
  traversal: TraversalStep[] | null
  traversalStep: number
  tab: IndexTab
}) {
  const W = 620, H = 360, PAD = 36
  const allPoints = queryPoint ? [...coords, queryPoint] : coords
  const xs = allPoints.map(c => c[0]), ys = allPoints.map(c => c[1])
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1

  function px(x: number) { return PAD + ((x - xMin) / xRange) * (W - PAD * 2) }
  function py(y: number) { return H - PAD - ((y - yMin) / yRange) * (H - PAD * 2) }

  const highlightSet = new Set(highlightedIndices)

  // HNSW layer data
  const layerData = hnswMeta?.layers.find(l => l.level === activeLayer)
  const layerNodeSet = new Set(layerData?.nodes ?? [])

  // Traversal visited set (all steps up to traversalStep, or all if -1)
  const traversalVisited = new Set<number>()
  const traversalBest = new Set<number>()
  if (traversal) {
    const steps = traversalStep === -1 ? traversal : traversal.slice(0, traversalStep + 1)
    steps.forEach(s => { s.visited.forEach(v => traversalVisited.add(v)); traversalBest.add(s.best) })
  }

  function nodeColor(i: number): string {
    if (tab === 'ivf' && clusterAssignments) {
      return IVF_CLUSTER_COLORS[clusterAssignments[i] % IVF_CLUSTER_COLORS.length]
    }
    if (tab === 'hnsw' && hnswMeta) {
      const level = hnswMeta.node_levels[i] ?? 0
      const colors = ['#6ee7b7', '#34d399', '#10b981', '#059669', '#047857']
      return colors[Math.min(level, colors.length - 1)]
    }
    const col = COLORS[i % COLORS.length]
    const map: Record<string, string> = {
      'text-violet-300': '#c4b5fd', 'text-sky-300': '#7dd3fc', 'text-emerald-300': '#6ee7b7',
      'text-amber-300': '#fcd34d', 'text-rose-300': '#fda4af', 'text-pink-300': '#f9a8d4',
      'text-cyan-300': '#67e8f9', 'text-lime-300': '#bef264', 'text-orange-300': '#fdba74', 'text-teal-300': '#5eead4',
    }
    return map[col.text] ?? '#a1a1aa'
  }

  const [hovered, setHovered] = useState<number | null>(null)

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
          {tab === 'flat' && 'Vector space — all chunks + query'}
          {tab === 'hnsw' && `HNSW graph — layer ${activeLayer}`}
          {tab === 'ivf' && 'IVF clusters'}
          {tab === 'mrl' && 'Vector space'}
        </p>
        {tab === 'hnsw' && hnswMeta && (
          <InfoTooltip
            title={`Layer ${activeLayer}`}
            body={activeLayer === 0
              ? `Layer 0 is the dense base layer — every node is here with up to M×2 = ${hnswMeta.M * 2} connections. This is where final precision comes from.`
              : `Layer ${activeLayer} is the express highway — only ${layerData?.nodes.length ?? 0} of ${coords.length} nodes appear here, with up to M = ${hnswMeta.M} connections each. Search enters here for fast navigation.`}
          />
        )}
        {tab === 'ivf' && queryPoint && searchedClusters && (
          <span className="ml-auto text-xs text-zinc-600">
            Bright = searched cluster · dim = skipped
          </span>
        )}
      </div>
      <div className="w-full overflow-x-auto">
        <svg width={W} height={H} className="block mx-auto">
          {/* Grid */}
          {[0.25, 0.5, 0.75].map(t => (
            <g key={t}>
              <line x1={PAD} x2={W-PAD} y1={PAD + t*(H-PAD*2)} y2={PAD + t*(H-PAD*2)} stroke="#27272a" strokeDasharray="3,3" />
              <line x1={PAD + t*(W-PAD*2)} x2={PAD + t*(W-PAD*2)} y1={PAD} y2={H-PAD} stroke="#27272a" strokeDasharray="3,3" />
            </g>
          ))}

          {/* HNSW edges */}
          {tab === 'hnsw' && layerData?.edges.map(([a, b], ei) => {
            const isTraversalEdge = traversalVisited.has(a) && traversalVisited.has(b)
            return (
              <line key={ei}
                x1={px(coords[a][0])} y1={py(coords[a][1])}
                x2={px(coords[b][0])} y2={py(coords[b][1])}
                stroke={isTraversalEdge ? '#f59e0b' : '#3f3f46'}
                strokeWidth={isTraversalEdge ? 1.5 : 0.8}
                opacity={isTraversalEdge ? 0.7 : 0.4}
              />
            )
          })}

          {/* Query → top-k lines (flat/mrl) */}
          {(tab === 'flat' || tab === 'mrl') && queryPoint && highlightedIndices.map(i => (
            <line key={i}
              x1={px(queryPoint[0])} y1={py(queryPoint[1])}
              x2={px(coords[i][0])} y2={py(coords[i][1])}
              stroke="#f59e0b" strokeWidth={1} strokeDasharray="3,3" opacity={0.5}
            />
          ))}

          {/* IVF: query → centroid lines */}
          {tab === 'ivf' && queryPoint && centroids2d && searchedClusters?.map(ci => (
            <line key={ci}
              x1={px(queryPoint[0])} y1={py(queryPoint[1])}
              x2={px(centroids2d[ci][0])} y2={py(centroids2d[ci][1])}
              stroke={IVF_CLUSTER_COLORS[ci % IVF_CLUSTER_COLORS.length]}
              strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}
            />
          ))}

          {/* IVF cluster boundaries — convex hull per cluster */}
          {tab === 'ivf' && clusterAssignments && (() => {
            const n_clusters = Math.max(...clusterAssignments) + 1
            return Array.from({ length: n_clusters }, (_, ci) => {
              const clusterPts = coords
                .map((c, i) => clusterAssignments[i] === ci ? [px(c[0]), py(c[1])] as [number, number] : null)
                .filter((p): p is [number, number] => p !== null)
              if (clusterPts.length === 0) return null
              const color = IVF_CLUSTER_COLORS[ci % IVF_CLUSTER_COLORS.length]
              const searched = !searchedClusters || searchedClusters.includes(ci)
              const sharedProps = {
                fill: color, fillOpacity: searched ? 0.12 : 0,
                stroke: color, strokeWidth: 1.5, strokeOpacity: searched ? 0.7 : 0.35, strokeDasharray: "4,3"
              }
              // For ≤2 points or degenerate hull, draw a circle that encompasses all points
              const cxAvg = clusterPts.reduce((s, p) => s + p[0], 0) / clusterPts.length
              const cyAvg = clusterPts.reduce((s, p) => s + p[1], 0) / clusterPts.length
              const maxR = clusterPts.reduce((m, [x, y]) => Math.max(m, Math.sqrt((x - cxAvg) ** 2 + (y - cyAvg) ** 2)), 0)
              if (clusterPts.length <= 2) {
                return <circle key={ci} cx={cxAvg} cy={cyAvg} r={maxR + 32} {...sharedProps} />
              }
              const hull = convexHull(clusterPts)
              if (hull.length <= 2) {
                return <circle key={ci} cx={cxAvg} cy={cyAvg} r={maxR + 32} {...sharedProps} />
              }
              const padded = padHull(hull, 24)
              const pathD = smoothPath(padded)
              return <path key={ci} d={pathD} {...sharedProps} />
            })
          })()}

          {/* Chunk nodes */}
          {coords.map((c, i) => {
            const fill = nodeColor(i)
            const isHighlighted = highlightedIndices.length === 0 || highlightSet.has(i)
            const inLayer = tab !== 'hnsw' || layerNodeSet.has(i)
            const clusterIdx = clusterAssignments?.[i] ?? 0
            const clusterSearched = !searchedClusters || searchedClusters.includes(clusterIdx)
            const isVisited = traversalVisited.has(i)
            const isBest = traversalBest.has(i)
            const r = isBest ? 9 : isHighlighted ? 7 : 5
            const opacity = !inLayer ? 0.1 : tab === 'ivf' && !clusterSearched && searchedClusters ? 0.2 : isHighlighted ? 0.9 : 0.35

            return (
              <g key={i} className="cursor-pointer"
                onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
                {isBest && (
                  <circle cx={px(c[0])} cy={py(c[1])} r={r + 5} fill={fill} fillOpacity={0.15} stroke={fill} strokeWidth={1} strokeDasharray="3,2" />
                )}
                {isVisited && !isBest && (
                  <circle cx={px(c[0])} cy={py(c[1])} r={r + 3} fill="#f59e0b" fillOpacity={0.1} />
                )}
                <circle cx={px(c[0])} cy={py(c[1])} r={r}
                  fill={fill} fillOpacity={opacity}
                  stroke={isBest ? '#f59e0b' : isHighlighted && isHighlighted ? fill : fill}
                  strokeWidth={isBest ? 2 : isHighlighted && highlightedIndices.length > 0 ? 1.5 : 0.5}
                />
                <text x={px(c[0])} y={py(c[1]) - r - 3} textAnchor="middle" fontSize={9} fill={fill}
                  fillOpacity={inLayer ? 0.8 : 0.15} className="select-none pointer-events-none font-mono">
                  #{i+1}
                </text>
              </g>
            )
          })}

          {/* IVF centroids */}
          {tab === 'ivf' && centroids2d?.map((c, ci) => {
            const color = IVF_CLUSTER_COLORS[ci % IVF_CLUSTER_COLORS.length]
            const searched = !searchedClusters || searchedClusters.includes(ci)
            return (
              <g key={ci}>
                <text x={px(c[0])} y={py(c[1])} textAnchor="middle" dominantBaseline="middle"
                  fontSize={16} fill={color} fillOpacity={searched ? 0.9 : 0.3}
                  className="select-none pointer-events-none font-bold">×</text>
                <text x={px(c[0])} y={py(c[1]) + 14} textAnchor="middle" fontSize={9}
                  fill={color} fillOpacity={searched ? 0.8 : 0.25} className="select-none pointer-events-none font-mono">
                  cluster {ci}
                </text>
              </g>
            )
          })}

          {/* Query point */}
          {queryPoint && (
            <g>
              <circle cx={px(queryPoint[0])} cy={py(queryPoint[1])} r={18} fill="#f59e0b" fillOpacity={0.18} stroke="#f59e0b" strokeWidth={1.5} strokeOpacity={0.6} />
              <text x={px(queryPoint[0])} y={py(queryPoint[1])} textAnchor="middle" dominantBaseline="middle"
                fontSize={16} fill="#f59e0b" className="select-none pointer-events-none">★</text>
              <text x={px(queryPoint[0])} y={py(queryPoint[1]) + 22} textAnchor="middle"
                fontSize={9} fill="#f59e0b" fillOpacity={0.9} fontWeight="bold" className="select-none pointer-events-none">query</text>
            </g>
          )}
        </svg>
      </div>

      {hovered !== null && (
        <div className="rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-xs text-zinc-300 leading-relaxed">
          <span className={`font-mono font-bold mr-2`} style={{ color: nodeColor(hovered) }}>#{hovered + 1}</span>
          {tab === 'hnsw' && hnswMeta && <span className="text-zinc-500 mr-2">level {hnswMeta.node_levels[hovered] ?? 0}</span>}
          {tab === 'ivf' && clusterAssignments && <span className="text-zinc-500 mr-2">cluster C{clusterAssignments[hovered]}</span>}
          {chunks[hovered]?.page_content.slice(0, 160)}{(chunks[hovered]?.page_content.length ?? 0) > 160 ? '…' : ''}
        </div>
      )}
      <p className="text-xs text-zinc-600">
        {tab === 'hnsw' ? 'Gold edges = traversal path · ring = best at that layer · node color = layer height' : ''}
        {tab === 'ivf' ? '× = cluster centroid · bright = searched · dim = skipped · dashed line = query→centroid path' : ''}
        {tab === 'flat' ? 'Dashed lines = query→top-k connections · ★ = query position (PCA projection)' : ''}
      </p>
    </div>
  )
}

// ── Stage 3: Results list ─────────────────────────────────────────────────────
function IndexResultsList({ results, label, color }: { results: IndexResult[]; label: string; color: string }) {
  if (!results.length) return null
  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-3">
      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">{label}</p>
      <div className="space-y-2">
        {results.map((r, rank) => (
          <div key={r.idx} className="flex items-start gap-3 rounded-lg bg-zinc-800/50 px-3 py-2.5">
            <span className={`text-xs font-bold font-mono w-5 shrink-0 ${color === 'amber' ? 'text-amber-500' : 'text-zinc-400'}`}>#{rank + 1}</span>
            <div className="flex-1 min-w-0">
              <p className="text-xs text-zinc-300 leading-relaxed line-clamp-2">{r.text}</p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span className={`text-xs font-mono font-semibold ${r.sim > 0.7 ? 'text-emerald-400' : r.sim > 0.4 ? 'text-amber-400' : 'text-zinc-400'}`}>
                {r.sim.toFixed(3)}
              </span>
              <span className="text-xs text-zinc-600">chunk #{r.idx + 1}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
