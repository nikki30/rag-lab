'use client'

import React, { useState, useMemo, useRef } from 'react'

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

  const tooltipStyle: React.CSSProperties = anchor ? {
    position: 'fixed',
    top: anchor.y - 8,
    left: pos === 'top-left'  ? anchor.x + 8  :
          pos === 'top-right' ? anchor.x - 8  :
          anchor.x,
    transform: pos === 'top-left'  ? 'translate(-100%, -100%)' :
               pos === 'top-right' ? 'translate(0, -100%)'     :
               'translate(-50%, -100%)',
    zIndex: 9999,
  } : {}

  return (
    <div className="relative inline-flex shrink-0"
      onMouseEnter={handleEnter}
      onMouseLeave={() => setAnchor(null)}
    >
      <span className="w-4 h-4 rounded-full border border-zinc-700 text-zinc-600 hover:text-zinc-300 hover:border-zinc-500 text-[10px] flex items-center justify-center transition-colors cursor-help select-none">?</span>
      {anchor && (
        <div style={tooltipStyle} className="w-72 rounded-xl bg-zinc-800 border border-zinc-700 p-3.5 pointer-events-none shadow-2xl">
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
          <span className="ml-2 text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded-full">chunker → embedder</span>
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
                  body: 'When two adjacent sentences have similarity below this value, the algorithm decides the topic changed and starts a new chunk.\n\nLow (0.1) → fewer, bigger chunks.\nHigh (0.8) → splits at every slight shift, many small chunks.\n\nWatch the bar chart below: red bars are where splits happen. Drag this slider and re-chunk to move the cutoff line.',
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
    </div>
  )
}

// ── Embed scatter plot ────────────────────────────────────────────────────────
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
