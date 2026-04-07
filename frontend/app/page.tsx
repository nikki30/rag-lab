'use client'

import { useState, useMemo } from 'react'

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

  const overlapError = strategy !== 'semantic' && chunkOverlap >= chunkSize
    ? 'Overlap must be less than chunk size' : null

  function resetResults() { setChunks([]); setStats(null); setSimilarityScores(null); setCompareData(null) }
  // When only the threshold changes, scores stay — only the coloring updates live
  function resetChunksOnly() { setChunks([]); setStats(null); setCompareData(null) }

  async function handleChunk(overrideStrategy?: Strategy) {
    const activeStrategy = overrideStrategy ?? strategy
    if (activeStrategy !== 'semantic' && chunkOverlap >= chunkSize) return
    setLoading(true); setError(null); resetResults()
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
          <span className="ml-2 text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded-full">chunker</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

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
      </main>
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
