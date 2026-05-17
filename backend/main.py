from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
import re
import math
from typing import Literal, Optional, List
import numpy as np
from sklearn.decomposition import PCA

# Lazy-loaded MiniLM model for semantic chunking (small, no user interaction required)
_semantic_model = None
# Cache: {text_hash: (sentences, embeddings, similarities)} — avoids re-embedding on threshold-only changes
_semantic_cache: dict = {}

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _semantic_model

app = FastAPI(title="RAG Lab Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ChunkStrategy = Literal["recursive", "fixed", "paragraph", "sentence", "semantic"]

class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 400
    chunk_overlap: int = 50
    strategy: ChunkStrategy = "recursive"
    breakpoint_threshold: float = 0.5  # only used by semantic

    @model_validator(mode="after")
    def overlap_must_be_smaller_than_size(self):
        if self.strategy != "semantic" and self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


# ── Chunking helpers ──────────────────────────────────────────────────────────

def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="",
    )
    return splitter.split_text(text)


def chunk_paragraph(text: str, chunk_size: int) -> list[str]:
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    current = ""
    for para in raw:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = para
    if current:
        merged.append(current)
    return merged


def chunk_sentence(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]
    result: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            overlap_buf = ""
            for prev in (result[-1].split(". ") if result else []):
                trial = (overlap_buf + ". " + prev).strip() if overlap_buf else prev
                if len(trial) <= chunk_overlap:
                    overlap_buf = trial
            current = (overlap_buf + " " + sent).strip() if overlap_buf else sent
    if current:
        result.append(current)
    return result


def tfidf_cosine(a: str, b: str, vocab: dict[str, int], idf: list[float]) -> float:
    """Lightweight TF-IDF cosine similarity between two strings."""
    def tf_vec(s: str) -> list[float]:
        tokens = re.findall(r'\w+', s.lower())
        counts: dict[int, float] = {}
        for t in tokens:
            if t in vocab:
                counts[vocab[t]] = counts.get(vocab[t], 0) + 1
        n = len(tokens) or 1
        return [counts.get(i, 0) / n * idf[i] for i in range(len(vocab))]

    va, vb = tf_vec(a), tf_vec(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    return dot / (na * nb) if na and nb else 0.0


def chunk_semantic(
    text: str,
    chunk_size: int,
    breakpoint_threshold: float,
) -> tuple[list[str], list[float]]:
    """
    Split on sentence boundaries where embedding cosine similarity between
    adjacent sentences drops below breakpoint_threshold.
    Uses MiniLM embeddings — real semantic similarity, not word overlap.
    Returns (chunks, similarity_scores) — one score per sentence boundary.
    """
    sentence_re = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_re.split(text) if s.strip()]

    if len(sentences) <= 1:
        return [text], []

    # Use cached embeddings if text hasn't changed — only threshold differs
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _semantic_cache:
        sentences, similarities = _semantic_cache[text_hash]
    else:
        model = get_semantic_model()
        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        similarities = [round(float(np.dot(embeddings[i], embeddings[i + 1])), 4)
                        for i in range(len(sentences) - 1)]
        _semantic_cache.clear()  # keep only the most recent text
        _semantic_cache[text_hash] = (sentences, similarities)

    # Build chunks: break when similarity < threshold or chunk_size exceeded
    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        next_sent = sentences[i + 1]
        prospective = " ".join(current_sentences + [next_sent])
        force_split = len(prospective) > chunk_size

        if sim < breakpoint_threshold or force_split:
            chunks.append(" ".join(current_sentences))
            current_sentences = [next_sent]
        else:
            current_sentences.append(next_sent)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks, similarities


# ── Shared response builder ───────────────────────────────────────────────────

def chunk_quality(chunk: str, chunk_size: int) -> dict:
    """
    Score a chunk on two axes (each 0.0–1.0):
      size_score    — how well it fills the target chunk_size budget
      boundary_score — does it start/end at a clean sentence boundary?
    """
    n = len(chunk)

    # Size score: 1.0 at exactly chunk_size, falls off symmetrically
    size_score = round(min(n, chunk_size) / max(n, chunk_size), 2)

    # Boundary score: +0.5 for clean start, +0.5 for clean end
    starts_clean = 1 if (chunk[0].isupper() or chunk[0].isdigit()) else 0
    ends_clean   = 1 if chunk[-1] in ".!?" else 0
    boundary_score = round((starts_clean + ends_clean) / 2, 2)

    quality = round((size_score + boundary_score) / 2, 2)
    return {
        "size_score": size_score,
        "boundary_score": boundary_score,
        "quality": quality,
    }


def build_response(
    text: str,
    chunks: list[str],
    chunk_size: int = 400,
    similarity_scores: Optional[list[float]] = None,
) -> dict:
    results = []
    search_start = 0
    for chunk in chunks:
        idx = text.find(chunk[:60], search_start)
        start_index = idx if idx != -1 else search_start
        results.append({
            "page_content": chunk,
            "metadata": {"start_index": start_index},
            "scores": chunk_quality(chunk, chunk_size),
        })
        search_start = max(0, start_index)

    sizes = [len(c) for c in chunks]
    avg = sum(sizes) / len(sizes) if sizes else 0
    variance = sum((s - avg) ** 2 for s in sizes) / len(sizes) if sizes else 0

    return {
        "chunks": results,
        "total": len(results),
        "stats": {
            "avg_size": round(avg),
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "std_dev": round(math.sqrt(variance)),
            "avg_quality": round(sum(chunk_quality(c, chunk_size)["quality"] for c in chunks) / len(chunks), 2) if chunks else 0,
        },
        "similarity_scores": similarity_scores,
    }


# ── Embedding ─────────────────────────────────────────────────────────────────

EmbedModelId = Literal["minilm", "bge-small", "mpnet", "nomic"]
ReductionId = Literal["pca", "umap", "pacmap"]

EMBED_MODELS: dict[str, str] = {
    "minilm":    "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "mpnet":     "sentence-transformers/all-mpnet-base-v2",
    "nomic":     "nomic-ai/nomic-embed-text-v1.5",
}

# Cache loaded models so we don't reload on every request
_model_cache: dict[str, object] = {}

# ── Stage 3 & 4 state ─────────────────────────────────────────────────────────
# Populated by /api/embed, consumed by /api/build-index, /api/query-index, /api/retrieve
_embed_store: dict = {"vectors": None, "chunks": None, "model": None, "pca": None}
_index_store: dict = {"hnsw": None, "hnsw_meta": None, "ivf_km": None, "ivf_assignments": None, "ivf_centroids_2d": None}

# Lazy cross-encoder for Stage 4 re-ranking (~80 MB, downloaded once)
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _cross_encoder


def _encode_tokens(model_obj, text: str, max_tokens: int = 40) -> tuple[list[str], np.ndarray]:
    """Per-token embeddings for ColBERT late interaction.
    Returns (token_strings, embeddings [seq_len, hidden_dim]) with CLS/SEP stripped."""
    import torch
    tokenizer = model_obj.tokenizer
    encoded = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens + 2, padding=False)
    input_ids = encoded["input_ids"][0]
    tokens = [tokenizer.decode([tid]).strip() for tid in input_ids[1:-1]]
    with torch.no_grad():
        out = model_obj[0].auto_model(**{k: v for k, v in encoded.items()})
    embeds = out.last_hidden_state[0, 1:-1].cpu().numpy()
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    return tokens, embeds / np.maximum(norms, 1e-8)

def get_model(model_id: EmbedModelId):
    if model_id not in _model_cache:
        from sentence_transformers import SentenceTransformer
        trust = model_id == "nomic"  # nomic requires trust_remote_code
        _model_cache[model_id] = SentenceTransformer(
            EMBED_MODELS[model_id], trust_remote_code=trust
        )
    return _model_cache[model_id]


class EmbedRequest(BaseModel):
    chunks: List[str]
    model: EmbedModelId = "minilm"
    reduction: ReductionId = "pca"


def reduce_pca(vectors: np.ndarray) -> list[list[float]]:
    n = len(vectors)
    if n < 2:
        return [[0.0, 0.0]] * n
    n_components = min(2, n)
    coords = PCA(n_components=n_components).fit_transform(vectors)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((n, 1))])
    return coords.tolist()


def reduce_umap(vectors: np.ndarray) -> list[list[float]]:
    import umap
    n = len(vectors)
    if n < 2:
        return [[0.0, 0.0]] * n
    n_neighbors = min(15, n - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    coords = reducer.fit_transform(vectors)
    return coords.tolist()


def reduce_pacmap(vectors: np.ndarray) -> list[list[float]]:
    import pacmap
    n = len(vectors)
    if n < 2:
        return [[0.0, 0.0]] * n
    n_neighbors = min(10, n - 1)
    # PaCMAP requires float32
    v32 = vectors.astype(np.float32)
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    coords = reducer.fit_transform(v32)
    return coords.tolist()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "RAG Lab Engine is Online", "version": "0.1.0"}


@app.post("/api/chunk")
async def chunk_text(request: ChunkRequest):
    similarity_scores = None

    if request.strategy == "recursive":
        chunks = chunk_recursive(request.text, request.chunk_size, request.chunk_overlap)
    elif request.strategy == "fixed":
        chunks = chunk_fixed(request.text, request.chunk_size, request.chunk_overlap)
    elif request.strategy == "paragraph":
        chunks = chunk_paragraph(request.text, request.chunk_size)
    elif request.strategy == "sentence":
        chunks = chunk_sentence(request.text, request.chunk_size, request.chunk_overlap)
    elif request.strategy == "semantic":
        chunks, similarity_scores = chunk_semantic(
            request.text, request.chunk_size, request.breakpoint_threshold
        )

    return build_response(request.text, chunks, request.chunk_size, similarity_scores)


@app.post("/api/embed")
async def embed_chunks(request: EmbedRequest):
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")
    try:
        model = get_model(request.model)
        # nomic requires a task prefix
        texts = (
            [f"search_document: {c}" for c in request.chunks]
            if request.model == "nomic" else request.chunks
        )
        vectors: np.ndarray = model.encode(texts, normalize_embeddings=True)

        # 2-D projection
        if request.reduction == "umap" and len(request.chunks) >= 4:
            coords_2d = reduce_umap(vectors)
        elif request.reduction == "pacmap" and len(request.chunks) >= 4:
            coords_2d = reduce_pacmap(vectors)
        else:
            coords_2d = reduce_pca(vectors)

        # Cosine similarity matrix (vectors are already L2-normalised → dot product = cosine)
        sim_matrix = (vectors @ vectors.T).tolist()

        # Store for Stage 3 indexing — fit PCA once for projecting query vectors later
        _embed_store["vectors"] = vectors
        _embed_store["chunks"] = list(request.chunks)
        _embed_store["model"] = request.model
        _embed_store["pca"] = PCA(n_components=2).fit(vectors) if len(request.chunks) >= 2 else None
        # Reset any stale indices
        for k in _index_store:
            _index_store[k] = None

        return {
            "model": request.model,
            "dimensions": int(vectors.shape[1]),
            "coords_2d": coords_2d,            # [[x, y], ...]  — one per chunk
            "similarity_matrix": sim_matrix,   # N×N cosine similarities
            "vectors": vectors.tolist(),        # raw vectors for inspector
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare_strategies(request: ChunkRequest):
    """Run all strategies with the same params and return stats for comparison."""
    strategies: list[ChunkStrategy] = ["recursive", "fixed", "paragraph", "sentence", "semantic"]
    results = {}

    for strat in strategies:
        try:
            if strat == "recursive":
                chunks = chunk_recursive(request.text, request.chunk_size, request.chunk_overlap)
            elif strat == "fixed":
                chunks = chunk_fixed(request.text, request.chunk_size, request.chunk_overlap)
            elif strat == "paragraph":
                chunks = chunk_paragraph(request.text, request.chunk_size)
            elif strat == "sentence":
                chunks = chunk_sentence(request.text, request.chunk_size, request.chunk_overlap)
            elif strat == "semantic":
                chunks, _ = chunk_semantic(
                    request.text, request.chunk_size, request.breakpoint_threshold
                )
            sizes = [len(c) for c in chunks]
            avg = sum(sizes) / len(sizes) if sizes else 0
            variance = sum((s - avg) ** 2 for s in sizes) / len(sizes) if sizes else 0
            avg_quality = round(sum(chunk_quality(c, request.chunk_size)["quality"] for c in chunks) / len(chunks), 2) if chunks else 0
            results[strat] = {
                "total": len(chunks),
                "avg_size": round(avg),
                "min_size": min(sizes) if sizes else 0,
                "max_size": max(sizes) if sizes else 0,
                "std_dev": round(math.sqrt(variance)),
                "avg_quality": avg_quality,
                "sizes": sizes,
                "preview_chunks": chunks[:2],  # first 2 chunks so UI can show actual splits
            }
        except Exception as e:
            results[strat] = {"error": str(e)}

    return results


# ── Stage 3: Indexing ─────────────────────────────────────────────────────────

class BuildIndexRequest(BaseModel):
    M: int = 16
    ef_construction: int = 100
    n_clusters: int = 4


def _hnsw_neighbors_at(offsets, neighbors_arr, node: int, level: int, node_max_level: int, M: int) -> list[int]:
    """Return actual graph neighbors of `node` at `level` using the FAISS offsets array.

    FAISS HNSW neighbor layout per node:
      level 0 → 2*M slots  (cum offset = 0)
      level 1 → M slots    (cum offset = 2*M)
      level l → M slots    (cum offset = 2*M + (l-1)*M = (l+1)*M)
    """
    if node_max_level < level:
        return []
    cum = 0 if level == 0 else (level + 1) * M
    nb_slots = 2 * M if level == 0 else M
    begin = int(offsets[node]) + cum
    end = begin + nb_slots
    return [int(x) for x in neighbors_arr[begin:end] if x != -1]


def _extract_hnsw_graph(index, vectors, M: int) -> dict:
    """Extract the real HNSW graph from a FAISS IndexHNSWFlat for visualization."""
    import faiss as faiss_mod
    n = len(vectors)
    levels_raw = faiss_mod.vector_to_array(index.hnsw.levels).tolist()
    offsets = faiss_mod.vector_to_array(index.hnsw.offsets)
    neighbors_arr = faiss_mod.vector_to_array(index.hnsw.neighbors)
    max_level = int(index.hnsw.max_level)
    entry_point = int(index.hnsw.entry_point)

    # levels_raw[i] = number of layers node i participates in (1-indexed)
    node_levels = [int(l) - 1 for l in levels_raw]  # convert to 0-indexed max level

    layers = []
    for lv in range(max_level + 1):
        lv_nodes = [i for i in range(n) if node_levels[i] >= lv]
        edges: set[tuple] = set()
        for nd in lv_nodes:
            for nb in _hnsw_neighbors_at(offsets, neighbors_arr, nd, lv, node_levels[nd], M):
                if nb != nd:
                    edges.add((min(nd, nb), max(nd, nb)))
        layers.append({"level": lv, "nodes": lv_nodes, "edges": [list(e) for e in edges]})

    return {
        "layers": layers,
        "node_levels": node_levels,
        "max_level": max_level,
        "entry_point": entry_point,
        "M": M,
    }


def _simulate_traversal(index, vectors: np.ndarray, query_vec: np.ndarray) -> list:
    """Greedy traversal through the FAISS HNSW graph — records visited nodes per layer."""
    import faiss as faiss_mod
    M = int(index.hnsw.M)
    levels_raw = faiss_mod.vector_to_array(index.hnsw.levels).tolist()
    offsets = faiss_mod.vector_to_array(index.hnsw.offsets)
    neighbors_arr = faiss_mod.vector_to_array(index.hnsw.neighbors)
    node_levels = [int(l) - 1 for l in levels_raw]
    max_level = int(index.hnsw.max_level)
    ep = int(index.hnsw.entry_point)
    traversal = []

    for lv in range(max_level, -1, -1):
        visited = [ep]
        current = ep
        best_sim = float(np.dot(query_vec, vectors[ep]))
        improved = True
        while improved:
            improved = False
            for nb in _hnsw_neighbors_at(offsets, neighbors_arr, current, lv, node_levels[current], M):
                if nb == current or nb in visited:
                    continue
                visited.append(nb)
                sim = float(np.dot(query_vec, vectors[nb]))
                if sim > best_sim:
                    best_sim = sim
                    current = nb
                    improved = True
                    break  # restart from new best
        traversal.append({"layer": lv, "visited": [int(v) for v in visited], "best": int(current)})
        ep = current

    return traversal


@app.post("/api/build-index")
async def build_index(request: BuildIndexRequest):
    if _embed_store["vectors"] is None:
        raise HTTPException(status_code=400, detail="No vectors — run /api/embed first")

    vectors = _embed_store["vectors"]
    n, dim = vectors.shape

    # ── HNSW via FAISS ────────────────────────────────────────────────────────
    try:
        import faiss
        hnsw_idx = faiss.IndexHNSWFlat(dim, request.M)
        hnsw_idx.hnsw.efConstruction = request.ef_construction
        hnsw_idx.add(vectors.astype(np.float32))
        hnsw_meta = _extract_hnsw_graph(hnsw_idx, vectors, request.M)
        _index_store["hnsw"] = hnsw_idx
        _index_store["hnsw_meta"] = hnsw_meta
    except Exception as e:
        import traceback
        hnsw_meta = {"error": traceback.format_exc()}

    # ── IVF via sklearn KMeans ────────────────────────────────────────────────
    from sklearn.cluster import KMeans
    k = min(request.n_clusters, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors).tolist()
    # Normalize centroids for cosine similarity queries
    centroids = km.cluster_centers_.copy()
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)

    centroids_2d: list = [[0.0, 0.0]] * k
    if _embed_store["pca"] is not None:
        centroids_2d = _embed_store["pca"].transform(centroids).tolist()

    _index_store["ivf_km"] = km
    _index_store["ivf_assignments"] = labels
    _index_store["ivf_centroids_2d"] = centroids_2d

    return {
        "num_vectors": n,
        "dimensions": dim,
        "hnsw": hnsw_meta,
        "ivf": {"cluster_assignments": labels, "n_clusters": k, "centroids_2d": centroids_2d},
    }


class QueryIndexRequest(BaseModel):
    query: str
    k: int = 5
    ef: int = 50
    nprobe: int = 2


@app.post("/api/query-index")
async def query_index(request: QueryIndexRequest):
    if _embed_store["vectors"] is None:
        raise HTTPException(status_code=400, detail="No vectors — build index first")

    vectors = _embed_store["vectors"]
    chunks = _embed_store["chunks"]
    model_id = _embed_store["model"]
    n, dim = vectors.shape
    k = min(request.k, n)

    # Embed the query (nomic uses search_query: prefix, not search_document:)
    model = get_model(model_id)
    prefix = "search_query: " if model_id == "nomic" else ""
    query_vec: np.ndarray = model.encode([prefix + request.query], normalize_embeddings=True)[0]

    # Project query to 2D using fitted PCA
    query_2d = [0.0, 0.0]
    if _embed_store["pca"] is not None:
        query_2d = _embed_store["pca"].transform(query_vec.reshape(1, -1)).tolist()[0]

    # ── Flat (exact brute-force) ──────────────────────────────────────────────
    flat_sims = (vectors @ query_vec).tolist()
    flat_order = sorted(range(n), key=lambda i: -flat_sims[i])[:k]
    flat_results = [{"idx": i, "sim": round(flat_sims[i], 4), "text": chunks[i]} for i in flat_order]
    flat_set = {r["idx"] for r in flat_results}

    # ── HNSW ─────────────────────────────────────────────────────────────────
    hnsw_results = flat_results
    hnsw_recall = 1.0
    hnsw_traversal: list = []
    if _index_store["hnsw"] is not None:
        try:
            hnsw_idx = _index_store["hnsw"]
            hnsw_idx.hnsw.efSearch = max(request.ef, k)
            _, I = hnsw_idx.search(query_vec.astype(np.float32).reshape(1, -1), k)
            hnsw_order = [int(i) for i in I[0] if i != -1]
            hnsw_results = [{"idx": i, "sim": round(float(np.dot(vectors[i], query_vec)), 4), "text": chunks[i]} for i in hnsw_order]
            hnsw_recall = round(len(flat_set & {r["idx"] for r in hnsw_results}) / len(flat_set), 3) if flat_set else 1.0
            hnsw_traversal = _simulate_traversal(hnsw_idx, vectors, query_vec)
        except Exception:
            pass

    # ── IVF ───────────────────────────────────────────────────────────────────
    ivf_results = flat_results
    ivf_recall = 1.0
    ivf_searched: list[int] = []
    if _index_store["ivf_km"] is not None:
        km = _index_store["ivf_km"]
        centroids = km.cluster_centers_.copy()
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        centroid_sims = (centroids @ query_vec).tolist()
        nprobe = min(request.nprobe, km.n_clusters)
        ivf_searched = sorted(range(km.n_clusters), key=lambda i: -centroid_sims[i])[:nprobe]
        searched_set = set(ivf_searched)
        assignments = _index_store["ivf_assignments"]
        candidates = sorted(
            [(float(np.dot(vectors[i], query_vec)), i) for i, lbl in enumerate(assignments) if lbl in searched_set],
            reverse=True,
        )[:k]
        ivf_results = [{"idx": i, "sim": round(sim, 4), "text": chunks[i]} for sim, i in candidates]
        ivf_recall = round(len(flat_set & {r["idx"] for r in ivf_results}) / len(flat_set), 3) if flat_set else 1.0

    # ── MRL (Nomic only — Matryoshka truncation) ──────────────────────────────
    mrl = None
    if model_id == "nomic":
        mrl_dims_list = [768, 512, 256, 128, 64]
        mrl_results: dict = {}
        mrl_recall: dict = {}
        for d in mrl_dims_list:
            d = min(d, dim)
            tv = vectors[:, :d].copy()
            tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-8)
            qv = query_vec[:d].copy()
            qv /= max(float(np.linalg.norm(qv)), 1e-8)
            sims = (tv @ qv).tolist()
            ranked = sorted(range(n), key=lambda i: -sims[i])[:k]
            mrl_results[str(d)] = [{"idx": i, "sim": round(sims[i], 4)} for i in ranked]
            mrl_recall[str(d)] = round(len(flat_set & {r["idx"] for r in mrl_results[str(d)]}) / len(flat_set), 3) if flat_set else 1.0
        mrl = {"results_by_dims": mrl_results, "recall": mrl_recall}

    return {
        "query_2d": query_2d,
        "flat_results": flat_results,
        "hnsw_results": hnsw_results,
        "hnsw_recall": hnsw_recall,
        "hnsw_traversal": hnsw_traversal,
        "ivf_results": ivf_results,
        "ivf_recall": ivf_recall,
        "ivf_searched_clusters": ivf_searched,
        "mrl": mrl,
    }


# ── Stage 4: Retrieval ────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    top_k_rerank: int = 20  # candidates passed to cross-encoder before final top-k


@app.post("/api/retrieve")
async def retrieve(request: RetrieveRequest):
    if _embed_store["vectors"] is None:
        raise HTTPException(status_code=400, detail="No vectors — run /api/embed first")

    vectors: np.ndarray = _embed_store["vectors"]
    chunks: list[str] = _embed_store["chunks"]
    model_id: str = _embed_store["model"]
    n = len(chunks)
    k = min(request.k, n)
    top_k = min(request.top_k_rerank, n)

    # ── Embed query ───────────────────────────────────────────────────────
    model = get_model(model_id)
    prefix = "search_query: " if model_id == "nomic" else ""
    query_vec: np.ndarray = model.encode([prefix + request.query], normalize_embeddings=True)[0]

    # ── Dense (cosine) ────────────────────────────────────────────────────
    dense_sims = (vectors @ query_vec).tolist()
    dense_order = sorted(range(n), key=lambda i: -dense_sims[i])
    dense_results = [{"idx": i, "score": round(dense_sims[i], 4), "text": chunks[i]}
                     for i in dense_order[:k]]

    # ── Sparse (BM25) ─────────────────────────────────────────────────────
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [re.findall(r"\w+", c.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens_bm25 = re.findall(r"\w+", request.query.lower())
    bm25_scores = bm25.get_scores(q_tokens_bm25).tolist()
    sparse_order = sorted(range(n), key=lambda i: -bm25_scores[i])
    sparse_results = [{"idx": i, "score": round(bm25_scores[i], 4), "text": chunks[i]}
                      for i in sparse_order[:k]]

    # ── Hybrid (RRF) ──────────────────────────────────────────────────────
    rrf_k = 60
    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_order[:top_k]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
    for rank, idx in enumerate(sparse_order[:top_k]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
    hybrid_order = sorted(rrf_scores, key=lambda i: -rrf_scores[i])
    hybrid_results = [{"idx": i, "score": round(rrf_scores[i], 6), "text": chunks[i]}
                      for i in hybrid_order[:k]]

    # ── Cross-encoder re-ranking ──────────────────────────────────────────
    # Candidates = union of top-top_k from all three strategies (deduped)
    seen: set[int] = set()
    candidates: list[int] = []
    for idx in hybrid_order[:top_k] + dense_order[:top_k] + sparse_order[:top_k]:
        if idx not in seen:
            seen.add(idx)
            candidates.append(idx)
    ce = get_cross_encoder()
    ce_pairs = [[request.query, chunks[i]] for i in candidates]
    ce_scores_arr = ce.predict(ce_pairs).tolist()
    ce_ranked = sorted(zip(candidates, ce_scores_arr), key=lambda x: -x[1])
    reranked_results = [{"idx": i, "score": round(s, 4), "text": chunks[i]}
                        for i, s in ce_ranked[:k]]

    # ── ColBERT late interaction (all reranked candidates) ───────────────
    colbert_data = None
    colbert_scores: dict[int, float] = {}
    if reranked_results:
        try:
            q_toks, q_embeds = _encode_tokens(model, request.query, max_tokens=24)
            top_c_toks, top_sim_matrix = None, None
            for i, res in enumerate(reranked_results):
                idx = res["idx"]
                c_toks_i, c_embeds_i = _encode_tokens(model, chunks[idx], max_tokens=48)
                sim_i = q_embeds @ c_embeds_i.T
                colbert_scores[idx] = float(sim_i.max(axis=1).sum())
                if i == 0:
                    top_c_toks = c_toks_i
                    top_sim_matrix = sim_i.tolist()
            colbert_data = {
                "query_tokens": q_toks,
                "chunk_tokens": top_c_toks,
                "sim_matrix": top_sim_matrix,
                "chunk_idx": reranked_results[0]["idx"],
                "scores": colbert_scores,
            }
        except Exception:
            pass

    # ColBERT ranked order (descending score)
    colbert_order = sorted(colbert_scores, key=lambda i: -colbert_scores[i])

    # ── Rank shift table ──────────────────────────────────────────────────
    all_top = list(dict.fromkeys(
        [r["idx"] for r in dense_results] + [r["idx"] for r in sparse_results] +
        [r["idx"] for r in hybrid_results] + [r["idx"] for r in reranked_results]
    ))

    def rank_in(idx: int, results: list[dict]) -> int | None:
        for pos, r in enumerate(results):
            if r["idx"] == idx:
                return pos + 1
        return None

    rank_shifts = [
        {
            "idx": idx,
            "text": chunks[idx][:100],
            "dense_rank": rank_in(idx, dense_results),
            "sparse_rank": rank_in(idx, sparse_results),
            "hybrid_rank": rank_in(idx, hybrid_results),
            "reranked_rank": rank_in(idx, reranked_results),
            "colbert_rank": (colbert_order.index(idx) + 1) if idx in colbert_scores else None,
        }
        for idx in all_top
    ]

    return {
        "dense": dense_results,
        "sparse": sparse_results,
        "hybrid": hybrid_results,
        "reranked": reranked_results,
        "colbert": colbert_data,
        "rank_shifts": rank_shifts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-6": 200_000,
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
    "gemma2-9b-it": 8_192,
    "mixtral-8x7b-32768": 32_768,
    # Ollama local models
    "llama3.2": 128_000,
    "gemma3": 128_000,
    "mistral": 32_000,
    "phi4": 16_000,
}

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_million_tokens, output_per_million_tokens)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "llama-3.1-8b-instant": (0.0, 0.0),
    "gemma2-9b-it": (0.0, 0.0),
    "mixtral-8x7b-32768": (0.0, 0.0),
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the provided context. "
    "Be concise and accurate. If the context doesn't contain enough information to fully answer, say so explicitly."
)


class GenerateChunk(BaseModel):
    idx: int
    score: float
    text: str


class GenerateRequest(BaseModel):
    query: str
    chunks: list[GenerateChunk]
    model: str
    api_key: str
    compaction: str = "raw"              # "raw" | "contextual"
    chunk_order: str = "relevance_desc"  # "relevance_desc" | "relevance_asc" | "sandwich"
    context_strategy: str = "stuffing"  # "stuffing" | "map_reduce" | "refine" | "map_rerank"
    temperature: float = 0.1


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _count_tokens(text: str, model: str) -> int:
    if model.startswith("gpt"):
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _apply_compaction(
    chunks: list[GenerateChunk], query: str, algo: str, embed_model
) -> list[tuple[GenerateChunk, str]]:
    if algo == "raw":
        return [(c, c.text) for c in chunks]

    if algo == "contextual":
        query_emb = embed_model.encode([query], normalize_embeddings=True)[0]
        result = []
        for chunk in chunks:
            sentences = _split_sentences(chunk.text)
            if len(sentences) <= 1:
                result.append((chunk, chunk.text))
                continue
            sent_embs = embed_model.encode(sentences, normalize_embeddings=True)
            sims = (sent_embs @ query_emb).tolist()
            kept = [s for s, sim in zip(sentences, sims) if sim > 0.2]
            if not kept:
                kept = [max(zip(sentences, sims), key=lambda x: x[1])[0]]
            result.append((chunk, " ".join(kept)))
        return result

    return [(c, c.text) for c in chunks]


def _apply_chunk_order(
    pairs: list[tuple[GenerateChunk, str]], order: str
) -> list[tuple[GenerateChunk, str]]:
    if order == "relevance_asc":
        return list(reversed(pairs))
    if order == "sandwich" and len(pairs) > 2:
        return [pairs[0]] + pairs[2:] + [pairs[1]]
    return pairs  # relevance_desc — already sorted best-first


def _score_grounding(
    answer: str, chunk_texts: list[str], embed_model
) -> list[dict]:
    sentences = _split_sentences(answer)
    if not sentences or not chunk_texts:
        return [{"sentence": s, "max_similarity": 0.0, "grounded": False} for s in sentences]
    try:
        all_embs = embed_model.encode(sentences + chunk_texts, normalize_embeddings=True)
        sent_embs = all_embs[:len(sentences)]
        chunk_embs = all_embs[len(sentences):]
        results = []
        for i, sent in enumerate(sentences):
            sims = (chunk_embs @ sent_embs[i]).tolist()
            max_sim = float(max(sims))
            results.append({
                "sentence": sent,
                "max_similarity": round(max_sim, 4),
                "grounded": max_sim >= 0.35,
            })
        return results
    except Exception:
        return [{"sentence": s, "max_similarity": 0.0, "grounded": False} for s in sentences]


@app.post("/api/generate")
async def generate_answer(request: GenerateRequest):
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")
    OLLAMA_MODELS = {"llama3.2", "gemma3", "mistral", "phi4"}
    if not request.api_key.strip() and request.model not in OLLAMA_MODELS:
        raise HTTPException(status_code=400, detail="API key required")
    if request.model not in MODEL_CONTEXT_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    embed_model = get_semantic_model()
    context_window = MODEL_CONTEXT_WINDOWS[request.model]

    # ── 1. Compact ────────────────────────────────────────────────────────────
    pairs = _apply_compaction(request.chunks, request.query, request.compaction, embed_model)
    original_chunk_tokens = sum(_count_tokens(c.text, request.model) for c, _ in pairs)

    # ── 2. Order ──────────────────────────────────────────────────────────────
    pairs = _apply_chunk_order(pairs, request.chunk_order)
    compressed_chunk_tokens = sum(_count_tokens(t, request.model) for _, t in pairs)

    # ── 3. Build prompt sections (for the context window visualisation) ───────
    sys_tokens = _count_tokens(SYSTEM_PROMPT, request.model)
    query_tokens = _count_tokens(request.query, request.model)

    sections = [{"label": "System prompt", "text": SYSTEM_PROMPT, "tokens": sys_tokens,
                 "role": "system", "chunk_idx": None, "original_tokens": None}]

    for chunk, compacted_text in pairs:
        orig_tok = _count_tokens(chunk.text, request.model)
        comp_tok = _count_tokens(compacted_text, request.model)
        sections.append({
            "label": f"Chunk #{chunk.idx + 1}",
            "text": compacted_text,
            "tokens": comp_tok,
            "role": "chunk",
            "chunk_idx": chunk.idx,
            "original_tokens": orig_tok if request.compaction != "raw" else None,
        })

    sections.append({"label": "User query", "text": request.query, "tokens": query_tokens,
                     "role": "query", "chunk_idx": None, "original_tokens": None})

    # ── 4. Call LLM ───────────────────────────────────────────────────────────
    context_str = "\n\n".join(
        f"[Context {i + 1}]:\n{text}" for i, (_, text) in enumerate(pairs)
    )
    user_message = f"{context_str}\n\nQuestion: {request.query}"

    answer = ""
    actual_input_tokens = sum(s["tokens"] for s in sections)
    actual_output_tokens = 0

    GROQ_MODELS = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"}

    def _openai_call(base_url: str | None, api_key: str) -> tuple[str, int, int]:
        from openai import OpenAI
        kwargs = {"api_key": api_key or "ollama"}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=request.temperature,
            max_tokens=1000,
        )
        return (
            resp.choices[0].message.content or "",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )

    try:
        if request.model.startswith("gpt"):
            answer, actual_input_tokens, actual_output_tokens = _openai_call(None, request.api_key)

        elif request.model.startswith("claude"):
            from anthropic import Anthropic
            client = Anthropic(api_key=request.api_key)
            resp = client.messages.create(
                model=request.model,
                max_tokens=1000,
                temperature=request.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = resp.content[0].text
            actual_input_tokens = resp.usage.input_tokens
            actual_output_tokens = resp.usage.output_tokens

        elif request.model in GROQ_MODELS:
            answer, actual_input_tokens, actual_output_tokens = _openai_call(
                "https://api.groq.com/openai/v1", request.api_key
            )

        else:
            # Ollama — local OpenAI-compatible server, no real API key needed
            answer, actual_input_tokens, actual_output_tokens = _openai_call(
                "http://localhost:11434/v1", "ollama"
            )

    except Exception as e:
        err = str(e)
        if "11434" in err or "connection" in err.lower() and request.model in OLLAMA_MODELS:
            raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve — then pull the model with: ollama pull " + request.model)
        status = 401 if any(k in err.lower() for k in ("api key", "authentication", "invalid", "unauthorized")) else 500
        raise HTTPException(status_code=status, detail=f"LLM error: {err}")

    # ── 5. Score grounding ────────────────────────────────────────────────────
    chunk_texts = [t for _, t in pairs]
    grounding = _score_grounding(answer, chunk_texts, embed_model)

    # ── 6. Cost ───────────────────────────────────────────────────────────────
    in_price, out_price = MODEL_PRICING.get(request.model, (0.0, 0.0))
    cost_usd = (actual_input_tokens / 1_000_000) * in_price + (actual_output_tokens / 1_000_000) * out_price

    return {
        "answer": answer,
        "sections": sections,
        "grounding": grounding,
        "total_input_tokens": actual_input_tokens,
        "total_output_tokens": actual_output_tokens,
        "cost_usd": round(cost_usd, 6),
        "model": request.model,
        "context_window": context_window,
        "compaction_stats": {
            "original_tokens": original_chunk_tokens,
            "compressed_tokens": compressed_chunk_tokens,
            "ratio": round(compressed_chunk_tokens / original_chunk_tokens, 3) if original_chunk_tokens > 0 else 1.0,
        },
    }
