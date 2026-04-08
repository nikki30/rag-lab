from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
import re
import math
from typing import Literal, Optional, List
import numpy as np
from sklearn.decomposition import PCA

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
    Split on sentence boundaries where cosine similarity between adjacent
    sentences drops below breakpoint_threshold.
    Returns (chunks, similarity_scores) — one score per sentence boundary.
    """
    sentence_re = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_re.split(text) if s.strip()]

    if len(sentences) <= 1:
        return [text], []

    # Build vocabulary + IDF over all sentences
    all_tokens = [set(re.findall(r'\w+', s.lower())) for s in sentences]
    vocab_set: set[str] = set()
    for ts in all_tokens:
        vocab_set |= ts
    vocab = {w: i for i, w in enumerate(sorted(vocab_set))}
    N = len(sentences)
    idf = [
        math.log((N + 1) / (1 + sum(1 for ts in all_tokens if w in ts))) + 1
        for w in sorted(vocab_set)
    ]

    # Compute similarity between each adjacent pair
    similarities: list[float] = []
    for i in range(len(sentences) - 1):
        sim = tfidf_cosine(sentences[i], sentences[i + 1], vocab, idf)
        similarities.append(round(sim, 4))

    # Build chunks: break when similarity < threshold
    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        next_sent = sentences[i + 1]
        prospective = " ".join(current_sentences + [next_sent])

        # Force a split if adding next sentence exceeds chunk_size
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

EMBED_MODELS: dict[str, str] = {
    "minilm":    "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "mpnet":     "sentence-transformers/all-mpnet-base-v2",
    "nomic":     "nomic-ai/nomic-embed-text-v1.5",
}

# Cache loaded models so we don't reload on every request
_model_cache: dict[str, object] = {}

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
    reduction: Literal["pca", "umap"] = "pca"


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
        else:
            coords_2d = reduce_pca(vectors)

        # Cosine similarity matrix (vectors are already L2-normalised → dot product = cosine)
        sim_matrix = (vectors @ vectors.T).tolist()

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
