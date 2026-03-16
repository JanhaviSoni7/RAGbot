"""ragbot/indexing/embedder.py

BGE-M3 embedding wrapper.

Supports two backends:
  1. FlagEmbedding (BGEM3FlagModel) — preferred, uses FlagEmbedding-specific API.
  2. sentence-transformers (SentenceTransformer) — fallback, standard ST API.

The backend is detected ONCE at model-load time and stored in a module-level
flag (_backend).  All encode() calls then dispatch to the right API so that
FlagEmbedding-only kwargs are NEVER passed to a SentenceTransformer model.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

from config import (
    EMBED_BATCH_SIZE,
    EMBED_MAX_SEQ_LEN,
    EMBED_MODEL_NAME,
    EMBED_NORMALIZE,
)
from ragbot.utils.logger import get_logger

if TYPE_CHECKING:
    from ragbot.ingestion.chunker import Chunk

log = get_logger(__name__)


# ── Singleton ─────────────────────────────────────────────────────────────────

_model   = None
_backend = None   # "flag" | "st"  — set exactly once at load time


def _get_model():
    """Load the embedding model on first call; return (model, backend_str)."""
    global _model, _backend
    if _model is None:
        log.info("Loading embedding model '%s' (first call)…", EMBED_MODEL_NAME)
        try:
            from FlagEmbedding import BGEM3FlagModel
            _model   = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=True)
            _backend = "flag"
            log.info("BGE-M3 loaded via FlagEmbedding backend.")
        except Exception as exc:
            log.warning(
                "FlagEmbedding unavailable (%s); using sentence-transformers.", exc
            )
            from sentence_transformers import SentenceTransformer
            _model   = SentenceTransformer(EMBED_MODEL_NAME)
            _backend = "st"
            log.info("BGE-M3 loaded via sentence-transformers backend.")
    return _model, _backend


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of strings and return a float32 numpy array of shape (N, D).

    Dispatches to the correct backend API so that each backend only receives
    the keyword arguments it understands.
    """
    model, backend = _get_model()
    if not texts:
        return np.empty((0,), dtype=np.float32)

    if backend == "flag":
        # ── FlagEmbedding BGEM3FlagModel path ────────────────────────────────
        out = model.encode(
            texts,
            batch_size          = EMBED_BATCH_SIZE,
            max_length          = EMBED_MAX_SEQ_LEN,
            return_dense        = True,
            return_sparse       = False,
            return_colbert_vecs = False,
            show_progress_bar   = show_progress,
        )
        vectors = np.array(out["dense_vecs"], dtype=np.float32)

        if EMBED_NORMALIZE:
            norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms   = np.where(norms == 0, 1, norms)
            vectors /= norms

    else:
        # ── sentence-transformers SentenceTransformer path ───────────────────
        # NOTE: Only pass kwargs that SentenceTransformer.encode() accepts.
        # Do NOT pass return_dense / return_sparse / max_length here.
        vectors = model.encode(
            texts,
            batch_size           = EMBED_BATCH_SIZE,
            show_progress_bar    = show_progress,
            normalize_embeddings = EMBED_NORMALIZE,
        ).astype(np.float32)

    log.debug("Embedded %d texts → shape %s (backend=%s)",
              len(texts), vectors.shape, backend)
    return vectors


def embed_chunks(chunks: List["Chunk"], show_progress: bool = True) -> np.ndarray:
    """Embed Chunk objects using chunk.text."""
    return embed_texts([c.text for c in chunks], show_progress=show_progress)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string and return a 1-D float32 array.

    Prepends 'Represent this sentence: ' as recommended by BGE-M3 for
    asymmetric dense retrieval (query vs. passage distinction).
    """
    prefixed = f"Represent this sentence: {query}"
    vec = embed_texts([prefixed], show_progress=False)
    return vec[0]
