"""ragbot/indexing/faiss_store.py

FAISS vector index — manages add, save, load, and search operations.

Index type: `IndexFlatIP` (inner product) on L2-normalised vectors
            ≡ cosine similarity, exact (no quantisation trade-offs at
            corpus sizes typical for a local RAG bot).

For very large corpora (>1M chunks), swap to IndexIVFFlat or HNSW.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    FAISS_TOP_K,
    EMBED_MODEL_NAME,
)
from ragbot.utils.logger import get_logger

if TYPE_CHECKING:
    from ragbot.ingestion.chunker import Chunk

log = get_logger(__name__)


class FAISSStore:
    """
    Thin wrapper around FAISS that also stores chunk metadata as a
    parallel list so that search results carry the full Chunk context.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_FILE,
        meta_path:  Path = FAISS_META_FILE,
    ):
        self.index_path = index_path
        self.meta_path  = meta_path
        self._index     = None        # faiss.Index
        self._chunks: List["Chunk"] = []
        self._dim: Optional[int]    = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List["Chunk"], vectors: np.ndarray) -> None:
        """Create a fresh index from chunks + their precomputed vectors."""
        import faiss

        assert len(chunks) == vectors.shape[0], "chunks / vector count mismatch"
        dim = vectors.shape[1]

        self._dim    = dim
        self._index  = faiss.IndexFlatIP(dim)   # Inner Product (cosine on normalised)
        self._index.add(vectors.astype(np.float32))
        self._chunks = list(chunks)

        log.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index + metadata to disk."""
        import faiss
        if self._index is None:
            raise RuntimeError("Index not built yet.")
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "dim": self._dim}, f)
        log.info("FAISS index saved → %s", self.index_path)

    def load(self) -> bool:
        """Load a previously saved index. Returns True on success."""
        import faiss
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                meta          = pickle.load(f)
                self._chunks  = meta["chunks"]
                self._dim     = meta["dim"]
            log.info("FAISS index loaded: %d vectors", self._index.ntotal)
            return True
        except Exception as exc:
            log.error("FAISS load failed: %s", exc)
            return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = FAISS_TOP_K,
    ) -> List[Tuple[float, "Chunk"]]:
        """
        Returns list of (score, Chunk) sorted descending by cosine similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((float(score), self._chunks[idx]))
        return results

    # ── Incremental add ───────────────────────────────────────────────────────

    def add(self, chunks: List["Chunk"], vectors: np.ndarray) -> None:
        """Append new chunks to an existing index."""
        import faiss
        if self._index is None:
            dim          = vectors.shape[1]
            self._dim    = dim
            self._index  = faiss.IndexFlatIP(dim)

        self._index.add(vectors.astype(np.float32))
        self._chunks.extend(chunks)
        log.info("Added %d chunks; index total: %d", len(chunks), self._index.ntotal)

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def chunks(self) -> List["Chunk"]:
        return list(self._chunks)
