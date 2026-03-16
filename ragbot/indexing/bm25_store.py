"""ragbot/indexing/bm25_store.py

BM25 sparse index using rank_bm25.

Provides a complementary lexical signal to the dense FAISS retrieval.
At query time, results from both indexes are fused via Reciprocal Rank
Fusion (RRF) in the HybridRetriever.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

from config import BM25_INDEX_FILE, BM25_TOP_K
from ragbot.utils.logger import get_logger

if TYPE_CHECKING:
    from ragbot.ingestion.chunker import Chunk

log = get_logger(__name__)


# ── Tokeniser (lightweight, no NLTK dependency required) ──────────────────────

_NON_WORD = re.compile(r"[^\w]", re.UNICODE)
_STOP_WORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for",
    "of","with","by","from","is","are","was","were","be","been",
    "that","this","it","its","as","not","no","nor","so","yet",
    "both","either","neither","whether","if","then","than","though",
})


def _tokenize(text: str) -> List[str]:
    tokens = _NON_WORD.sub(" ", text.lower()).split()
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]


class BM25Store:
    """BM25 index with persistence and ranked search."""

    def __init__(self, index_path: Path = BM25_INDEX_FILE):
        self.index_path = index_path
        self._bm25      = None
        self._chunks:   List["Chunk"] = []
        self._tokenized: List[List[str]] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List["Chunk"]) -> None:
        from rank_bm25 import BM25Okapi

        tokenized        = [_tokenize(c.text) for c in chunks]
        self._bm25       = BM25Okapi(tokenized)
        self._chunks     = list(chunks)
        self._tokenized  = tokenized
        log.info("BM25 index built: %d documents", len(chunks))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built yet.")
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25":      self._bm25,
                "chunks":    self._chunks,
                "tokenized": self._tokenized,
            }, f)
        log.info("BM25 index saved → %s", self.index_path)

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path, "rb") as f:
                data             = pickle.load(f)
                self._bm25       = data["bm25"]
                self._chunks     = data["chunks"]
                self._tokenized  = data["tokenized"]
            log.info("BM25 index loaded: %d documents", len(self._chunks))
            return True
        except Exception as exc:
            log.error("BM25 load failed: %s", exc)
            return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> List[Tuple[float, "Chunk"]]:
        """Returns (bm25_score, Chunk) pairs sorted descending."""
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        k      = min(top_k, len(self._chunks))

        # argsort descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]

        results = [
            (float(scores[i]), self._chunks[i])
            for i in top_indices
            if scores[i] > 0
        ]
        return results

    # ── Incremental add ───────────────────────────────────────────────────────

    def add(self, chunks: List["Chunk"]) -> None:
        """Rebuild index with new chunks appended. BM25 requires full rebuild."""
        from rank_bm25 import BM25Okapi

        new_tokenized    = [_tokenize(c.text) for c in chunks]
        self._chunks    += list(chunks)
        self._tokenized += new_tokenized
        self._bm25       = BM25Okapi(self._tokenized)
        log.info("BM25 rebuilt; total: %d documents", len(self._chunks))

    @property
    def size(self) -> int:
        return len(self._chunks)
