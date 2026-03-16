"""ragbot/retrieval/reranker.py

Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Cross-encoders jointly encode (query, passage) pairs and output a
relevance score that is much more accurate than bi-encoder similarity
alone, at the cost of O(N) forward passes per query.

We run this on the top HYBRID_TOP_K candidates → return RERANKER_TOP_K.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

from config import RERANKER_MODEL_NAME, RERANKER_TOP_K
from ragbot.utils.logger import get_logger

if TYPE_CHECKING:
    from ragbot.ingestion.chunker import Chunk

log = get_logger(__name__)


# ── Singleton ─────────────────────────────────────────────────────────────────

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        log.info("Loading cross-encoder reranker '%s'…", RERANKER_MODEL_NAME)
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        log.info("Cross-encoder reranker loaded.")
    return _reranker


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    query:      str,
    candidates: List[Tuple[float, "Chunk"]],
    top_k:      int = RERANKER_TOP_K,
) -> List[Tuple[float, "Chunk"]]:
    """
    Rerank candidates using a cross-encoder.

    Args:
        query:      User/rewritten query string.
        candidates: List of (retrieval_score, Chunk) from HybridRetriever.
        top_k:      Number of results to keep after reranking.

    Returns:
        List of (reranker_score, Chunk) sorted descending.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs    = [(query, c.text) for _, c in candidates]

    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        log.error("Reranker prediction failed: %s — returning original order.", exc)
        return candidates[:top_k]

    scored = list(zip(scores, [c for _, c in candidates]))
    scored.sort(key=lambda x: x[0], reverse=True)

    result = [(float(s), chunk) for s, chunk in scored[:top_k]]
    log.info("Reranked %d → top %d (best score: %.4f)",
             len(candidates), len(result),
             result[0][0] if result else 0.0)
    return result
