"""ragbot/retrieval/hybrid_retriever.py

Hybrid (dense + sparse) retriever using Reciprocal Rank Fusion (RRF).

RRF formula:
    score(d) = Σ_{r ∈ rankings} 1 / (k + rank_r(d))

where k=60 is the standard smoothing constant and rank_r(d) is the
1-indexed position of document d in ranking r.

This fuses FAISS and BM25 rankings into a single, calibrated score
that is resilient to individual ranker weaknesses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

from config import FAISS_TOP_K, BM25_TOP_K, HYBRID_TOP_K, RRF_K
from ragbot.utils.logger import get_logger

if TYPE_CHECKING:
    from ragbot.ingestion.chunker import Chunk
    from ragbot.indexing.faiss_store import FAISSStore
    from ragbot.indexing.bm25_store  import BM25Store
    from ragbot.indexing.embedder    import embed_query as EmbedQuery

log = get_logger(__name__)


def reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[float, "Chunk"]],
    k:      int = RRF_K,
    top_k:  int = HYBRID_TOP_K,
) -> List[Tuple[float, "Chunk"]]:
    """
    Fuse N ranked lists via RRF.

    Args:
        *ranked_lists: Each list is [(score, Chunk), …] sorted desc by score.
        k:             RRF smoothing constant.
        top_k:         Number of results to return.

    Returns:
        Fused [(rrf_score, Chunk), …] sorted desc.
    """
    rrf_scores: Dict[str, float]  = defaultdict(float)
    chunk_map:  Dict[str, "Chunk"] = {}

    for ranked in ranked_lists:
        for rank, (_, chunk) in enumerate(ranked, start=1):
            cid                   = chunk.chunk_id
            rrf_scores[cid]      += 1.0 / (k + rank)
            chunk_map[cid]        = chunk

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, chunk_map[cid]) for cid, score in fused[:top_k]]


class HybridRetriever:
    """
    Orchestrates FAISS + BM25 retrieval and fuses them with RRF.
    """

    def __init__(
        self,
        faiss_store: "FAISSStore",
        bm25_store:  "BM25Store",
    ):
        self.faiss = faiss_store
        self.bm25  = bm25_store

    def retrieve(
        self,
        query:        str,
        query_vec:    "np.ndarray",
        faiss_top_k:  int = FAISS_TOP_K,
        bm25_top_k:   int = BM25_TOP_K,
        hybrid_top_k: int = HYBRID_TOP_K,
    ) -> List[Tuple[float, "Chunk"]]:
        """
        Run dense + sparse retrieval and fuse.

        Args:
            query:        Raw query string (for BM25).
            query_vec:    Precomputed dense embedding (for FAISS).
            *_top_k:      Per-index retrieval budget.

        Returns:
            List of (rrf_score, Chunk) pairs, descending.
        """
        dense_results  = self.faiss.search(query_vec, top_k=faiss_top_k)
        sparse_results = self.bm25.search(query, top_k=bm25_top_k)

        log.debug("Dense hits: %d | Sparse hits: %d",
                  len(dense_results), len(sparse_results))

        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k     = RRF_K,
            top_k = hybrid_top_k,
        )
        log.info("Hybrid retrieval → %d candidates", len(fused))
        return fused
