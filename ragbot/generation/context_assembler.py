"""ragbot/generation/context_assembler.py

Context assembly layer.

Responsibilities:
  1. De-duplicate retrieved chunks (same text ≈ same source, keep highest score).
  2. Sort by reranker score (already sorted; preserve).
  3. Truncate to MAX_CONTEXT_TOKENS.
  4. Format each chunk with a citeable source header.
  5. Return both the formatted context string AND a list of source dicts
     for UI attribution.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

from config import CONTEXT_SEPARATOR, MAX_CONTEXT_TOKENS
from ragbot.utils.logger import get_logger

if TYPE_CHECKING:
    from ragbot.ingestion.chunker import Chunk

log = get_logger(__name__)


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _deduplicate(
    ranked: List[Tuple[float, "Chunk"]],
) -> List[Tuple[float, "Chunk"]]:
    """
    Remove near-duplicate chunks.
    Strategy: if two chunks share the same chunk_id, keep the first
    (highest score). A more sophisticated version could use Jaccard
    similarity, but chunk_id dedup handles the common case where the
    same chunk appears in multiple sub-query results.
    """
    seen: set = set()
    deduped   = []
    for score, chunk in ranked:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduped.append((score, chunk))
    return deduped


def assemble_context(
    ranked: List[Tuple[float, "Chunk"]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> Tuple[str, List[dict]]:
    """
    Build the context string to inject into the LLM prompt.

    Args:
        ranked:     [(score, Chunk), …] from reranker, descending.
        max_tokens: Approximate token budget for the whole context block.

    Returns:
        (context_str, sources)
        • context_str: formatted text to paste into prompt.
        • sources:     list of dicts {file_name, page_no, score, snippet}
                       for UI attribution.
    """
    ranked    = _deduplicate(ranked)
    sources   = []
    blocks    = []
    used_toks = 0

    for idx, (score, chunk) in enumerate(ranked, start=1):
        meta     = chunk.metadata
        filename = meta.get("file_name", "unknown")
        page_no  = meta.get("page_no",   meta.get("page", "?"))
        header   = f"[Source {idx}] {filename} | Page {page_no} | Score: {score:.3f}"
        block    = f"{header}\n{chunk.text}"
        btoks    = _approx_tokens(block)

        if used_toks + btoks > max_tokens and blocks:
            log.debug("Context budget hit at chunk %d; stopping.", idx)
            break

        blocks.append(block)
        used_toks += btoks
        sources.append({
            "idx":       idx,
            "file_name": filename,
            "page_no":   page_no,
            "score":     round(score, 4),
            "snippet":   chunk.text[:200],
            "chunk_id":  chunk.chunk_id,
        })

    context_str = CONTEXT_SEPARATOR.join(blocks)
    log.info("Context assembled: %d chunks, ~%d tokens", len(blocks), used_toks)
    return context_str, sources
