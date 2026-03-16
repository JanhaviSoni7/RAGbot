"""ragbot/retrieval/query_rewriter.py

Query rewriting + sub-query decomposition via the local Ollama LLM.

Two techniques are applied in sequence:

1. Clarification rewrite — rephrase the user query to be self-contained
   and unambiguous, resolving pronouns/coreferences using STM context.

2. Sub-query decomposition — break complex questions into NUM_SUBQUERIES
   simpler sub-questions, each targeting a different aspect of the answer.
   All sub-queries are also embedded and retrieved, giving much wider
   coverage of the relevant corpus.  Results are merged before reranking.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config import OLLAMA_MODEL, NUM_SUBQUERIES, REWRITE_ENABLED
from ragbot.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Fire a short completion at Ollama and return the raw response text."""
    try:
        import ollama
        response = ollama.generate(
            model   = model,
            prompt  = prompt,
            options = {"temperature": 0.0, "num_predict": 128},  # Reduced from 512
        )
        return response["response"].strip()
    except Exception as exc:
        log.warning("Ollama call failed in query_rewriter: %s", exc)
        return ""


def _extract_lines(text: str, expected: int) -> List[str]:
    """
    Extract a numbered list from LLM output.
    Tries to parse 'N. ...' or '- ...' patterns; falls back to splitting.
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)   # strip "1. " prefix
        line = re.sub(r"^[-•]\s*",       "", line)   # strip "- " prefix
        if line:
            lines.append(line)

    # Trim / pad to expected count
    lines = lines[:expected]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_query(
    query:   str,
    history: Optional[List[dict]] = None,
) -> str:
    """
    Clarify the user query using conversation history (STM).

    Returns a rewritten, self-contained query string.
    If rewriting fails or is disabled, returns the original query.
    """
    if not REWRITE_ENABLED:
        return query

    history_str = ""
    if history:
        turns = []
        for turn in history[-4:]:    # Last 2 exchanges
            role = turn.get("role", "user")
            msg  = turn.get("content", "")
            turns.append(f"{role.capitalize()}: {msg}")
        history_str = "\n".join(turns)

    prompt = f"""You are a search query optimizer. Rewrite the following query to be clear, specific, and self-contained. If there is conversation history, resolve any pronouns or references.

Conversation history:
{history_str if history_str else "(none)"}

Original query: {query}

Rewritten query (one line, no explanation):"""

    rewritten = _call_ollama(prompt)
    if not rewritten or len(rewritten) > 500:
        return query   # fallback

    log.info("Query rewritten: '%s' → '%s'", query[:60], rewritten[:60])
    return rewritten


def decompose_query(query: str, n: int = NUM_SUBQUERIES) -> List[str]:
    """
    Decompose the query into N sub-questions.

    Returns a list of sub-query strings (may be shorter than n if the
    LLM produces fewer, or just [query] on failure).
    """
    if not REWRITE_ENABLED:
        return [query]

    prompt = f"""You are a research assistant. Break the following question into {n} specific, independent sub-questions that together cover the full answer.

Question: {query}

Output exactly {n} sub-questions, one per line, numbered:"""

    raw = _call_ollama(prompt)
    if not raw:
        return [query]

    sub_qs = _extract_lines(raw, n)
    if not sub_qs:
        return [query]

    # Always include original query as anchor
    all_qs = [query] + sub_qs
    log.info("Query decomposed into %d sub-queries", len(all_qs))
    return all_qs
