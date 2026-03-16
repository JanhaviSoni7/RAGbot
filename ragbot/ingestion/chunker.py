"""ragbot/ingestion/chunker.py

Semantic + structure-aware chunking.

Strategy (in order of priority):
1. Structural split  — respect headings / paragraph boundaries from markdown.
2. Sentence-window   — group sentences into windows of CHUNK_SIZE tokens.
3. Overlap           — add CHUNK_OVERLAP tokens of context from adjacent chunks.

Each Chunk carries a rich metadata dict for downstream filtering /
source attribution.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_LEN,
    SENTENCE_WINDOW,
)
from ragbot.ingestion.loader import RawDocument
from ragbot.utils.logger import get_logger

log = get_logger(__name__)

# Lazy NLTK import
_sent_tokenize = None

def _get_sent_tokenize():
    global _sent_tokenize
    if _sent_tokenize is None:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            log.info("Downloading NLTK punkt tokenizer …")
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize as _st
        _sent_tokenize = _st
    return _sent_tokenize


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:  str
    text:      str
    metadata:  dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text.split())

    # Ensure consistent pickling regardless of import path
    def __reduce__(self):
        # Return constructor and args to recreate the object
        return (Chunk, (self.chunk_id, self.text, self.metadata))


# Register with copyreg in case other pickling mechanisms are used
import copyreg

def _pickle_chunk(chunk: "Chunk"):
    return Chunk, (chunk.chunk_id, chunk.text, chunk.metadata)

copyreg.pickle(Chunk, _pickle_chunk)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BLANK_RE   = re.compile(r"\n{2,}")


def _clean_text(text: str) -> str:
    """Normalise unicode, collapse excess whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _approx_tokens(text: str) -> int:
    """Fast approximation: 1 token ≈ 4 characters (BPE heuristic)."""
    return len(text) // 4


def _split_by_structure(text: str) -> List[str]:
    """
    Split on headings first, then on blank lines (paragraphs).
    Returns a list of logical 'sections'.
    """
    # Find heading positions
    sections: List[str] = []
    last = 0
    for m in _HEADING_RE.finditer(text):
        if m.start() > last:
            chunk = text[last:m.start()].strip()
            if chunk:
                sections.append(chunk)
        last = m.start()
    if last < len(text):
        chunk = text[last:].strip()
        if chunk:
            sections.append(chunk)

    if len(sections) <= 1:
        # No headings — split on double newlines
        sections = [s.strip() for s in _BLANK_RE.split(text) if s.strip()]

    return sections


def _sentence_window_chunks(
    section: str,
    chunk_size: int  = CHUNK_SIZE,
    overlap: int     = CHUNK_OVERLAP,
    win: int         = SENTENCE_WINDOW,
) -> List[str]:
    """
    Group sentences into token-bounded windows with overlap.
    Returns raw text strings for the section.
    """
    sent_tokenize = _get_sent_tokenize()
    sentences     = sent_tokenize(section)
    if not sentences:
        return []

    chunks:  List[str] = []
    current: List[str] = []
    current_len        = 0

    for i, sent in enumerate(sentences):
        tok_est = _approx_tokens(sent)

        # If adding this sentence would overflow, flush and start new chunk
        if current_len + tok_est > chunk_size and current:
            # Window context: prepend last `win` sentences of current
            window_prefix = " ".join(current[-win:]) if win > 0 else ""
            chunks.append(" ".join(current))
            # Retain overlap sentences for next chunk
            overlap_sents = current[-win:] if win > 0 else []
            current       = overlap_sents + [sent]
            current_len   = sum(_approx_tokens(s) for s in current)
        else:
            current.append(sent)
            current_len += tok_est

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(doc: RawDocument) -> List[Chunk]:
    """
    Chunk a single RawDocument into Chunk objects.

    Applies:
      • Text cleaning
      • Structural split (headings → paragraphs)
      • Sentence-window chunking within each section
      • Overlap between adjacent chunks
    """
    text     = _clean_text(doc.content)
    sections = _split_by_structure(text)
    chunks:  List[Chunk] = []
    chunk_idx = 0

    for sec_idx, section in enumerate(sections):
        if _approx_tokens(section) <= CHUNK_SIZE:
            # Short section — keep as-is
            raw_chunks = [section]
        else:
            raw_chunks = _sentence_window_chunks(section)

        for raw in raw_chunks:
            raw = raw.strip()
            if len(raw) < MIN_CHUNK_LEN:
                continue

            chunk_id = (
                f"{doc.metadata.get('file_name', 'doc')}"
                f"_p{doc.page_no}"
                f"_s{sec_idx}"
                f"_c{chunk_idx}"
            )
            chunks.append(Chunk(
                chunk_id = chunk_id,
                text     = raw,
                metadata = {
                    **doc.metadata,
                    "page_no":    doc.page_no,
                    "section_no": sec_idx,
                    "chunk_no":   chunk_idx,
                    "source":     doc.source,
                    "char_len":   len(raw),
                    "approx_tok": _approx_tokens(raw),
                },
            ))
            chunk_idx += 1

    log.debug("Chunked '%s' page %d → %d chunks",
              doc.metadata.get("file_name", "?"), doc.page_no, len(chunks))
    return chunks


def chunk_documents(docs: List[RawDocument]) -> List[Chunk]:
    """Chunk all documents; returns flat list of Chunk objects."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    log.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks
