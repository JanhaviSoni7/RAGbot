"""ragbot/pipeline.py

RAGPipeline — the single class that wires together every subsystem.

Lifecycle:
    1.  ingest(paths)   — load, chunk, embed, index
    2.  query(q)        — rewrite, retrieve, rerank, assemble, generate

This is deliberately a stateful object (one per Streamlit session) so
that the FAISS / BM25 indexes and the short-term memory survive across
requests without disk round-trips.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from config import FAISS_TOP_K, BM25_TOP_K, HYBRID_TOP_K, RERANKER_TOP_K
from ragbot.ingestion        import load_documents, chunk_documents, RawDocument, Chunk
from ragbot.indexing         import embed_query, embed_chunks, FAISSStore, BM25Store
from ragbot.retrieval        import HybridRetriever, rerank, rewrite_query, decompose_query
from ragbot.generation       import ShortTermMemory, assemble_context, stream_response
from ragbot.utils.logger     import get_logger

log = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Attributes:
        faiss   : FAISSStore
        bm25    : BM25Store
        memory  : ShortTermMemory
        chunks  : All indexed Chunk objects
        retriever: HybridRetriever
    """

    def __init__(self):
        self.faiss    = FAISSStore()
        self.bm25     = BM25Store()
        self.memory   = ShortTermMemory()
        self.chunks:  List[Chunk] = []
        self.retriever: Optional[HybridRetriever] = None
        self._index_ready = False

        # Try loading persisted indexes from a previous session
        self._try_load_indexes()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _try_load_indexes(self) -> bool:
        faiss_ok = self.faiss.load()
        bm25_ok  = self.bm25.load()
        if faiss_ok and bm25_ok:
            self.chunks        = self.faiss.chunks
            self.retriever     = HybridRetriever(self.faiss, self.bm25)
            self._index_ready  = True
            log.info("Loaded persisted indexes (%d chunks).", len(self.chunks))
            return True
        return False

    def ingest(
        self,
        file_paths: List[str | Path],
        incremental: bool = False,
    ) -> int:
        """
        Load, chunk, embed, and index documents.

        Args:
            file_paths:  List of file paths to process.
            incremental: If True, ADD to existing index; otherwise rebuild.

        Returns:
            Number of new chunks indexed.
        """
        t0 = time.time()
        log.info("Ingestion started for %d file(s).", len(file_paths))

        # 1. Load
        raw_docs: List[RawDocument] = load_documents(file_paths)
        if not raw_docs:
            log.warning("No documents loaded.")
            return 0

        # 2. Chunk
        new_chunks = chunk_documents(raw_docs)
        if not new_chunks:
            log.warning("Chunking produced no output.")
            return 0

        # 3. Embed
        log.info("Embedding %d chunks…", len(new_chunks))
        vectors = embed_chunks(new_chunks, show_progress=True)

        # 4. Index
        if incremental and self._index_ready:
            self.faiss.add(new_chunks, vectors)
            self.bm25.add(new_chunks)
            self.chunks.extend(new_chunks)
        else:
            self.faiss.build(new_chunks, vectors)
            self.bm25.build(new_chunks)
            self.chunks = list(new_chunks)

        # 5. Persist
        self.faiss.save()
        self.bm25.save()

        # 6. Wire retriever
        self.retriever    = HybridRetriever(self.faiss, self.bm25)
        self._index_ready = True

        elapsed = time.time() - t0
        log.info("Ingestion complete: %d chunks in %.1fs.", len(new_chunks), elapsed)
        return len(new_chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        model:      str = "",
        stream:     bool = True,
        selected_docs: list = None,
    ) -> Tuple[Generator | str, List[dict], dict]:
        """
        Full RAG query pipeline with timing instrumentation.

        Args:
            user_query: Raw user input.
            model:      Ollama model name (falls back to config default).
            stream:     If True returns a generator for st.write_stream.
            selected_docs: List of file names to limit retrieval to. None = use all.

        Returns:
            (answer, sources, timings)
            • answer  : streaming generator or full string
            • sources : list of source attribution dicts
            • timings : dict with elapsed time for each stage
        """
        if not self._index_ready:
            err = "⚠️ No documents indexed yet. Please upload documents first."
            return (t for t in [err]), [], {}

        from config import OLLAMA_MODEL
        model = model or OLLAMA_MODEL

        timings = {}

        # ── Step 1: Query Rewriting ───────────────────────────────────────────
        t_start = time.time()
        history_dicts = self.memory.to_dict_list()
        rewritten = rewrite_query(user_query, history=history_dicts)
        timings["query_rewriting"] = time.time() - t_start

        # ── Step 2: Sub-query Decomposition ──────────────────────────────────
        sub_queries = decompose_query(rewritten)

        # ── Step 3: Hybrid Retrieval for each sub-query ───────────────────────
        t_start = time.time()
        all_candidates: List[Tuple[float, Chunk]] = []
        seen_ids: set = set()

        for sq in sub_queries:
            q_vec   = embed_query(sq)
            results = self.retriever.retrieve(
                query        = sq,
                query_vec    = q_vec,
                faiss_top_k  = FAISS_TOP_K,
                bm25_top_k   = BM25_TOP_K,
                hybrid_top_k = HYBRID_TOP_K,
            )
            for score, chunk in results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_candidates.append((score, chunk))

        # ── Filter by selected documents ──────────────────────────────────────
        if selected_docs:
            selected_set = set(doc.lower() for doc in selected_docs)
            all_candidates = [
                (score, chunk) for score, chunk in all_candidates
                if chunk.metadata.get("file_name", "").lower() in selected_set
            ]
            log.info("Filtered to %d candidates from selected %d documents.",
                     len(all_candidates), len(selected_docs))

        log.info("Total unique candidates from all sub-queries: %d", len(all_candidates))
        timings["hybrid_retrieval"] = time.time() - t_start

        # ── Step 4: Cross-encoder Reranking ──────────────────────────────────
        t_start = time.time()
        reranked = rerank(
            query      = rewritten,
            candidates = all_candidates,
            top_k      = RERANKER_TOP_K,
        )
        timings["reranking"] = time.time() - t_start

        # ── Step 5: Context Assembly ──────────────────────────────────────────
        t_start = time.time()
        context, sources = assemble_context(reranked)
        timings["context_assembly"] = time.time() - t_start

        if not context.strip():
            msg = "⚠️ No relevant context found in the documents for this query."
            self.memory.add_user(user_query)
            self.memory.add_assistant(msg)
            return (t for t in [msg]), [], timings

        # ── Step 6: Generation ────────────────────────────────────────────────
        # Update STM with user turn BEFORE generating so history is consistent
        self.memory.add_user(user_query)

        t_gen_start = time.time()
        if stream:
            # We capture the full text to update STM after streaming
            # by wrapping the generator
            gen = stream_response(
                context = context,
                query   = rewritten,
                history = self.memory.to_dict_list()[:-1],  # exclude current user
                model   = model,
            )
            # Wrap generator to capture output for STM and timing
            return self._wrap_stream(gen, timings, t_gen_start), sources, timings
        else:
            from ragbot.generation.llm import generate_response
            answer = generate_response(
                context = context,
                query   = rewritten,
                history = self.memory.to_dict_list()[:-1],
                model   = model,
            )
            timings["llm_generation"] = time.time() - t_gen_start
            self.memory.add_assistant(answer)
            return answer, sources, timings

    def _wrap_stream(self, gen: Generator, timings: dict, t_start: float) -> Generator:
        """Wrap streaming generator to update STM and record timing after completion."""
        buffer = []
        for token in gen:
            buffer.append(token)
            yield token
        timings["llm_generation"] = time.time() - t_start
        self.memory.add_assistant("".join(buffer))

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear_memory(self) -> None:
        self.memory.clear()

    def reset_index(self) -> None:
        """Wipe all indexes and memory (fresh start)."""
        self.faiss    = FAISSStore()
        self.bm25     = BM25Store()
        self.chunks   = []
        self.retriever = None
        self._index_ready = False
        from config import FAISS_INDEX_FILE, FAISS_META_FILE, BM25_INDEX_FILE
        for f in [FAISS_INDEX_FILE, FAISS_META_FILE, BM25_INDEX_FILE]:
            if Path(f).exists():
                Path(f).unlink()
        log.info("Index reset complete.")

    @property
    def doc_count(self) -> int:
        """Number of unique source files indexed."""
        return len({c.metadata.get("file_name", "") for c in self.chunks})

    @property
    def document_names(self) -> List[str]:
        """List of unique document file names in the index."""
        return sorted({c.metadata.get("file_name", "") for c in self.chunks if c.metadata.get("file_name", "")})

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def is_ready(self) -> bool:
        return self._index_ready
