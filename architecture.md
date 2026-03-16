# DeepCytes RAGBot — System Architecture

## Overview

A fully-local, enterprise-grade Retrieval-Augmented Generation (RAG) chatbot that supports multi-document ingestion, hybrid retrieval, cross-encoder reranking, and short-term memory — all running on-device using Ollama.

---

## Pipeline Flow

```
Documents (PDF/TXT/IMG)
       │
       ▼
[Document Loader]         ← PyMuPDF, pytesseract, Pillow
       │
       ▼
[Chunker]                 ← Semantic + structure-aware (sentence-window, paragraph)
       │
       ▼
[Embedder]                ← BGE-M3 via sentence-transformers (local)
       │
    ┌──┴──┐
    │     │
[FAISS]  [BM25]           ← Dense + sparse indexes
    │     │
    └──┬──┘
       │
       ▼
[Hybrid Retriever]        ← RRF (Reciprocal Rank Fusion)
       │
       ▼
[Cross-Encoder Reranker]  ← ms-marco-MiniLM-L-6-v2 (local)
       │
       ▼
[Query Rewriter]          ← Sub-query decomposition, HyDE
       │
       ▼
[Context Assembler]       ← De-duplicate, truncate, format
       │
       ▼
[Short-Term Memory]       ← Last N turns, entity tracking
       │
       ▼
[Ollama LLM Generator]    ← llama3 / mistral / phi3 (local)
       │
       ▼
[Streamlit UI]            ← Document upload, chat, source inspector
```

---

## Module Map

```
ragbot1/
├── app.py                     # Streamlit entry point
├── config.py                  # All configuration (paths, model names, params)
├── requirements.txt
│
├── ragbot/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # PDF/TXT/image loading
│   │   └── chunker.py         # Semantic + structural chunking
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embedder.py        # BGE-M3 embeddings
│   │   ├── faiss_store.py     # FAISS index management
│   │   └── bm25_store.py      # BM25 sparse index
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py# RRF fusion of FAISS + BM25
│   │   ├── reranker.py        # Cross-encoder reranking
│   │   └── query_rewriter.py  # Query expansion / rewriting
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context_assembler.py # Context build & formatting
│   │   ├── memory.py           # Short-term memory (STM)
│   │   └── llm.py              # Ollama LLM interface
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Structured JSON logging
```
