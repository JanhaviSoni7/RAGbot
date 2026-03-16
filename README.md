# RAGBOT: Enterprise-Grade Retrieval-Augmented Generation Chatbot

A fully-local, production-ready Retrieval-Augmented Generation (RAG) chatbot that processes multi-format documents and provides accurate, source-attributed responses using hybrid search and local LLMs.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Overview

RAGBOT is an advanced chatbot system designed for enterprise document analysis. It ingests documents from various formats (PDFs, Word docs, spreadsheets, images), creates intelligent indexes using dense and sparse embeddings, and generates responses grounded in the provided context using local Ollama models. The system runs entirely on-device, ensuring privacy and no external API dependencies.

### Core Components
- **Ingestion Pipeline**: Multi-format document loading and semantic chunking
- **Indexing Engine**: Hybrid FAISS (dense) + BM25 (sparse) indexing
- **Retrieval System**: Query rewriting, hybrid search with RRF fusion, and cross-encoder reranking
- **Generation Module**: Context assembly, short-term memory, and streaming LLM responses
- **Web UI**: Modern Streamlit interface with dark glassmorphism design

## Key Features

✅ **Multi-Format Support**: PDFs, DOCX, XLSX, PPTX, images (with OCR), TXT, MD, JSON  
✅ **Hybrid Retrieval**: Combines semantic (dense) and keyword (sparse) search for superior accuracy  
✅ **Local Execution**: Runs on Ollama models with no cloud dependencies  
✅ **Source Attribution**: Citations with file names, page numbers, and relevance scores  
✅ **Conversation Memory**: Maintains context across turns with configurable memory limits  
✅ **Streaming Responses**: Real-time token generation for responsive UI  
✅ **Enterprise-Ready**: Structured logging, error resilience, and configurable parameters  

## Architecture

RAGBOT follows a modular, pipeline-based architecture:

```
Data Ingestion → Chunking → Dense + Sparse Indexing
        ↓            ↓              ↓
   Loader      Embedder      FAISS & BM25
           
           ↓ (Query enters here)
       
Hybrid Retrieval (RRF) → Cross-Encoder Reranking → Query Rewriting
           ↓                    ↓                        ↓
    30 candidates          8 top chunks           Sub-query decomposition
                                ↓
                          Context Assembly
                                ↓
                          Short-Term Memory (STM)
                                ↓
                          Ollama LLM Generator
                                ↓
                          Streamlit UI (Response + Sources)
```

### Module Breakdown

#### Ingestion Module (`ragbot/ingestion/`)
- **Loader**: Handles 9+ file formats with format-specific parsers and OCR fallback
- **Chunker**: Semantic chunking with structure awareness, sentence windows, and configurable overlap

#### Indexing Module (`ragbot/indexing/`)
- **Embedder**: BGE-M3 embeddings with batch processing and query prefixing
- **FAISS Store**: Dense vector indexing with incremental additions
- **BM25 Store**: Sparse lexical indexing with custom tokenization

#### Retrieval Module (`ragbot/retrieval/`)
- **Query Rewriter**: LLM-powered query clarification and decomposition
- **Hybrid Retriever**: RRF fusion of dense and sparse results
- **Reranker**: Cross-encoder for final relevance scoring

#### Generation Module (`ragbot/generation/`)
- **LLM Interface**: Streaming/non-streaming Ollama integration
- **Context Assembler**: Citation-formatted context with token budgeting
- **Memory**: Short-term conversation history management

#### Utils (`ragbot/utils/`)
- **Logger**: Structured JSONL logging with console output

## Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running locally
- Tesseract OCR (for image processing)

### Setup
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required Ollama models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Workflow
1. **Upload Documents**: Use the file uploader to add PDFs, docs, etc.
2. **Ingest Data**: Click "Ingest Documents" to process and index files
3. **Chat**: Ask questions in the chat interface; responses include source citations

### Example Interaction
```
User: What are the key features of RAGBOT?

Assistant: RAGBOT offers multi-format document support, hybrid retrieval combining dense and sparse search, and local LLM execution. [Source 1] ragbot/README.md | Page 1 | Score: 0.85
```

### API Usage
For programmatic access, use the `RAGPipeline` class:

```python
from ragbot.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Ingest documents
pipeline.ingest(["document.pdf", "data.xlsx"])

# Query
response = pipeline.query("What is the main topic?")
print(response)
```

## Configuration

All settings are centralized in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_MODEL` | "qwen2.5:3b" | LLM for generation |
| `EMBEDDING_MODEL` | "BAAI/bge-m3" | Embedding model |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `RERANKER_TOP_K` | 8 | Final chunks for LLM |
| `STM_MAX_TURNS` | 6 | Conversation memory turns |

## API Reference

### Ingestion
- `load_document(path)`: Load single file → `List[RawDocument]`
- `chunk_document(doc)`: Chunk document → `List[Chunk]`

### Indexing
- `embed_texts(texts)`: Generate embeddings → `np.ndarray`
- `FAISSStore.build(chunks)`: Create dense index
- `BM25Store.build(chunks)`: Create sparse index

### Retrieval
- `rewrite_query(query, history)`: Clarify query
- `HybridRetriever.retrieve(query)`: Get candidates
- `rerank(query, candidates)`: Score and rank

### Generation
- `stream_response(context, query, history)`: Streaming generator
- `assemble_context(ranked_chunks)`: Format context string
- `ShortTermMemory.add_user(content)`: Update conversation
