"""
config.py — Central configuration for DeepCytes RAGBot.

All hyper-parameters, model paths, and runtime flags live here so
every other module stays clean and import-only.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = ROOT_DIR / "data"
INDEX_DIR  = ROOT_DIR / "indexes"
LOG_DIR    = ROOT_DIR / "logs"
UPLOAD_DIR = ROOT_DIR / "uploads"

for d in (DATA_DIR, INDEX_DIR, LOG_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME      = "BAAI/bge-m3"          # HuggingFace model id
EMBED_BATCH_SIZE      = 32
EMBED_MAX_SEQ_LEN     = 8192
EMBED_NORMALIZE       = True                    # L2 normalise for cosine sim

# ── Cross-encoder reranker ────────────────────────────────────────────────────
RERANKER_MODEL_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K        = 8                       # Final chunks passed to LLM (↑ for depth)

# ── FAISS index ───────────────────────────────────────────────────────────────
FAISS_INDEX_FILE      = INDEX_DIR / "faiss.index"
FAISS_META_FILE       = INDEX_DIR / "faiss_meta.pkl"
FAISS_TOP_K           = 20                      # Initial dense retrieval k

# ── BM25 index ───────────────────────────────────────────────────────────────
BM25_INDEX_FILE       = INDEX_DIR / "bm25.pkl"
BM25_TOP_K            = 20                      # Initial sparse retrieval k

# ── Hybrid retrieval (RRF) ───────────────────────────────────────────────────
RRF_K                 = 60                      # RRF constant (standard = 60)
HYBRID_TOP_K          = 30                      # Candidates fed to reranker

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE            = 512                     # tokens (approx chars/4)
CHUNK_OVERLAP         = 64                      # overlap between chunks
MIN_CHUNK_LEN         = 50                      # discard chunks shorter than this
SENTENCE_WINDOW       = 2                       # sentences before+after anchor

# ── Ollama LLM ────────────────────────────────────────────────────────────────
OLLAMA_HOST           = "http://localhost:11434"
OLLAMA_MODEL          = "qwen2.5:3b"            # Fast, concise model
OLLAMA_TEMPERATURE    = 0.05                    # Very low for factual, to-the-point answers
OLLAMA_CONTEXT_WINDOW = 8192
OLLAMA_NUM_PREDICT    = 1024                    # Max tokens to generate (shorter for concise responses)
OLLAMA_TIMEOUT        = 120                     # seconds

# ── Short-term memory ─────────────────────────────────────────────────────────
STM_MAX_TURNS         = 6                       # Keep last N user+assistant turns
STM_MAX_TOKENS        = 1024                    # Hard cap on memory in prompt

# ── Context assembly ─────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS    = 5500                    # Space reserved for retrieved chunks (↑ for depth)
CONTEXT_SEPARATOR     = "\n\n---\n\n"

# ── Query rewriting ──────────────────────────────────────────────────────────
REWRITE_ENABLED       = True
NUM_SUBQUERIES        = 1                       # Set to 1 to skip decomposition

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE              = LOG_DIR / "ragbot.jsonl"
LOG_LEVEL             = "INFO"

# ── Supported document types ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS  = {
    # Text formats
    ".pdf", ".txt", ".md", ".json",
    # Office formats
    ".docx", ".csv", ".xlsx", ".xls", ".pptx",
    # Image formats (with OCR)
    ".png", ".jpg", ".jpeg", ".tiff"
}
