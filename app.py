"""app.py — DeepCytes RAGBot Streamlit frontend.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure ragbot/ is importable from project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title       = "RAGBot",
    page_icon        = "🧠",
    layout           = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
    padding-top: 1rem;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc;
}

/* ── Main area ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Custom header ── */
.ragbot-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.ragbot-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ragbot-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.stat-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.1rem;
}

/* ── Chat messages ── */
.stChatMessage {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}
[data-testid="stChatMessage-user"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stChatMessage-assistant"] {
    background: rgba(16,20,40,0.6) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── Source cards ── */
.source-card {
    background: rgba(15,22,46,0.8);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
}
.source-header {
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 0.25rem;
}
.source-snippet {
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    border-left: 2px solid rgba(99,102,241,0.3);
    padding-left: 0.5rem;
    margin-top: 0.3rem;
    line-height: 1.5;
}
.source-score {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.04);
    border: 1.5px dashed rgba(99,102,241,0.35);
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
}

/* ── Model selector ── */
.stSelectbox [data-testid="stSelectbox"] {
    background: rgba(15,22,46,0.8);
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── Status badges ── */
.badge-ready {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-empty {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-error {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(99,102,241,0.05) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Imports (after CSS — faster perceived load)
# ─────────────────────────────────────────────────────────────────────────────
import tempfile
import time

from config import SUPPORTED_EXTENSIONS, UPLOAD_DIR
from ragbot.pipeline          import RAGPipeline
from ragbot.generation.llm    import list_available_models


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline:   RAGPipeline  = RAGPipeline()
if "messages" not in st.session_state:
    st.session_state.messages:   list         = []    # UI display history
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files: list     = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources: list       = []
if "last_timings" not in st.session_state:
    st.session_state.last_timings: dict       = {}
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs: list      = None
if "doc_filters" not in st.session_state:
    st.session_state.doc_filters: dict        = {}

pipeline: RAGPipeline = st.session_state.pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 RAGBot")
    # st.markdown("*Fully local · No API keys · NotebookLM-style*")
    st.divider()

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Model")
    try:
        available_models = list_available_models()
    except Exception:
        available_models = ["llama3"]

    from config import OLLAMA_MODEL
    default_idx = 0
    if OLLAMA_MODEL in available_models:
        default_idx = available_models.index(OLLAMA_MODEL)

    selected_model = st.selectbox(
        "Ollama model",
        options   = available_models,
        index     = default_idx,
        help      = "Only locally available Ollama models are listed.",
    )
    st.session_state.selected_model = selected_model

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("#### 📁 Documents")

    ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    uploaded_files = st.file_uploader(
        f"Upload ({ext_list})",
        type    = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files = True,
        help    = "Files are processed locally. Nothing leaves your machine.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        add_btn  = st.button("➕ Add",    use_container_width=True,
                             help="Incrementally add to existing index")
    with col_b:
        build_btn = st.button("🔄 Rebuild", use_container_width=True,
                              help="Wipe and rebuild index from these files")

    if (add_btn or build_btn) and uploaded_files:
        saved_paths = []
        for uf in uploaded_files:
            dest = UPLOAD_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.read())
            saved_paths.append(dest)

        incremental = add_btn and pipeline.is_ready
        mode_str    = "Adding" if incremental else "Rebuilding"

        with st.spinner(f"⚙️ {mode_str} index…"):
            try:
                n = pipeline.ingest(saved_paths, incremental=incremental)
                if n > 0:
                    st.success(f"✅ Indexed {n} chunks from {len(saved_paths)} file(s).")
                    for sp in saved_paths:
                        fname = Path(sp).name
                        if fname not in st.session_state.ingested_files:
                            st.session_state.ingested_files.append(fname)
                else:
                    st.warning("⚠️ No content extracted — check file types / content.")
            except Exception as exc:
                st.error(f"❌ Ingestion error: {exc}")

    st.divider()

    # ── Document filter ───────────────────────────────────────────────────────
    # Populate from pipeline if empty (e.g., after loading persisted index)
    if not st.session_state.ingested_files and pipeline.is_ready:
        st.session_state.ingested_files = pipeline.document_names

    if st.session_state.ingested_files:
        st.markdown("**🎯 Filter sources**")
        st.caption("Select documents to search within")
        
        # Initialize filter state if needed
        if "doc_filters" not in st.session_state:
            st.session_state.doc_filters = {doc: True for doc in st.session_state.ingested_files}
        
        # Create checkboxes for each document
        for doc in sorted(st.session_state.ingested_files):
            st.session_state.doc_filters[doc] = st.checkbox(
                doc,
                value=st.session_state.doc_filters.get(doc, True),
                key=f"filter_{doc}"
            )
        
        # Get selected documents from checkboxes
        selected_docs = [doc for doc, checked in st.session_state.doc_filters.items() if checked]
        st.session_state.selected_docs = selected_docs if selected_docs else None
        st.divider()
    else:
        st.session_state.selected_docs = None

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.last_sources = []
            pipeline.clear_memory()
            st.rerun()
    with c2:
        if st.button("🔥 Reset All", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.last_sources = []
            st.session_state.ingested_files = []
            pipeline.reset_index()
            pipeline.clear_memory()
            st.rerun()

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Index Stats")
    if pipeline.is_ready:
        st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">{pipeline.doc_count}</div>
    <div class="stat-label">Documents</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{pipeline.chunk_count}</div>
    <div class="stat-label">Chunks</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(pipeline.memory.turns)}</div>
    <div class="stat-label">Memory turns</div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown('<span class="badge-ready">● Index ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-empty">○ No documents indexed</span>', unsafe_allow_html=True)

    st.divider()
    st.caption("DeepCytes RAGBot v1.0 · Fully local · BGE-M3 + Ollama")


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ragbot-header">
  <span style="font-size:2.2rem;">🧠</span>
  <span class="ragbot-title">RAGBot</span>
</div>
<p class="ragbot-subtitle">
  Hybrid retrieval · Cross-encoder reranking · Short-term memory · Fully local
</p>
""", unsafe_allow_html=True)

# ── Source inspector (last query) ─────────────────────────────────────────────
if st.session_state.last_sources:
    with st.expander(f"🔍 Source attribution — {len(st.session_state.last_sources)} chunk(s) used", expanded=False):
        for src in st.session_state.last_sources:
            score_badge = f'<span class="source-score">Score {src["score"]:.3f}</span>'
            st.markdown(f"""
<div class="source-card">
  <div class="source-header">
    [{src["idx"]}] {src["file_name"]} | Page {src["page_no"]} {score_badge}
  </div>
  <div class="source-snippet">{src["snippet"]}…</div>
</div>
""", unsafe_allow_html=True)

# ── Timing inspector ──────────────────────────────────────────────────────────
if st.session_state.last_timings:
    # always show the timings by default
    with st.expander("⏱️ Pipeline Timing", expanded=True):
        cols = st.columns(5)
        timing_items = [
            ("Rewriting", st.session_state.last_timings.get("query_rewriting", 0)),
            ("Retrieval", st.session_state.last_timings.get("hybrid_retrieval", 0)),
            ("Reranking", st.session_state.last_timings.get("reranking", 0)),
            ("Assembly", st.session_state.last_timings.get("context_assembly", 0)),
            ("Generation", st.session_state.last_timings.get("llm_generation", 0)),
        ]
        for col, (label, elapsed) in zip(cols, timing_items):
            with col:
                st.metric(label, f"{elapsed:.2f}s")
        
        total_time = sum(st.session_state.last_timings.values())
        st.caption(f"**Total: {total_time:.2f}s**")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if not pipeline.is_ready:
    st.info("👈 Upload documents in the sidebar to get started.")

user_input = st.chat_input(
    "Ask a question about your documents…",
    disabled = not pipeline.is_ready,
)

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run pipeline and stream assistant response
    with st.chat_message("assistant"):
        model = st.session_state.get("selected_model", "qwen2.5:3b")
        selected_docs = st.session_state.get("selected_docs", None)

        answer_stream, sources, timings = pipeline.query(
            user_query = user_input,
            model      = model,
            stream     = True,
            selected_docs = selected_docs,
        )
        # Stream to UI
        full_answer = st.write_stream(answer_stream)

    # Save to display history
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.last_sources = sources
    st.session_state.last_timings = timings
