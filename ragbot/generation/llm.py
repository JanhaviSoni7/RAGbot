"""ragbot/generation/llm.py

Ollama LLM interface.

Wraps the Ollama Python client to provide:
  • Streaming generation (token-by-token, compatible with Streamlit's
    st.write_stream).
  • Non-streaming generation (for convenience in batch/test contexts).
  • System prompt injection with explicit grounding instructions.
  • STM history injection in Ollama's message format.
  • Graceful error handling.
"""

from __future__ import annotations

from typing import Generator, List, Optional

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
)
from ragbot.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DeepCytes RAGBot, an expert analyst with precise, concise communication style.

Your mission is to answer the user's questions using ONLY the provided CONTEXT documents.
Keep answers **brief, focused, and directly on-topic**. Avoid verbosity.

## ANSWERING STANDARDS

1. **Be concise.** Answer directly without filler or lengthy introductions. 

2. **Cite every claim.** Append source in brackets after each fact: [Source 1], [Source 2, 3].
   Do not claim anything you cannot cite.

3. **Use all context.** Review ALL provided context blocks before answering.

4. **Grounding only.** Use ONLY information in the CONTEXT. If not found, state:
   > ⚠️ Not found in the provided documents.

5. **No hallucination.** Never invent facts, data, dates, or technical details. Admit uncertainty.

6. **Minimal formatting.** Write in clear, professional English. Use markdown ONLY when essential.
   For code/formulas use ``` or $$ syntax.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(
    context:    str,
    query:      str,
    history:    Optional[List[dict]] = None,
) -> List[dict]:
    """
    Build the Ollama messages list.

    Format:
        [system]
        [history turns …]
        [user: context + question]
    """
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject STM history (exclude the last user message — we'll add it below)
    if history:
        for turn in history:
            if turn["role"] in {"user", "assistant"}:
                messages.append(turn)

    # Current user turn with context
    user_content = f"""## CONTEXT DOCUMENTS

{context}

---

## QUESTION
{query}

## ANSWER (concise and direct)
Answer briefly, directly addressing the question. Cite sources. Do not add lengthy preambles or summaries."""
    messages.append({"role": "user", "content": user_content})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def stream_response(
    context: str,
    query:   str,
    history: Optional[List[dict]] = None,
    model:   str = OLLAMA_MODEL,
) -> Generator[str, None, None]:
    """
    Stream an LLM response token-by-token.

    Yields:
        str — text tokens as they arrive.

    Usage in Streamlit:
        response_text = st.write_stream(stream_response(context, query, history))
    """
    import ollama

    messages = _build_messages(context, query, history)
    log.info("Streaming response from Ollama (model=%s)…", model)

    try:
        stream = ollama.chat(
            model    = model,
            messages = messages,
            stream   = True,
            options  = {
                "temperature":  OLLAMA_TEMPERATURE,
                "num_ctx":      OLLAMA_CONTEXT_WINDOW,
                "num_predict":  OLLAMA_NUM_PREDICT,
            },
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except Exception as exc:
        error_msg = f"\n\n⚠️ Generation error: {exc}"
        log.error("Ollama streaming error: %s", exc)
        yield error_msg


def generate_response(
    context: str,
    query:   str,
    history: Optional[List[dict]] = None,
    model:   str = OLLAMA_MODEL,
) -> str:
    """
    Non-streaming response — returns the full answer as a string.
    """
    import ollama

    messages = _build_messages(context, query, history)
    log.info("Non-streaming response from Ollama (model=%s)…", model)

    try:
        response = ollama.chat(
            model    = model,
            messages = messages,
            stream   = False,
            options  = {
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx":     OLLAMA_CONTEXT_WINDOW,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        )
        return response["message"]["content"]
    except Exception as exc:
        log.error("Ollama generation error: %s", exc)
        return f"⚠️ Error generating response: {exc}"


def list_available_models() -> List[str]:
    """Return a list of locally available Ollama model names."""
    try:
        import ollama
        models = ollama.list()
        return [m["model"] for m in models.get("models", [])]
    except Exception as exc:
        log.warning("Could not list Ollama models: %s", exc)
        return [OLLAMA_MODEL]
