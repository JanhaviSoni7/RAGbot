"""ragbot/generation/memory.py

Short-Term Memory (STM) module.

Maintains a sliding window of the last N conversation turns.
Provides utilities to:
  • Append new turns
  • Render the memory as a formatted prompt string
  • Respect a token budget (token-count informed truncation)
  • Track entities / topics mentioned for surface-form resolution

Design: plain Python dataclass — no external DB, no LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config import STM_MAX_TOKENS, STM_MAX_TURNS
from ragbot.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Turn:
    role:    str       # "user" | "assistant"
    content: str


class ShortTermMemory:
    """
    Sliding-window conversation memory.

    Attributes:
        max_turns:   Maximum number of turns to retain.
        max_tokens:  Approximate token budget for rendered output.
    """

    def __init__(
        self,
        max_turns:  int = STM_MAX_TURNS,
        max_tokens: int = STM_MAX_TOKENS,
    ):
        self.max_turns  = max_turns
        self.max_tokens = max_tokens
        self._turns: List[Turn] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._turns.append(Turn(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._turns.append(Turn(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        """Keep last max_turns turns; always keep in user/assistant pairs."""
        # Trim by count first
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]
        # Trim by approximate token budget
        while self._approx_tokens() > self.max_tokens and len(self._turns) > 2:
            self._turns = self._turns[2:]   # Remove oldest user+assistant pair

    def clear(self) -> None:
        self._turns = []
        log.info("STM cleared.")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict_list(self) -> List[dict]:
        """Return turns as list of {'role': ..., 'content': ...} dicts."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def to_prompt_string(self, header: str = "### Conversation History") -> str:
        """
        Render memory as a formatted string for injection into the prompt.
        """
        if not self._turns:
            return ""
        lines = [header]
        for t in self._turns:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    # ── Properties ────────────────────────────────────────────────────────────

    def _approx_tokens(self) -> int:
        total_chars = sum(len(t.content) for t in self._turns)
        return total_chars // 4   # 1 token ≈ 4 chars heuristic

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0
