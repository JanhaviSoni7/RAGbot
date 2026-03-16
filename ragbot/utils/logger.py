"""ragbot/utils/logger.py

Structured JSON logger for the entire RAG pipeline.
Every module imports `get_logger(__name__)` for consistent output.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_FILE, LOG_LEVEL


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line (JSONL)."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "module":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that writes JSON to file + pretty text to stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # ── File handler (JSONL) ──────────────────────────────────────────────────
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    # ── Console handler (pretty) ──────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(ch)
    logger.propagate = False
    return logger
