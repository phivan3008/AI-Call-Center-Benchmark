"""Structured JSON logging (structlog).

Every log line is a JSON object. Context such as ``run_id``, ``model_id``, ``layer``
and ``sample_id`` is attached with :func:`bind_context` and propagates via contextvars.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, TextIO

import structlog

_file_handle: TextIO | None = None


class _Tee:
    """Writes each log line to stderr and, optionally, a JSONL file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def configure_logging(level: str = "INFO", jsonl_path: Path | None = None) -> None:
    """Configure structlog for JSON output. Safe to call more than once."""
    global _file_handle
    if _file_handle is not None:
        _file_handle.close()
        _file_handle = None

    streams: list[TextIO] = [sys.stderr]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        _file_handle = jsonl_path.open("a", encoding="utf-8")
        streams.append(_file_handle)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=_Tee(*streams)),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )


def close_logging() -> None:
    """Close the JSONL file handle, if any, and fall back to stderr-only logging."""
    configure_logging()


def bind_context(**values: Any) -> None:
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
