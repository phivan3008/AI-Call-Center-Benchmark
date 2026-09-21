"""Time sources.

Latency metrics use the harness-side monotonic clock only (ARCHITECTURE §6.2);
wall-clock time is used for provenance timestamps. Both are injectable for tests.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...

    def utc_now(self) -> datetime: ...


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def utc_now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """Manually advanced clock for deterministic tests."""

    def __init__(self, start_ns: int = 0, start_utc: datetime | None = None) -> None:
        self._ns = start_ns
        self._utc = start_utc or datetime(2026, 1, 1, tzinfo=UTC)

    def advance_ms(self, ms: float) -> None:
        self._ns += int(ms * 1_000_000)

    def monotonic_ns(self) -> int:
        return self._ns

    def utc_now(self) -> datetime:
        return self._utc
