from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.core.clock import FakeClock, SystemClock
from benchmark.core.logging import bind_context, close_logging, configure_logging, get_logger
from benchmark.core.registry import Registry, RegistryError


def test_json_logs_with_context(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"
    configure_logging("INFO", path)
    bind_context(run_id="R1", layer="L1")
    log = get_logger("t")
    log.info("hello", value="日本語")
    log.debug("hidden")
    close_logging()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "run_id": "R1",
        "layer": "L1",
        "event": "hello",
        "value": "日本語",
        "level": "info",
        "timestamp": record["timestamp"],
    }


def test_registry() -> None:
    registry: Registry[int] = Registry("thing")

    @registry.register("a")
    def _unused() -> None:  # pragma: no cover - only the decorator matters
        pass

    registry.add("b", 2)
    assert registry.get("b") == 2
    assert "a" in registry
    assert registry.names() == ["a", "b"]
    with pytest.raises(RegistryError, match="already registered"):
        registry.add("b", 3)
    with pytest.raises(RegistryError, match="Unknown thing 'c'"):
        registry.get("c")


def test_clocks() -> None:
    clock = FakeClock(start_ns=10)
    clock.advance_ms(1.5)
    assert clock.monotonic_ns() == 1_500_010
    system = SystemClock()
    assert system.monotonic_ns() <= system.monotonic_ns()
    assert system.utc_now().tzinfo is not None
