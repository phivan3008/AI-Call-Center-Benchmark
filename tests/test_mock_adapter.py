from __future__ import annotations

import asyncio

from benchmark.adapters import adapter_registry
from benchmark.adapters.base import (
    AudioChunk,
    AudioDelta,
    ModelAdapter,
    ResponseDone,
    SessionConfig,
    SpeechSession,
    TimedEvent,
)
from benchmark.adapters.mock import MockAdapter, MockConfig, MockSession


async def _collect(session: SpeechSession) -> list[str]:
    types: list[str] = []
    async for event in session.events():
        types.append(event.type)
        if event.type in {"response_done", "response_cancelled", "error"}:
            break
    return types


async def test_mock_response_sequence() -> None:
    adapter = MockAdapter(MockConfig(n_chunks=3))
    assert isinstance(adapter, ModelAdapter)
    assert (await adapter.health()).ok
    session = await adapter.open_session(SessionConfig(session_id="s"))
    assert isinstance(session, SpeechSession)
    await session.send_audio(AudioChunk(pcm16=b"\x00\x00" * 320, sample_rate_hz=16000))
    assert isinstance(session, MockSession)
    assert session.received_audio_ms == 20.0
    await session.commit_input()
    assert await _collect(session) == [
        "text_delta",
        "audio_delta",
        "audio_delta",
        "audio_delta",
        "response_done",
    ]
    await session.close()
    await session.close()  # idempotent


async def test_mock_cancel() -> None:
    adapter = MockAdapter(MockConfig(first_audio_delay_ms=500))
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.commit_input()
    await asyncio.sleep(0.01)
    await session.cancel_response()
    assert await _collect(session) == ["response_cancelled"]
    await session.close()


async def test_mock_injected_failure() -> None:
    adapter = MockAdapter(MockConfig(failure_rate=1.0))
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.commit_input()
    assert await _collect(session) == ["error"]
    await session.close()


async def test_mock_tool_call_then_result() -> None:
    adapter = MockAdapter(MockConfig(emit_tool_call="check_availability", n_chunks=1))
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.commit_input()
    assert await _collect(session) == ["tool_call", "response_done"]
    await session.send_tool_result("c1", {"ok": True})
    assert await _collect(session) == ["text_delta", "audio_delta", "response_done"]
    await session.close()


async def test_close_ends_event_stream_and_cancels_pending_response() -> None:
    adapter = MockAdapter(MockConfig(first_audio_delay_ms=1000))
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.commit_input()
    await session.close()
    assert await _collect(session) == []


def test_registry_factory_validates_options() -> None:
    adapter = adapter_registry.get("mock")({"n_chunks": 2})
    assert adapter.is_mock
    assert "mock" in adapter_registry.names()


def test_timed_event_summarizes_audio() -> None:
    delta = AudioDelta(response_id="r", pcm16=b"\x00\x00" * 2400, sample_rate_hz=24000)
    timed = TimedEvent.from_event(42, delta)
    assert timed.type == "audio_delta"
    assert timed.response_id == "r"
    assert timed.payload["bytes"] == 4800
    assert timed.payload["duration_ms"] == 100.0
    assert "pcm16" not in timed.payload
    done = TimedEvent.from_event(43, ResponseDone(response_id="r"))
    assert done.payload == {"usage": {}}
