"""Deterministic CPU-only adapter for tests and pipeline dry runs.

Runs produced with this adapter are always ``is_mock=True`` and can never be ranked
(ARCHITECTURE §10.3). Its latencies and outputs are configuration, not measurements.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import struct
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmark.adapters.base import (
    AudioChunk,
    AudioDelta,
    CapabilityStatus,
    ErrorEvent,
    ModelAdapter,
    ModelCapabilities,
    ResponseCancelled,
    ResponseDone,
    RuntimeHealth,
    SessionConfig,
    SpeechSession,
    TextDelta,
    ToolCall,
    adapter_registry,
)

MOCK_MODEL_ID = "mock"

_Event = AudioDelta | TextDelta | ToolCall | ResponseDone | ResponseCancelled | ErrorEvent


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_audio_delay_ms: float = Field(default=5.0, ge=0)
    chunk_interval_ms: float = Field(default=1.0, ge=0)
    chunk_ms: int = Field(default=40, gt=0)
    n_chunks: int = Field(default=5, gt=0)
    output_sample_rate_hz: int = 24000
    reply_text: str = "かしこまりました。"
    emit_tool_call: str | None = None
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    seed: int = 0


def sine_pcm16(duration_ms: int, sample_rate_hz: int, freq_hz: float = 220.0) -> bytes:
    n = int(sample_rate_hz * duration_ms / 1000)
    amplitude = 0.2 * 32767
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate_hz)) for i in range(n)),
    )


class MockSession:
    def __init__(self, cfg: SessionConfig, mock: MockConfig, fail: bool) -> None:
        self._cfg = cfg
        self._mock = mock
        self._fail = fail
        self._queue: asyncio.Queue[_Event | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._response_seq = 0
        self._received_audio_ms = 0.0
        self._closed = False

    @property
    def received_audio_ms(self) -> float:
        return self._received_audio_ms

    async def send_audio(self, chunk: AudioChunk) -> None:
        self._received_audio_ms += chunk.duration_ms

    async def commit_input(self) -> None:
        self._start_response()

    async def send_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        self._start_response(tool_result=(call_id, result))

    async def cancel_response(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            await self._queue.put(ResponseCancelled(response_id=self._current_id))

    def events(self) -> AsyncIterator[_Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[_Event]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._queue.put(None)

    @property
    def _current_id(self) -> str:
        return f"{self._cfg.session_id}-r{self._response_seq}"

    def _start_response(self, tool_result: tuple[str, dict[str, Any]] | None = None) -> None:
        self._response_seq += 1
        self._task = asyncio.create_task(self._respond(self._current_id, tool_result))

    async def _respond(
        self, response_id: str, tool_result: tuple[str, dict[str, Any]] | None
    ) -> None:
        mock = self._mock
        await asyncio.sleep(mock.first_audio_delay_ms / 1000)
        if self._fail:
            await self._queue.put(
                ErrorEvent(response_id=response_id, code="mock_failure", message="injected")
            )
            return
        if mock.emit_tool_call and tool_result is None:
            await self._queue.put(
                ToolCall(
                    response_id=response_id,
                    call_id=f"{response_id}-c1",
                    name=mock.emit_tool_call,
                    arguments_json=json.dumps({}),
                )
            )
            await self._queue.put(ResponseDone(response_id=response_id))
            return
        pcm = sine_pcm16(mock.chunk_ms, mock.output_sample_rate_hz)
        await self._queue.put(TextDelta(response_id=response_id, text=mock.reply_text))
        for index in range(mock.n_chunks):
            if index:
                await asyncio.sleep(mock.chunk_interval_ms / 1000)
            await self._queue.put(
                AudioDelta(
                    response_id=response_id, pcm16=pcm, sample_rate_hz=mock.output_sample_rate_hz
                )
            )
        await self._queue.put(ResponseDone(response_id=response_id, usage={"mock": True}))


class MockAdapter:
    model_id = MOCK_MODEL_ID
    is_mock = True

    def __init__(self, config: MockConfig | None = None) -> None:
        self.config = config or MockConfig()
        self.capabilities = ModelCapabilities(
            native_full_duplex=CapabilityStatus.UNSUPPORTED,
            streaming_audio_output=CapabilityStatus.VERIFIED,
            native_tool_calling=CapabilityStatus.VERIFIED,
            text_output_channel=CapabilityStatus.VERIFIED,
            output_sample_rate_hz=self.config.output_sample_rate_hz,
        )
        self._rng = random.Random(self.config.seed)

    async def health(self) -> RuntimeHealth:
        return RuntimeHealth(ok=True, detail="mock adapter", engine="mock", engine_version="0")

    async def open_session(self, cfg: SessionConfig) -> SpeechSession:
        fail = self._rng.random() < self.config.failure_rate
        return MockSession(cfg, self.config, fail=fail)


@adapter_registry.register(MOCK_MODEL_ID)
def create_mock_adapter(options: dict[str, Any]) -> ModelAdapter:
    return MockAdapter(MockConfig.model_validate(options))
