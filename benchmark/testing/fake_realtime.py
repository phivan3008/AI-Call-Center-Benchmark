"""In-process fake realtime server (vLLM-Omni / OpenAI dialect subset) for CPU-only tests.

It implements just enough of docs/REALTIME_PROTOCOL.md to exercise the adapter, runner and
capability report without a GPU. Its outputs are synthetic and never benchmark data.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve


def _tone(duration_ms: int, rate: int = 24000) -> bytes:
    n = rate * duration_ms // 1000
    return struct.pack(
        f"<{n}h", *(int(6000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n))
    )


@dataclass
class FakeRealtimeConfig:
    legacy_event_names: bool = False  # response.audio.delta instead of response.output_audio.delta
    n_audio_chunks: int = 4
    chunk_ms: int = 40
    first_delay_s: float = 0.01
    chunk_interval_s: float = 0.005
    emit_tool_call: bool = True  # when the session declares tools
    reject_session: bool = False
    fail_response: bool = False


@dataclass
class FakeRealtimeServer:
    config: FakeRealtimeConfig = field(default_factory=FakeRealtimeConfig)
    received: list[dict[str, Any]] = field(default_factory=list)
    _server: Server | None = None
    port: int = 0

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def __aenter__(self) -> FakeRealtimeServer:
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = next(iter(self._server.sockets)).getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: ServerConnection) -> None:
        cfg = self.config
        tools: list[Any] = []
        response_task: asyncio.Task[None] | None = None
        tool_called = False
        seq = 0

        async def send(payload: dict[str, Any]) -> None:
            await ws.send(json.dumps(payload))

        async def respond(response_id: str, with_tool: bool) -> None:
            await asyncio.sleep(cfg.first_delay_s)
            await send({"type": "response.created", "response": {"id": response_id}})
            if cfg.fail_response:
                await send(
                    {
                        "type": "response.done",
                        "response": {"id": response_id, "status": "failed"},
                    }
                )
                return
            if with_tool:
                await send(
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": response_id,
                        "call_id": f"call_{response_id}",
                        "name": tools[0]["name"],
                        "arguments": json.dumps({"summary": "予約の依頼"}, ensure_ascii=False),
                    }
                )
                await send(
                    {
                        "type": "response.done",
                        "response": {"id": response_id, "status": "completed"},
                    }
                )
                return
            audio_type = (
                "response.audio.delta" if cfg.legacy_event_names else "response.output_audio.delta"
            )
            text_type = (
                "response.audio_transcript.delta"
                if cfg.legacy_event_names
                else "response.output_audio_transcript.delta"
            )
            await send({"type": text_type, "response_id": response_id, "delta": "はい、"})
            for index in range(cfg.n_audio_chunks):
                if index:
                    await asyncio.sleep(cfg.chunk_interval_s)
                await send(
                    {
                        "type": audio_type,
                        "response_id": response_id,
                        "delta": base64.b64encode(_tone(cfg.chunk_ms)).decode(),
                        "sample_rate_hz": 24000,
                    }
                )
            await send(
                {
                    "type": "response.done",
                    "response": {"id": response_id, "status": "completed", "usage": {"fake": 1}},
                }
            )

        try:
            async for raw in ws:
                message = json.loads(raw)
                self.received.append(message)
                kind = message.get("type")
                if kind == "session.update":
                    if cfg.reject_session:
                        await send({"type": "error", "error": {"code": "bad_session"}})
                        continue
                    tools = list(message.get("session", {}).get("tools") or [])
                    await send({"type": "session.updated", "session": message.get("session")})
                elif kind == "response.create":
                    seq += 1
                    with_tool = bool(tools) and cfg.emit_tool_call and not tool_called
                    tool_called = tool_called or with_tool
                    response_task = asyncio.create_task(respond(f"resp_{seq}", with_tool))
                elif kind == "response.cancel":
                    if response_task is not None and not response_task.done():
                        response_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await response_task
                        await send(
                            {
                                "type": "response.done",
                                "response": {"id": f"resp_{seq}", "status": "cancelled"},
                            }
                        )
                elif kind == "conversation.item.create":
                    await send({"type": "conversation.item.added", "item": message.get("item")})
        finally:
            if response_task is not None:
                response_task.cancel()
