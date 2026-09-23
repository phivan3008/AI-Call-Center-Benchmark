"""In-process fake ``/v1/chat/completions`` server (httpx MockTransport) for CPU-only tests.

It reproduces the streaming shape of vLLM-Omni: SSE ``data:`` lines whose chunks carry
``modality`` at the top level and the payload in ``choices[].delta.content`` (base64 audio
when ``modality == "audio"``), plus OpenAI-style ``tool_calls`` deltas. Its outputs are
synthetic and never benchmark data.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from benchmark.audio.pcm import wav_bytes


def _tone(duration_ms: int, rate: int = 24000) -> bytes:
    n = rate * duration_ms // 1000
    return struct.pack(
        f"<{n}h", *(int(6000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n))
    )


@dataclass
class FakeChatConfig:
    n_audio_chunks: int = 3
    chunk_ms: int = 40
    chunk_delay_s: float = 0.0
    text: str = "かしこまりました。"
    wav_container: bool = True  # audio chunks as WAV containers (else raw PCM16)
    emit_tool_call: bool = True  # when the request declares tools
    prompted_tool_text: str | None = None  # text reply used for the prompted protocol
    status_code: int = 200
    error_body: str = '{"error": "bad request"}'
    audio_rate_hz: int = 24000


@dataclass
class FakeChatServer:
    config: FakeChatConfig = field(default_factory=FakeChatConfig)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        self.requests.append(body)
        if self.config.status_code >= 400:
            return httpx.Response(self.config.status_code, text=self.config.error_body)
        wants_tool = bool(body.get("tools")) and self.config.emit_tool_call
        already_called = any(
            m.get("role") == "tool"
            or (m.get("role") == "user" and "ツールの実行結果" in str(m.get("content")))
            for m in body.get("messages", [])
        )
        if self.config.prompted_tool_text is not None and not already_called:
            return httpx.Response(200, content=self._stream_text(self.config.prompted_tool_text))
        return httpx.Response(
            200, content=self._stream(wants_tool and not already_called), headers={}
        )

    async def _stream_text(self, text: str) -> AsyncIterator[bytes]:
        chunk = {"choices": [{"delta": {"content": text}}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def _stream(self, with_tool: bool) -> AsyncIterator[bytes]:
        cfg = self.config

        def sse(chunk: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

        if with_tool:
            yield sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "record_request",
                                            "arguments": '{"summary":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
            yield sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"予約の依頼"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            yield b"data: [DONE]\n\n"
            return

        yield sse({"choices": [{"delta": {"content": cfg.text}}]})
        pcm = _tone(cfg.chunk_ms, cfg.audio_rate_hz)
        payload = wav_bytes(pcm, cfg.audio_rate_hz) if cfg.wav_container else pcm
        for index in range(cfg.n_audio_chunks):
            if index and cfg.chunk_delay_s:
                await asyncio.sleep(cfg.chunk_delay_s)
            yield sse(
                {
                    "modality": "audio",
                    "choices": [{"delta": {"content": base64.b64encode(payload).decode()}}],
                }
            )
        yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"fake": 1}})
        yield b"data: [DONE]\n\n"
