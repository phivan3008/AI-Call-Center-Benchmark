"""OpenAI-compatible chat-completions adapter (turn mode).

vLLM-Omni's turn-based ``/v1/realtime`` endpoint is upstream vLLM's speech-to-text realtime
API: it accepts only ``session.update {model}``, ``input_audio_buffer.append`` and
``.commit``, so it carries no system prompt, no tools and no cancel (checkpoint 2,
2026-09-23). Quality layers therefore run over ``/v1/chat/completions``, which takes the
system prompt in ``messages``, audio input as a base64 ``audio_url`` and returns text plus
audio (``modalities: ["text", "audio"]``).

Streaming shape (vLLM-Omni): each SSE chunk carries ``modality`` at the top level; when it is
``audio`` the base64 payload sits in ``choices[].delta.content``, otherwise the delta is text.
Audio chunks may be WAV containers or raw PCM16; both are handled.

The realtime WebSocket adapter stays for the duplex/barge-in work (Phase 6) and for
GPT-Realtime.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from benchmark.adapters.base import (
    AudioChunk,
    AudioDelta,
    CapabilityStatus,
    ErrorEvent,
    ModelCapabilities,
    ResponseCancelled,
    ResponseDone,
    RuntimeHealth,
    SessionConfig,
    SpeechSession,
    TextDelta,
    ToolCall,
)
from benchmark.audio.pcm import read_wav_bytes, resample_pcm16, wav_bytes
from benchmark.core.logging import get_logger
from benchmark.core.modelspec import Capability, ChatEndpoint, ModelSpec

log = get_logger(__name__)

_Event = AudioDelta | TextDelta | ToolCall | ResponseDone | ResponseCancelled | ErrorEvent
DATA_PREFIX = "data: "
DONE = "[DONE]"


class ChatError(Exception):
    pass


def audio_data_url(pcm16: bytes, sample_rate_hz: int) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes(pcm16, sample_rate_hz)).decode()


def decode_audio_delta(payload: str, default_rate_hz: int) -> tuple[bytes, int]:
    """Decode one streamed audio chunk: WAV container or raw PCM16."""
    raw = base64.b64decode(payload)
    if raw[:4] == b"RIFF":
        return read_wav_bytes(raw)
    return raw, default_rate_hz


class ChatSession:
    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: ChatEndpoint,
        cfg: SessionConfig,
        headers: Mapping[str, str],
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._cfg = cfg
        self._headers = dict(headers)
        self._queue: asyncio.Queue[_Event | None] = asyncio.Queue()
        self._input = bytearray()
        self._input_rate = endpoint.input_sample_rate_hz
        self._messages: list[dict[str, Any]] = []
        if cfg.system_prompt:
            self._messages.append(
                {"role": "system", "content": [{"type": "text", "text": cfg.system_prompt}]}
            )
        self._task: asyncio.Task[None] | None = None
        self._response_seq = 0
        self._closed = False

    # --- SpeechSession ---------------------------------------------------------------

    async def send_audio(self, chunk: AudioChunk) -> None:
        self._input.extend(
            resample_pcm16(chunk.pcm16, chunk.sample_rate_hz, self._endpoint.input_sample_rate_hz)
        )

    async def commit_input(self) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_data_url(bytes(self._input), self._input_rate)},
                    }
                ],
            }
        )
        self._input.clear()
        self._start()

    async def cancel_response(self) -> None:
        """Abort the streaming request: closing the stream stops generation server side."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            await self._queue.put(ResponseCancelled(response_id=self._response_id))

    async def send_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
        self._start()

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

    # --- request/response ------------------------------------------------------------

    @property
    def _response_id(self) -> str:
        return f"{self._cfg.session_id}-r{self._response_seq}"

    def _start(self) -> None:
        self._response_seq += 1
        self._task = asyncio.create_task(self._stream(self._response_id))

    def build_request(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._endpoint.api_model,
            "messages": self._messages,
            "modalities": list(self._endpoint.modalities),
            "stream": True,
            **self._endpoint.extra_body,
        }
        if self._endpoint.chat_template_kwargs:
            body["chat_template_kwargs"] = dict(self._endpoint.chat_template_kwargs)
        if self._cfg.tools:
            body["tools"] = [
                {"type": "function", "function": tool.model_dump()} for tool in self._cfg.tools
            ]
            body["tool_choice"] = "auto"
        body.update(self._cfg.sampling)
        return body

    async def _stream(self, response_id: str) -> None:
        body = self.build_request()
        assistant_text: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        try:
            async with self._client.stream(
                "POST",
                self._endpoint.url,
                json=body,
                headers=self._headers,
                timeout=self._endpoint.request_timeout_s,
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:500]
                    await self._queue.put(
                        ErrorEvent(
                            response_id=response_id,
                            code=f"http_{response.status_code}",
                            message=detail,
                        )
                    )
                    return
                async for line in response.aiter_lines():
                    if not line.startswith(DATA_PREFIX):
                        continue
                    payload = line[len(DATA_PREFIX) :].strip()
                    if payload == DONE:
                        break
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or usage
                    for event in self._chunk_events(chunk, response_id, assistant_text, tool_calls):
                        await self._queue.put(event)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            await self._queue.put(
                ErrorEvent(
                    response_id=response_id,
                    code=type(exc).__name__,
                    message=str(exc)[:500],
                )
            )
            return

        if tool_calls:
            self._messages.append(
                {
                    "role": "assistant",
                    "content": "".join(assistant_text) or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in tool_calls.values()
                    ],
                }
            )
            for call in tool_calls.values():
                await self._queue.put(
                    ToolCall(
                        response_id=response_id,
                        call_id=call["id"],
                        name=call["name"],
                        arguments_json=call["arguments"] or "{}",
                    )
                )
        elif assistant_text:
            self._messages.append({"role": "assistant", "content": "".join(assistant_text)})
        await self._queue.put(ResponseDone(response_id=response_id, usage=usage))

    def _chunk_events(
        self,
        chunk: Mapping[str, Any],
        response_id: str,
        assistant_text: list[str],
        tool_calls: dict[int, dict[str, Any]],
    ) -> list[_Event]:
        events: list[_Event] = []
        modality = chunk.get("modality")
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if modality == "audio":
                    pcm, rate = decode_audio_delta(content, self._endpoint.output_sample_rate_hz)
                    events.append(
                        AudioDelta(response_id=response_id, pcm16=pcm, sample_rate_hz=rate)
                    )
                else:
                    assistant_text.append(content)
                    events.append(TextDelta(response_id=response_id, text=content))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                entry = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                entry["id"] = call.get("id") or entry["id"] or f"{response_id}-c{index}"
                function = call.get("function") or {}
                entry["name"] = function.get("name") or entry["name"]
                entry["arguments"] += function.get("arguments") or ""
        return events


class ChatAdapter:
    is_mock = False

    def __init__(
        self,
        spec: ModelSpec,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if spec.chat is None:
            raise ChatError(f"{spec.model_id}: transport 'chat' needs a `chat:` section")
        self.spec = spec
        self.model_id = spec.model_id
        self.endpoint: ChatEndpoint = spec.chat
        self._env = os.environ if env is None else env
        self._client = client or httpx.AsyncClient()
        caps = spec.capabilities

        def status(cap: Capability) -> CapabilityStatus:
            return cap.verified if cap.verified is not CapabilityStatus.UNKNOWN else cap.claimed

        self.capabilities = ModelCapabilities(
            native_full_duplex=CapabilityStatus.UNSUPPORTED,  # turn mode by construction
            streaming_audio_output=status(caps.streaming_audio_output),
            native_tool_calling=status(caps.native_tool_calling),
            text_output_channel=status(caps.text_output_channel),
            input_sample_rate_hz=self.endpoint.input_sample_rate_hz,
            output_sample_rate_hz=self.endpoint.output_sample_rate_hz,
        )

    def _headers(self) -> dict[str, str]:
        if not self.endpoint.auth_env:
            return {}
        token = self._env.get(self.endpoint.auth_env)
        if not token:
            raise ChatError(f"{self.model_id}: {self.endpoint.auth_env} is not set")
        return {"Authorization": f"Bearer {token}"}

    async def health(self) -> RuntimeHealth:
        url = self.endpoint.url.replace("/chat/completions", "/models")
        try:
            response = await self._client.get(url, headers=self._headers(), timeout=10)
        except (httpx.HTTPError, ChatError) as exc:
            return RuntimeHealth(ok=False, detail=f"{type(exc).__name__}: {exc}"[:300])
        return RuntimeHealth(ok=response.status_code < 400, detail=f"HTTP {response.status_code}")

    async def open_session(self, cfg: SessionConfig) -> SpeechSession:
        return ChatSession(self._client, self.endpoint, cfg, self._headers())

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._client.aclose()
