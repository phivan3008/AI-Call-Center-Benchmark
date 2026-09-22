"""Realtime WebSocket adapter for OpenAI-Realtime-style servers (docs/REALTIME_PROTOCOL.md).

One adapter serves two dialects:

* ``vllm_omni``: vLLM-Omni ``/v1/realtime`` (Qwen3-Omni, MiniCPM-o 4.5), 16 kHz PCM16 in.
* ``openai``: OpenAI Realtime API (GPT-Realtime baseline), 24 kHz PCM16 in, bearer auth.

Server events are mapped to :mod:`benchmark.adapters.base` events. Both current and legacy
event names are accepted (e.g. ``response.output_audio.delta`` and ``response.audio.delta``)
because vLLM-Omni's turn-based and duplex handlers use different names.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import urllib.parse
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

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
from benchmark.audio.pcm import resample_pcm16
from benchmark.core.logging import get_logger
from benchmark.core.modelspec import Capability, ModelSpec, RealtimeEndpoint

log = get_logger(__name__)

_Event = AudioDelta | TextDelta | ToolCall | ResponseDone | ResponseCancelled | ErrorEvent

AUDIO_DELTA_TYPES = frozenset({"response.output_audio.delta", "response.audio.delta"})
TEXT_DELTA_TYPES = frozenset(
    {
        "response.output_audio_transcript.delta",
        "response.audio_transcript.delta",
        "response.output_text.delta",
        "response.text.delta",
    }
)
SESSION_ACK_TYPES = frozenset({"session.created", "session.updated"})

Connect = Callable[..., Any]


class RealtimeError(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_session_update(endpoint: RealtimeEndpoint, cfg: SessionConfig) -> dict[str, Any]:
    """``session.update`` payload for the endpoint's dialect."""
    if endpoint.dialect == "openai":
        session: dict[str, Any] = {
            "type": "realtime",
            "model": endpoint.api_model,
            "instructions": cfg.system_prompt,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": endpoint.input_sample_rate_hz},
                    "turn_detection": None,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": endpoint.output_sample_rate_hz},
                    **({"voice": endpoint.voice} if endpoint.voice else {}),
                },
            },
        }
        if cfg.tools:
            session["tools"] = [{"type": "function", **tool.model_dump()} for tool in cfg.tools]
            session["tool_choice"] = "auto"
    else:
        session = {
            "model": endpoint.api_model,
            "modalities": ["audio", "text"],
            "instructions": cfg.system_prompt,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            "voice": endpoint.voice,
        }
        if cfg.tools:
            session["tools"] = [{"type": "function", **tool.model_dump()} for tool in cfg.tools]
        session.update(cfg.sampling)
    return {"type": "session.update", "session": session}


def map_server_event(
    message: Mapping[str, Any], endpoint: RealtimeEndpoint, seen_calls: set[str]
) -> _Event | None:
    """Translate one server event into a harness event (None = not relevant)."""
    kind = str(message.get("type", ""))
    response = message.get("response") if isinstance(message.get("response"), dict) else {}
    response_id = str(message.get("response_id") or (response or {}).get("id") or "")

    if kind in AUDIO_DELTA_TYPES:
        return AudioDelta(
            response_id=response_id,
            pcm16=base64.b64decode(message.get("delta", "")),
            sample_rate_hz=int(message.get("sample_rate_hz") or endpoint.output_sample_rate_hz),
        )
    if kind in TEXT_DELTA_TYPES:
        return TextDelta(response_id=response_id, text=str(message.get("delta", "")))
    if kind == "response.function_call_arguments.done" or (
        kind == "response.output_item.done"
        and isinstance(message.get("item"), dict)
        and message["item"].get("type") == "function_call"
    ):
        source = message["item"] if kind == "response.output_item.done" else message
        call_id = str(source.get("call_id") or source.get("id") or "")
        if call_id in seen_calls:
            return None
        seen_calls.add(call_id)
        return ToolCall(
            response_id=response_id,
            call_id=call_id,
            name=str(source.get("name") or ""),
            arguments_json=str(source.get("arguments") or "{}"),
        )
    if kind == "response.done":
        status = str((response or {}).get("status") or "completed")
        if status == "cancelled":
            return ResponseCancelled(response_id=response_id)
        if status in {"failed", "incomplete"}:
            details = (response or {}).get("status_details") or {}
            return ErrorEvent(
                response_id=response_id,
                code=f"response_{status}",
                message=json.dumps(details, ensure_ascii=False)[:500],
            )
        return ResponseDone(
            response_id=response_id, usage=dict((response or {}).get("usage") or {})
        )
    if kind == "error":
        error = message.get("error") if isinstance(message.get("error"), dict) else {}
        return ErrorEvent(
            response_id=response_id or None,
            code=str((error or {}).get("code") or (error or {}).get("type") or "error"),
            message=str((error or {}).get("message") or "")[:500],
        )
    return None


class RealtimeSession:
    def __init__(self, ws: ClientConnection, endpoint: RealtimeEndpoint) -> None:
        self._ws = ws
        self._endpoint = endpoint
        self._queue: asyncio.Queue[_Event | None] = asyncio.Queue()
        self._seen_calls: set[str] = set()
        self._current_response: str | None = None
        self.server_event_counts: Counter[str] = Counter()
        self._reader = asyncio.create_task(self._read())
        self._closed = False

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                kind = str(message.get("type", ""))
                self.server_event_counts[kind] += 1
                if kind == "response.created":
                    self._current_response = str((message.get("response") or {}).get("id") or "")
                event = map_server_event(message, self._endpoint, self._seen_calls)
                if event is not None:
                    await self._queue.put(event)
        except websockets.ConnectionClosed as exc:
            if not self._closed:
                await self._queue.put(ErrorEvent(code="connection_closed", message=str(exc)[:500]))
        finally:
            await self._queue.put(None)

    async def _send(self, payload: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def send_audio(self, chunk: AudioChunk) -> None:
        pcm = resample_pcm16(chunk.pcm16, chunk.sample_rate_hz, self._endpoint.input_sample_rate_hz)
        payload: dict[str, Any] = {"type": "input_audio_buffer.append", "audio": _b64(pcm)}
        if self._endpoint.dialect == "vllm_omni":
            payload["sample_rate_hz"] = self._endpoint.input_sample_rate_hz
        await self._send(payload)

    async def commit_input(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send(self._response_create())

    async def cancel_response(self) -> None:
        payload: dict[str, Any] = {"type": "response.cancel"}
        if self._current_response:
            payload["response_id"] = self._current_response
        await self._send(payload)

    async def send_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                },
            }
        )
        await self._send(self._response_create())

    def _response_create(self) -> dict[str, Any]:
        if self._endpoint.dialect == "vllm_omni":
            return {"type": "response.create", "response": {"modalities": ["audio", "text"]}}
        return {"type": "response.create"}

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
        with contextlib.suppress(Exception):
            await self._ws.close()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(self._reader, timeout=5)


class RealtimeAdapter:
    is_mock = False

    def __init__(
        self,
        spec: ModelSpec,
        env: Mapping[str, str] | None = None,
        connect: Connect = websockets.connect,
        handshake_timeout_s: float = 30.0,
    ) -> None:
        self.spec = spec
        self.model_id = spec.model_id
        self.endpoint = spec.realtime
        self._env = os.environ if env is None else env
        self._connect = connect
        self._handshake_timeout_s = handshake_timeout_s
        caps = spec.capabilities

        def status(cap: Capability) -> CapabilityStatus:
            return cap.verified if cap.verified is not CapabilityStatus.UNKNOWN else cap.claimed

        self.capabilities = ModelCapabilities(
            native_full_duplex=status(caps.native_full_duplex),
            streaming_audio_output=status(caps.streaming_audio_output),
            native_tool_calling=status(caps.native_tool_calling),
            text_output_channel=status(caps.text_output_channel),
            input_sample_rate_hz=self.endpoint.input_sample_rate_hz,
            output_sample_rate_hz=self.endpoint.output_sample_rate_hz,
        )

    def _url_and_headers(self) -> tuple[str, dict[str, str]]:
        url = self.endpoint.url
        headers: dict[str, str] = {}
        if self.endpoint.dialect == "openai":
            if not self.endpoint.api_model:
                raise RealtimeError(f"{self.model_id}: realtime.api_model is not configured")
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}model={urllib.parse.quote(self.endpoint.api_model)}"
        if self.endpoint.auth_env:
            token = self._env.get(self.endpoint.auth_env)
            if not token:
                raise RealtimeError(f"{self.model_id}: {self.endpoint.auth_env} is not set")
            headers["Authorization"] = f"Bearer {token}"
        return url, headers

    async def health(self) -> RuntimeHealth:
        try:
            url, headers = self._url_and_headers()
            ws = await self._connect(url, additional_headers=headers, open_timeout=10)
            await ws.close()
        except Exception as exc:
            return RuntimeHealth(ok=False, detail=f"{type(exc).__name__}: {exc}"[:300])
        return RuntimeHealth(ok=True, detail="websocket handshake ok")

    async def open_session(self, cfg: SessionConfig) -> SpeechSession:
        url, headers = self._url_and_headers()
        ws = await self._connect(url, additional_headers=headers, open_timeout=30, max_size=None)
        await ws.send(json.dumps(build_session_update(self.endpoint, cfg), ensure_ascii=False))
        try:
            async with asyncio.timeout(self._handshake_timeout_s):
                while True:
                    message = json.loads(await ws.recv())
                    kind = message.get("type")
                    if kind in SESSION_ACK_TYPES:
                        break
                    if kind == "error":
                        raise RealtimeError(f"session.update rejected: {message.get('error')}")
        except TimeoutError as exc:
            await ws.close()
            raise RealtimeError("no session acknowledgement from server") from exc
        except RealtimeError:
            await ws.close()
            raise
        return RealtimeSession(ws, self.endpoint)
