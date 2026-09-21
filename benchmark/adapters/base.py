"""Model adapter interface (ARCHITECTURE §6.2).

Adapters translate between the harness and a model runtime. They never timestamp events
for latency purposes: the harness stamps every event on receipt with its own monotonic
clock (:class:`TimedEvent`) so latency is comparable across models.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from benchmark.core.registry import Registry


class CapabilityStatus(StrEnum):
    VERIFIED = "verified"
    CLAIMED = "claimed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    native_full_duplex: CapabilityStatus = CapabilityStatus.UNKNOWN
    streaming_audio_output: CapabilityStatus = CapabilityStatus.UNKNOWN
    native_tool_calling: CapabilityStatus = CapabilityStatus.UNKNOWN
    text_output_channel: CapabilityStatus = CapabilityStatus.UNKNOWN
    max_context_tokens: int | None = None
    input_sample_rate_hz: int = 16000
    output_sample_rate_hz: int = 24000


class AudioChunk(BaseModel):
    """Mono PCM16 little-endian audio."""

    model_config = ConfigDict(frozen=True)

    pcm16: bytes
    sample_rate_hz: int

    @property
    def duration_ms(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate_hz * 1000


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    system_prompt: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    voice: str | None = None
    sampling: dict[str, Any] = Field(default_factory=dict)


# --- Events -------------------------------------------------------------------------------


class AudioDelta(BaseModel):
    type: Literal["audio_delta"] = "audio_delta"
    response_id: str
    pcm16: bytes = Field(repr=False)
    sample_rate_hz: int


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    response_id: str
    text: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    response_id: str
    call_id: str
    name: str
    arguments_json: str


class ResponseDone(BaseModel):
    type: Literal["response_done"] = "response_done"
    response_id: str
    usage: dict[str, Any] = Field(default_factory=dict)


class ResponseCancelled(BaseModel):
    type: Literal["response_cancelled"] = "response_cancelled"
    response_id: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    response_id: str | None = None
    code: str
    message: str


ModelEvent = Annotated[
    AudioDelta | TextDelta | ToolCall | ResponseDone | ResponseCancelled | ErrorEvent,
    Field(discriminator="type"),
]

TERMINAL_EVENT_TYPES = frozenset({"response_done", "response_cancelled", "error"})


class TimedEvent(BaseModel):
    """An event as recorded by the harness: receipt time + event.

    Audio payloads are summarized (byte count, duration) rather than stored inline; the
    audio itself goes to the sample's output audio file.
    """

    received_ns: int
    type: str
    response_id: str | None
    payload: dict[str, Any]

    @classmethod
    def from_event(cls, received_ns: int, event: BaseModel) -> TimedEvent:
        data = event.model_dump(exclude={"pcm16"})
        if isinstance(event, AudioDelta):
            data["bytes"] = len(event.pcm16)
            data["duration_ms"] = len(event.pcm16) / 2 / event.sample_rate_hz * 1000
        event_type = str(data.pop("type"))
        response_id = data.pop("response_id", None)
        return cls(received_ns=received_ns, type=event_type, response_id=response_id, payload=data)


class RuntimeHealth(BaseModel):
    ok: bool
    detail: str = ""
    engine: str | None = None
    engine_version: str | None = None


# --- Interfaces ---------------------------------------------------------------------------


@runtime_checkable
class SpeechSession(Protocol):
    async def send_audio(self, chunk: AudioChunk) -> None: ...

    async def commit_input(self) -> None: ...

    async def cancel_response(self) -> None: ...

    async def send_tool_result(self, call_id: str, result: dict[str, Any]) -> None: ...

    def events(
        self,
    ) -> AsyncIterator[
        AudioDelta | TextDelta | ToolCall | ResponseDone | ResponseCancelled | ErrorEvent
    ]: ...

    async def close(self) -> None: ...


@runtime_checkable
class ModelAdapter(Protocol):
    model_id: str
    capabilities: ModelCapabilities
    is_mock: bool

    async def health(self) -> RuntimeHealth: ...

    async def open_session(self, cfg: SessionConfig) -> SpeechSession: ...


AdapterFactory = Callable[[dict[str, Any]], ModelAdapter]

adapter_registry: Registry[AdapterFactory] = Registry("adapter")
