from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from benchmark.adapters.base import AudioChunk, SessionConfig, SpeechSession, ToolSpec
from benchmark.adapters.chat import ChatAdapter, ChatError, audio_data_url, decode_audio_delta
from benchmark.audio.pcm import read_wav_bytes
from benchmark.core.config import RunProfile, Settings
from benchmark.core.modelspec import ModelSpec, load_model_spec
from benchmark.core.schemas import ModelVersion, RunStatus
from benchmark.runner import Stimulus, execute_run
from benchmark.testing.fake_chat import FakeChatConfig, FakeChatServer

TOOL = ToolSpec(
    name="record_request",
    description="d",
    parameters={"type": "object", "properties": {"summary": {"type": "string"}}},
)


def _spec(**chat: Any) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "model_id": "fake",
            "display_name": "Fake",
            "license": "n/a",
            "runtime": {"kind": "remote_api"},
            "transport": "chat",
            "realtime": {"dialect": "vllm_omni", "url": "ws://unused"},
            "chat": {
                "url": "http://server/v1/chat/completions",
                "api_model": "fake-model",
                **chat,
            },
        }
    )


async def _drain(session: SpeechSession) -> list[str]:
    kinds = []
    async for event in session.events():
        kinds.append(event.type)
        if event.type in {"response_done", "response_cancelled", "error"}:
            break
    return kinds


def test_repo_models_use_chat_transport(settings: Settings) -> None:
    for model_id in ("minicpm-o-4_5", "qwen3-omni-30b-a3b"):
        spec = load_model_spec(settings, model_id)
        assert spec.transport == "chat"
        assert spec.chat is not None
        assert spec.chat.url.endswith("/v1/chat/completions")
        assert spec.realtime.url.endswith("duplex=1")  # duplex stays for Phase 6


async def test_turn_request_and_streamed_audio() -> None:
    server = FakeChatServer(FakeChatConfig(n_audio_chunks=3))
    adapter = ChatAdapter(
        _spec(chat_template_kwargs={"use_tts_template": True}), client=server.client()
    )
    assert (await adapter.health()).ok
    session = await adapter.open_session(SessionConfig(session_id="s", system_prompt="指示"))
    await session.send_audio(AudioChunk(pcm16=b"\x00\x10" * 16000, sample_rate_hz=16000))
    await session.commit_input()
    assert await _drain(session) == [
        "text_delta",
        "audio_delta",
        "audio_delta",
        "audio_delta",
        "response_done",
    ]
    await session.close()

    body = server.requests[0]
    assert body["model"] == "fake-model"
    assert body["stream"] is True
    assert body["modalities"] == ["text", "audio"]
    assert body["chat_template_kwargs"] == {"use_tts_template": True}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"][0]["text"] == "指示"
    audio_part = body["messages"][1]["content"][0]
    assert audio_part["type"] == "audio_url"
    url = audio_part["audio_url"]["url"]
    assert url.startswith("data:audio/wav;base64,")
    pcm, rate = read_wav_bytes(base64.b64decode(url.split(",", 1)[1]))
    assert rate == 16000
    assert len(pcm) == 2 * 16000


async def test_raw_pcm_audio_chunks_and_resampling() -> None:
    server = FakeChatServer(FakeChatConfig(wav_container=False, n_audio_chunks=1))
    adapter = ChatAdapter(_spec(input_sample_rate_hz=24000), client=server.client())
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.send_audio(AudioChunk(pcm16=b"\x00\x10" * 16000, sample_rate_hz=16000))
    await session.commit_input()
    events = []
    async for event in session.events():
        events.append(event)
        if event.type == "response_done":
            break
    await session.close()
    audio = next(e for e in events if e.type == "audio_delta")
    assert audio.sample_rate_hz == 24000  # type: ignore[union-attr]
    url = server.requests[0]["messages"][0]["content"][0]["audio_url"]["url"]
    pcm, rate = read_wav_bytes(base64.b64decode(url.split(",", 1)[1]))
    assert rate == 24000
    assert len(pcm) == pytest.approx(2 * 24000, rel=0.01)


async def test_tool_call_then_result_continues_the_conversation() -> None:
    server = FakeChatServer()
    adapter = ChatAdapter(_spec(), client=server.client())
    session = await adapter.open_session(SessionConfig(session_id="s", tools=[TOOL]))
    await session.commit_input()
    events = []
    async for event in session.events():
        events.append(event)
        if event.type == "response_done":
            break
    call = next(e for e in events if e.type == "tool_call")
    assert call.name == "record_request"  # type: ignore[union-attr]
    assert json.loads(call.arguments_json) == {"summary": "予約の依頼"}  # type: ignore[union-attr]

    await session.send_tool_result(call.call_id, {"status": "recorded"})  # type: ignore[union-attr]
    assert (await _drain(session))[-1] == "response_done"
    await session.close()

    assert server.requests[0]["tools"][0]["function"]["name"] == "record_request"
    assert server.requests[0]["tool_choice"] == "auto"
    messages = server.requests[1]["messages"]
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "record_request"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"status": "recorded"}',
    }


async def test_cancel_stops_the_stream() -> None:
    server = FakeChatServer(FakeChatConfig(n_audio_chunks=20, chunk_delay_s=0.05))
    adapter = ChatAdapter(_spec(), client=server.client())
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.commit_input()
    kinds = []
    async for event in session.events():
        kinds.append(event.type)
        if event.type == "audio_delta" and kinds.count("audio_delta") == 2:
            await session.cancel_response()
        if event.type in {"response_cancelled", "response_done"}:
            break
    await session.close()
    assert kinds[-1] == "response_cancelled"
    assert kinds.count("audio_delta") < 20


async def test_http_error_becomes_an_error_event() -> None:
    server = FakeChatServer(FakeChatConfig(status_code=422, error_body="no such model"))
    adapter = ChatAdapter(_spec(), client=server.client())
    session = await adapter.open_session(SessionConfig(session_id="s"))
    await session.commit_input()
    events = []
    async for event in session.events():
        events.append(event)
        break
    await session.close()
    assert events[0].type == "error"
    assert events[0].code == "http_422"  # type: ignore[union-attr]
    assert "no such model" in events[0].message  # type: ignore[union-attr]


async def test_health_failure_and_missing_config() -> None:
    adapter = ChatAdapter(_spec(url="http://127.0.0.1:9/v1/chat/completions"))
    assert not (await adapter.health()).ok
    await adapter.aclose()

    spec = _spec()
    bare = spec.model_copy(update={"chat": None})
    with pytest.raises(ChatError, match="needs a `chat:` section"):
        ChatAdapter(bare)

    authed = ChatAdapter(_spec(auth_env="TOKEN"), env={})
    assert not (await authed.health()).ok
    with pytest.raises(ChatError, match="TOKEN is not set"):
        await authed.open_session(SessionConfig(session_id="s"))
    ok = ChatAdapter(_spec(auth_env="TOKEN"), env={"TOKEN": "x"}, client=FakeChatServer().client())
    session = await ok.open_session(SessionConfig(session_id="s"))
    await session.close()


def test_audio_helpers() -> None:
    url = audio_data_url(b"\x00\x01" * 100, 16000)
    assert url.startswith("data:audio/wav;base64,")
    raw = b"\x01\x02" * 10
    assert decode_audio_delta(base64.b64encode(raw).decode(), 24000) == (raw, 24000)


async def test_runner_smoke_over_chat_transport(settings: Settings) -> None:
    server = FakeChatServer(FakeChatConfig(n_audio_chunks=20, chunk_delay_s=0.02))
    adapter = ChatAdapter(_spec(), client=server.client())
    profile = RunProfile(name="t", layers=["L1"], samples_per_layer=1, sample_timeout_s=10)
    pcm = b"\x00\x10" * 16000
    stimuli = [
        Stimulus("turn_0", "L1", pcm, 16000, 1.0, task="turn", system_prompt="p"),
        Stimulus("tool_0", "L3", pcm, 16000, 1.0, task="tool", tools=(TOOL,)),
        Stimulus("cancel_0", "L2", pcm, 16000, 1.0, task="cancel", cancel_after_ms=30),
    ]
    manifest = await execute_run(settings, profile, adapter, stimuli, ModelVersion(model_id="fake"))
    assert manifest.status is RunStatus.COMPLETED, manifest
    run_dir = settings.raw_dir / manifest.run_id
    report = json.loads((run_dir / "capability_report.json").read_text("utf-8"))
    observations = report["observations"]
    assert observations["streaming_audio_output"]["observation"] == "observed"
    assert observations["native_tool_calling"]["observation"] == "observed"
    assert observations["response_cancel"]["observation"] == "observed"
    assert (run_dir / "L1" / "turn_0" / "output.wav").is_file()


async def test_prompted_tool_protocol() -> None:
    """Fallback for servers without --tool-call-parser: tools described in the prompt."""
    tool_json = '{"tool": "record_request", "arguments": {"summary": "予約の依頼"}}'
    server = FakeChatServer(FakeChatConfig(prompted_tool_text=tool_json))
    adapter = ChatAdapter(_spec(tool_protocol="prompted"), client=server.client())
    assert adapter.capabilities.native_tool_calling.value == "unsupported"

    session = await adapter.open_session(
        SessionConfig(session_id="s", system_prompt="指示", tools=[TOOL])
    )
    await session.commit_input()
    events = []
    async for event in session.events():
        events.append(event)
        if event.type == "response_done":
            break
    call = next(e for e in events if e.type == "tool_call")
    assert call.name == "record_request"  # type: ignore[union-attr]
    assert json.loads(call.arguments_json) == {"summary": "予約の依頼"}  # type: ignore[union-attr]

    await session.send_tool_result(call.call_id, {"status": "recorded"})  # type: ignore[union-attr]
    assert (await _drain(session))[-1] == "response_done"
    await session.close()

    first = server.requests[0]
    assert "tools" not in first  # never sent to the server in prompted mode
    system_text = first["messages"][0]["content"][0]["text"]
    assert "record_request" in system_text
    assert '{"tool":' in system_text.replace(" ", "")
    follow_up = server.requests[1]["messages"][-1]
    assert follow_up["role"] == "user"
    assert "ツールの実行結果" in follow_up["content"][0]["text"]


def test_parse_prompted_tool_call() -> None:
    from benchmark.adapters.chat import parse_prompted_tool_call

    tools = [TOOL]
    assert parse_prompted_tool_call('{"tool": "record_request", "arguments": {"a": 1}}', tools) == (
        "record_request",
        '{"a": 1}',
    )
    # Embedded in surrounding text, with nested objects.
    embedded = 'はい。{"tool": "record_request", "arguments": {"x": {"y": 2}}} です'
    assert parse_prompted_tool_call(embedded, tools) == ("record_request", '{"x": {"y": 2}}')
    # A hallucinated tool name is still reported, so evaluators can count it.
    assert parse_prompted_tool_call('{"tool": "unknown_tool", "arguments": {}}', tools) == (
        "unknown_tool",
        "{}",
    )
    assert parse_prompted_tool_call("ご用件をお伺いします。", tools) is None
    assert parse_prompted_tool_call('{"tool": ', tools) is None
    assert parse_prompted_tool_call('{"no_tool_key": 1}', tools) is None
    assert parse_prompted_tool_call('{"tool": "record_request"}', tools) == ("record_request", "{}")
