from __future__ import annotations

import json
from typing import Any

import pytest

from benchmark.adapters.base import AudioChunk, SessionConfig, SpeechSession, ToolSpec
from benchmark.adapters.realtime import (
    RealtimeAdapter,
    RealtimeError,
    build_session_update,
    map_server_event,
)
from benchmark.core.config import RunProfile, Settings
from benchmark.core.modelspec import ModelSpec, RealtimeEndpoint, list_model_ids, load_model_spec
from benchmark.core.schemas import ModelVersion, RunStatus
from benchmark.runner import Stimulus, execute_run
from benchmark.testing.fake_realtime import FakeRealtimeConfig, FakeRealtimeServer

TOOL = ToolSpec(
    name="record_request",
    description="d",
    parameters={"type": "object", "properties": {"summary": {"type": "string"}}},
)


def _spec(url: str, dialect: str = "vllm_omni", **realtime: Any) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "model_id": "fake",
            "display_name": "Fake",
            "license": "n/a",
            "runtime": {"kind": "remote_api"},
            "realtime": {"dialect": dialect, "url": url, "api_model": "fake-model", **realtime},
        }
    )


async def _drain(session: SpeechSession) -> list[str]:
    kinds = []
    async for event in session.events():
        kinds.append(event.type)
        if event.type in {"response_done", "response_cancelled", "error"}:
            break
    return kinds


def test_repo_model_configs_are_valid(settings: Settings) -> None:
    ids = list_model_ids(settings)
    assert {"qwen3-omni-30b-a3b", "minicpm-o-4_5", "gpt-realtime"} <= set(ids)
    for model_id in ids:
        spec = load_model_spec(settings, model_id)
        assert spec.model_id == model_id


@pytest.mark.parametrize("legacy", [False, True])
async def test_turn_roundtrip_both_event_namings(legacy: bool) -> None:
    async with FakeRealtimeServer(FakeRealtimeConfig(legacy_event_names=legacy)) as server:
        adapter = RealtimeAdapter(_spec(server.url))
        assert (await adapter.health()).ok
        session = await adapter.open_session(SessionConfig(session_id="s", system_prompt="p"))
        await session.send_audio(AudioChunk(pcm16=b"\x00\x01" * 1600, sample_rate_hz=16000))
        await session.commit_input()
        kinds = await _drain(session)
        await session.close()
        await session.close()
    assert kinds == ["text_delta"] + ["audio_delta"] * 4 + ["response_done"]
    sent = [m["type"] for m in server.received]
    assert sent == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "response.create",
    ]
    append = server.received[1]
    assert append["sample_rate_hz"] == 16000
    assert server.received[0]["session"]["instructions"] == "p"


async def test_tool_call_then_result() -> None:
    async with FakeRealtimeServer() as server:
        adapter = RealtimeAdapter(_spec(server.url))
        session = await adapter.open_session(SessionConfig(session_id="s", tools=[TOOL]))
        await session.commit_input()
        events = []
        async for event in session.events():
            events.append(event)
            if event.type == "response_done":
                break
        call = next(e for e in events if e.type == "tool_call")
        assert call.name == "record_request"  # type: ignore[union-attr]
        assert json.loads(call.arguments_json)["summary"]  # type: ignore[union-attr]
        await session.send_tool_result(call.call_id, {"status": "recorded"})  # type: ignore[union-attr]
        assert (await _drain(session))[-1] == "response_done"
        await session.close()
    item = next(m for m in server.received if m["type"] == "conversation.item.create")
    assert item["item"]["type"] == "function_call_output"
    assert server.received[0]["session"]["tools"][0]["type"] == "function"


async def test_cancel_and_failed_and_rejected() -> None:
    config = FakeRealtimeConfig(first_delay_s=0.0, chunk_interval_s=0.5)
    async with FakeRealtimeServer(config) as server:
        session = await RealtimeAdapter(_spec(server.url)).open_session(
            SessionConfig(session_id="s")
        )
        await session.commit_input()
        async for event in session.events():
            if event.type == "audio_delta":
                await session.cancel_response()
            if event.type in {"response_cancelled", "response_done"}:
                assert event.type == "response_cancelled"
                break
        await session.close()
    assert any(m["type"] == "response.cancel" and m["response_id"] for m in server.received)

    async with FakeRealtimeServer(FakeRealtimeConfig(fail_response=True)) as server:
        session = await RealtimeAdapter(_spec(server.url)).open_session(
            SessionConfig(session_id="s")
        )
        await session.commit_input()
        assert await _drain(session) == ["error"]
        await session.close()

    async with FakeRealtimeServer(FakeRealtimeConfig(reject_session=True)) as server:
        with pytest.raises(RealtimeError, match="rejected"):
            await RealtimeAdapter(_spec(server.url)).open_session(SessionConfig(session_id="s"))


async def test_connection_loss_and_health_failure() -> None:
    adapter = RealtimeAdapter(_spec("ws://127.0.0.1:9/v1/realtime"))
    health = await adapter.health()
    assert not health.ok


def test_openai_url_auth_and_session_shape() -> None:
    spec = _spec(
        "wss://api.example/v1/realtime",
        dialect="openai",
        auth_env="TOKEN",
        input_sample_rate_hz=24000,
        voice="alloy",
    )
    captured: dict[str, Any] = {}

    async def fake_connect(url: str, **kwargs: Any) -> Any:
        captured["url"] = url
        captured["headers"] = kwargs["additional_headers"]
        raise OSError("stop here")

    adapter = RealtimeAdapter(spec, env={"TOKEN": "secret"}, connect=fake_connect)
    import asyncio

    with pytest.raises(OSError):
        asyncio.run(adapter.open_session(SessionConfig(session_id="s")))
    assert captured["url"] == "wss://api.example/v1/realtime?model=fake-model"
    assert captured["headers"] == {"Authorization": "Bearer secret"}

    payload = build_session_update(spec.realtime, SessionConfig(session_id="s", tools=[TOOL]))
    session = payload["session"]
    assert session["type"] == "realtime"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "alloy"
    assert session["tool_choice"] == "auto"

    with pytest.raises(RealtimeError, match="TOKEN is not set"):
        RealtimeAdapter(spec, env={})._url_and_headers()
    no_model = _spec("wss://x", dialect="openai", api_model=None)
    with pytest.raises(RealtimeError, match="api_model"):
        RealtimeAdapter(no_model, env={})._url_and_headers()


def test_event_mapping_edge_cases() -> None:
    endpoint = RealtimeEndpoint(dialect="vllm_omni", url="ws://x")
    seen: set[str] = set()
    item = {
        "type": "response.output_item.done",
        "response_id": "r",
        "item": {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
    }
    assert map_server_event(item, endpoint, seen) is not None
    assert map_server_event(item, endpoint, seen) is None  # duplicate call id
    assert map_server_event({"type": "session.updated"}, endpoint, seen) is None
    incomplete = map_server_event(
        {"type": "response.done", "response": {"id": "r", "status": "incomplete"}}, endpoint, seen
    )
    assert incomplete is not None and incomplete.type == "error"
    err = map_server_event(
        {"type": "error", "error": {"type": "x", "message": "m"}}, endpoint, seen
    )
    assert err is not None and err.code == "x"  # type: ignore[union-attr]


async def test_runner_tasks_and_capability_report(settings: Settings) -> None:
    profile = RunProfile(name="t", layers=["L1"], samples_per_layer=1, sample_timeout_s=10)
    pcm = b"\x00\x10" * 16000
    stimuli = [
        Stimulus("turn_0", "L1", pcm, 16000, 1.0, task="turn", system_prompt="p"),
        Stimulus("tool_0", "L3", pcm, 16000, 1.0, task="tool", tools=(TOOL,)),
        Stimulus("cancel_0", "L2", pcm, 16000, 1.0, task="cancel", cancel_after_ms=0),
    ]
    config = FakeRealtimeConfig(n_audio_chunks=6, chunk_interval_s=0.05)
    async with FakeRealtimeServer(config) as server:
        adapter = RealtimeAdapter(_spec(server.url))
        manifest = await execute_run(
            settings, profile, adapter, stimuli, ModelVersion(model_id="fake")
        )
    assert manifest.status is RunStatus.COMPLETED, manifest
    assert manifest.is_mock is False
    run_dir = settings.raw_dir / manifest.run_id
    tool_result = json.loads((run_dir / "L3" / "tool_0" / "result.json").read_text("utf-8"))
    assert tool_result["tool_calls"][0]["name"] == "record_request"
    assert tool_result["terminal_event"] == "response_done"
    cancel_result = json.loads((run_dir / "L2" / "cancel_0" / "result.json").read_text("utf-8"))
    assert cancel_result["cancel_sent"] is True
    assert cancel_result["terminal_event"] == "response_cancelled"
    assert (run_dir / "L1" / "turn_0" / "output_text.txt").read_text("utf-8").strip() == "はい、"

    report = json.loads((run_dir / "capability_report.json").read_text("utf-8"))
    obs = report["observations"]
    assert obs["streaming_audio_output"]["observation"] == "observed"
    assert obs["text_output_channel"]["observation"] == "observed"
    assert obs["native_tool_calling"]["observation"] == "observed"
    assert obs["response_cancel"]["observation"] == "observed"
    assert obs["native_full_duplex"]["observation"] == "not_tested"


async def test_open_session_failure_is_an_error_sample(settings: Settings) -> None:
    profile = RunProfile(name="t", layers=["L1"], samples_per_layer=1, sample_timeout_s=5)
    adapter = RealtimeAdapter(_spec("ws://127.0.0.1:9/v1/realtime"))
    stimuli = [Stimulus("s", "L1", b"\x00\x00" * 160, 16000, 0.01)]
    manifest = await execute_run(settings, profile, adapter, stimuli, ModelVersion(model_id="x"))
    assert manifest.status is RunStatus.FAILED
    result = json.loads(
        (settings.raw_dir / manifest.run_id / "L1" / "s" / "result.json").read_text("utf-8")
    )
    assert result["reason"].startswith("open_session failed")
    report = json.loads(
        (settings.raw_dir / manifest.run_id / "capability_report.json").read_text("utf-8")
    )
    assert report["observations"]["streaming_audio_output"]["observation"] == "not_tested"
