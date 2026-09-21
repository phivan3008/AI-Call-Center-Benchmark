from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from benchmark.adapters.mock import MockAdapter, MockConfig
from benchmark.core.artifacts import verify_sha256sums
from benchmark.core.config import RunProfile, Settings
from benchmark.core.schemas import ModelVersion, RunManifest, RunStatus
from benchmark.runner import Stimulus, execute_run, mock_stimuli

MODEL = ModelVersion(model_id="mock", engine="mock")


def _events(sample_dir: Path) -> list[dict[str, object]]:
    lines = (sample_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


async def test_mock_run_writes_complete_raw_artifacts(
    settings: Settings, profile: RunProfile
) -> None:
    manifest = await execute_run(settings, profile, MockAdapter(), mock_stimuli(profile), MODEL)
    run_dir = settings.raw_dir / manifest.run_id
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.is_mock is True
    assert manifest.samples_total == manifest.samples_completed == 2
    assert manifest.finished_at is not None

    stored = RunManifest.model_validate_json((run_dir / "manifest.json").read_text("utf-8"))
    assert stored == manifest
    assert verify_sha256sums(run_dir) == []

    sample_dir = run_dir / "L1" / "mock_l1_0000"
    events = _events(sample_dir)
    types = [e["type"] for e in events]
    assert types[0] == "harness.user_speech_end"
    assert "harness.input_committed" in types
    assert types[-1] == "response_done"
    times = [int(e["received_ns"]) for e in events]  # type: ignore[call-overload]
    assert times == sorted(times)

    result = json.loads((sample_dir / "result.json").read_text("utf-8"))
    assert result["status"] == "completed"
    assert result["t_user_speech_end_ns"] <= result["t_first_event_ns"]
    with wave.open(str(sample_dir / "output.wav")) as wav:
        assert wav.getframerate() == 24000
    assert result["output_audio_ms"] == pytest.approx(5 * 40)

    log_lines = (run_dir / "logs" / "run.jsonl").read_text("utf-8").splitlines()
    first = json.loads(log_lines[0])
    assert first["event"] == "run_started"
    assert first["run_id"] == manifest.run_id


async def test_failures_are_recorded_not_raised(settings: Settings, profile: RunProfile) -> None:
    adapter = MockAdapter(MockConfig(failure_rate=1.0))
    manifest = await execute_run(settings, profile, adapter, mock_stimuli(profile), MODEL)
    assert manifest.status is RunStatus.FAILED
    assert manifest.samples_failed == 2
    result = json.loads(
        (settings.raw_dir / manifest.run_id / "L1" / "mock_l1_0000" / "result.json").read_text(
            "utf-8"
        )
    )
    assert result["reason"] == "model error: mock_failure"


async def test_timeout_is_an_error_sample(settings: Settings) -> None:
    profile = RunProfile(name="t", layers=["L2"], samples_per_layer=1, sample_timeout_s=0.05)
    adapter = MockAdapter(MockConfig(first_audio_delay_ms=2000))
    manifest = await execute_run(settings, profile, adapter, mock_stimuli(profile), MODEL)
    assert manifest.status is RunStatus.FAILED
    result = json.loads(
        (settings.raw_dir / manifest.run_id / "L2" / "mock_l2_0000" / "result.json").read_text(
            "utf-8"
        )
    )
    assert result["reason"].startswith("timeout")


async def test_repeats_use_separate_directories(settings: Settings) -> None:
    profile = RunProfile(name="r", layers=["L1"], samples_per_layer=1, repeats=2)
    manifest = await execute_run(settings, profile, MockAdapter(), mock_stimuli(profile), MODEL)
    sample_dir = settings.raw_dir / manifest.run_id / "L1" / "mock_l1_0000"
    assert (sample_dir / "r0" / "events.jsonl").is_file()
    assert (sample_dir / "r1" / "events.jsonl").is_file()
    assert manifest.samples_total == 2


async def test_speech_end_timestamp_uses_speech_end_not_file_end(
    settings: Settings, profile: RunProfile
) -> None:
    rate = 16000
    stimulus = Stimulus(
        sample_id="s",
        layer="L1",
        pcm16=b"\x00\x00" * rate,  # 1.0 s of audio, 50 frames of 20 ms
        sample_rate_hz=rate,
        speech_end_s=0.5,
    )
    manifest = await execute_run(settings, profile, MockAdapter(), [stimulus], MODEL)
    events = _events(settings.raw_dir / manifest.run_id / "L1" / "s")
    speech_end = next(e for e in events if e["type"] == "harness.user_speech_end")
    committed = next(e for e in events if e["type"] == "harness.input_committed")
    assert int(speech_end["received_ns"]) <= int(committed["received_ns"])  # type: ignore[call-overload]


def test_mock_stimuli_cover_all_layers() -> None:
    profile = RunProfile(name="m", layers=["L1", "L3"], samples_per_layer=2)
    stimuli = mock_stimuli(profile)
    assert [s.sample_id for s in stimuli] == [
        "mock_l1_0000",
        "mock_l1_0001",
        "mock_l3_0000",
        "mock_l3_0001",
    ]
