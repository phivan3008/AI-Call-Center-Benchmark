"""Turn-mode collection runner (Stage 2 "Collect", ARCHITECTURE §4 and §7).

Phase 1 scope: drives any :class:`ModelAdapter` through a list of stimuli, recording a
harness-timestamped event timeline, input/output audio and a per-sample result. Metric
computation happens later in evaluators; this module writes raw artifacts only.
"""

from __future__ import annotations

import asyncio
import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from benchmark.adapters.base import (
    TERMINAL_EVENT_TYPES,
    AudioChunk,
    AudioDelta,
    ModelAdapter,
    SessionConfig,
    SpeechSession,
    TimedEvent,
)
from benchmark.core.artifacts import (
    JsonlWriter,
    RunPaths,
    atomic_write_json,
    write_sha256sums,
)
from benchmark.core.clock import Clock, SystemClock
from benchmark.core.config import RunProfile, Settings
from benchmark.core.logging import bind_context, configure_logging, get_logger
from benchmark.core.provenance import capture_env, config_sha256, git_state, new_run_id
from benchmark.core.schemas import Layer, ModelVersion, RunManifest, RunStatus

log = get_logger(__name__)


@dataclass(frozen=True)
class Stimulus:
    sample_id: str
    layer: Layer
    pcm16: bytes
    sample_rate_hz: int
    speech_end_s: float


class SampleResult(BaseModel):
    sample_id: str
    layer: Layer
    repeat: int
    status: Literal["completed", "error"]
    reason: str | None = None
    n_events: int
    output_audio_ms: float
    t_user_speech_end_ns: int | None
    t_first_event_ns: int | None


def mock_stimuli(profile: RunProfile) -> list[Stimulus]:
    """Silence-only placeholder stimuli for mock runs. Never used with real models."""
    rate = profile.input_sample_rate_hz
    duration_s = 1.0
    pcm = b"\x00\x00" * int(rate * duration_s)
    return [
        Stimulus(
            sample_id=f"mock_{layer.lower()}_{index:04d}",
            layer=layer,  # type: ignore[arg-type]
            pcm16=pcm,
            sample_rate_hz=rate,
            speech_end_s=duration_s,
        )
        for layer in profile.layers
        for index in range(profile.samples_per_layer)
    ]


def write_wav(path: Path, pcm16: bytes, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm16)


class TurnRunner:
    def __init__(
        self,
        adapter: ModelAdapter,
        profile: RunProfile,
        paths: RunPaths,
        clock: Clock | None = None,
    ) -> None:
        self.adapter = adapter
        self.profile = profile
        self.paths = paths
        self.clock = clock or SystemClock()

    async def run_sample(self, stimulus: Stimulus, repeat: int) -> SampleResult:
        sample_dir = self.paths.sample_dir(stimulus.layer, stimulus.sample_id)
        if self.profile.repeats > 1:
            sample_dir = sample_dir / f"r{repeat}"
        write_wav(sample_dir / "input.wav", stimulus.pcm16, stimulus.sample_rate_hz)
        bind_context(sample_id=stimulus.sample_id, layer=stimulus.layer, repeat=repeat)

        timeline: list[TimedEvent] = []
        output = bytearray()
        output_rate: int | None = None
        status: Literal["completed", "error"] = "completed"
        reason: str | None = None
        t_eos: int | None = None

        session = await self.adapter.open_session(
            SessionConfig(session_id=f"{self.paths.run_id}-{stimulus.sample_id}-{repeat}")
        )

        async def consume(sess: SpeechSession) -> TimedEvent | None:
            nonlocal output_rate
            async for event in sess.events():
                timed = TimedEvent.from_event(self.clock.monotonic_ns(), event)
                timeline.append(timed)
                if isinstance(event, AudioDelta):
                    output.extend(event.pcm16)
                    output_rate = event.sample_rate_hz
                if timed.type in TERMINAL_EVENT_TYPES:
                    return timed
            return None

        consumer = asyncio.create_task(consume(session))
        try:
            t_eos = await self._send_input(session, stimulus, timeline)
            terminal = await asyncio.wait_for(consumer, timeout=self.profile.sample_timeout_s)
            if terminal is None:
                status, reason = "error", "event stream ended without a terminal event"
            elif terminal.type == "error":
                status, reason = "error", f"model error: {terminal.payload.get('code')}"
        except TimeoutError:
            status, reason = "error", f"timeout after {self.profile.sample_timeout_s}s"
        finally:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
            await session.close()

        with JsonlWriter(sample_dir / "events.jsonl") as writer:
            for timed in sorted(timeline, key=lambda e: e.received_ns):
                writer.write(timed)
        if output and output_rate:
            write_wav(sample_dir / "output.wav", bytes(output), output_rate)

        result = SampleResult(
            sample_id=stimulus.sample_id,
            layer=stimulus.layer,
            repeat=repeat,
            status=status,
            reason=reason,
            n_events=len(timeline),
            output_audio_ms=(len(output) / 2 / output_rate * 1000) if output_rate else 0.0,
            t_user_speech_end_ns=t_eos,
            t_first_event_ns=min(
                (e.received_ns for e in timeline if not e.type.startswith("harness.")),
                default=None,
            ),
        )
        atomic_write_json(sample_dir / "result.json", result)
        if status == "error":
            log.warning("sample_failed", reason=reason)
        return result

    async def _send_input(
        self, session: SpeechSession, stimulus: Stimulus, timeline: list[TimedEvent]
    ) -> int | None:
        """Send the stimulus in frames (turn mode: not real-time paced) and commit.

        Returns the send timestamp of the frame containing the last speech sample (t_eos,
        METRIC_DEFINITIONS §3.1).
        """
        frame_bytes = stimulus.sample_rate_hz * self.profile.frame_ms // 1000 * 2
        speech_end_byte = int(stimulus.speech_end_s * stimulus.sample_rate_hz) * 2
        t_eos: int | None = None
        for offset in range(0, len(stimulus.pcm16), frame_bytes):
            frame = stimulus.pcm16[offset : offset + frame_bytes]
            await session.send_audio(
                AudioChunk(pcm16=frame, sample_rate_hz=stimulus.sample_rate_hz)
            )
            if t_eos is None and offset + len(frame) >= speech_end_byte:
                t_eos = self.clock.monotonic_ns()
                timeline.append(
                    TimedEvent(
                        received_ns=t_eos,
                        type="harness.user_speech_end",
                        response_id=None,
                        payload={"speech_end_s": stimulus.speech_end_s},
                    )
                )
        await session.commit_input()
        timeline.append(
            TimedEvent(
                received_ns=self.clock.monotonic_ns(),
                type="harness.input_committed",
                response_id=None,
                payload={},
            )
        )
        return t_eos


def start_run(
    settings: Settings,
    profile: RunProfile,
    adapter: ModelAdapter,
    model: ModelVersion,
    clock: Clock | None = None,
) -> tuple[RunPaths, RunManifest]:
    clock = clock or SystemClock()
    run_id = new_run_id(clock)
    paths = RunPaths.create(settings.raw_dir, run_id)
    commit, dirty = git_state(settings.repo_root)
    manifest = RunManifest(
        run_id=run_id,
        profile=profile.name,
        is_mock=adapter.is_mock,
        started_at=clock.utc_now(),
        git_commit=commit,
        git_dirty=dirty,
        config_sha256=config_sha256(
            {"profile": profile.model_dump(mode="json"), "model": model.model_dump(mode="json")}
        ),
        model=model,
        environment=capture_env(),
        seeds={"profile": profile.seed},
        layers=profile.layers,  # type: ignore[arg-type]
    )
    atomic_write_json(paths.manifest, manifest)
    return paths, manifest


async def execute_run(
    settings: Settings,
    profile: RunProfile,
    adapter: ModelAdapter,
    stimuli: list[Stimulus],
    model: ModelVersion,
    clock: Clock | None = None,
) -> RunManifest:
    """Run all stimuli x repeats, then finalize manifest and SHA256SUMS."""
    clock = clock or SystemClock()
    paths, manifest = start_run(settings, profile, adapter, model, clock)
    configure_logging(settings.log_level, paths.run_log)
    bind_context(run_id=manifest.run_id, model_id=model.model_id)
    log.info("run_started", profile=profile.name, samples=len(stimuli), mock=adapter.is_mock)

    runner = TurnRunner(adapter, profile, paths, clock)
    results: list[SampleResult] = []
    try:
        for stimulus in stimuli:
            for repeat in range(profile.repeats):
                results.append(await runner.run_sample(stimulus, repeat))
        failed = sum(r.status == "error" for r in results)
        status = RunStatus.COMPLETED if failed == 0 else RunStatus.PARTIAL
        if results and failed == len(results):
            status = RunStatus.FAILED
    except Exception:
        log.exception("run_aborted")
        status = RunStatus.FAILED
        failed = sum(r.status == "error" for r in results)
        raise
    finally:
        manifest = manifest.model_copy(
            update={
                "finished_at": clock.utc_now(),
                "status": status,
                "samples_total": len(stimuli) * profile.repeats,
                "samples_completed": len(results) - failed,
                "samples_failed": failed,
            }
        )
        atomic_write_json(paths.manifest, manifest)
        log.info("run_finished", status=str(status), failed=failed)
        configure_logging(settings.log_level)  # release run.jsonl before checksumming
        write_sha256sums(paths.root)
    return manifest
