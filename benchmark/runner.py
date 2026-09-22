"""Turn-mode collection runner (Stage 2 "Collect", ARCHITECTURE §4 and §7).

Drives any :class:`ModelAdapter` through a list of stimuli, recording a harness-timestamped
event timeline, input/output audio, output text and a per-sample result. Three task kinds:

* ``turn``: send the utterance, commit, wait for the response.
* ``tool``: like turn, with tools declared; the first tool call gets a fixed tool result and
  the run waits for the follow-up response.
* ``cancel``: like turn; ``cancel_after_ms`` after the first audio delta the harness sends
  ``cancel_response`` (orchestrated barge-in check).

Metric computation happens later in evaluators; this module writes raw artifacts only.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from benchmark.adapters.base import (
    TERMINAL_EVENT_TYPES,
    AudioChunk,
    AudioDelta,
    ModelAdapter,
    SessionConfig,
    SpeechSession,
    TextDelta,
    TimedEvent,
    ToolCall,
    ToolSpec,
)
from benchmark.audio.pcm import write_wav
from benchmark.core.artifacts import (
    JsonlWriter,
    RunPaths,
    atomic_write_json,
    atomic_write_text,
    write_sha256sums,
)
from benchmark.core.clock import Clock, SystemClock
from benchmark.core.config import RunProfile, Settings
from benchmark.core.logging import bind_context, configure_logging, get_logger
from benchmark.core.provenance import capture_env, config_sha256, new_run_id, source_state
from benchmark.core.schemas import DatasetRef, Layer, ModelVersion, RunManifest, RunStatus

log = get_logger(__name__)

TaskKind = Literal["turn", "tool", "cancel"]
TOOL_RESULT: dict[str, Any] = {"status": "recorded"}


@dataclass(frozen=True)
class Stimulus:
    sample_id: str
    layer: Layer
    pcm16: bytes
    sample_rate_hz: int
    speech_end_s: float
    task: TaskKind = "turn"
    system_prompt: str = ""
    tools: tuple[ToolSpec, ...] = field(default_factory=tuple)
    cancel_after_ms: int = 1000
    reference_text: str | None = None


class SampleResult(BaseModel):
    sample_id: str
    layer: Layer
    task: TaskKind = "turn"
    repeat: int
    status: Literal["completed", "error"]
    reason: str | None = None
    terminal_event: str | None = None
    n_events: int
    n_audio_deltas: int = 0
    n_text_deltas: int = 0
    tool_calls: list[dict[str, str]] = []
    cancel_sent: bool = False
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


def _harness_event(clock: Clock, kind: str, **payload: Any) -> TimedEvent:
    return TimedEvent(
        received_ns=clock.monotonic_ns(), type=f"harness.{kind}", response_id=None, payload=payload
    )


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
        bind_context(
            sample_id=stimulus.sample_id, layer=stimulus.layer, task=stimulus.task, repeat=repeat
        )

        timeline: list[TimedEvent] = []
        output = bytearray()
        output_rate: int | None = None
        text_parts: list[str] = []
        tool_calls: list[dict[str, str]] = []
        state: dict[str, Any] = {"cancel_sent": False, "tool_result_sent": False}
        status: Literal["completed", "error"] = "completed"
        reason: str | None = None
        terminal_type: str | None = None
        t_eos: int | None = None
        cancel_task: asyncio.Task[None] | None = None

        try:
            session = await self.adapter.open_session(
                SessionConfig(
                    session_id=f"{self.paths.run_id}-{stimulus.sample_id}-{repeat}",
                    system_prompt=stimulus.system_prompt,
                    tools=list(stimulus.tools),
                )
            )
        except Exception as exc:
            session = None
            status, reason = "error", f"open_session failed: {type(exc).__name__}: {exc}"[:500]

        async def delayed_cancel(sess: SpeechSession) -> None:
            await asyncio.sleep(stimulus.cancel_after_ms / 1000)
            timeline.append(_harness_event(self.clock, "cancel_sent"))
            state["cancel_sent"] = True
            await sess.cancel_response()

        async def consume(sess: SpeechSession) -> TimedEvent | None:
            nonlocal output_rate, cancel_task
            pending_call: ToolCall | None = None
            async for event in sess.events():
                timed = TimedEvent.from_event(self.clock.monotonic_ns(), event)
                timeline.append(timed)
                if isinstance(event, AudioDelta):
                    output.extend(event.pcm16)
                    output_rate = event.sample_rate_hz
                    if stimulus.task == "cancel" and cancel_task is None:
                        cancel_task = asyncio.create_task(delayed_cancel(sess))
                elif isinstance(event, TextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, ToolCall):
                    tool_calls.append(
                        {
                            "call_id": event.call_id,
                            "name": event.name,
                            "arguments": event.arguments_json,
                        }
                    )
                    if pending_call is None:
                        pending_call = event
                if timed.type in TERMINAL_EVENT_TYPES:
                    needs_result = (
                        timed.type == "response_done"
                        and stimulus.task == "tool"
                        and pending_call is not None
                        and not state["tool_result_sent"]
                    )
                    if needs_result and pending_call is not None:
                        state["tool_result_sent"] = True
                        timeline.append(
                            _harness_event(
                                self.clock, "tool_result_sent", call_id=pending_call.call_id
                            )
                        )
                        await sess.send_tool_result(pending_call.call_id, TOOL_RESULT)
                        continue
                    return timed
            return None

        if session is not None:
            consumer = asyncio.create_task(consume(session))
            try:
                t_eos = await self._send_input(session, stimulus, timeline)
                terminal = await asyncio.wait_for(consumer, timeout=self.profile.sample_timeout_s)
                if terminal is None:
                    status, reason = "error", "event stream ended without a terminal event"
                else:
                    terminal_type = terminal.type
                    if terminal.type == "error":
                        status = "error"
                        reason = f"model error: {terminal.payload.get('code')}"
            except TimeoutError:
                status, reason = "error", f"timeout after {self.profile.sample_timeout_s}s"
            except Exception as exc:
                status, reason = "error", f"{type(exc).__name__}: {exc}"[:500]
            finally:
                for task in (consumer, cancel_task):
                    if task is not None and not task.done():
                        task.cancel()
                    if task is not None:
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await task
                await session.close()

        with JsonlWriter(sample_dir / "events.jsonl") as writer:
            for timed in sorted(timeline, key=lambda e: e.received_ns):
                writer.write(timed)
        if output and output_rate:
            write_wav(sample_dir / "output.wav", bytes(output), output_rate)
        if text_parts:
            atomic_write_text(sample_dir / "output_text.txt", "".join(text_parts) + "\n")

        result = SampleResult(
            sample_id=stimulus.sample_id,
            layer=stimulus.layer,
            task=stimulus.task,
            repeat=repeat,
            status=status,
            reason=reason,
            terminal_event=terminal_type,
            n_events=len(timeline),
            n_audio_deltas=sum(e.type == "audio_delta" for e in timeline),
            n_text_deltas=sum(e.type == "text_delta" for e in timeline),
            tool_calls=tool_calls,
            cancel_sent=bool(state["cancel_sent"]),
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
                event = _harness_event(
                    self.clock, "user_speech_end", speech_end_s=stimulus.speech_end_s
                )
                t_eos = event.received_ns
                timeline.append(event)
        await session.commit_input()
        timeline.append(_harness_event(self.clock, "input_committed"))
        return t_eos


def start_run(
    settings: Settings,
    profile: RunProfile,
    adapter: ModelAdapter,
    model: ModelVersion,
    clock: Clock | None = None,
    datasets: list[DatasetRef] | None = None,
) -> tuple[RunPaths, RunManifest]:
    clock = clock or SystemClock()
    run_id = new_run_id(clock)
    paths = RunPaths.create(settings.raw_dir, run_id)
    source = source_state(settings.repo_root)
    manifest = RunManifest(
        run_id=run_id,
        profile=profile.name,
        is_mock=adapter.is_mock,
        started_at=clock.utc_now(),
        git_commit=source.commit,
        git_dirty=source.dirty,
        source_kind=source.kind,
        release_name=source.release_name,
        notes=[f"source: {problem}" for problem in source.problems[:20]],
        config_sha256=config_sha256(
            {"profile": profile.model_dump(mode="json"), "model": model.model_dump(mode="json")}
        ),
        model=model,
        datasets=datasets or [],
        environment=capture_env(),
        seeds={"profile": profile.seed},
        layers=sorted(set(profile.layers) | {t.layer for t in profile.tasks}),  # type: ignore[arg-type]
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
    datasets: list[DatasetRef] | None = None,
) -> RunManifest:
    """Run all stimuli x repeats, then write the capability report, manifest and SHA256SUMS."""
    from benchmark.capabilities import build_capability_report

    clock = clock or SystemClock()
    paths, manifest = start_run(settings, profile, adapter, model, clock, datasets)
    configure_logging(settings.log_level, paths.run_log)
    bind_context(run_id=manifest.run_id, model_id=model.model_id)
    log.info("run_started", profile=profile.name, samples=len(stimuli), mock=adapter.is_mock)

    runner = TurnRunner(adapter, profile, paths, clock)
    results: list[SampleResult] = []
    status = RunStatus.FAILED
    failed = 0
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
        failed = sum(r.status == "error" for r in results)
        raise
    finally:
        atomic_write_json(
            paths.root / "capability_report.json",
            build_capability_report(model.model_id, results, adapter.capabilities),
        )
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
