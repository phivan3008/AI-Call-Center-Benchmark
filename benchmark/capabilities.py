"""Capability report written at the end of every run (ARCHITECTURE §6.3).

The report records what was *observed* in this run, with the sample ids as evidence. It is
not a verdict: ``verified`` fields in ``configs/models/*.yaml`` are only updated after a
person has reviewed the report.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from benchmark.adapters.base import ModelCapabilities

if TYPE_CHECKING:
    from benchmark.runner import SampleResult

Observation = Literal["observed", "not_observed", "not_tested", "inconclusive"]


def _entry(status: Observation, detail: str, evidence: Sequence[str]) -> dict[str, Any]:
    return {"observation": status, "detail": detail, "evidence": list(evidence)}


def _valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def build_capability_report(
    model_id: str, results: Sequence[SampleResult], declared: ModelCapabilities
) -> dict[str, Any]:
    completed = [r for r in results if r.status == "completed"]
    turn = [r for r in completed if r.task in {"turn", "cancel"}]

    streaming = [r.sample_id for r in turn if r.n_audio_deltas >= 2]
    if not turn:
        stream = _entry("not_tested", "no completed turn samples", [])
    elif streaming:
        stream = _entry(
            "observed",
            f"{len(streaming)}/{len(turn)} responses arrived in >=2 audio deltas",
            streaming,
        )
    else:
        stream = _entry("not_observed", "every response arrived as a single audio chunk", [])

    with_text = [r.sample_id for r in turn if r.n_text_deltas > 0]
    if not turn:
        text = _entry("not_tested", "no completed turn samples", [])
    elif with_text:
        text = _entry(
            "observed", f"{len(with_text)}/{len(turn)} responses had text deltas", with_text
        )
    else:
        text = _entry("not_observed", "no text deltas in any response", [])

    tool_samples = [r for r in completed if r.task == "tool"]
    good_calls = [
        r.sample_id
        for r in tool_samples
        if any(c["name"] and _valid_json(c["arguments"]) for c in r.tool_calls)
    ]
    if not tool_samples:
        tools = _entry("not_tested", "no completed tool samples", [])
    elif good_calls:
        tools = _entry(
            "observed",
            f"{len(good_calls)}/{len(tool_samples)} tool samples emitted a named JSON call",
            good_calls,
        )
    else:
        tools = _entry("not_observed", "tools were declared but no valid tool call was emitted", [])

    cancel_samples = [r for r in results if r.task == "cancel"]
    cancelled = [
        r.sample_id
        for r in cancel_samples
        if r.cancel_sent and r.terminal_event == "response_cancelled"
    ]
    ignored = [
        r.sample_id for r in cancel_samples if r.cancel_sent and r.terminal_event == "response_done"
    ]
    if not cancel_samples:
        cancel = _entry("not_tested", "no cancel samples", [])
    elif cancelled:
        cancel = _entry(
            "observed",
            f"{len(cancelled)}/{len(cancel_samples)} responses were cancelled",
            cancelled,
        )
    elif ignored:
        cancel = _entry(
            "not_observed", "cancel was sent but the response completed normally", ignored
        )
    else:
        cancel = _entry(
            "inconclusive", "responses ended before the cancel was sent (or errored)", []
        )

    return {
        "model_id": model_id,
        "note": (
            "Observations from this run only. Update `verified` in configs/models/*.yaml "
            "after review; never automatically."
        ),
        "declared": declared.model_dump(mode="json"),
        "observations": {
            "streaming_audio_output": stream,
            "text_output_channel": text,
            "native_tool_calling": tools,
            "response_cancel": cancel,
            "native_full_duplex": _entry(
                "not_tested", "turn-based session (duplex is tested in Phase 6)", []
            ),
        },
    }
