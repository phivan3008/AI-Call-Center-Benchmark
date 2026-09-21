from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from benchmark.core.schemas import DatasetSample, MetricRecord, MetricStatus


def _metric(**overrides: Any) -> MetricRecord:
    data: dict[str, Any] = {
        "run_id": "R",
        "model_id": "m",
        "model_revision": "abc",
        "layer": "L2",
        "metric": "l2.ttfa_ms",
        "value": 123.0,
        "unit": "ms",
        "status": MetricStatus.MEASURED,
        "method": "ttfa_v1",
        "measured_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return MetricRecord.model_validate(data)


def test_measured_metric_requires_value() -> None:
    assert _metric().value == 123.0
    with pytest.raises(ValidationError, match="requires a value"):
        _metric(value=None)


@pytest.mark.parametrize("status", ["not_measured", "unsupported", "error"])
def test_non_measured_metric_has_no_value_and_a_reason(status: str) -> None:
    ok = _metric(status=status, value=None, reason="model crashed")
    assert ok.value is None
    with pytest.raises(ValidationError, match="must have value=None"):
        _metric(status=status, value=1.0, reason="x")
    with pytest.raises(ValidationError, match="requires a reason"):
        _metric(status=status, value=None)


def test_metric_rejects_unknown_layer_and_fields() -> None:
    with pytest.raises(ValidationError):
        _metric(layer="L9")
    with pytest.raises(ValidationError):
        _metric(extra_field=1)


def _sample(**overrides: Any) -> DatasetSample:
    data: dict[str, Any] = {
        "sample_id": "ds.v1.0001",
        "audio_path": "audio/0001.flac",
        "audio_sha256": "0" * 64,
        "duration_s": 3.0,
        "sample_rate_hz": 16000,
        "speech_start_s": 0.1,
        "speech_end_s": 2.9,
        "text": "お問い合わせありがとうございます。",
        "channel": "clean_16k",
        "speaker": {"id": "spk_01"},
        "synthetic": False,
        "source": {"corpus": "test"},
        "license": "CC0-1.0",
    }
    data.update(overrides)
    return DatasetSample.model_validate(data)


GENERATOR = {"type": "tts", "engine": "e", "version": "1", "voice": "v"}


def test_dataset_sample_valid() -> None:
    assert _sample().labels.slots == []


def test_tts_audio_requires_generator_and_synthetic_flag() -> None:
    voice = {"id": "tts_01", "synthetic_voice": True}
    with pytest.raises(ValidationError, match="requires a 'generator'"):
        _sample(synthetic=True, speaker=voice)
    with pytest.raises(ValidationError, match="implies synthetic=true"):
        _sample(synthetic=False, speaker=voice, generator=GENERATOR)
    assert _sample(synthetic=True, speaker=voice, generator=GENERATOR).generator is not None


def test_generator_requires_synthetic() -> None:
    with pytest.raises(ValidationError, match="must be marked synthetic"):
        _sample(generator=GENERATOR)


def test_human_read_authored_text_is_synthetic_without_generator() -> None:
    assert _sample(synthetic=True, text_origin="authored").generator is None


def test_speech_bounds_checked() -> None:
    with pytest.raises(ValidationError, match="speech_start_s"):
        _sample(speech_end_s=3.5)
