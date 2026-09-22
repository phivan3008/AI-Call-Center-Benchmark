"""Shared Pydantic schemas (ARCHITECTURE §9, DATASET_SPEC §3, METRIC_DEFINITIONS §1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Layer = Literal["L1", "L2", "L3", "L4", "L5"]


class MetricStatus(StrEnum):
    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelVersion(_Strict):
    model_id: str
    repo: str | None = None
    revision: str | None = None
    engine: str | None = None
    engine_version: str | None = None
    runtime_lock_sha256: str | None = None


class DatasetRef(_Strict):
    name: str
    version: str
    manifest_sha256: str


class GpuInfo(_Strict):
    index: int
    name: str
    memory_total_mb: int
    uuid: str | None = None


class EnvInfo(_Strict):
    hostname: str
    platform: str
    python_version: str
    packages: dict[str, str] = Field(default_factory=dict)
    gpus: list[GpuInfo] = Field(default_factory=list)
    driver_version: str | None = None
    cuda_driver_version: str | None = None


class RunManifest(_Strict):
    run_id: str
    profile: str
    is_mock: bool
    started_at: datetime
    finished_at: datetime | None = None
    git_commit: str
    git_dirty: bool
    source_kind: Literal["git", "release", "unknown"] = "unknown"
    release_name: str | None = None
    config_sha256: str
    model: ModelVersion
    datasets: list[DatasetRef] = Field(default_factory=list)
    environment: EnvInfo
    seeds: dict[str, int] = Field(default_factory=dict)
    layers: list[Layer]
    status: RunStatus = RunStatus.RUNNING
    samples_total: int = 0
    samples_completed: int = 0
    samples_failed: int = 0
    gpu_contended: bool = False
    notes: list[str] = Field(default_factory=list)


class MetricRecord(_Strict):
    run_id: str
    model_id: str
    model_revision: str | None
    layer: Layer
    metric: str
    sample_id: str | None = None
    value: float | None
    unit: str
    status: MetricStatus
    reason: str | None = None
    method: str
    source_files: list[str] = Field(default_factory=list)
    measured_at: datetime
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _value_matches_status(self) -> MetricRecord:
        if self.status is MetricStatus.MEASURED and self.value is None:
            raise ValueError("status 'measured' requires a value")
        if self.status is not MetricStatus.MEASURED:
            if self.value is not None:
                raise ValueError(f"status '{self.status}' must have value=None")
            if not self.reason:
                raise ValueError(f"status '{self.status}' requires a reason")
        return self


# --- Dataset manifest (DATASET_SPEC §3) ---------------------------------------------------


class Speaker(_Strict):
    id: str
    gender: Literal["f", "m", "x"] | None = None
    age_band: str | None = None
    synthetic_voice: bool = False


class Generator(_Strict):
    type: Literal["tts"]
    engine: str
    version: str
    voice: str
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class SampleSource(_Strict):
    corpus: str
    item: str | None = None
    url: str | None = None


class SlotLabel(_Strict):
    name: str
    value: str


class SampleLabels(_Strict):
    intent: str | None = None
    slots: list[SlotLabel] = Field(default_factory=list)


class DatasetSample(_Strict):
    sample_id: str
    audio_path: str
    audio_sha256: str
    duration_s: float = Field(gt=0)
    sample_rate_hz: int = Field(gt=0)
    speech_start_s: float = Field(ge=0)
    speech_end_s: float = Field(ge=0)
    text: str
    text_reading_kana: str | None = None
    reading_source: Literal["corpus", "analyzer", "human"] | None = None
    labels: SampleLabels = Field(default_factory=SampleLabels)
    channel: Literal["clean_16k", "clean_24k", "telephone_8k"]
    speaker: Speaker
    synthetic: bool
    generator: Generator | None = None
    text_origin: Literal["corpus", "authored", "llm_draft_human_reviewed"] | None = None
    source: SampleSource
    license: str
    split: str = "test"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> DatasetSample:
        # Authored text read by a human is synthetic without a generator; TTS audio is not.
        if self.speaker.synthetic_voice and not self.synthetic:
            raise ValueError("a synthetic voice implies synthetic=true")
        if self.speaker.synthetic_voice and self.generator is None:
            raise ValueError("synthetic (TTS) audio requires a 'generator' record")
        if self.generator is not None and not self.synthetic:
            raise ValueError("a sample with a 'generator' must be marked synthetic")
        if not self.speech_start_s <= self.speech_end_s <= self.duration_s:
            raise ValueError("require speech_start_s <= speech_end_s <= duration_s")
        return self
