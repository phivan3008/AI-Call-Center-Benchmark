"""Build runner stimuli from a prepared dataset, a run profile and its prompt file."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from benchmark.adapters.base import ToolSpec
from benchmark.audio.pcm import read_wav
from benchmark.core.artifacts import sha256_file
from benchmark.core.config import ConfigError, RunProfile, Settings, load_model
from benchmark.core.schemas import DatasetRef
from benchmark.datasets.fleurs import build_fleurs_subset, load_manifest
from benchmark.runner import Stimulus

DATASET_VERSION = "v1"


class PromptEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str
    tools: list[ToolSpec] = Field(default_factory=list)


class PromptFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    prompts: dict[str, PromptEntry]


# Dataset name -> builder(out_dir). Each builder is deterministic and version-pinned.
DATASET_BUILDERS: dict[str, Callable[[Path], Path]] = {
    "fleurs_ja_smoke": lambda out: build_fleurs_subset(out, "fleurs_ja_smoke", n=12),
}


def dataset_dir(settings: Settings, name: str) -> Path:
    return settings.home / "datasets_audio" / name / DATASET_VERSION


def reference_manifest(settings: Settings, name: str) -> Path:
    """Manifest committed in the repo (``datasets/<name>/<version>/manifest.jsonl``)."""
    return settings.datasets_dir / name / DATASET_VERSION / "manifest.jsonl"


def compare_with_reference(built: Path, reference: Path) -> list[str]:
    """Differences between a freshly built manifest and the committed reference."""
    got = {s.sample_id: s for s in load_manifest(built)}
    want = {s.sample_id: s for s in load_manifest(reference)}
    problems = [f"missing sample: {sid}" for sid in sorted(set(want) - set(got))]
    problems += [f"unexpected sample: {sid}" for sid in sorted(set(got) - set(want))]
    for sid in sorted(set(got) & set(want)):
        if got[sid] != want[sid]:
            fields = [
                f
                for f in type(got[sid]).model_fields
                if getattr(got[sid], f) != getattr(want[sid], f)
            ]
            problems.append(f"{sid}: differs in {', '.join(fields)}")
    return problems


def prepare_dataset(settings: Settings, name: str) -> Path:
    """Build a dataset; if the repo has a reference manifest, the build must match it."""
    try:
        builder = DATASET_BUILDERS[name]
    except KeyError:
        known = ", ".join(sorted(DATASET_BUILDERS))
        raise ConfigError(f"Unknown dataset '{name}'. Known: {known}") from None
    built = builder(dataset_dir(settings, name))
    reference = reference_manifest(settings, name)
    if reference.is_file():
        problems = compare_with_reference(built, reference)
        if problems:
            raise ConfigError(
                f"built dataset differs from {reference}: " + "; ".join(problems[:10])
            )
    return built


def load_prompts(settings: Settings, name: str) -> PromptFile:
    return load_model(settings.configs_dir / "prompts" / f"{name}.yaml", PromptFile)


def build_stimuli(
    settings: Settings, profile: RunProfile
) -> tuple[list[Stimulus], list[DatasetRef]]:
    """Assign dataset samples to the profile's tasks in manifest order."""
    if not profile.dataset or not profile.prompts or not profile.tasks:
        raise ConfigError(f"profile '{profile.name}' needs dataset, prompts and tasks")
    manifest = dataset_dir(settings, profile.dataset) / "manifest.jsonl"
    if not manifest.is_file():
        raise ConfigError(
            f"dataset '{profile.dataset}' is not prepared; run: "
            f"vbench data prepare --dataset {profile.dataset}"
        )
    samples = load_manifest(manifest)
    needed = sum(task.count for task in profile.tasks)
    if len(samples) < needed:
        raise ConfigError(f"dataset has {len(samples)} samples, profile needs {needed}")
    prompts = load_prompts(settings, profile.prompts)

    stimuli: list[Stimulus] = []
    cursor = 0
    for task in profile.tasks:
        try:
            prompt = prompts.prompts[task.prompt]
        except KeyError:
            raise ConfigError(f"prompt '{task.prompt}' not found in {profile.prompts}") from None
        for sample in samples[cursor : cursor + task.count]:
            audio_path = manifest.parent / sample.audio_path
            if sha256_file(audio_path) != sample.audio_sha256:
                raise ConfigError(f"checksum mismatch for {audio_path}; re-run data prepare")
            pcm, rate = read_wav(audio_path)
            stimuli.append(
                Stimulus(
                    sample_id=sample.sample_id,
                    layer=task.layer,
                    pcm16=pcm,
                    sample_rate_hz=rate,
                    speech_end_s=sample.speech_end_s,
                    task=task.kind,
                    system_prompt=prompt.system,
                    tools=tuple(prompt.tools),
                    cancel_after_ms=task.cancel_after_ms,
                    reference_text=sample.text,
                )
            )
        cursor += task.count
    ref = DatasetRef(
        name=profile.dataset, version=DATASET_VERSION, manifest_sha256=sha256_file(manifest)
    )
    return stimuli, [ref]
