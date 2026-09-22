from __future__ import annotations

import io
import math
import struct
import tarfile
import wave
from pathlib import Path
from typing import Any

import pytest

from benchmark.core.config import ConfigError, RunProfile, Settings, TaskSpec
from benchmark.datasets import fleurs
from benchmark.stimuli import build_stimuli, dataset_dir, load_prompts, prepare_dataset

RATE = 16000


def _wav_bytes(seconds: float) -> bytes:
    n = int(seconds * RATE)
    pcm = b"\x00\x00" * (RATE // 5) + struct.pack(
        f"<{n}h", *(int(8000 * math.sin(2 * math.pi * 300 * i / RATE)) for i in range(n))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def fake_fleurs(tmp_path: Path) -> Any:
    """A miniature FLEURS ja_jp layout: TSV + tar.gz with a few 16 kHz WAV files."""
    rows = []
    tar_path = tmp_path / "test.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for index, seconds in enumerate([3.0, 1.0, 4.0, 2.5, 12.0, 3.5]):
            name = f"{9000 + index}.wav"
            data = _wav_bytes(seconds)
            info = tarfile.TarInfo(f"test/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            samples = int((seconds + 0.2) * RATE)
            gender = "FEMALE" if index % 2 else "MALE"
            rows.append(
                f"{index}\t{name}\t文{index}です。\t文{index}です\t文 {index}\t{samples}\t{gender}"
            )
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text("\n".join(rows) + "\nbroken\trow\n", encoding="utf-8")

    def downloader(**kwargs: Any) -> str:
        assert kwargs["revision"] == fleurs.REVISION
        assert kwargs["repo_type"] == "dataset"
        return str(tsv_path if kwargs["filename"].endswith(".tsv") else tar_path)

    return downloader


def test_build_fleurs_subset(tmp_path: Path, fake_fleurs: Any) -> None:
    manifest = fleurs.build_fleurs_subset(tmp_path / "out", "t", n=3, downloader=fake_fleurs)
    samples = fleurs.load_manifest(manifest)
    # 1.2 s and 12.2 s rows are outside [2, 10] s; order follows file name.
    assert [s.source.item for s in samples] == ["9000.wav", "9002.wav", "9003.wav"]
    first = samples[0]
    assert first.synthetic is False
    assert first.license == "CC-BY-4.0"
    assert first.text == "文0です。"
    assert first.speaker.gender == "m"
    assert samples[1].speaker.gender == "m"
    assert 0.15 <= first.speech_start_s <= 0.25
    assert first.speech_end_s <= first.duration_s
    assert (tmp_path / "out" / "dataset_info.json").is_file()

    with pytest.raises(ValueError, match="only 4"):
        fleurs.build_fleurs_subset(tmp_path / "o2", "t", n=10, downloader=fake_fleurs)


def test_prepare_and_build_stimuli(
    settings: Settings, fake_fleurs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        __import__("benchmark.stimuli", fromlist=["x"]).DATASET_BUILDERS,
        "fake",
        lambda out: fleurs.build_fleurs_subset(out, "fake", n=4, downloader=fake_fleurs),
    )
    manifest = prepare_dataset(settings, "fake")
    assert manifest.parent == dataset_dir(settings, "fake")

    profile = RunProfile(
        name="p",
        layers=["L1"],
        samples_per_layer=1,
        dataset="fake",
        prompts="ja/smoke_v1",
        tasks=[
            TaskSpec(kind="turn", layer="L1", count=2, prompt="turn"),
            TaskSpec(kind="tool", layer="L3", count=1, prompt="tool"),
            TaskSpec(kind="cancel", layer="L2", count=1, prompt="cancel", cancel_after_ms=500),
        ],
    )
    stimuli, refs = build_stimuli(settings, profile)
    assert [s.task for s in stimuli] == ["turn", "turn", "tool", "cancel"]
    assert stimuli[2].tools[0].name == "record_request"
    assert stimuli[3].cancel_after_ms == 500
    assert stimuli[0].reference_text == "文0です。"
    assert refs[0].name == "fake" and len(refs[0].manifest_sha256) == 64

    too_many = profile.model_copy(
        update={"tasks": [TaskSpec(kind="turn", layer="L1", count=9, prompt="turn")]}
    )
    with pytest.raises(ConfigError, match="needs 9"):
        build_stimuli(settings, too_many)
    bad_prompt = profile.model_copy(
        update={"tasks": [TaskSpec(kind="turn", layer="L1", count=1, prompt="nope")]}
    )
    with pytest.raises(ConfigError, match="prompt 'nope'"):
        build_stimuli(settings, bad_prompt)

    audio = next((dataset_dir(settings, "fake") / "audio").glob("*.wav"))
    audio.write_bytes(audio.read_bytes() + b"\x00\x00")
    with pytest.raises(ConfigError, match="checksum mismatch"):
        build_stimuli(settings, profile)


def test_stimuli_errors(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="Unknown dataset"):
        prepare_dataset(settings, "nope")
    bare = RunProfile(name="p", layers=["L1"], samples_per_layer=1)
    with pytest.raises(ConfigError, match="needs dataset"):
        build_stimuli(settings, bare)
    unprepared = bare.model_copy(
        update={
            "dataset": "fleurs_ja_smoke",
            "prompts": "ja/smoke_v1",
            "tasks": [TaskSpec(kind="turn", layer="L1", count=1, prompt="turn")],
        }
    )
    with pytest.raises(ConfigError, match="not prepared"):
        build_stimuli(settings, unprepared)
    assert "turn" in load_prompts(settings, "ja/smoke_v1").prompts


def test_prepare_rejects_build_that_differs_from_reference(
    settings: Settings, fake_fleurs: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import benchmark.stimuli as stimuli_module

    monkeypatch.setitem(
        stimuli_module.DATASET_BUILDERS,
        "fake",
        lambda out: fleurs.build_fleurs_subset(out, "fake", n=3, downloader=fake_fleurs),
    )
    repo = tmp_path / "repo"
    (repo / "datasets" / "fake" / "v1").mkdir(parents=True)
    local = settings.model_copy(update={"repo_root": repo})
    built = prepare_dataset(local, "fake")  # no reference yet: accepted
    reference = repo / "datasets" / "fake" / "v1" / "manifest.jsonl"
    lines = built.read_text(encoding="utf-8").splitlines()
    reference.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert prepare_dataset(local, "fake") == built  # identical rebuild: accepted

    tampered = lines[0].replace('"license":"CC-BY-4.0"', '"license":"other"')
    extra = lines[1].replace("fake.v1.0001", "fake.v1.0099")
    reference.write_text("\n".join([tampered, extra]) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="differs") as excinfo:
        prepare_dataset(local, "fake")
    message = str(excinfo.value)
    assert "fake.v1.0000: differs in license" in message
    assert "missing sample: fake.v1.0099" in message
    assert "unexpected sample: fake.v1.0001" in message
