"""FLEURS Japanese (``google/fleurs``, ``ja_jp``) stimuli.

FLEURS is human read speech with transcripts, licensed CC-BY-4.0 (HF dataset card). It is
used for the Phase 2 smoke set (pipeline/capability checks, not scored) and later as the
real-human-speech source for Layer 1 ASR. Selection is deterministic: rows sorted by file
name, filtered by duration, first N taken. Labels are the corpus transcripts, unmodified.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

from benchmark.audio.pcm import duration_s, read_wav, speech_bounds, write_wav
from benchmark.core.artifacts import atomic_write_text, sha256_file
from benchmark.core.schemas import DatasetSample, SampleSource, Speaker

REPO_ID = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # HF API, 2026-09-22
LICENSE = "CC-BY-4.0"
TSV_PATH = "data/ja_jp/test.tsv"
TAR_PATH = "data/ja_jp/audio/test.tar.gz"

Downloader = Callable[..., str]


@dataclass(frozen=True)
class FleursRow:
    row_id: str
    file_name: str
    raw_transcription: str
    num_samples: int
    gender: str


def parse_tsv(text: str) -> list[FleursRow]:
    """FLEURS TSV columns: id, file name, raw transcription, transcription,
    character-split transcription, num_samples, gender (no header row)."""
    rows = []
    for record in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(record) < 7:
            continue
        rows.append(
            FleursRow(
                row_id=record[0],
                file_name=record[1],
                raw_transcription=record[2],
                num_samples=int(record[5]),
                gender=record[6].strip().lower(),
            )
        )
    return rows


def select_rows(
    rows: list[FleursRow], n: int, min_s: float, max_s: float, rate: int = 16000
) -> list[FleursRow]:
    eligible = [r for r in rows if min_s <= r.num_samples / rate <= max_s]
    return sorted(eligible, key=lambda r: r.file_name)[:n]


def build_fleurs_subset(
    out_dir: Path,
    name: str,
    n: int,
    min_s: float = 2.0,
    max_s: float = 10.0,
    downloader: Downloader = hf_hub_download,
) -> Path:
    """Download (pinned revision), select, convert and write ``manifest.jsonl``.

    Returns the manifest path. Existing outputs are overwritten deterministically.
    """
    tsv_file = Path(
        downloader(repo_id=REPO_ID, filename=TSV_PATH, repo_type="dataset", revision=REVISION)
    )
    tar_file = Path(
        downloader(repo_id=REPO_ID, filename=TAR_PATH, repo_type="dataset", revision=REVISION)
    )
    chosen = select_rows(parse_tsv(tsv_file.read_text(encoding="utf-8")), n, min_s, max_s)
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} FLEURS rows match the duration filter, need {n}")
    wanted = {row.file_name: row for row in chosen}

    audio_dir = out_dir / "audio"
    extracted: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(tar_file, "r:gz") as tar:
        for member in tar:
            base = PurePosixPath(member.name).name
            if member.isfile() and base in wanted:
                data = tar.extractfile(member)
                if data is None:  # pragma: no cover - isfile() members always have data
                    continue
                tmp = out_dir / ".tmp.wav"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(data.read())
                extracted[base] = read_wav(tmp)
                tmp.unlink()
    missing = sorted(set(wanted) - set(extracted))
    if missing:
        raise ValueError(f"audio missing from FLEURS archive: {missing[:5]}")

    lines = []
    for index, row in enumerate(chosen):
        pcm, rate = extracted[row.file_name]
        path = audio_dir / f"{index:04d}.wav"
        write_wav(path, pcm, rate)
        bounds = speech_bounds(pcm, rate) or (0.0, duration_s(pcm, rate))
        sample = DatasetSample(
            sample_id=f"{name}.v1.{index:04d}",
            audio_path=f"audio/{path.name}",
            audio_sha256=sha256_file(path),
            duration_s=round(duration_s(pcm, rate), 3),
            sample_rate_hz=rate,
            speech_start_s=bounds[0],
            speech_end_s=min(bounds[1], round(duration_s(pcm, rate), 3)),
            text=row.raw_transcription,
            reading_source=None,
            channel="clean_16k",
            speaker=Speaker(
                id=f"fleurs_{row.gender}_{row.file_name.removesuffix('.wav')}",
                gender="f" if row.gender == "female" else "m" if row.gender == "male" else "x",
            ),
            synthetic=False,
            text_origin="corpus",
            source=SampleSource(
                corpus=f"{REPO_ID}@{REVISION}",
                item=row.file_name,
                url=f"https://huggingface.co/datasets/{REPO_ID}",
            ),
            license=LICENSE,
            split="test",
            tags=["fleurs", "ja_jp"],
        )
        lines.append(sample.model_dump_json())
    manifest = out_dir / "manifest.jsonl"
    atomic_write_text(manifest, "\n".join(lines) + "\n")
    atomic_write_text(
        out_dir / "dataset_info.json",
        json.dumps(
            {
                "name": name,
                "version": "v1",
                "source": REPO_ID,
                "revision": REVISION,
                "license": LICENSE,
                "files": [TSV_PATH, TAR_PATH],
                "selection": {"n": n, "min_s": min_s, "max_s": max_s, "order": "file_name"},
                "speech_bounds_method": "stim_vad_v1 (adaptive energy threshold)",
            },
            indent=2,
        )
        + "\n",
    )
    return manifest


def load_manifest(manifest: Path) -> list[DatasetSample]:
    return [
        DatasetSample.model_validate_json(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
