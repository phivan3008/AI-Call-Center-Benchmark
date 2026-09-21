"""Result bundles for the GPU-server -> developer round trip (DEPLOYMENT_GUIDE §7).

A bundle is ``<run_id>.tar.zst`` plus a sidecar ``<run_id>.SHA256SUMS`` holding the
archive hash. Inside, the run directory carries its own ``SHA256SUMS``; verification checks
both levels and validates the manifest before any analysis.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import zstandard

from benchmark.core.artifacts import (
    MANIFEST,
    atomic_write_text,
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)
from benchmark.core.schemas import RunManifest

AUDIO_SUFFIXES = frozenset({".wav", ".flac"})
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".log", ".yaml", ".yml", ".csv", ""})
SECRET_ENV_VARS = ("OPENAI_API_KEY", "HF_TOKEN")
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
)
_MIN_SECRET_LEN = 8


class BundleError(Exception):
    pass


@dataclass
class VerifyResult:
    archive: Path
    ok: bool
    problems: list[str] = field(default_factory=list)
    manifest: RunManifest | None = None


def scan_for_secrets(root: Path, env: Mapping[str, str] | None = None) -> list[str]:
    """Return relative paths of text files containing secret-looking strings.

    Checks the literal values of known secret env vars plus generic key patterns.
    Secret values are never included in the returned findings.
    """
    env = os.environ if env is None else env
    literals = [env[name] for name in SECRET_ENV_VARS if len(env.get(name, "")) >= _MIN_SECRET_LEN]
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(lit in text for lit in literals) or any(p.search(text) for p in SECRET_PATTERNS):
            findings.append(path.relative_to(root).as_posix())
    return findings


def _members(run_dir: Path, include_audio: bool) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and (include_audio or path.suffix.lower() not in AUDIO_SUFFIXES):
            yield path


def create_bundle(
    run_dir: Path,
    out_dir: Path,
    include_audio: bool = True,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Create ``<out_dir>/<run_id>.tar.zst`` and its sidecar checksum file."""
    if not (run_dir / MANIFEST).is_file():
        raise BundleError(f"Not a run directory (no {MANIFEST}): {run_dir}")
    leaks = scan_for_secrets(run_dir, env)
    if leaks:
        raise BundleError(f"Refusing to bundle: secret-like strings found in {leaks}")

    write_sha256sums(run_dir)
    run_id = run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{run_id}.tar.zst"

    with archive.open("wb") as fh:
        compressor = zstandard.ZstdCompressor(level=10)
        with (
            compressor.stream_writer(fh, closefd=False) as writer,
            tarfile.open(fileobj=writer, mode="w|") as tar,
        ):
            for path in _members(run_dir, include_audio):
                tar.add(path, arcname=f"{run_id}/{path.relative_to(run_dir).as_posix()}")
            if not include_audio:
                # The inner SHA256SUMS still lists the omitted audio files; record why.
                note = b"audio excluded from this bundle (--no-audio)\n"
                info = tarfile.TarInfo(f"{run_id}.BUNDLE_NOTE")
                info.size = len(note)
                tar.addfile(info, io.BytesIO(note))
    atomic_write_text(out_dir / f"{run_id}.SHA256SUMS", f"{sha256_file(archive)}  {archive.name}\n")
    return archive


def _extract_stream(archive: Path, dest: Path) -> list[str]:
    """Stream-extract a ``.tar.zst``; rejects links and paths escaping ``dest``."""
    dest_resolved = dest.resolve()
    names: list[str] = []
    with (
        archive.open("rb") as fh,
        zstandard.ZstdDecompressor().stream_reader(fh) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        for member in tar:
            target = (dest / member.name).resolve()
            if member.issym() or member.islnk() or not target.is_relative_to(dest_resolved):
                raise BundleError(f"Unsafe archive member: {member.name}")
            tar.extract(member, dest, filter="data")
            names.append(member.name)
    return names


def verify_bundle(archive: Path, extract_to: Path | None = None) -> VerifyResult:
    """Verify archive hash, inner checksums and manifest schema.

    With ``extract_to``, the run directory is extracted there (kept after verification).
    """
    result = VerifyResult(archive=archive, ok=False)
    sidecar = archive.with_name(archive.name.removesuffix(".tar.zst") + ".SHA256SUMS")
    if not sidecar.is_file():
        result.problems.append(f"missing sidecar checksum file: {sidecar.name}")
        return result
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(archive) != expected:
        result.problems.append("archive checksum mismatch")
        return result

    with tempfile.TemporaryDirectory() as tmp:
        dest = extract_to or Path(tmp)
        dest.mkdir(parents=True, exist_ok=True)
        names = _extract_stream(archive, dest)

        run_id = archive.name.removesuffix(".tar.zst")
        run_dir = dest / run_id
        audio_excluded = f"{run_id}.BUNDLE_NOTE" in names
        problems = verify_sha256sums(run_dir) if run_dir.is_dir() else ["run directory missing"]
        if audio_excluded:
            problems = [
                p
                for p in problems
                if not (
                    p.startswith("missing file: ")
                    and Path(p.removeprefix("missing file: ")).suffix in AUDIO_SUFFIXES
                )
            ]
        result.problems.extend(problems)
        try:
            result.manifest = RunManifest.model_validate_json(
                (run_dir / MANIFEST).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            result.problems.append(f"invalid manifest: {exc}")
    result.ok = not result.problems
    return result
