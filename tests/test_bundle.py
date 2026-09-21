from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import zstandard

from benchmark.adapters.mock import MockAdapter
from benchmark.bundle import BundleError, create_bundle, scan_for_secrets, verify_bundle
from benchmark.core.artifacts import sha256_file
from benchmark.core.config import RunProfile, Settings
from benchmark.core.schemas import ModelVersion
from benchmark.runner import execute_run, mock_stimuli


@pytest.fixture
async def run_dir(settings: Settings, profile: RunProfile) -> Path:
    manifest = await execute_run(
        settings,
        profile,
        MockAdapter(),
        mock_stimuli(profile),
        ModelVersion(model_id="mock"),
    )
    return settings.raw_dir / manifest.run_id


def test_bundle_roundtrip(run_dir: Path, tmp_path: Path) -> None:
    archive = create_bundle(run_dir, tmp_path / "bundles", env={})
    assert archive.name == f"{run_dir.name}.tar.zst"
    result = verify_bundle(archive, extract_to=tmp_path / "extracted")
    assert result.ok, result.problems
    assert result.manifest is not None
    assert result.manifest.is_mock is True
    assert (tmp_path / "extracted" / run_dir.name / "manifest.json").is_file()


def test_bundle_without_audio_still_verifies(run_dir: Path, tmp_path: Path) -> None:
    archive = create_bundle(run_dir, tmp_path, include_audio=False, env={})
    out = tmp_path / "x"
    result = verify_bundle(archive, extract_to=out)
    assert result.ok, result.problems
    assert not list(out.rglob("*.wav"))


def test_tampered_archive_is_rejected(run_dir: Path, tmp_path: Path) -> None:
    archive = create_bundle(run_dir, tmp_path, env={})
    archive.write_bytes(archive.read_bytes() + b"x")
    result = verify_bundle(archive)
    assert not result.ok
    assert result.problems == ["archive checksum mismatch"]


def test_tampered_content_with_matching_sidecar_is_rejected(run_dir: Path, tmp_path: Path) -> None:
    (run_dir / "L1" / "mock_l1_0000" / "result.json").write_text("{}", encoding="utf-8")
    # Build an archive by hand from the modified directory without refreshing SHA256SUMS.
    archive = tmp_path / f"{run_dir.name}.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))
    (tmp_path / f"{run_dir.name}.SHA256SUMS").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8"
    )
    result = verify_bundle(archive)
    assert not result.ok
    assert any("checksum mismatch" in p for p in result.problems)


def test_missing_sidecar(tmp_path: Path) -> None:
    archive = tmp_path / "RUN.tar.zst"
    archive.write_bytes(b"")
    assert verify_bundle(archive).problems == ["missing sidecar checksum file: RUN.SHA256SUMS"]


def test_unsafe_member_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "RUN.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))
    (tmp_path / "RUN.SHA256SUMS").write_text(f"{sha256_file(archive)}  RUN.tar.zst\n", "utf-8")
    with pytest.raises(BundleError, match="Unsafe"):
        verify_bundle(archive, extract_to=tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_secret_scan_blocks_bundle(run_dir: Path, tmp_path: Path) -> None:
    secret = "my-very-secret-openai-key-value"
    (run_dir / "logs" / "leak.log").write_text(f"key={secret}", encoding="utf-8")
    env = {"OPENAI_API_KEY": secret}
    assert scan_for_secrets(run_dir, env) == ["logs/leak.log"]
    with pytest.raises(BundleError, match="secret-like") as excinfo:
        create_bundle(run_dir, tmp_path, env=env)
    assert secret not in str(excinfo.value)


def test_secret_pattern_detected_without_env(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('{"k": "sk-' + "a" * 30 + '"}', encoding="utf-8")
    (tmp_path / "audio.wav").write_bytes(b"sk-" + b"a" * 30)  # binary files are not scanned
    assert scan_for_secrets(tmp_path, env={}) == ["a.json"]


def test_bundle_requires_run_directory(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="Not a run directory"):
        create_bundle(tmp_path, tmp_path / "out", env={})
