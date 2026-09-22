from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark.cli import app
from benchmark.core import provenance
from benchmark.core.provenance import source_state
from benchmark.release import (
    BUILD_INFO,
    ReleaseError,
    build_release,
    load_build_info,
    verify_archive_digest,
    verify_release,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

BUILT_AT = datetime(2026, 9, 22, tzinfo=UTC)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "benchmark").mkdir(parents=True)
    (root / "configs").mkdir()
    # write_bytes: exact LF bytes regardless of the host platform
    (root / "benchmark" / "mod.py").write_bytes(b"X = 1\n")
    (root / "configs" / "a.yaml").write_bytes(b"a: 1\n")
    (root / "README.md").write_bytes(b"readme\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _extract(archive: Path, dest: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest / archive.name.removesuffix(".zip")


def test_build_extract_verify_roundtrip(repo: Path, tmp_path: Path) -> None:
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT, tag="t")
    assert verify_archive_digest(archive)
    extracted = _extract(archive, tmp_path / "server")
    info = load_build_info(extracted)
    assert info is not None
    assert info.release_name == archive.name.removesuffix(".zip")
    assert info.version == "0.1.1"
    assert info.tag == "t"
    assert set(info.files) == {"README.md", "benchmark/mod.py", "configs/a.yaml"}

    state = verify_release(extracted)
    assert state.ok, state.problems
    assert state.commit == info.commit

    # Generated files in ignored locations do not affect integrity.
    (extracted / ".venv").mkdir()
    (extracted / ".venv" / "x.py").write_text("", encoding="utf-8")
    (extracted / "benchmark" / "__pycache__").mkdir()
    (extracted / "benchmark" / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
    (extracted / "notes.txt").write_text("operator notes", encoding="utf-8")
    assert verify_release(extracted).ok


def test_verify_detects_modified_missing_and_unexpected(repo: Path, tmp_path: Path) -> None:
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)
    extracted = _extract(archive, tmp_path / "server")
    (extracted / "benchmark" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    (extracted / "configs" / "a.yaml").unlink()
    (extracted / "benchmark" / "extra.py").write_text("", encoding="utf-8")
    state = verify_release(extracted)
    assert not state.ok
    assert set(state.problems) == {
        "modified file: benchmark/mod.py",
        "missing file: configs/a.yaml",
        "unexpected file: benchmark/extra.py",
    }


def test_crlf_conversion_is_detected(repo: Path, tmp_path: Path) -> None:
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)
    extracted = _extract(archive, tmp_path / "server")
    (extracted / "benchmark" / "mod.py").write_bytes(b"X = 1\r\n")
    assert verify_release(extracted).problems == ["modified file: benchmark/mod.py"]


def test_dirty_tree_refused(repo: Path, tmp_path: Path) -> None:
    (repo / "benchmark" / "mod.py").write_bytes(b"X = 3\n")
    with pytest.raises(ReleaseError, match="uncommitted"):
        build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)


def test_package_content_comes_from_commit_not_working_tree(repo: Path, tmp_path: Path) -> None:
    # Untracked files never ship, even though they sit in the working tree.
    (repo / "benchmark" / "scratch.py").write_bytes(b"untracked\n")
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert not any(n.endswith("scratch.py") for n in names)
    assert any(n.endswith(BUILD_INFO) for n in names)
    info = load_build_info(_extract(archive, tmp_path / "server"))
    assert info is not None
    assert info.files["benchmark/mod.py"] == hashlib.sha256(b"X = 1\n").hexdigest()


def test_archive_digest_mismatch_and_missing_sidecar(repo: Path, tmp_path: Path) -> None:
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)
    archive.write_bytes(archive.read_bytes() + b"x")
    assert verify_archive_digest(archive) is False
    (tmp_path / "dist" / f"{archive.name}.sha256").unlink()
    with pytest.raises(ReleaseError, match="missing checksum"):
        verify_archive_digest(archive)


def test_verify_without_build_info(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match=BUILD_INFO):
        verify_release(tmp_path)


def test_build_requires_git(monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path) -> None:
    monkeypatch.setattr("benchmark.release.shutil.which", lambda _: None)
    with pytest.raises(ReleaseError, match="git is required"):
        build_release(repo, tmp_path, version="0", built_at=BUILT_AT)


def test_build_outside_git_repo_fails(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="failed"):
        build_release(tmp_path, tmp_path / "dist", version="0", built_at=BUILT_AT)


# --- source_state -------------------------------------------------------------------------


def test_source_state_git_checkout(repo: Path) -> None:
    state = source_state(repo)
    assert state.kind == "git"
    assert state.dirty is False
    assert len(state.commit) == 40


def test_source_state_release_clean_and_modified(repo: Path, tmp_path: Path) -> None:
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)
    extracted = _extract(archive, tmp_path / "server")
    clean = source_state(extracted)
    assert clean.kind == "release"
    assert clean.dirty is False
    assert clean.release_name == archive.name.removesuffix(".zip")
    (extracted / "configs" / "a.yaml").write_text("a: 2\n", encoding="utf-8")
    modified = source_state(extracted)
    assert modified.dirty is True
    assert modified.problems == ["modified file: configs/a.yaml"]


def test_source_state_unknown(tmp_path: Path) -> None:
    state = source_state(tmp_path)
    assert (state.kind, state.commit, state.dirty) == ("unknown", "unknown", True)
    (tmp_path / BUILD_INFO).write_text("{not json", encoding="utf-8")
    broken = source_state(tmp_path)
    assert broken.kind == "unknown"
    assert broken.problems


def test_source_state_falls_back_when_git_unusable(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    monkeypatch.setattr(provenance, "git_state", lambda _: ("unknown", True))
    assert source_state(repo).kind == "unknown"


# --- CLI ----------------------------------------------------------------------------------


def test_cli_release_verify(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = build_release(repo, tmp_path / "dist", version="0.1.1", built_at=BUILT_AT)
    extracted = _extract(archive, tmp_path / "server")
    monkeypatch.setattr(
        "benchmark.cli.Settings.from_env",
        classmethod(
            lambda cls: cls(repo_root=extracted, home=tmp_path / "home")  # type: ignore[misc]
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["release", "verify", "--archive", str(archive)])
    assert result.exit_code == 0, result.output
    assert '"ok": true' in result.stdout

    (extracted / "configs" / "b.yaml").write_bytes(b"b: 1\n")
    result = runner.invoke(app, ["release", "verify"])
    assert result.exit_code == 1
    assert "unexpected file: configs/b.yaml" in result.stdout

    archive.write_bytes(b"corrupt")
    result = runner.invoke(app, ["release", "verify", "--archive", str(archive)])
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output


def test_cli_release_build_and_errors(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "benchmark.cli.Settings.from_env",
        classmethod(lambda cls: cls(repo_root=repo, home=tmp_path / "home")),  # type: ignore[misc]
    )
    runner = CliRunner()
    out = tmp_path / "dist"
    result = runner.invoke(app, ["release", "build", "--out", str(out), "--tag", "v0.1.1"])
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.zip"))) == 1

    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    result = runner.invoke(app, ["release", "build", "--out", str(out)])
    assert result.exit_code == 1

    result = runner.invoke(app, ["release", "verify"])  # git checkout has no BUILD_INFO.json
    assert result.exit_code == 1
