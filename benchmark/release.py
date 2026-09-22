"""Offline release packages for servers without GitHub access.

The GPU server cannot reach GitHub; source code arrives as a file copied from the company
PC. A release package is a zip of the tracked files at one commit plus ``BUILD_INFO.json``
(commit, version, per-file SHA-256). On the server, the build info replaces git as the
source of provenance: :func:`verify_release` proves the extracted tree is exactly that
commit, so runs stay traceable and rankable without a ``.git`` directory.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from benchmark.core.artifacts import atomic_write_text, sha256_file

BUILD_INFO = "BUILD_INFO.json"
PACKAGE_NAME = "ai-callcenter-benchmark"

# Extra files under these top-level directories change behaviour, so they count as
# modifications. Anything else (venvs, caches, run outputs) is ignored.
GUARDED_DIRS = frozenset({"benchmark", "configs", "datasets", "scripts", "tests"})
IGNORED_PARTS = frozenset(
    {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "audio"}
)


class ReleaseError(Exception):
    pass


class BuildInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package: str
    version: str
    release_name: str
    commit: str
    tag: str | None = None
    dirty: bool = False
    built_at: datetime
    files: dict[str, str]


class ReleaseState(BaseModel):
    release_name: str
    commit: str
    ok: bool
    problems: list[str]


def _git(repo_root: Path, *args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ReleaseError("git is required to build a release")
    try:
        return subprocess.run(
            [git, *args], cwd=repo_root, capture_output=True, text=True, check=True, timeout=30
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        raise ReleaseError(f"git {' '.join(args)} failed: {exc}") from exc


def build_release(
    repo_root: Path,
    out_dir: Path,
    version: str,
    built_at: datetime,
    tag: str | None = None,
    allow_dirty: bool = False,
) -> Path:
    """Zip the tracked files at HEAD with ``BUILD_INFO.json``; write a ``.sha256`` sidecar.

    Refuses a dirty working tree (unless ``allow_dirty``) so that the package content always
    equals the recorded commit.
    """
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=no").strip())
    if dirty and not allow_dirty:
        raise ReleaseError("working tree has uncommitted changes; commit them first")
    paths = [p for p in _git(repo_root, "ls-files", "-z").split("\0") if p]

    release_name = f"{PACKAGE_NAME}-{version}-{commit[:8]}"
    files = {p: sha256_file(repo_root / p) for p in sorted(paths) if (repo_root / p).is_file()}
    info = BuildInfo(
        package=PACKAGE_NAME,
        version=version,
        release_name=release_name,
        commit=commit,
        tag=tag,
        dirty=dirty,
        built_at=built_at,
        files=files,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{release_name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(repo_root / rel, arcname=f"{release_name}/{rel}")
        zf.writestr(f"{release_name}/{BUILD_INFO}", info.model_dump_json(indent=2) + "\n")
    digest = sha256_file(archive)
    atomic_write_text(out_dir / f"{archive.name}.sha256", f"{digest}  {archive.name}\n")
    return archive


def load_build_info(repo_root: Path) -> BuildInfo | None:
    path = repo_root / BUILD_INFO
    if not path.is_file():
        return None
    return BuildInfo.model_validate_json(path.read_text(encoding="utf-8"))


def _is_ignored(rel: PurePosixPath) -> bool:
    return any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in rel.parts)


def verify_release(repo_root: Path, info: BuildInfo | None = None) -> ReleaseState:
    """Compare the extracted tree with ``BUILD_INFO.json``."""
    info = info or load_build_info(repo_root)
    if info is None:
        raise ReleaseError(f"{BUILD_INFO} not found in {repo_root}")
    problems: list[str] = []
    for rel, expected in info.files.items():
        path = repo_root / rel
        if not path.is_file():
            problems.append(f"missing file: {rel}")
        elif sha256_file(path) != expected:
            problems.append(f"modified file: {rel}")
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = PurePosixPath(path.relative_to(repo_root).as_posix())
        guarded = rel_path.parts[0] in GUARDED_DIRS and not _is_ignored(rel_path)
        if guarded and str(rel_path) not in info.files:
            problems.append(f"unexpected file: {rel_path}")
    return ReleaseState(
        release_name=info.release_name, commit=info.commit, ok=not problems, problems=problems
    )


def verify_archive_digest(archive: Path) -> bool:
    """Check ``<archive>.sha256`` written by :func:`build_release`."""
    sidecar = archive.with_name(archive.name + ".sha256")
    if not sidecar.is_file():
        raise ReleaseError(f"missing checksum file: {sidecar.name}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    return sha256_file(archive) == expected
