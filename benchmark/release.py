"""Offline release packages for servers without GitHub access.

The GPU server cannot reach GitHub; source code arrives as a file copied from the company
PC. A release package is a zip of the tracked files at one commit plus ``BUILD_INFO.json``
(commit, version, per-file SHA-256). On the server, the build info replaces git as the
source of provenance: :func:`verify_release` proves the extracted tree is exactly that
commit, so runs stay traceable and rankable without a ``.git`` directory.
"""

from __future__ import annotations

import hashlib
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
) -> Path:
    """Package the committed tree at HEAD as ``<name>.zip`` + ``<name>.zip.sha256``.

    Content comes from git objects (``git archive``), not the working tree, so the package
    is byte-identical to the commit regardless of local line-ending conversion. A dirty
    working tree is refused so nobody assumes uncommitted edits were shipped.
    """
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    if _git(repo_root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ReleaseError("working tree has uncommitted changes; commit them first")

    release_name = f"{PACKAGE_NAME}-{version}-{commit[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{release_name}.zip"
    archive.unlink(missing_ok=True)
    # autocrlf=false: Git for Windows would otherwise convert LF to CRLF inside the archive.
    _git(
        repo_root,
        "-c",
        "core.autocrlf=false",
        "archive",
        "--format=zip",
        f"--prefix={release_name}/",
        f"--output={archive.resolve()}",
        "HEAD",
    )

    prefix = f"{release_name}/"
    files: dict[str, str] = {}
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            if not entry.is_dir():
                files[entry.filename.removeprefix(prefix)] = hashlib.sha256(
                    zf.read(entry)
                ).hexdigest()
    info = BuildInfo(
        package=PACKAGE_NAME,
        version=version,
        release_name=release_name,
        commit=commit,
        tag=tag,
        built_at=built_at,
        files=dict(sorted(files.items())),
    )
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}{BUILD_INFO}", info.model_dump_json(indent=2) + "\n")
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
