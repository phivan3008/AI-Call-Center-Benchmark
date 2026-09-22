"""Provenance capture: run IDs, git state, config hashes, environment info."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from benchmark.core.clock import Clock, SystemClock
from benchmark.core.schemas import EnvInfo, GpuInfo

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

TRACKED_PACKAGES = (
    "ai-callcenter-benchmark",
    "pydantic",
    "typer",
    "structlog",
    "zstandard",
    "nvidia-ml-py",
    "numpy",
    "torch",
    "vllm",
    "transformers",
)


def new_run_id(clock: Clock | None = None) -> str:
    """Return a ULID: 48-bit millisecond timestamp + 80 random bits, Crockford base32.

    ULIDs sort lexicographically by creation time, which keeps ``artifacts/raw`` ordered.
    """
    ms = int((clock or SystemClock()).utc_now().timestamp() * 1000)
    value = (ms << 80) | secrets.randbits(80)
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def canonical_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_sha256(obj: Any) -> str:
    """Hash of a fully resolved config, independent of key order."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def git_state(repo_root: Path) -> tuple[str, bool]:
    """Return ``(commit, dirty)``. Unknown state is reported as dirty (non-reproducible)."""
    git = shutil.which("git")
    if git is None:
        return "unknown", True
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown", True
    return commit, bool(status)


class SourceState(BaseModel):
    """Where the running code came from: a git checkout or an offline release package."""

    kind: Literal["git", "release", "unknown"]
    commit: str
    dirty: bool
    release_name: str | None = None
    problems: list[str] = []


def source_state(repo_root: Path) -> SourceState:
    """Provenance of the code tree.

    A git checkout is used when present. Otherwise ``BUILD_INFO.json`` from an offline
    release package (GPU server without GitHub access) is verified file by file; any
    missing, modified or unexpected file marks the tree dirty.
    """
    from benchmark.release import ReleaseError, load_build_info, verify_release

    if (repo_root / ".git").exists():
        commit, dirty = git_state(repo_root)
        if commit != "unknown":
            return SourceState(kind="git", commit=commit, dirty=dirty)
    try:
        info = load_build_info(repo_root)
    except ValueError as exc:
        return SourceState(kind="unknown", commit="unknown", dirty=True, problems=[str(exc)])
    if info is None:
        return SourceState(
            kind="unknown",
            commit="unknown",
            dirty=True,
            problems=["no .git directory and no BUILD_INFO.json"],
        )
    try:
        state = verify_release(repo_root, info)
    except ReleaseError as exc:  # pragma: no cover - info is already loaded
        return SourceState(kind="unknown", commit="unknown", dirty=True, problems=[str(exc)])
    return SourceState(
        kind="release",
        commit=info.commit,
        dirty=info.dirty or not state.ok,
        release_name=info.release_name,
        problems=state.problems,
    )


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def gpu_info() -> tuple[list[GpuInfo], str | None, str | None]:
    """Query NVML. Returns ``([], None, None)`` when no NVIDIA GPU/driver is available."""
    try:
        import pynvml
    except ImportError:
        return [], None, None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return [], None, None
    try:
        driver = _to_str(pynvml.nvmlSystemGetDriverVersion())
        cuda_raw = int(pynvml.nvmlSystemGetCudaDriverVersion())
        cuda = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=_to_str(pynvml.nvmlDeviceGetName(handle)),
                    memory_total_mb=int(memory.total) // (1024 * 1024),
                    uuid=_to_str(pynvml.nvmlDeviceGetUUID(handle)),
                )
            )
        return gpus, driver, cuda
    except pynvml.NVMLError:
        return [], None, None
    finally:
        pynvml.nvmlShutdown()


def _to_str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def capture_env() -> EnvInfo:
    gpus, driver, cuda = gpu_info()
    return EnvInfo(
        hostname=socket.gethostname(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        python_version=platform.python_version(),
        packages=package_versions(),
        gpus=gpus,
        driver_version=driver,
        cuda_driver_version=cuda,
    )


def cpu_count() -> int:
    return os.cpu_count() or 0
