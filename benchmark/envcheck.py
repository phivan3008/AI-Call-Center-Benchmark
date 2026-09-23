"""Preflight environment check (``vbench env check``, DEPLOYMENT_GUIDE §6.2).

Reports facts about the machine; it never prints secret values (only present/missing).
Role ``server`` treats GPU and API reachability as required; role ``dev`` downgrades
them to warnings so the command is also usable on the developer machine.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from benchmark.core.clock import Clock, SystemClock
from benchmark.core.config import Settings
from benchmark.core.provenance import capture_env, source_state
from benchmark.core.schemas import EnvInfo
from benchmark.runtime.manager import dir_size_bytes, disk_budget_gb

Role = Literal["server", "dev"]

NETWORK_TARGETS = {
    "huggingface": "https://huggingface.co/api/models?limit=1",
    "openai": "https://api.openai.com/v1/models",
    "pypi": "https://pypi.org/simple/uv/",
}
TOOLS = ("uv", "git", "ffmpeg", "zstd", "nvidia-smi")


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    required: bool


class EnvReport(BaseModel):
    role: Role
    generated_at: str
    git_commit: str
    git_dirty: bool
    source_kind: str
    release_name: str | None
    vbench_home: str
    environment: EnvInfo
    disk_free_gb: float
    disk_total_gb: float
    memory_total_gb: float | None
    cpu_count: int
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status is not CheckStatus.FAIL for c in self.checks)


Probe = Callable[[str], tuple[bool, str]]


def http_probe(url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Reachable means an HTTP response of any status (401 without a key is fine)."""
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "vbench-envcheck"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"unreachable: {exc}"


def memory_total_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    return None


def _check(name: str, ok: bool, detail: str, required: bool) -> CheckResult:
    status = CheckStatus.PASS if ok else CheckStatus.FAIL if required else CheckStatus.WARN
    return CheckResult(name=name, status=status, detail=detail, required=required)


def run_env_check(
    settings: Settings,
    role: Role = "server",
    env: Mapping[str, str] | None = None,
    probe: Probe = http_probe,
    which: Callable[[str], str | None] = shutil.which,
    clock: Clock | None = None,
) -> EnvReport:
    env = os.environ if env is None else env
    clock = clock or SystemClock()
    server = role == "server"
    info = capture_env()
    source = source_state(settings.repo_root)

    settings.home.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(settings.home)
    checks: list[CheckResult] = []

    checks.append(
        _check(
            "python_version",
            sys.version_info >= (3, 11),
            platform.python_version(),
            required=True,
        )
    )
    checks.append(
        _check(
            "gpu",
            bool(info.gpus),
            ", ".join(f"{g.name} ({g.memory_total_mb} MiB)" for g in info.gpus)
            or "no NVIDIA GPU visible via NVML",
            required=server,
        )
    )
    checks.append(
        _check(
            "nvidia_driver",
            info.driver_version is not None,
            f"driver={info.driver_version}, cuda_driver={info.cuda_driver_version}",
            required=server,
        )
    )
    try:
        with tempfile.NamedTemporaryFile(dir=settings.home):
            writable = True
    except OSError:
        writable = False
    checks.append(_check("home_writable", writable, str(settings.home), required=True))

    # On the server the code directory is replaced by each new release package, so run
    # outputs must live outside it (VBENCH_HOME set in .env).
    home_separate = settings.home.resolve() != settings.repo_root.resolve()
    checks.append(
        _check(
            "home_outside_repo",
            home_separate,
            f"VBENCH_HOME={settings.home}" if home_separate else "VBENCH_HOME is the code dir",
            required=server,
        )
    )

    if source.kind == "release":
        source_detail = f"release {source.release_name} (commit {source.commit[:12]})"
    else:
        source_detail = f"{source.kind} (commit {source.commit[:12]})"
    if source.dirty:
        source_detail += ", DIRTY (code differs from the recorded commit)"
    if source.problems:
        source_detail += f"; {len(source.problems)} problem(s): " + "; ".join(source.problems[:5])
    checks.append(
        _check(
            "source_integrity",
            source.kind != "unknown" and not source.dirty,
            source_detail,
            required=server,
        )
    )

    for tool in TOOLS:
        path = which(tool)
        required = server and tool in {"uv", "nvidia-smi"}
        checks.append(_check(f"tool:{tool}", path is not None, path or "not found", required))

    # Model runtimes need these system libraries: libsndfile (audio I/O) and libGL/libglib
    # (OpenCV, imported by vLLM-Omni workers). Install with:
    #   apt-get update && apt-get install -y libgl1 libglib2.0-0 libsndfile1
    for label, lib in (("sndfile", "sndfile"), ("GL", "GL"), ("glib", "glib-2.0")):
        found = ctypes.util.find_library(lib)
        checks.append(_check(f"lib:{label}", found is not None, found or "not found", False))

    for name, url in NETWORK_TARGETS.items():
        reachable, detail = probe(url)
        checks.append(_check(f"network:{name}", reachable, detail, required=server))

    # OPENAI_API_KEY is only needed for the GPT-Realtime baseline (and an OpenAI LLM judge);
    # `vbench run --model gpt-realtime` refuses to start without it.
    for var, purpose in (
        ("OPENAI_API_KEY", "needed for GPT-Realtime runs"),
        ("HF_TOKEN", "needed only for gated models"),
    ):
        present = bool(env.get(var))
        detail = "present" if present else f"missing ({purpose})"
        checks.append(_check(f"env:{var}", present, detail, False))

    budget = disk_budget_gb(env)
    used_gb = round(dir_size_bytes(settings.home) / 1024**3, 1)
    checks.append(
        _check(
            "disk_budget",
            used_gb <= budget,
            f"VBENCH_HOME uses {used_gb} GB of the {budget:.0f} GB budget",
            required=server,
        )
    )

    return EnvReport(
        role=role,
        generated_at=clock.utc_now().isoformat(),
        git_commit=source.commit,
        git_dirty=source.dirty,
        source_kind=source.kind,
        release_name=source.release_name,
        vbench_home=str(settings.home),
        environment=info,
        disk_free_gb=round(usage.free / 1024**3, 1),
        disk_total_gb=round(usage.total / 1024**3, 1),
        memory_total_gb=memory_total_gb(),
        cpu_count=os.cpu_count() or 0,
        checks=checks,
    )
