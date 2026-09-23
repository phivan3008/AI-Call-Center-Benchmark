"""Model runtime lifecycle on the GPU server: prepare, serve, stop, status, evict.

* One ``uv`` venv per serving stack (``configs/runtimes/*.yaml``), shared by the models that
  use it, under ``$VBENCH_HOME/runtimes/<runtime_id>/.venv``.
* Weights are downloaded from Hugging Face at the pinned revision into ``$HF_HOME``.
* The server runs as a detached process group; stdout/stderr go to
  ``$VBENCH_HOME/logs/runtime_<model_id>.log``. A GPU lock file prevents two GPU workloads.
* Disk use of ``$VBENCH_HOME`` is checked against ``VBENCH_DISK_BUDGET_GB`` (default 200,
  the usable share of the 300 GB SSD) before downloading.

All sizes reported here are measured (HF file metadata or bytes on disk), never estimated.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from benchmark.core.artifacts import atomic_write_json
from benchmark.core.config import Settings
from benchmark.core.logging import get_logger
from benchmark.core.modelspec import LocalRuntime, ModelSpec, RuntimeSpec

log = get_logger(__name__)

DEFAULT_DISK_BUDGET_GB = 200.0
GB = 1024**3


class RuntimeError_(Exception):
    """Lifecycle failure with an operator-facing message."""


@dataclass(frozen=True)
class RuntimePaths:
    settings: Settings
    model_id: str
    runtime_id: str

    @property
    def venv(self) -> Path:
        return self.settings.home / "runtimes" / self.runtime_id / ".venv"

    @property
    def python(self) -> Path:
        return self.venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def executable(self, name: str) -> Path:
        return self.venv / ("Scripts" if sys.platform == "win32" else "bin") / name

    @property
    def state_dir(self) -> Path:
        return self.settings.home / "runtimes" / "models" / self.model_id

    @property
    def prepare_report(self) -> Path:
        return self.state_dir / "prepare_report.json"

    @property
    def server_state(self) -> Path:
        return self.state_dir / "server.json"

    @property
    def log_file(self) -> Path:
        return self.settings.home / "logs" / f"runtime_{self.model_id}.log"

    @property
    def gpu_lock(self) -> Path:
        return self.settings.home / ".gpu.lock"


def disk_budget_gb(env: Mapping[str, str] | None = None) -> float:
    env = os.environ if env is None else env
    return float(env.get("VBENCH_DISK_BUDGET_GB", DEFAULT_DISK_BUDGET_GB))


def dir_size_bytes(path: Path) -> int:
    """Bytes on disk under ``path``, following symlinks once (HF snapshots link to blobs)."""
    if not path.exists():
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                st = (Path(root) / name).stat()
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    return total


def hf_repo_size_bytes(repo: str, revision: str, api: HfApi | None = None) -> int:
    info = (api or HfApi()).model_info(repo, revision=revision, files_metadata=True)
    return sum(int(s.size or 0) for s in (info.siblings or []))


def _local(spec: ModelSpec) -> LocalRuntime:
    if not isinstance(spec.runtime, LocalRuntime) or spec.source is None:
        raise RuntimeError_(f"{spec.model_id} is not a locally served model")
    return spec.runtime


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run(cmd: list[str], runner: Runner, env: Mapping[str, str] | None = None) -> str:
    log.info("exec", cmd=" ".join(cmd))
    proc = runner(cmd, capture_output=True, text=True, env=dict(env) if env else None)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError_(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout or ""


def prepare(
    settings: Settings,
    spec: ModelSpec,
    runtime: RuntimeSpec,
    *,
    runner: Runner = subprocess.run,
    api: HfApi | None = None,
    download: Callable[..., str] = snapshot_download,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create/refresh the runtime venv, download pinned weights, write prepare_report.json."""
    local = _local(spec)
    assert spec.source is not None
    paths = RuntimePaths(settings, spec.model_id, local.runtime_id)
    budget = disk_budget_gb(env)

    weights_bytes = hf_repo_size_bytes(spec.source.repo, spec.source.revision, api)
    used_bytes = dir_size_bytes(settings.home)
    if (used_bytes + weights_bytes) / GB > budget:
        raise RuntimeError_(
            f"disk budget exceeded: VBENCH_HOME uses {used_bytes / GB:.1f} GB, "
            f"{spec.source.repo} needs {weights_bytes / GB:.1f} GB, budget {budget:.0f} GB. "
            "Evict another model first (vbench model evict --model <id>)."
        )

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError_("uv not found on PATH")
    if not paths.python.exists():
        _run([uv, "venv", "--python", runtime.python, str(paths.venv)], runner)
    for package in runtime.packages:
        cmd = [uv, "pip", "install", "--python", str(paths.python), package]
        if runtime.torch_backend:
            cmd += ["--torch-backend", runtime.torch_backend]
        _run(cmd, runner)
    versions = json.loads(
        _run(
            [
                str(paths.python),
                "-c",
                "import json, importlib.metadata as m\n"
                "out = {}\n"
                "for p in ('vllm', 'vllm-omni', 'torch', 'transformers'):\n"
                "    try: out[p] = m.version(p)\n"
                "    except m.PackageNotFoundError: pass\n"
                "print(json.dumps(out))",
            ],
            runner,
        ).strip()
        or "{}"
    )
    if runtime.extra_packages:
        constraints = paths.venv.parent / "constraints.txt"
        constraints.parent.mkdir(parents=True, exist_ok=True)
        constraints.write_text(
            _run([uv, "pip", "freeze", "--python", str(paths.python)], runner), encoding="utf-8"
        )
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(paths.python),
                "--constraints",
                str(constraints),
                *runtime.extra_packages,
            ],
            runner,
        )

    for step in runtime.post_install:
        args = [arg.format(python=str(paths.python)) for arg in step.args]
        try:
            _run([uv, *args], runner)
        except RuntimeError_:
            if not step.ignore_failure:
                raise
            log.warning("post_install_step_failed", args=args, why=step.why)
    check_imports(paths.python, runtime.verify_imports, runner)
    freeze = _run([uv, "pip", "freeze", "--python", str(paths.python)], runner)

    snapshot = download(repo_id=spec.source.repo, revision=spec.source.revision)
    report = {
        "model_id": spec.model_id,
        "repo": spec.source.repo,
        "revision": spec.source.revision,
        "snapshot_path": str(snapshot),
        "runtime_id": local.runtime_id,
        "packages_requested": runtime.packages,
        "packages_installed": versions,
        "freeze_sha256": _sha256_text(freeze),
        "weights_gb_hf_metadata": round(weights_bytes / GB, 2),
        "weights_gb_on_disk": round(dir_size_bytes(Path(snapshot)) / GB, 2),
        "venv_gb_on_disk": round(dir_size_bytes(paths.venv) / GB, 2),
        "home_gb_after": round(dir_size_bytes(settings.home) / GB, 2),
        "disk_budget_gb": budget,
        "prepared_at": datetime.now(UTC).isoformat(),
    }
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    (paths.state_dir / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    atomic_write_json(paths.prepare_report, report)
    return report


IMPORT_HINTS = {
    "cv2": (
        "Two known causes:\n"
        "  1) the container lacks the system libraries libGL.so.1 / libglib2.0 -> install "
        "them once as root: apt-get update && apt-get install -y libgl1 libglib2.0-0 "
        "(older images call the package libgl1-mesa-glx);\n"
        "  2) opencv-python and opencv-python-headless were both installed and removing one "
        "deleted the shared cv2 files -> re-run `vbench model prepare`, which reinstalls the "
        "headless build."
    ),
}


def check_imports(python: Path, modules: list[str], runner: Runner) -> None:
    """Import each module inside the runtime venv.

    vLLM-Omni imports these in worker subprocesses; without this check a missing system
    library only shows up as an orchestrator start-up timeout minutes later.
    """
    for module in modules:
        try:
            _run([str(python), "-c", f"import {module}"], runner)
        except RuntimeError_ as exc:
            hint = IMPORT_HINTS.get(module, "")
            raise RuntimeError_(
                f"the runtime venv cannot import '{module}'. {hint}\n{exc}"
            ) from exc


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prepare_report(settings: Settings, spec: ModelSpec) -> dict[str, Any] | None:
    local = _local(spec)
    path = RuntimePaths(settings, spec.model_id, local.runtime_id).prepare_report
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def health_ok(port: int, path: str = "/health", timeout_s: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout_s) as resp:
            return bool(200 <= resp.status < 300)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def build_serve_command(
    settings: Settings, spec: ModelSpec, runtime: RuntimeSpec, snapshot: str
) -> list[str]:
    local = _local(spec)
    assert spec.source is not None
    paths = RuntimePaths(settings, spec.model_id, local.runtime_id)
    exe, *rest = runtime.serve_command
    cmd = [str(paths.executable(exe)), *rest, snapshot]
    cmd += ["--host", "127.0.0.1", "--port", str(local.port)]
    cmd += ["--served-model-name", spec.realtime.api_model or spec.source.repo]
    cmd += local.serve_args
    if local.deploy_config:
        cmd += ["--deploy-config", str(settings.repo_root / local.deploy_config)]
    return cmd


def serve(
    settings: Settings,
    spec: ModelSpec,
    runtime: RuntimeSpec,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    health: Callable[[int, str], bool] = health_ok,
    alive: Callable[[int], bool] = pid_alive,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 5.0,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Start the model server detached and wait until its health endpoint answers."""
    local = _local(spec)
    paths = RuntimePaths(settings, spec.model_id, local.runtime_id)
    report = load_prepare_report(settings, spec)
    if report is None:
        raise RuntimeError_(
            f"{spec.model_id} is not prepared; run: vbench model prepare --model {spec.model_id}"
        )
    if paths.gpu_lock.exists():
        holder = json.loads(paths.gpu_lock.read_text(encoding="utf-8"))
        if alive(int(holder.get("pid", -1))):
            raise RuntimeError_(
                f"GPU is locked by {holder.get('model_id')} (pid {holder.get('pid')}); "
                f"stop it first: vbench model stop --model {holder.get('model_id')}"
            )
        paths.gpu_lock.unlink()

    cmd = build_serve_command(settings, spec, runtime, str(report["snapshot_path"]))
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **local.env}
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    with paths.log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== vbench serve {datetime.now(UTC).isoformat()} ===\n{' '.join(cmd)}\n")
    log_fh = paths.log_file.open("ab")
    kwargs: dict[str, Any] = {"stdout": log_fh, "stderr": subprocess.STDOUT, "env": env}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    proc = popen(cmd, **kwargs)
    state = {
        "model_id": spec.model_id,
        "pid": proc.pid,
        "port": local.port,
        "cmd": cmd,
        "log": str(paths.log_file),
        "started_at": datetime.now(UTC).isoformat(),
    }
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.server_state, state)
    atomic_write_json(paths.gpu_lock, {"model_id": spec.model_id, "pid": proc.pid})

    limit_s = local.launch_timeout_s if timeout_s is None else timeout_s
    deadline = time.monotonic() + limit_s
    started = time.monotonic()
    while time.monotonic() < deadline:
        if not alive(proc.pid):
            _cleanup(paths)
            raise RuntimeError_(
                f"server exited during startup; see {paths.log_file}\n{tail_log(paths.log_file)}"
            )
        if health(local.port, local.health_path):
            state["ready_after_s"] = round(time.monotonic() - started, 1)
            atomic_write_json(paths.server_state, state)
            return state
        sleep(poll_s)
    stop(settings, spec)
    raise RuntimeError_(
        f"server not healthy after {limit_s}s; see {paths.log_file}\n{tail_log(paths.log_file)}"
    )


def tail_log(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _cleanup(paths: RuntimePaths) -> None:
    paths.server_state.unlink(missing_ok=True)
    if paths.gpu_lock.exists():
        holder = json.loads(paths.gpu_lock.read_text(encoding="utf-8"))
        if holder.get("model_id") == paths.model_id:
            paths.gpu_lock.unlink()


def _terminate(pid: int, sig: int) -> None:
    killpg = getattr(os, "killpg", None)
    try:
        if killpg is not None:
            killpg(pid, sig)
        else:  # pragma: no cover - Windows developer machines only
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop(
    settings: Settings,
    spec: ModelSpec,
    *,
    alive: Callable[[int], bool] = pid_alive,
    terminate: Callable[[int, int], None] = _terminate,
    sleep: Callable[[float], None] = time.sleep,
    grace_s: float = 60.0,
) -> bool:
    """Stop the server's process group. Returns False if nothing was running."""
    local = _local(spec)
    paths = RuntimePaths(settings, spec.model_id, local.runtime_id)
    if not paths.server_state.is_file():
        _cleanup(paths)
        return False
    pid = int(json.loads(paths.server_state.read_text(encoding="utf-8"))["pid"])
    if alive(pid):
        terminate(pid, signal.SIGTERM)
        waited = 0.0
        while alive(pid) and waited < grace_s:
            sleep(1.0)
            waited += 1.0
        if alive(pid):
            terminate(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    _cleanup(paths)
    return True


def status(
    settings: Settings,
    spec: ModelSpec,
    *,
    alive: Callable[[int], bool] = pid_alive,
    health: Callable[[int, str], bool] = health_ok,
) -> dict[str, Any]:
    local = _local(spec)
    paths = RuntimePaths(settings, spec.model_id, local.runtime_id)
    prepared = load_prepare_report(settings, spec)
    info: dict[str, Any] = {
        "model_id": spec.model_id,
        "prepared": prepared is not None,
        "running": False,
        "healthy": False,
        "log": str(paths.log_file),
    }
    if paths.server_state.is_file():
        state = json.loads(paths.server_state.read_text(encoding="utf-8"))
        info["pid"] = state["pid"]
        info["running"] = alive(int(state["pid"]))
        info["healthy"] = info["running"] and health(local.port, local.health_path)
    return info


def evict(
    settings: Settings,
    spec: ModelSpec,
    *,
    with_runtime: bool = False,
    stopper: Callable[[Settings, ModelSpec], bool] = stop,
) -> dict[str, float]:
    """Stop the server, delete the model's weights (and optionally the shared venv)."""
    local = _local(spec)
    paths = RuntimePaths(settings, spec.model_id, local.runtime_id)
    stopper(settings, spec)
    freed = 0
    report = load_prepare_report(settings, spec)
    if report is not None:
        # snapshots/<rev> lives inside models--org--name; remove the whole repo cache entry.
        repo_cache = Path(report["snapshot_path"]).parent.parent
        if repo_cache.name.startswith("models--"):
            freed += dir_size_bytes(repo_cache)
            shutil.rmtree(repo_cache, ignore_errors=True)
    if with_runtime and paths.venv.exists():
        freed += dir_size_bytes(paths.venv)
        shutil.rmtree(paths.venv, ignore_errors=True)
    paths.prepare_report.unlink(missing_ok=True)
    return {
        "freed_gb": round(freed / GB, 2),
        "home_gb_after": round(dir_size_bytes(settings.home) / GB, 2),
    }
