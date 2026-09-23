from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.core.config import Settings
from benchmark.core.modelspec import ModelSpec, load_model_spec, load_runtime_spec
from benchmark.runtime import manager
from benchmark.runtime.manager import RuntimeError_, RuntimePaths

MODEL = "minicpm-o-4_5"


class FakeApi:
    def __init__(self, size: int) -> None:
        self.size = size

    def model_info(self, repo: str, revision: str, files_metadata: bool) -> Any:
        return SimpleNamespace(
            siblings=[SimpleNamespace(size=self.size), SimpleNamespace(size=None)]
        )


class FakeRunner:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        if cmd[0].endswith("venv") or "venv" in cmd[1:2]:
            Path(cmd[-1], "bin").mkdir(parents=True, exist_ok=True)
        if "-c" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"vllm": "0.28.0"}), "")
        if "freeze" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "vllm==0.28.0\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def spec(settings: Settings) -> ModelSpec:
    return load_model_spec(settings, MODEL)


def _prepare(settings: Settings, spec: ModelSpec, tmp_path: Path, **kw: Any) -> dict[str, Any]:
    snapshot = tmp_path / "hf" / "hub" / "models--openbmb--MiniCPM-o-4_5" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "w.bin").write_bytes(b"x" * 1000)

    def download(**kwargs: Any) -> str:
        return str(snapshot)

    runtime = load_runtime_spec(settings, "vllm_omni_0_28")
    return manager.prepare(
        settings,
        spec,
        runtime,
        runner=kw.get("runner", FakeRunner()),
        api=FakeApi(kw.get("size", 1000)),  # type: ignore[arg-type]
        download=download,
        env=kw.get("env", {}),
    )


def test_prepare_writes_report(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda _: "/usr/bin/uv")
    runner = FakeRunner()
    report = _prepare(settings, spec, tmp_path, runner=runner)
    assert report["revision"] == spec.source.revision  # type: ignore[union-attr]
    assert report["packages_installed"] == {"vllm": "0.28.0"}
    assert report["disk_budget_gb"] == 200.0
    installs = [c for c in runner.calls if "install" in c and "--torch-backend" in c]
    assert [c[5] for c in installs] == ["vllm==0.28.0", "vllm-omni==0.28.0"]
    # post_install swaps OpenCV for the headless build; imports are checked in the venv.
    uninstalled = [c[-1] for c in runner.calls if "uninstall" in c]
    assert uninstalled == ["opencv-python", "opencv-python-headless"]
    reinstall = next(c for c in runner.calls if "--reinstall-package" in c)
    assert reinstall[-1] == "opencv-python-headless>=4.13"
    imported = [c[2].removeprefix("import ") for c in runner.calls if c[1:2] == ["-c"]]
    assert imported[-4:] == ["vllm", "vllm_omni", "cv2", "soundfile"]
    assert manager.load_prepare_report(settings, spec) == report


def test_prepare_reports_missing_system_library(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda _: "/usr/bin/uv")
    with pytest.raises(RuntimeError_, match=r"(?s)cannot import .cv2.*libgl1"):
        _prepare(settings, spec, tmp_path, runner=FakeRunner(fail_on="import cv2"))


def test_post_install_failure_can_be_ignored(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda _: "/usr/bin/uv")
    # `uv pip uninstall opencv-python` fails when it is not installed: ignore_failure=true.
    report = _prepare(settings, spec, tmp_path, runner=FakeRunner(fail_on="uninstall"))
    assert report["model_id"] == MODEL


def test_prepare_refuses_over_budget_and_failures(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda _: "/usr/bin/uv")
    with pytest.raises(RuntimeError_, match="disk budget exceeded"):
        _prepare(
            settings, spec, tmp_path / "a", size=2 * 1024**3, env={"VBENCH_DISK_BUDGET_GB": "1"}
        )
    with pytest.raises(RuntimeError_, match="command failed"):
        _prepare(settings, spec, tmp_path / "b", runner=FakeRunner(fail_on="vllm-omni"))
    monkeypatch.setattr(manager.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError_, match="uv not found"):
        _prepare(settings, spec, tmp_path / "c")


def test_remote_model_rejected(settings: Settings) -> None:
    remote = load_model_spec(settings, "gpt-realtime")
    with pytest.raises(RuntimeError_, match="not a locally served"):
        manager.status(settings, remote)


class FakeProc:
    pid = 4242


def _prepared(settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(manager.shutil, "which", lambda _: "/usr/bin/uv")
    _prepare(settings, spec, tmp_path)


def test_serve_status_stop(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_runtime_spec(settings, "vllm_omni_0_28")
    with pytest.raises(RuntimeError_, match="not prepared"):
        manager.serve(settings, spec, runtime)
    _prepared(settings, spec, tmp_path, monkeypatch)

    launched: dict[str, Any] = {}

    def popen(cmd: list[str], **kwargs: Any) -> FakeProc:
        launched["cmd"] = cmd
        launched["env"] = kwargs["env"]
        return FakeProc()

    polls = iter([False, True])
    state = manager.serve(
        settings,
        spec,
        runtime,
        popen=popen,
        health=lambda port, path: next(polls),
        alive=lambda pid: True,
        sleep=lambda s: None,
    )
    cmd = launched["cmd"]
    assert cmd[1:3] == [
        "serve",
        str(Path(manager.load_prepare_report(settings, spec)["snapshot_path"])),
    ]  # type: ignore[index]
    assert "--omni" in cmd and "--served-model-name" in cmd
    assert cmd[cmd.index("--port") + 1] == "18002"
    assert launched["env"]["HF_HUB_OFFLINE"] == "1"
    assert state["pid"] == 4242
    paths = RuntimePaths(settings, spec.model_id, "vllm_omni_0_28")
    assert json.loads(paths.gpu_lock.read_text())["model_id"] == MODEL

    info = manager.status(settings, spec, alive=lambda pid: True, health=lambda p, q: True)
    assert info["running"] and info["healthy"] and info["prepared"]

    with pytest.raises(RuntimeError_, match="GPU is locked"):
        manager.serve(settings, spec, runtime, popen=popen, alive=lambda pid: True)

    signals: list[int] = []
    alive_calls = iter([True, True, False, False])
    assert manager.stop(
        settings,
        spec,
        alive=lambda pid: next(alive_calls),
        terminate=lambda pid, sig: signals.append(sig),
        sleep=lambda s: None,
    )
    assert len(signals) == 1
    assert not paths.gpu_lock.exists()
    assert manager.stop(settings, spec) is False


def test_serve_failures(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_runtime_spec(settings, "vllm_omni_0_28")
    _prepared(settings, spec, tmp_path, monkeypatch)
    paths = RuntimePaths(settings, spec.model_id, "vllm_omni_0_28")

    with pytest.raises(RuntimeError_, match="exited during startup"):
        manager.serve(
            settings, spec, runtime, popen=lambda c, **k: FakeProc(), alive=lambda pid: False
        )
    assert not paths.gpu_lock.exists()

    # A stale lock from a dead process is cleared; the server then never becomes healthy.
    paths.gpu_lock.write_text(json.dumps({"model_id": "other", "pid": 1}))
    killed: list[int] = []
    monkeypatch.setattr(manager, "stop", lambda s, sp, **k: killed.append(1) or True)
    alive_seq = iter([False] + [True] * 10)
    with pytest.raises(RuntimeError_, match="not healthy"):
        manager.serve(
            settings,
            spec,
            runtime,
            popen=lambda c, **k: FakeProc(),
            health=lambda p, q: False,
            alive=lambda pid: next(alive_seq),
            sleep=lambda s: None,
            timeout_s=0,
        )
    assert killed == [1]


def test_evict_removes_weights(
    settings: Settings, spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepared(settings, spec, tmp_path, monkeypatch)
    report = manager.load_prepare_report(settings, spec)
    assert report is not None
    repo_cache = Path(report["snapshot_path"]).parent.parent
    paths = RuntimePaths(settings, spec.model_id, "vllm_omni_0_28")
    paths.venv.mkdir(parents=True, exist_ok=True)
    (paths.venv / "lib.so").write_bytes(b"y" * 10)
    result = manager.evict(settings, spec, with_runtime=True, stopper=lambda s, sp: False)
    assert not repo_cache.exists()
    assert not paths.venv.exists()
    assert result["freed_gb"] >= 0
    assert manager.load_prepare_report(settings, spec) is None


def test_helpers(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"x" * 10)
    assert manager.dir_size_bytes(tmp_path) == 10
    assert manager.dir_size_bytes(tmp_path / "missing") == 0
    assert manager.disk_budget_gb({"VBENCH_DISK_BUDGET_GB": "150"}) == 150.0
    assert manager.health_ok(9, timeout_s=0.2) is False
    assert manager.pid_alive(999999999) is False
    assert manager.tail_log(tmp_path / "none.log") == ""
    log = tmp_path / "x.log"
    log.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
    assert manager.tail_log(log, 2) == "98\n99"
