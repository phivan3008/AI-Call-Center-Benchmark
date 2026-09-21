from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.core import provenance
from benchmark.core.clock import FakeClock
from benchmark.core.provenance import (
    capture_env,
    config_sha256,
    cpu_count,
    git_state,
    new_run_id,
    package_versions,
)


def test_run_id_is_ulid_and_time_sortable() -> None:
    early = new_run_id(FakeClock(start_utc=datetime(2026, 1, 1, tzinfo=UTC)))
    late = new_run_id(FakeClock(start_utc=datetime(2026, 6, 1, tzinfo=UTC)))
    assert len(early) == 26
    assert set(early) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
    assert early < late


def test_config_hash_ignores_key_order() -> None:
    assert config_sha256({"a": 1, "b": [1, 2]}) == config_sha256({"b": [1, 2], "a": 1})
    assert config_sha256({"a": 1}) != config_sha256({"a": 2})


def test_git_state_outside_repo_is_unknown_and_dirty(tmp_path: Path) -> None:
    assert git_state(tmp_path) == ("unknown", True)


def test_git_state_without_git_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda _: None)
    assert git_state(tmp_path) == ("unknown", True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_state_clean_then_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("a", encoding="utf-8")
    git("add", "f.txt")
    git("commit", "-q", "-m", "init")
    commit, dirty = git_state(tmp_path)
    assert len(commit) == 40
    assert dirty is False
    (tmp_path / "f.txt").write_text("b", encoding="utf-8")
    assert git_state(tmp_path)[1] is True


def test_environment_capture() -> None:
    info = capture_env()
    assert info.python_version
    assert "pydantic" in package_versions()
    assert package_versions(("definitely-not-installed-pkg",)) == {}
    assert cpu_count() >= 1


class _FakeNvml:
    """Minimal stand-in for pynvml to exercise GPU capture without a GPU."""

    class NVMLError(Exception):
        pass

    class _Mem:
        total = 80 * 1024**3

    def __init__(self, fail_init: bool = False) -> None:
        self.fail_init = fail_init

    def nvmlInit(self) -> None:
        if self.fail_init:
            raise self.NVMLError("no driver")

    def nvmlShutdown(self) -> None:
        pass

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"999.99"

    def nvmlSystemGetCudaDriverVersion(self) -> int:
        return 12040

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> _Mem:
        return self._Mem()

    def nvmlDeviceGetName(self, handle: int) -> str:
        return "Fake H100"

    def nvmlDeviceGetUUID(self, handle: int) -> str:
        return "GPU-fake"


def test_gpu_info_with_fake_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml())
    gpus, driver, cuda = provenance.gpu_info()
    assert driver == "999.99"
    assert cuda == "12.4"
    assert gpus[0].name == "Fake H100"
    assert gpus[0].memory_total_mb == 80 * 1024

    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml(fail_init=True))
    assert provenance.gpu_info() == ([], None, None)
