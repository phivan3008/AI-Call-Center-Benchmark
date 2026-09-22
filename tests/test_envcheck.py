from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any

import pytest

from benchmark import envcheck
from benchmark.core.config import Settings
from benchmark.core.provenance import SourceState
from benchmark.envcheck import CheckStatus, http_probe, run_env_check


def _status(report: envcheck.EnvReport, name: str) -> CheckStatus:
    return next(c.status for c in report.checks if c.name == name)


def _no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envcheck, "capture_env", _fake_env(gpus=False))


def _fake_env(gpus: bool) -> Any:
    from benchmark.core.schemas import EnvInfo, GpuInfo

    def capture() -> EnvInfo:
        return EnvInfo(
            hostname="h",
            platform="Linux",
            python_version="3.11.9",
            gpus=[GpuInfo(index=0, name="H100", memory_total_mb=81559)] if gpus else [],
            driver_version="1.0" if gpus else None,
            cuda_driver_version="12.4" if gpus else None,
        )

    return capture


def test_dev_role_without_gpu_or_network_passes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_gpu(monkeypatch)
    report = run_env_check(
        settings, role="dev", env={}, probe=lambda _: (False, "offline"), which=lambda _: None
    )
    assert report.ok
    assert _status(report, "gpu") is CheckStatus.WARN
    assert _status(report, "network:openai") is CheckStatus.WARN
    assert _status(report, "env:OPENAI_API_KEY") is CheckStatus.WARN
    assert _status(report, "home_writable") is CheckStatus.PASS


def test_server_role_is_strict(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch)
    report = run_env_check(
        settings, role="server", env={}, probe=lambda _: (False, "offline"), which=lambda _: None
    )
    assert not report.ok
    for name in ("gpu", "nvidia_driver", "network:huggingface", "tool:uv", "env:OPENAI_API_KEY"):
        assert _status(report, name) is CheckStatus.FAIL
    assert _status(report, "tool:ffmpeg") is CheckStatus.WARN
    assert _status(report, "env:HF_TOKEN") is CheckStatus.WARN


def test_server_role_all_good_and_no_secret_values(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envcheck, "capture_env", _fake_env(gpus=True))
    monkeypatch.setattr(
        envcheck,
        "source_state",
        lambda _: SourceState(kind="release", commit="a" * 40, dirty=False, release_name="r1"),
    )
    monkeypatch.setattr(envcheck.ctypes.util, "find_library", lambda _: "libsndfile.so")
    secret = "sk-secret-value-that-must-not-leak"
    report = run_env_check(
        settings,
        role="server",
        env={"OPENAI_API_KEY": secret, "HF_TOKEN": "hf_x"},
        probe=lambda _: (True, "HTTP 401"),
        which=lambda tool: f"/usr/bin/{tool}",
    )
    assert report.ok, [c for c in report.checks if c.status is not CheckStatus.PASS]
    assert secret not in report.model_dump_json()
    assert report.disk_total_gb > 0


class _Resp:
    status = 200

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        pass


def test_http_probe_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert http_probe("https://x") == (True, "HTTP 200")

    def raise_http(*a: object, **k: object) -> None:
        raise urllib.error.HTTPError("https://x", 401, "unauthorized", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", raise_http)
    assert http_probe("https://x") == (True, "HTTP 401")

    def raise_url(*a: object, **k: object) -> None:
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    ok, detail = http_probe("https://x")
    assert not ok
    assert "unreachable" in detail


def test_server_requires_home_outside_code_dir_and_clean_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.conftest import REPO_ROOT

    _no_gpu(monkeypatch)
    monkeypatch.setattr(
        envcheck,
        "source_state",
        lambda _: SourceState(
            kind="release", commit="b" * 40, dirty=True, release_name="r", problems=["x"]
        ),
    )
    settings = Settings(repo_root=REPO_ROOT, home=REPO_ROOT)
    report = run_env_check(
        settings, role="server", env={}, probe=lambda _: (True, "ok"), which=lambda _: "/bin/x"
    )
    assert _status(report, "home_outside_repo") is CheckStatus.FAIL
    source = next(c for c in report.checks if c.name == "source_integrity")
    assert source.status is CheckStatus.FAIL
    assert "DIRTY" in source.detail
    assert report.source_kind == "release"
