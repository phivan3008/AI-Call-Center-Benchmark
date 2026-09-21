from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark import __version__, envcheck
from benchmark.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VBENCH_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return tmp_path


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_mock_run_bundle_and_verify(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--model", "mock", "--profile", "mock_smoke"])
    assert result.exit_code == 0, result.output
    match = re.search(r"RUN_ID=(\w+)", result.stdout)
    assert match
    run_id = match.group(1)

    result = runner.invoke(app, ["bundle", "create", run_id, "--no-audio"])
    assert result.exit_code == 0, result.output
    archive = tmp_path / "bundles" / f"{run_id}.tar.zst"
    assert archive.is_file()

    result = runner.invoke(app, ["bundle", "verify", str(archive)])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["ok"] is True
    assert summary["is_mock"] is True
    assert summary["run_id"] == run_id


def test_real_models_not_available_in_phase_1() -> None:
    result = runner.invoke(app, ["run", "--model", "qwen3-omni", "--profile", "mock_smoke"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_unknown_profile() -> None:
    result = runner.invoke(app, ["run", "--model", "mock", "--profile", "nope"])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_bundle_unknown_run() -> None:
    result = runner.invoke(app, ["bundle", "create", "NOPE"])
    assert result.exit_code == 1


def test_bundle_verify_failure(tmp_path: Path) -> None:
    archive = tmp_path / "X.tar.zst"
    archive.write_bytes(b"")
    result = runner.invoke(app, ["bundle", "verify", str(archive)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_env_check_dev_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envcheck, "http_probe", lambda url: (True, "HTTP 401"))
    monkeypatch.setattr(
        "benchmark.cli.run_env_check",
        lambda settings, role: envcheck.run_env_check(
            settings, role=role, probe=lambda _: (True, "HTTP 401")
        ),
    )
    out = tmp_path / "env_report.json"
    result = runner.invoke(app, ["env", "check", "--role", "dev", "--output", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["role"] == "dev"
    assert any(c["name"] == "gpu" for c in report["checks"])


def test_env_check_rejects_bad_role() -> None:
    result = runner.invoke(app, ["env", "check", "--role", "cloud"])
    assert result.exit_code != 0
