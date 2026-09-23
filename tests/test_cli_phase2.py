from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchmark import cli
from benchmark.adapters.base import ModelCapabilities
from benchmark.capabilities import build_capability_report
from benchmark.cli import app
from benchmark.runner import SampleResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VBENCH_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def test_model_list() -> None:
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "qwen3-omni-30b-a3b" in result.stdout
    assert "gpt-realtime" in result.stdout


def test_remote_model_cannot_be_prepared() -> None:
    result = runner.invoke(app, ["model", "prepare", "--model", "gpt-realtime"])
    assert result.exit_code == 2
    assert "remote API" in result.output


def test_status_stop_evict_on_unprepared_model() -> None:
    result = runner.invoke(app, ["model", "status", "--model", "minicpm-o-4_5"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["prepared"] is False
    assert runner.invoke(app, ["model", "stop", "--model", "minicpm-o-4_5"]).stdout.strip() == (
        "not running"
    )
    evicted = runner.invoke(app, ["model", "evict", "--model", "minicpm-o-4_5"])
    assert evicted.exit_code == 0
    assert json.loads(evicted.stdout)["freed_gb"] == 0


def test_prepare_and_serve_errors_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise cli.manager.RuntimeError_("disk budget exceeded: test")

    monkeypatch.setattr(cli.manager, "prepare", boom)
    monkeypatch.setattr(cli.manager, "serve", boom)
    for command in ("prepare", "serve"):
        result = runner.invoke(app, ["model", command, "--model", "qwen3-omni-30b-a3b"])
        assert result.exit_code == 1
        assert "disk budget exceeded" in result.output


def test_prepare_and_serve_success_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.manager, "prepare", lambda *a, **k: {"model_id": "x"})
    monkeypatch.setattr(cli.manager, "serve", lambda *a, **k: {"pid": 1})
    assert json.loads(
        runner.invoke(app, ["model", "prepare", "--model", "minicpm-o-4_5"]).stdout
    ) == {"model_id": "x"}
    assert json.loads(
        runner.invoke(app, ["model", "serve", "--model", "minicpm-o-4_5"]).stdout
    ) == {"pid": 1}


def test_run_real_model_preconditions(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["run", "--model", "minicpm-o-4_5", "--profile", "smoke"])
    assert result.exit_code == 2
    assert "not healthy" in result.output

    result = runner.invoke(app, ["run", "--model", "gpt-realtime", "--profile", "smoke"])
    assert result.exit_code == 2
    assert "api_model" in result.output

    monkeypatch.setattr(cli.manager, "health_ok", lambda *a: True)
    result = runner.invoke(app, ["run", "--model", "minicpm-o-4_5", "--profile", "smoke"])
    assert result.exit_code == 2
    assert "not prepared" in result.output


def test_run_real_model_end_to_end_with_fake_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full CLI path with a fake realtime server and a tiny prepared dataset."""
    import asyncio
    import threading

    from benchmark.runner import Stimulus
    from benchmark.testing.fake_realtime import FakeRealtimeServer

    loop = asyncio.new_event_loop()
    server = FakeRealtimeServer()
    started = threading.Event()

    def serve() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.__aenter__())
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    started.wait(5)
    try:
        original = cli.load_model_spec

        def patched_spec(settings: Any, model: str) -> Any:
            spec = original(settings, model)
            return spec.model_copy(
                update={"realtime": spec.realtime.model_copy(update={"url": server.url})}
            )

        pcm = b"\x00\x10" * 16000
        stimuli = [Stimulus("s0", "L1", pcm, 16000, 1.0, task="turn")]
        monkeypatch.setattr(cli, "load_model_spec", patched_spec)
        monkeypatch.setattr(cli.manager, "health_ok", lambda *a: True)
        monkeypatch.setattr(cli, "build_stimuli", lambda s, p: (stimuli, []))
        result = runner.invoke(app, ["run", "--model", "minicpm-o-4_5", "--profile", "smoke"])
        assert result.exit_code == 0, result.output
        run_id = re.search(r"RUN_ID=(\w+)", result.stdout).group(1)  # type: ignore[union-attr]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)

    manifest = json.loads((tmp_path / "artifacts" / "raw" / run_id / "manifest.json").read_text())
    assert manifest["model"]["engine"] == "vllm-omni"
    assert manifest["is_mock"] is False

    # Bundle attaches runtime logs and prepare reports.
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "runtime_minicpm-o-4_5.log").write_text("server log", encoding="utf-8")
    report_dir = tmp_path / "runtimes" / "models" / "minicpm-o-4_5"
    report_dir.mkdir(parents=True)
    (report_dir / "prepare_report.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["bundle", "create", run_id])
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "artifacts" / "raw" / run_id
    assert (run_dir / "runtime" / "runtime_minicpm-o-4_5.log").is_file()
    assert (run_dir / "runtime" / "minicpm-o-4_5_prepare_report.json").is_file()


def test_data_prepare_unknown_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["data", "prepare", "--dataset", "nope"])
    assert result.exit_code == 1
    assert "Unknown dataset" in result.output
    monkeypatch.setattr(cli, "prepare_dataset", lambda s, n: Path("m.jsonl"))
    result = runner.invoke(app, ["data", "prepare", "--dataset", "fleurs_ja_smoke"])
    assert result.exit_code == 0
    assert "m.jsonl" in result.stdout


def _result(**kw: Any) -> SampleResult:
    base: dict[str, Any] = {
        "sample_id": "s",
        "layer": "L1",
        "repeat": 0,
        "status": "completed",
        "n_events": 3,
        "output_audio_ms": 0.0,
        "t_user_speech_end_ns": 0,
        "t_first_event_ns": 1,
    }
    base.update(kw)
    return SampleResult.model_validate(base)


def test_capability_report_negative_branches() -> None:
    results = [
        _result(sample_id="t", task="turn", n_audio_deltas=1, n_text_deltas=0),
        _result(sample_id="tool", task="tool", tool_calls=[]),
        _result(sample_id="c", task="cancel", cancel_sent=True, terminal_event="response_done"),
    ]
    obs = build_capability_report("m", results, ModelCapabilities())["observations"]
    assert obs["streaming_audio_output"]["observation"] == "not_observed"
    assert obs["text_output_channel"]["observation"] == "not_observed"
    assert obs["native_tool_calling"]["observation"] == "not_observed"
    assert obs["response_cancel"]["observation"] == "not_observed"

    early = [
        _result(sample_id="c", task="cancel", cancel_sent=False, terminal_event="response_done")
    ]
    obs = build_capability_report("m", early, ModelCapabilities())["observations"]
    assert obs["response_cancel"]["observation"] == "inconclusive"
    assert obs["native_tool_calling"]["observation"] == "not_tested"

    bad_json = [_result(task="tool", tool_calls=[{"call_id": "1", "name": "f", "arguments": "{"}])]
    obs = build_capability_report("m", bad_json, ModelCapabilities())["observations"]
    assert obs["native_tool_calling"]["observation"] == "not_observed"


def test_model_check_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["model", "check", "--model", "minicpm-o-4_5"])
    assert result.exit_code == 1
    assert "no runtime venv" in result.output

    venv_bin = tmp_path / "runtimes" / "vllm_omni_0_28" / ".venv"
    (venv_bin / "Scripts").mkdir(parents=True)
    (venv_bin / "bin").mkdir(parents=True)
    for exe in (venv_bin / "Scripts" / "python.exe", venv_bin / "bin" / "python"):
        exe.write_text("", encoding="utf-8")

    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli.manager, "check_imports", lambda python, modules, runner_: calls.append(modules)
    )
    result = runner.invoke(app, ["model", "check", "--model", "minicpm-o-4_5"])
    assert result.exit_code == 0, result.output
    assert calls == [["vllm", "vllm_omni", "cv2", "soundfile"]]
    assert "ok: vllm, vllm_omni, cv2, soundfile" in result.stdout

    def boom(*args: Any, **kwargs: Any) -> None:
        raise cli.manager.RuntimeError_("cannot import 'cv2'. install libgl1")

    monkeypatch.setattr(cli.manager, "check_imports", boom)
    result = runner.invoke(app, ["model", "check", "--model", "minicpm-o-4_5"])
    assert result.exit_code == 1
    assert "libgl1" in result.output
