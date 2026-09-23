"""``vbench`` command line interface (ARCHITECTURE §13)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from benchmark import __version__
from benchmark.adapters import adapter_registry
from benchmark.adapters.base import ModelAdapter
from benchmark.adapters.mock import MOCK_MODEL_ID
from benchmark.adapters.realtime import RealtimeAdapter
from benchmark.bundle import BundleError, create_bundle, verify_bundle
from benchmark.core.artifacts import atomic_write_json
from benchmark.core.config import ConfigError, Settings, load_profile
from benchmark.core.logging import configure_logging, get_logger
from benchmark.core.modelspec import (
    LocalRuntime,
    ModelSpec,
    RuntimeSpec,
    list_model_ids,
    load_model_spec,
    load_runtime_spec,
)
from benchmark.core.schemas import DatasetRef, ModelVersion, RunStatus
from benchmark.envcheck import CheckStatus, Role, run_env_check
from benchmark.release import ReleaseError, build_release, verify_archive_digest, verify_release
from benchmark.runner import execute_run, mock_stimuli
from benchmark.runtime import manager
from benchmark.stimuli import build_stimuli, prepare_dataset

app = typer.Typer(help="Japanese AI call center speech-to-speech benchmark.", no_args_is_help=True)
env_app = typer.Typer(help="Environment checks.", no_args_is_help=True)
bundle_app = typer.Typer(help="Create and verify result bundles.", no_args_is_help=True)
release_app = typer.Typer(
    help="Offline release packages (GPU server has no GitHub access).", no_args_is_help=True
)
app.add_typer(env_app, name="env")
app.add_typer(bundle_app, name="bundle")
app.add_typer(release_app, name="release")
data_app = typer.Typer(help="Benchmark datasets.", no_args_is_help=True)
model_app = typer.Typer(help="Model runtimes on the GPU server.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")

log = get_logger("vbench")


def _settings() -> Settings:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    configure_logging(settings.log_level)
    return settings


@app.command()
def version() -> None:
    """Show the framework version."""
    typer.echo(__version__)


@env_app.command("check")
def env_check(
    output: Annotated[Path | None, typer.Option(help="Write the JSON report here.")] = None,
    role: Annotated[str, typer.Option(help="server (strict) or dev (GPU optional).")] = "server",
) -> None:
    """Preflight check: GPU, driver, disk, tools, network, API keys (presence only)."""
    if role not in ("server", "dev"):
        raise typer.BadParameter("role must be 'server' or 'dev'")
    settings = _settings()
    role_value: Role = "server" if role == "server" else "dev"
    report = run_env_check(settings, role=role_value)
    for check in report.checks:
        colour = {
            CheckStatus.PASS: typer.colors.GREEN,
            CheckStatus.WARN: typer.colors.YELLOW,
            CheckStatus.FAIL: typer.colors.RED,
        }[check.status]
        typer.secho(f"[{check.status.value:4}] {check.name:24} {check.detail}", fg=colour)
    typer.echo(f"disk free: {report.disk_free_gb} GB / {report.disk_total_gb} GB")
    if output is not None:
        atomic_write_json(output, report.model_dump(mode="json"))
        typer.echo(f"report written to {output}")
    raise typer.Exit(0 if report.ok else 1)


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code)


def _spec(settings: Settings, model: str) -> ModelSpec:
    try:
        return load_model_spec(settings, model)
    except ConfigError as exc:
        known = ", ".join([MOCK_MODEL_ID, *list_model_ids(settings)])
        raise _fail(f"{exc}\nKnown models: {known}") from exc


def _local_spec(settings: Settings, model: str) -> tuple[ModelSpec, RuntimeSpec]:
    spec = _spec(settings, model)
    if not isinstance(spec.runtime, LocalRuntime):
        raise _fail(f"{model} is a remote API model; nothing to prepare or serve")
    try:
        return spec, load_runtime_spec(settings, spec.runtime.runtime_id)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def _model_version(settings: Settings, spec: ModelSpec) -> ModelVersion:
    if not isinstance(spec.runtime, LocalRuntime):
        return ModelVersion(
            model_id=spec.model_id, repo=spec.realtime.api_model, engine=spec.realtime.dialect
        )
    report = manager.load_prepare_report(settings, spec) or {}
    installed = report.get("packages_installed", {})
    return ModelVersion(
        model_id=spec.model_id,
        repo=spec.source.repo if spec.source else None,
        revision=spec.source.revision if spec.source else None,
        engine="vllm-omni",
        engine_version=",".join(f"{k}=={v}" for k, v in sorted(installed.items())) or None,
        runtime_lock_sha256=report.get("freeze_sha256"),
    )


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Model id (configs/models/<id>.yaml) or 'mock'.")],
    profile: Annotated[str, typer.Option(help="Run profile name in configs/runs/.")],
) -> None:
    """Collect raw artifacts for one model with one run profile."""
    settings = _settings()
    try:
        run_profile = load_profile(settings, profile)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    datasets: list[DatasetRef] = []
    if model == MOCK_MODEL_ID:
        adapter: ModelAdapter = adapter_registry.get(model)({})
        stimuli = mock_stimuli(run_profile)
        version = ModelVersion(model_id=model, engine="mock")
    else:
        if run_profile.mock_only:
            raise _fail(f"profile '{profile}' is mock-only; use --model mock")
        spec = _spec(settings, model)
        if isinstance(spec.runtime, LocalRuntime) and not manager.health_ok(
            spec.runtime.port, spec.runtime.health_path
        ):
            raise _fail(f"{model} server is not healthy; run: vbench model serve --model {model}")
        if spec.realtime.dialect == "openai" and not spec.realtime.api_model:
            raise _fail(f"{model}: set realtime.api_model in configs/models/{model}.yaml first")
        if spec.realtime.auth_env and not os.environ.get(spec.realtime.auth_env):
            raise _fail(f"{model}: {spec.realtime.auth_env} is not set (load .env first)")
        try:
            stimuli, datasets = build_stimuli(settings, run_profile)
        except ConfigError as exc:
            raise _fail(str(exc)) from exc
        adapter = RealtimeAdapter(spec)
        version = _model_version(settings, spec)

    manifest = asyncio.run(
        execute_run(settings, run_profile, adapter, stimuli, version, datasets=datasets)
    )
    typer.echo(f"RUN_ID={manifest.run_id}")
    typer.echo(
        f"status={manifest.status.value} completed={manifest.samples_completed} "
        f"failed={manifest.samples_failed} dir={settings.raw_dir / manifest.run_id}"
    )
    raise typer.Exit(0 if manifest.status is RunStatus.COMPLETED else 1)


@data_app.command("prepare")
def data_prepare(
    dataset: Annotated[str, typer.Option(help="Dataset name, e.g. fleurs_ja_smoke.")],
) -> None:
    """Download (pinned revision) and build a dataset under VBENCH_HOME/datasets_audio/."""
    settings = _settings()
    try:
        manifest = prepare_dataset(settings, dataset)
    except (ConfigError, ValueError) as exc:
        raise _fail(str(exc), 1) from exc
    typer.echo(f"manifest: {manifest}")


@model_app.command("list")
def model_list() -> None:
    """List configured models."""
    settings = _settings()
    for model_id in list_model_ids(settings):
        spec = load_model_spec(settings, model_id)
        typer.echo(f"{model_id:24} {spec.runtime.kind:12} {spec.display_name}")


@model_app.command("prepare")
def model_prepare(model: Annotated[str, typer.Option(help="Model id.")]) -> None:
    """Create the runtime venv (PyPI) and download pinned weights (Hugging Face)."""
    settings = _settings()
    spec, runtime = _local_spec(settings, model)
    try:
        report = manager.prepare(settings, spec, runtime)
    except manager.RuntimeError_ as exc:
        raise _fail(str(exc), 1) from exc
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@model_app.command("serve")
def model_serve(
    model: Annotated[str, typer.Option(help="Model id.")],
    timeout: Annotated[int | None, typer.Option(help="Startup timeout in seconds.")] = None,
) -> None:
    """Start the model server in the background and wait until it is healthy."""
    settings = _settings()
    spec, runtime = _local_spec(settings, model)
    try:
        state = manager.serve(settings, spec, runtime, timeout_s=timeout)
    except manager.RuntimeError_ as exc:
        raise _fail(str(exc), 1) from exc
    typer.echo(json.dumps(state, indent=2, ensure_ascii=False))


@model_app.command("check")
def model_check(model: Annotated[str, typer.Option(help="Model id.")]) -> None:
    """Import the runtime's modules inside its venv (catches missing system libraries)."""
    settings = _settings()
    spec, runtime = _local_spec(settings, model)
    paths = manager.RuntimePaths(settings, spec.model_id, runtime.runtime_id)
    if not paths.python.exists():
        raise _fail(f"{model} has no runtime venv; run: vbench model prepare --model {model}", 1)
    try:
        manager.check_imports(paths.python, runtime.verify_imports, subprocess.run)
    except manager.RuntimeError_ as exc:
        raise _fail(str(exc), 1) from exc
    typer.echo(f"ok: {', '.join(runtime.verify_imports)}")


@model_app.command("stop")
def model_stop(model: Annotated[str, typer.Option(help="Model id.")]) -> None:
    """Stop the model server and release the GPU lock."""
    settings = _settings()
    spec, _ = _local_spec(settings, model)
    typer.echo("stopped" if manager.stop(settings, spec) else "not running")


@model_app.command("status")
def model_status(model: Annotated[str, typer.Option(help="Model id.")]) -> None:
    """Show whether the model is prepared, running and healthy."""
    settings = _settings()
    spec, _ = _local_spec(settings, model)
    typer.echo(json.dumps(manager.status(settings, spec), indent=2))


@model_app.command("evict")
def model_evict(
    model: Annotated[str, typer.Option(help="Model id.")],
    with_runtime: Annotated[
        bool, typer.Option("--with-runtime", help="Also delete the shared runtime venv.")
    ] = False,
) -> None:
    """Stop the server and delete the model's weights to free disk."""
    settings = _settings()
    spec, _ = _local_spec(settings, model)
    typer.echo(json.dumps(manager.evict(settings, spec, with_runtime=with_runtime), indent=2))


def _attach_runtime_files(settings: Settings, run_dir: Path) -> list[str]:
    """Copy runtime logs and prepare reports into the run so they travel in the bundle."""
    target = run_dir / "runtime"
    copied = []
    sources = [
        *sorted((settings.home / "logs").glob("runtime_*.log")),
        *sorted((settings.home / "runtimes" / "models").glob("*/prepare_report.json")),
    ]
    for src in sources:
        name = src.name if src.suffix == ".log" else f"{src.parent.name}_{src.name}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target / name)
        copied.append(name)
    return copied


@bundle_app.command("create")
def bundle_create(
    run_id: Annotated[str, typer.Argument(help="Run id under artifacts/raw/.")],
    no_audio: Annotated[bool, typer.Option("--no-audio", help="Exclude audio files.")] = False,
    no_runtime_logs: Annotated[
        bool, typer.Option("--no-runtime-logs", help="Do not attach model server logs.")
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Output dir (default: bundles/).")] = None,
) -> None:
    """Archive a run as <run_id>.tar.zst + <run_id>.SHA256SUMS."""
    settings = _settings()
    run_dir = settings.raw_dir / run_id
    if not no_runtime_logs and run_dir.is_dir():
        _attach_runtime_files(settings, run_dir)
    try:
        archive = create_bundle(run_dir, out or settings.bundles_dir, include_audio=not no_audio)
    except BundleError as exc:
        raise _fail(str(exc), 1) from exc
    typer.echo(f"bundle: {archive}")


@bundle_app.command("verify")
def bundle_verify(
    archive: Annotated[Path, typer.Argument(help="Path to <run_id>.tar.zst.")],
    extract_to: Annotated[
        Path | None, typer.Option(help="Extract the verified run here (e.g. artifacts/raw).")
    ] = None,
) -> None:
    """Verify archive checksum, inner SHA256SUMS and manifest schema."""
    _settings()
    result = verify_bundle(archive, extract_to=extract_to)
    summary = {
        "archive": str(result.archive),
        "ok": result.ok,
        "problems": result.problems,
        "run_id": result.manifest.run_id if result.manifest else None,
        "is_mock": result.manifest.is_mock if result.manifest else None,
        "status": result.manifest.status.value if result.manifest else None,
    }
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    raise typer.Exit(0 if result.ok else 1)


@release_app.command("build")
def release_build(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("dist"),
    tag: Annotated[str | None, typer.Option(help="Git tag this release belongs to.")] = None,
) -> None:
    """Build <name>.zip + <name>.zip.sha256 from the committed tree at HEAD."""
    settings = _settings()
    try:
        archive = build_release(
            settings.repo_root,
            out,
            version=__version__,
            built_at=datetime.now(UTC),
            tag=tag,
        )
    except ReleaseError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"release: {archive}")


@release_app.command("verify")
def release_verify(
    archive: Annotated[
        Path | None,
        typer.Option(help="Also check a release .zip against its .sha256 file."),
    ] = None,
) -> None:
    """Verify that the extracted code tree matches its BUILD_INFO.json."""
    settings = _settings()
    try:
        if archive is not None and not verify_archive_digest(archive):
            typer.secho(f"checksum mismatch: {archive.name}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        state = verify_release(settings.repo_root)
    except ReleaseError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(state.model_dump(), indent=2, ensure_ascii=False))
    raise typer.Exit(0 if state.ok else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
