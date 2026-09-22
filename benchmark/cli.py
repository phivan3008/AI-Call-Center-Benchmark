"""``vbench`` command line interface (ARCHITECTURE §13)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from benchmark import __version__
from benchmark.adapters import adapter_registry
from benchmark.adapters.mock import MOCK_MODEL_ID
from benchmark.bundle import BundleError, create_bundle, verify_bundle
from benchmark.core.artifacts import atomic_write_json
from benchmark.core.config import ConfigError, Settings, load_profile
from benchmark.core.logging import configure_logging, get_logger
from benchmark.core.schemas import ModelVersion, RunStatus
from benchmark.envcheck import CheckStatus, Role, run_env_check
from benchmark.release import ReleaseError, build_release, verify_archive_digest, verify_release
from benchmark.runner import execute_run, mock_stimuli

app = typer.Typer(help="Japanese AI call center speech-to-speech benchmark.", no_args_is_help=True)
env_app = typer.Typer(help="Environment checks.", no_args_is_help=True)
bundle_app = typer.Typer(help="Create and verify result bundles.", no_args_is_help=True)
release_app = typer.Typer(
    help="Offline release packages (GPU server has no GitHub access).", no_args_is_help=True
)
app.add_typer(env_app, name="env")
app.add_typer(bundle_app, name="bundle")
app.add_typer(release_app, name="release")

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


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Model id (Phase 1: only 'mock').")],
    profile: Annotated[str, typer.Option(help="Run profile name in configs/runs/.")],
) -> None:
    """Collect raw artifacts for one model with one run profile."""
    settings = _settings()
    if model != MOCK_MODEL_ID:
        typer.secho(
            f"Model '{model}' is not available yet: real model adapters arrive in Phase 2. "
            f"Available: {', '.join(adapter_registry.names())}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    try:
        run_profile = load_profile(settings, profile)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    adapter = adapter_registry.get(model)({})
    manifest = asyncio.run(
        execute_run(
            settings,
            run_profile,
            adapter,
            mock_stimuli(run_profile),
            ModelVersion(model_id=model, engine="mock"),
        )
    )
    typer.echo(f"RUN_ID={manifest.run_id}")
    typer.echo(
        f"status={manifest.status.value} completed={manifest.samples_completed} "
        f"failed={manifest.samples_failed} dir={settings.raw_dir / manifest.run_id}"
    )
    raise typer.Exit(0 if manifest.status is RunStatus.COMPLETED else 1)


@bundle_app.command("create")
def bundle_create(
    run_id: Annotated[str, typer.Argument(help="Run id under artifacts/raw/.")],
    no_audio: Annotated[bool, typer.Option("--no-audio", help="Exclude audio files.")] = False,
    out: Annotated[Path | None, typer.Option(help="Output dir (default: bundles/).")] = None,
) -> None:
    """Archive a run as <run_id>.tar.zst + <run_id>.SHA256SUMS."""
    settings = _settings()
    run_dir = settings.raw_dir / run_id
    try:
        archive = create_bundle(run_dir, out or settings.bundles_dir, include_audio=not no_audio)
    except BundleError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
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
