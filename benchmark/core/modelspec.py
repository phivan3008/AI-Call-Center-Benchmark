"""Model registry entries (``configs/models/<model_id>.yaml``) and runtime definitions
(``configs/runtimes/<runtime_id>.yaml``), ARCHITECTURE §6.

Capabilities keep ``claimed`` (with a source URL) separate from ``verified``, which is only
set after a smoke run's capability report has been reviewed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmark.adapters.base import CapabilityStatus
from benchmark.core.config import Settings, load_model


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HubSource(_Strict):
    hub: Literal["huggingface"] = "huggingface"
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # pinned commit, never a branch


class Capability(_Strict):
    claimed: CapabilityStatus = CapabilityStatus.UNKNOWN
    verified: CapabilityStatus = CapabilityStatus.UNKNOWN
    source: str | None = None


class Capabilities(_Strict):
    native_full_duplex: Capability = Capability()
    streaming_audio_output: Capability = Capability()
    native_tool_calling: Capability = Capability()
    text_output_channel: Capability = Capability()


class RealtimeEndpoint(_Strict):
    dialect: Literal["openai", "vllm_omni"]
    url: str  # e.g. ws://127.0.0.1:18001/v1/realtime?duplex=0
    api_model: str | None = None  # value sent as `model` (OpenAI: required query param)
    input_sample_rate_hz: int = 16000
    output_sample_rate_hz: int = 24000
    voice: str | None = None
    auth_env: str | None = None  # env var holding a bearer token, e.g. OPENAI_API_KEY


class LocalRuntime(_Strict):
    kind: Literal["vllm_omni"]
    runtime_id: str  # configs/runtimes/<runtime_id>.yaml (shared venv)
    port: int
    serve_args: list[str] = Field(default_factory=list)
    deploy_config: str | None = None  # path relative to repo root
    env: dict[str, str] = Field(default_factory=dict)
    health_path: str = "/health"
    launch_timeout_s: int = Field(default=1800, gt=0)


class RemoteRuntime(_Strict):
    kind: Literal["remote_api"]


class ModelSpec(_Strict):
    model_id: str
    display_name: str
    license: str
    commercial_use: Literal["allowed", "restricted", "unknown"] = "unknown"
    source: HubSource | None = None
    runtime: LocalRuntime | RemoteRuntime = Field(discriminator="kind")
    realtime: RealtimeEndpoint
    capabilities: Capabilities = Capabilities()
    sampling: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PostInstallStep(_Strict):
    """A ``uv`` command run after the packages are installed.

    ``{python}`` in an argument is replaced by the venv interpreter path.
    """

    args: list[str] = Field(min_length=1)
    ignore_failure: bool = False
    why: str = ""


class RuntimeSpec(_Strict):
    """A shared Python environment for one serving stack (e.g. vLLM-Omni 0.28.0)."""

    runtime_id: str
    python: str
    packages: list[str] = Field(min_length=1)  # exact pins, installed in order
    torch_backend: str | None = "auto"
    # Installed after the core packages, pinned by a constraints file built from the
    # already-installed set so they cannot move torch/vllm versions.
    extra_packages: list[str] = Field(default_factory=list)
    post_install: list[PostInstallStep] = Field(default_factory=list)
    verify_imports: list[str] = Field(default_factory=list)  # checked inside the venv
    serve_command: list[str] = Field(min_length=1)  # executable relative to venv bin/


def load_model_spec(settings: Settings, model_id: str) -> ModelSpec:
    return load_model(settings.configs_dir / "models" / f"{model_id}.yaml", ModelSpec)


def load_runtime_spec(settings: Settings, runtime_id: str) -> RuntimeSpec:
    return load_model(settings.configs_dir / "runtimes" / f"{runtime_id}.yaml", RuntimeSpec)


def list_model_ids(settings: Settings) -> list[str]:
    return sorted(p.stem for p in (settings.configs_dir / "models").glob("*.yaml"))
