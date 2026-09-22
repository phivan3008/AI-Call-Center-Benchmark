"""Settings and YAML configuration loading.

Two roots are distinguished (DEPLOYMENT_GUIDE §3):

* ``repo_root``: this repository (configs, dataset manifests, docs).
* ``home`` (``VBENCH_HOME``): data root on the benchmark SSD (artifacts, bundles,
  model caches, runtimes). Defaults to ``repo_root`` on a developer machine.

No path is hardcoded; everything derives from these two roots or env overrides.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ENV_PREFIX = "VBENCH_"

M = TypeVar("M", bound=BaseModel)


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file) to the directory holding ``pyproject.toml``
    and ``configs/``."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigError(f"Could not locate repository root from {here}")


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path
    home: Path
    log_level: str = "INFO"

    @property
    def configs_dir(self) -> Path:
        return self.repo_root / "configs"

    @property
    def datasets_dir(self) -> Path:
        return self.repo_root / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def raw_dir(self) -> Path:
        return self.artifacts_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.artifacts_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def bundles_dir(self) -> Path:
        return self.home / "bundles"

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, repo_root: Path | None = None
    ) -> Settings:
        env = os.environ if env is None else env
        root = repo_root or find_repo_root()
        home = Path(env[f"{ENV_PREFIX}HOME"]) if env.get(f"{ENV_PREFIX}HOME") else root
        return cls(
            repo_root=root,
            home=home.expanduser().resolve(),
            log_level=env.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO"),
        )


class TaskSpec(BaseModel):
    """One block of samples in a profile (Phase 2: turn, tool and cancel checks)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["turn", "tool", "cancel"]
    layer: Literal["L1", "L2", "L3", "L4", "L5"]
    count: int = Field(gt=0)
    prompt: str  # key in the profile's prompt file
    cancel_after_ms: int = Field(default=1000, ge=0)


class RunProfile(BaseModel):
    """A run profile (``configs/runs/<name>.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    layers: list[str] = Field(min_length=1)
    samples_per_layer: int = Field(gt=0)
    repeats: int = Field(default=1, gt=0)
    seed: int = 1234
    sample_timeout_s: float = Field(default=60.0, gt=0)
    frame_ms: int = Field(default=20, gt=0)
    input_sample_rate_hz: int = Field(default=16000, gt=0)
    mock_only: bool = False
    dataset: str | None = None  # dataset built by `vbench data prepare`
    prompts: str | None = None  # configs/prompts/<path>.yaml
    tasks: list[TaskSpec] = Field(default_factory=list)


def load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc


def load_model(path: Path, model_cls: type[M]) -> M:
    """Load a YAML file and validate it into ``model_cls``."""
    data = load_yaml(path)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc


def load_profile(settings: Settings, name: str) -> RunProfile:
    return load_model(settings.configs_dir / "runs" / f"{name}.yaml", RunProfile)
