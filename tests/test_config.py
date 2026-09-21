from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.core.config import (
    ConfigError,
    RunProfile,
    Settings,
    find_repo_root,
    load_model,
    load_profile,
)
from tests.conftest import REPO_ROOT


def test_home_defaults_to_repo_root() -> None:
    settings = Settings.from_env(env={}, repo_root=REPO_ROOT)
    assert settings.home == REPO_ROOT.resolve()
    assert settings.raw_dir == settings.home / "artifacts" / "raw"
    assert settings.configs_dir == REPO_ROOT / "configs"
    assert settings.log_level == "INFO"


def test_env_overrides(tmp_path: Path) -> None:
    env = {"VBENCH_HOME": str(tmp_path), "VBENCH_LOG_LEVEL": "DEBUG"}
    settings = Settings.from_env(env=env, repo_root=REPO_ROOT)
    assert settings.home == tmp_path.resolve()
    assert settings.bundles_dir == tmp_path.resolve() / "bundles"
    assert settings.processed_dir.name == "processed"
    assert settings.reports_dir.name == "reports"
    assert settings.datasets_dir == REPO_ROOT / "datasets"
    assert settings.log_level == "DEBUG"


def test_find_repo_root_fails_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        find_repo_root(tmp_path)


def test_load_repo_mock_profile(settings: Settings) -> None:
    profile = load_profile(settings, "mock_smoke")
    assert profile.mock_only is True
    assert profile.layers


def test_load_model_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model(tmp_path / "missing.yaml", RunProfile)
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_model(bad_yaml, RunProfile)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name: x\nlayers: []\nsamples_per_layer: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid config"):
        load_model(invalid, RunProfile)
