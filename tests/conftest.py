from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmark.core.config import RunProfile, Settings, find_repo_root
from benchmark.core.logging import clear_context, configure_logging

REPO_ROOT = find_repo_root()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=REPO_ROOT, home=tmp_path / "home")


@pytest.fixture
def profile() -> RunProfile:
    return RunProfile(
        name="test",
        layers=["L1"],
        samples_per_layer=2,
        sample_timeout_s=5,
        input_sample_rate_hz=16000,
    )


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    configure_logging()
    clear_context()
