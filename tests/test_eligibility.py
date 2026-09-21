from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from benchmark.core.schemas import EnvInfo, ModelVersion, RunManifest, RunStatus
from benchmark.scoring.eligibility import (
    IneligibleRunError,
    assert_rankable,
    ineligibility_reasons,
)


def _manifest(**overrides: Any) -> RunManifest:
    data: dict[str, Any] = {
        "run_id": "R",
        "profile": "full",
        "is_mock": False,
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "git_commit": "abc",
        "git_dirty": False,
        "config_sha256": "0" * 64,
        "model": ModelVersion(model_id="m"),
        "environment": EnvInfo(hostname="h", platform="p", python_version="3.11"),
        "layers": ["L1"],
        "status": RunStatus.COMPLETED,
    }
    data.update(overrides)
    return RunManifest.model_validate(data)


def test_clean_completed_run_is_rankable() -> None:
    assert_rankable(_manifest())
    assert ineligibility_reasons(_manifest(status=RunStatus.PARTIAL)) == []


def test_mock_run_is_never_rankable() -> None:
    with pytest.raises(IneligibleRunError, match="mock"):
        assert_rankable(_manifest(is_mock=True), allow_dirty=True)


def test_dirty_run_requires_explicit_override() -> None:
    dirty = _manifest(git_dirty=True)
    with pytest.raises(IneligibleRunError, match="dirty"):
        assert_rankable(dirty)
    assert_rankable(dirty, allow_dirty=True)


@pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.RUNNING])
def test_failed_or_running_run_rejected(status: RunStatus) -> None:
    assert ineligibility_reasons(_manifest(status=status)) == [f"run status is '{status}'"]


def test_gpu_contended_run_rejected() -> None:
    assert ineligibility_reasons(_manifest(gpu_contended=True))
