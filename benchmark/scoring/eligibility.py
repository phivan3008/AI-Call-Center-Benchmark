"""Leaderboard eligibility rules (ARCHITECTURE §10.3).

Mock runs never enter a leaderboard. Dirty-tree runs are rejected unless explicitly
allowed, and are then flagged. Incomplete runs are rejected.
"""

from __future__ import annotations

from benchmark.core.schemas import RunManifest, RunStatus


class IneligibleRunError(ValueError):
    pass


def ineligibility_reasons(manifest: RunManifest, allow_dirty: bool = False) -> list[str]:
    reasons: list[str] = []
    if manifest.is_mock:
        reasons.append("mock run (is_mock=true) can never be ranked")
    if manifest.git_dirty and not allow_dirty:
        reasons.append("run was produced from a dirty git tree (use --allow-dirty to override)")
    if manifest.status is not RunStatus.COMPLETED and manifest.status is not RunStatus.PARTIAL:
        reasons.append(f"run status is '{manifest.status}'")
    if manifest.gpu_contended:
        reasons.append("GPU was shared with foreign processes during the run")
    return reasons


def assert_rankable(manifest: RunManifest, allow_dirty: bool = False) -> None:
    reasons = ineligibility_reasons(manifest, allow_dirty)
    if reasons:
        raise IneligibleRunError(f"Run {manifest.run_id} is not rankable: " + "; ".join(reasons))
