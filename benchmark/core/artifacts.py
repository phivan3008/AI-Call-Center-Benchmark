"""Artifact layout, atomic writers and checksums (ARCHITECTURE §9.3)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

SHA256SUMS = "SHA256SUMS"
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class RunPaths:
    """Paths inside ``artifacts/raw/<run_id>/``."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def run_log(self) -> Path:
        return self.logs_dir / "run.jsonl"

    @property
    def monitoring_dir(self) -> Path:
        return self.root / "monitoring"

    @property
    def checksums(self) -> Path:
        return self.root / SHA256SUMS

    def sample_dir(self, layer: str, sample_id: str) -> Path:
        return self.root / layer / sample_id

    @classmethod
    def create(cls, raw_dir: Path, run_id: str) -> RunPaths:
        root = raw_dir / run_id
        if root.exists():
            raise FileExistsError(f"Run directory already exists: {root}")
        (root / "logs").mkdir(parents=True)
        return cls(root)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory and ``os.replace`` it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    atomic_write_text(path, text + "\n")


class JsonlWriter:
    """Append-only JSONL writer, flushed per line so crashes keep completed lines."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def write(self, obj: BaseModel | dict[str, Any]) -> None:
        line = (
            obj.model_dump_json()
            if isinstance(obj, BaseModel)
            else json.dumps(obj, ensure_ascii=False, sort_keys=True)
        )
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _listed_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name != SHA256SUMS and not p.name.endswith(".tmp")
    )


def write_sha256sums(root: Path) -> Path:
    """Write ``SHA256SUMS`` (``<hash>  <relative posix path>`` per line) covering all files."""
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in _listed_files(root)]
    target = root / SHA256SUMS
    atomic_write_text(target, "\n".join(lines) + ("\n" if lines else ""))
    return target


def verify_sha256sums(root: Path) -> list[str]:
    """Return a list of problems; empty means every listed file matches and none is unlisted."""
    sums_path = root / SHA256SUMS
    if not sums_path.is_file():
        return [f"missing {SHA256SUMS}"]
    problems: list[str] = []
    listed: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, rel = line.partition("  ")
        listed.add(rel)
        target = root / rel
        if not target.is_file():
            problems.append(f"missing file: {rel}")
        elif sha256_file(target) != expected:
            problems.append(f"checksum mismatch: {rel}")
    for path in _listed_files(root):
        rel = path.relative_to(root).as_posix()
        if rel not in listed:
            problems.append(f"unlisted file: {rel}")
    return problems
