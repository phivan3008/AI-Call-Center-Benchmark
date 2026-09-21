from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from benchmark.core.artifacts import (
    JsonlWriter,
    RunPaths,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)


class _Row(BaseModel):
    a: int


def test_run_paths_layout(tmp_path: Path) -> None:
    paths = RunPaths.create(tmp_path, "RUN1")
    assert paths.run_id == "RUN1"
    assert paths.logs_dir.is_dir()
    assert paths.manifest.name == "manifest.json"
    assert paths.run_log.parent == paths.logs_dir
    assert paths.monitoring_dir.name == "monitoring"
    assert paths.checksums.name == "SHA256SUMS"
    assert paths.sample_dir("L1", "s1") == tmp_path / "RUN1" / "L1" / "s1"
    with pytest.raises(FileExistsError):
        RunPaths.create(tmp_path, "RUN1")


def test_atomic_writes(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "sub" / "a.txt", "日本語")
    assert (tmp_path / "sub" / "a.txt").read_text(encoding="utf-8") == "日本語"
    atomic_write_json(tmp_path / "b.json", {"b": 1, "a": "あ"})
    assert json.loads((tmp_path / "b.json").read_text(encoding="utf-8")) == {"a": "あ", "b": 1}
    atomic_write_json(tmp_path / "c.json", _Row(a=3))
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8")) == {"a": 3}
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        atomic_write_json(tmp_path / "bad.json", {"x": object()})
    assert not list(tmp_path.iterdir())


def test_jsonl_writer(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    with JsonlWriter(path) as writer:
        writer.write(_Row(a=1))
        writer.write({"a": 2})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"a": 1}, {"a": 2}]


def test_sha256sums_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    sums = write_sha256sums(tmp_path)
    assert "x/a.txt" in sums.read_text(encoding="utf-8")
    assert verify_sha256sums(tmp_path) == []

    (tmp_path / "b.bin").write_bytes(b"changed")
    (tmp_path / "new.txt").write_text("n", encoding="utf-8")
    (tmp_path / "x" / "a.txt").unlink()
    problems = verify_sha256sums(tmp_path)
    assert "checksum mismatch: b.bin" in problems
    assert "unlisted file: new.txt" in problems
    assert "missing file: x/a.txt" in problems


def test_verify_without_sums_file(tmp_path: Path) -> None:
    assert verify_sha256sums(tmp_path) == ["missing SHA256SUMS"]


def test_empty_directory_sums(tmp_path: Path) -> None:
    write_sha256sums(tmp_path)
    assert verify_sha256sums(tmp_path) == []
    assert len(sha256_file(tmp_path / "SHA256SUMS")) == 64
