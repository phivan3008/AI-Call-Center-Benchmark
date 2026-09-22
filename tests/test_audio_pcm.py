from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmark.audio.pcm import (
    active_frames,
    duration_s,
    float_to_pcm16,
    pcm16_to_float,
    read_wav,
    resample_pcm16,
    speech_bounds,
    write_wav,
)


def _tone(seconds: float, rate: int, amp: float = 0.3) -> bytes:
    n = int(seconds * rate)
    return struct.pack(
        f"<{n}h", *(int(amp * 32767 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n))
    )


def test_pcm_roundtrip() -> None:
    pcm = _tone(0.1, 16000)
    back = float_to_pcm16(pcm16_to_float(pcm))
    assert np.max(np.abs(np.frombuffer(back, "<i2") - np.frombuffer(pcm, "<i2"))) <= 1


def test_resample_changes_length_and_keeps_identity() -> None:
    pcm = _tone(1.0, 16000)
    up = resample_pcm16(pcm, 16000, 24000)
    assert duration_s(up, 24000) == pytest.approx(1.0, abs=0.01)
    assert resample_pcm16(pcm, 16000, 16000) is pcm
    assert resample_pcm16(b"", 16000, 24000) == b""


def test_wav_io_and_stereo_downmix(tmp_path: Path) -> None:
    pcm = _tone(0.2, 16000)
    write_wav(tmp_path / "a.wav", pcm, 16000)
    assert read_wav(tmp_path / "a.wav") == (pcm, 16000)

    stereo = tmp_path / "s.wav"
    with wave.open(str(stereo), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<4h", 1000, 3000, -1000, -3000))
    mono, rate = read_wav(stereo)
    assert rate == 8000
    assert list(np.frombuffer(mono, "<i2")) == [1999, -1999]

    bad = tmp_path / "b.wav"
    with wave.open(str(bad), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(8000)
        wav.writeframes(b"\x80" * 10)
    with pytest.raises(ValueError, match="unsupported WAV format"):
        read_wav(bad)


def test_speech_bounds_find_the_tone() -> None:
    rate = 16000
    pcm = b"\x00\x00" * int(0.5 * rate) + _tone(1.0, rate) + b"\x00\x00" * int(0.3 * rate)
    bounds = speech_bounds(pcm, rate)
    assert bounds is not None
    assert bounds[0] == pytest.approx(0.5, abs=0.02)
    assert bounds[1] == pytest.approx(1.5, abs=0.02)
    assert speech_bounds(b"\x00\x00" * rate, rate) is None
    assert active_frames(b"\x00\x00", rate).size == 0
