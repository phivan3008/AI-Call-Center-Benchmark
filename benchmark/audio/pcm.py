"""PCM16 helpers.

All audio inside the harness is mono PCM16 little-endian. Resampling uses soxr (HQ), whose
library name and version are recorded in run manifests via ``package_versions``.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soxr

FloatArray = npt.NDArray[np.float32]


def pcm16_to_float(pcm16: bytes) -> FloatArray:
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: FloatArray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample_pcm16(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm16:
        return pcm16
    out = soxr.resample(pcm16_to_float(pcm16), src_rate, dst_rate, quality="HQ")
    return float_to_pcm16(np.asarray(out, dtype=np.float32))


def to_mono(samples: npt.NDArray[np.float32], channels: int) -> FloatArray:
    if channels == 1:
        return samples
    mono: FloatArray = samples.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return mono


_WAVE_PCM = 1
_WAVE_FLOAT = 3
_WAVE_EXTENSIBLE = 0xFFFE


def read_wav(path: Path) -> tuple[bytes, int]:
    """Read a WAV file as mono PCM16 (see :func:`read_wav_bytes`)."""
    try:
        return read_wav_bytes(path.read_bytes())
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def wav_bytes(pcm16: bytes, sample_rate_hz: int) -> bytes:
    """A mono PCM16 WAV container in memory (for audio sent to HTTP APIs)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm16)
    return buffer.getvalue()


def read_wav_bytes(data: bytes) -> tuple[bytes, int]:
    """Decode WAV bytes as mono PCM16.

    Supports 16-bit integer PCM and 32-bit IEEE float (FLEURS ships float WAVs), including
    WAVE_FORMAT_EXTENSIBLE headers. The stdlib ``wave`` module only reads integer PCM.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    fmt: tuple[int, int, int, int] | None = None
    payload: bytes | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            tag = int.from_bytes(body[0:2], "little")
            if tag == _WAVE_EXTENSIBLE and len(body) >= 26:
                tag = int.from_bytes(body[24:26], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            payload = body
        pos += 8 + size + (size % 2)
    if fmt is None or payload is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    if tag == _WAVE_PCM and bits == 16:
        samples = pcm16_to_float(payload[: len(payload) // 2 * 2])
    elif tag == _WAVE_FLOAT and bits == 32:
        samples = np.frombuffer(payload[: len(payload) // 4 * 4], dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format (tag={tag}, bits={bits})")
    if channels == 1 and tag == _WAVE_PCM:
        return payload[: len(payload) // 2 * 2], rate
    return float_to_pcm16(to_mono(samples, channels)), rate


def write_wav(path: Path, pcm16: bytes, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm16)


def duration_s(pcm16: bytes, sample_rate_hz: int) -> float:
    return len(pcm16) / 2 / sample_rate_hz


def active_frames(
    pcm16: bytes, sample_rate_hz: int, frame_ms: int = 10, threshold_dbfs: float = -45.0
) -> npt.NDArray[np.bool_]:
    """``vad_v1`` activity mask: a frame is active if its RMS level exceeds the threshold
    (METRIC_DEFINITIONS §1.5)."""
    samples = pcm16_to_float(pcm16)
    frame_len = max(1, sample_rate_hz * frame_ms // 1000)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    dbfs = 20 * np.log10(np.maximum(rms, 1e-10))
    return np.asarray(dbfs > threshold_dbfs, dtype=bool)


def frame_dbfs(pcm16: bytes, sample_rate_hz: int, frame_ms: int = 10) -> npt.NDArray[np.float64]:
    samples = pcm16_to_float(pcm16)
    frame_len = max(1, sample_rate_hz * frame_ms // 1000)
    n_frames = len(samples) // frame_len
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1)) if n_frames else np.zeros(0)
    dbfs: npt.NDArray[np.float64] = 20 * np.log10(np.maximum(rms, 1e-10))
    return dbfs


def adaptive_threshold_dbfs(dbfs: npt.NDArray[np.float64]) -> float:
    """``stim_vad_v1`` threshold for recorded stimuli:
    max(min(max(P10 + 10 dB, P90 - 30 dB, -70 dBFS), P90 - 6 dB), -90 dBFS).

    Real recordings differ in level and noise floor (FLEURS clips range from about -95 to
    -30 dBFS background), so a fixed threshold cannot locate speech in all of them. The
    P90 - 6 dB cap keeps speech detectable when silence is under 10% of the clip; the
    -90 dBFS floor keeps digital silence inactive.
    """
    p10, p90 = np.percentile(dbfs, [10, 90])
    return float(max(min(max(p10 + 10.0, p90 - 30.0, -70.0), p90 - 6.0), -90.0))


def speech_bounds(
    pcm16: bytes,
    sample_rate_hz: int,
    frame_ms: int = 10,
    threshold_dbfs: float | None = None,
    min_run: int = 3,
) -> tuple[float, float] | None:
    """Start/end of speech in seconds, or None if there is no activity.

    A boundary needs ``min_run`` consecutive active frames, so isolated clicks are ignored.
    With ``threshold_dbfs=None`` the adaptive ``stim_vad_v1`` threshold is used.
    """
    dbfs = frame_dbfs(pcm16, sample_rate_hz, frame_ms)
    if dbfs.size == 0:
        return None
    threshold = adaptive_threshold_dbfs(dbfs) if threshold_dbfs is None else threshold_dbfs
    mask = dbfs > threshold
    runs = np.convolve(mask.astype(int), np.ones(min_run, dtype=int), mode="valid") == min_run
    idx = np.flatnonzero(runs)
    if idx.size == 0:
        return None
    return float(idx[0] * frame_ms / 1000), float((idx[-1] + min_run) * frame_ms / 1000)
