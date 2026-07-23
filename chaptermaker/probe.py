"""Extract time-series brightness and loudness signals from a video via ffmpeg.

We do two cheap passes:

  * video luma  -> signalstats YAVG (average luma per sampled frame, 0..255)
  * audio level -> astats RMS_level (dBFS per short window)

The file is always passed as a normal ``-i`` argument (never embedded in a
filtergraph), so paths with spaces, apostrophes, ``+``, drive letters and
backslashes — e.g. Windows rips like ``X:\\Little Bear\\... Bear's Bath.mkv``
— work without any fragile filter escaping. The numbers are computed from the
raw decoded bytes ffmpeg pipes to stdout (grayscale pixels / mono float PCM),
which is byte-identical across ffmpeg versions — nothing depends on parsing
ffmpeg's human-readable log output.

Both signals are downsampled in time so a 2-hour tape costs seconds of parsing,
not minutes. Video frames are also scaled down (luma average is essentially
unchanged by downscaling) to keep decoding fast.

Nothing here interprets the numbers — that's detect.py's job. This module just
turns a video file into two numpy arrays of (timestamp, value).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np


class FfmpegNotFound(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FfmpegNotFound(
            f"'{tool}' was not found on your PATH. Install ffmpeg "
            f"(which includes {tool}) and make sure it's on PATH."
        )
    return path


@dataclass
class Signals:
    """Raw sampled signals for one file."""

    duration: float
    # video: parallel arrays, one entry per sampled frame
    v_times: np.ndarray  # seconds
    v_luma: np.ndarray   # average luma, 0..255 (0 = black)
    # audio: parallel arrays, one entry per sampled window
    a_times: np.ndarray  # seconds
    a_rms: np.ndarray    # RMS level in dBFS (e.g. -90 = near silent, 0 = max)
    has_audio: bool

    def as_dict(self) -> dict:
        return {
            "duration": self.duration,
            "v_times": self.v_times.tolist(),
            "v_luma": self.v_luma.tolist(),
            "a_times": self.a_times.tolist(),
            "a_rms": self.a_rms.tolist(),
            "has_audio": self.has_audio,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Signals":
        return cls(
            duration=d["duration"],
            v_times=np.asarray(d["v_times"], dtype=np.float64),
            v_luma=np.asarray(d["v_luma"], dtype=np.float64),
            a_times=np.asarray(d["a_times"], dtype=np.float64),
            a_rms=np.asarray(d["a_rms"], dtype=np.float64),
            has_audio=d["has_audio"],
        )


def probe_duration(path: str) -> float:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _has_audio_stream(path: str) -> bool:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


# We extract brightness and loudness by piping *raw* decoded bytes out of
# ffmpeg (grayscale pixels / mono float PCM) and computing the numbers in Python.
# Raw output formats are byte-identical across every ffmpeg version and platform,
# so there is nothing to parse from ffmpeg's log text — the previous approach
# depended on the exact wording/stream of stderr, which varies between builds.


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes, or fewer only at end-of-stream."""
    parts = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _popen(args: list[str]):
    ffmpeg = _require("ffmpeg")
    return subprocess.Popen(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _finish(proc, produced: bool, what: str) -> None:
    err = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0 and not produced:
        msg = err.decode("utf-8", "replace").strip()[:500]
        raise RuntimeError(f"ffmpeg {what} decode failed (exit {ret}): {msg}")


def _sample_video_luma(path: str, fps: float, scale_w: int, scale_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, luma 0..255) by averaging raw grayscale frames at `fps`."""
    frame_bytes = scale_w * scale_h
    proc = _popen([
        "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={scale_w}:{scale_h}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    luma: list[float] = []
    while True:
        buf = _read_exact(proc.stdout, frame_bytes)
        if len(buf) < frame_bytes:
            break  # clean EOF (ignore any partial trailing frame)
        luma.append(float(np.frombuffer(buf, dtype=np.uint8).mean()))
    _finish(proc, produced=bool(luma), what="video")
    times = np.arange(len(luma), dtype=np.float64) / fps
    return times, np.asarray(luma, dtype=np.float64)


def _sample_audio_rms(path: str, window: float, sr: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, rms dBFS) from raw mono float PCM, in `window`-sec bins."""
    win = max(1, int(sr * window))
    win_bytes = win * 4  # float32
    proc = _popen([
        "-i", path, "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ])
    rms: list[float] = []
    while True:
        buf = _read_exact(proc.stdout, win_bytes)
        if not buf:
            break
        x = np.frombuffer(buf, dtype=np.float32)
        if x.size == 0:
            break
        r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        rms.append(20.0 * np.log10(r) if r > 1e-7 else -120.0)
        if len(buf) < win_bytes:
            break  # final short window
    _finish(proc, produced=bool(rms), what="audio")
    times = np.arange(len(rms), dtype=np.float64) * window
    return times, np.asarray(rms, dtype=np.float64)


def probe_file(
    path: str,
    *,
    video_fps: float = 12.0,
    audio_window: float = 0.1,
    scale_w: int = 128,
    scale_h: int = 72,
) -> Signals:
    """Profile one file into brightness and loudness time series."""
    duration = probe_duration(path)
    v_times, v_luma = _sample_video_luma(path, video_fps, scale_w, scale_h)
    has_audio = _has_audio_stream(path)
    if has_audio:
        a_times, a_rms = _sample_audio_rms(path, audio_window)
    else:
        a_times, a_rms = np.empty(0), np.empty(0)
    return Signals(
        duration=duration or (float(v_times[-1]) if v_times.size else 0.0),
        v_times=v_times, v_luma=v_luma,
        a_times=a_times, a_rms=a_rms,
        has_audio=has_audio,
    )
