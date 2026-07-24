"""Extract time-series brightness and loudness signals from a video via ffmpeg.

We do two cheap passes:

  * video luma  -> average luma per sampled frame, 0..255
  * video peak  -> a high percentile of each frame's pixels, so a scene sitting
                   on a black background (low average, but bright objects) can be
                   told apart from a genuinely blank/fade frame (dark everywhere)
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
    # per-frame high-percentile luma (brightest region). None for legacy caches
    # profiled before this signal existed -> detection skips the blank guard.
    v_peak: np.ndarray | None = None

    def as_dict(self) -> dict:
        d = {
            "duration": self.duration,
            "v_times": self.v_times.tolist(),
            "v_luma": self.v_luma.tolist(),
            "a_times": self.a_times.tolist(),
            "a_rms": self.a_rms.tolist(),
            "has_audio": self.has_audio,
        }
        if self.v_peak is not None:
            d["v_peak"] = self.v_peak.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Signals":
        peak = d.get("v_peak")
        return cls(
            duration=d["duration"],
            v_times=np.asarray(d["v_times"], dtype=np.float64),
            v_luma=np.asarray(d["v_luma"], dtype=np.float64),
            a_times=np.asarray(d["a_times"], dtype=np.float64),
            a_rms=np.asarray(d["a_rms"], dtype=np.float64),
            has_audio=d["has_audio"],
            v_peak=None if peak is None else np.asarray(peak, dtype=np.float64),
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


# High percentile used for the per-frame "peak" (brightest region) signal.
# We want to catch even a small, dim highlight in an otherwise-black frame (a
# dark scene is not a blank), so this is very high — the top ~0.1% of pixels.
# It works only because the peak is sampled at a fairly high resolution (see
# probe_file's scale below): at a small size, area-averaging blends sparse
# bright pixels into the surrounding black and no percentile can recover them.
_PEAK_PCT = 99.9


def _sample_video(path: str, fps: float, scale_w: int, scale_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, luma, peak), each 0..255, from raw grayscale frames at `fps`.

    luma = frame mean (scale-invariant, so cheap to compute at any size); peak =
    the frame's high-percentile brightness. A blank frame is dark in both; a
    scene on a black background — even a very dark one with only faint detail —
    has a low mean but a peak that rises above the file's darkest frames, which
    is how detection avoids marking dark-scene frames as breaks."""
    frame_bytes = scale_w * scale_h
    proc = _popen([
        "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={scale_w}:{scale_h}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    luma: list[float] = []
    peak: list[float] = []
    while True:
        buf = _read_exact(proc.stdout, frame_bytes)
        if len(buf) < frame_bytes:
            break  # clean EOF (ignore any partial trailing frame)
        px = np.frombuffer(buf, dtype=np.uint8)
        luma.append(float(px.mean()))
        peak.append(float(np.percentile(px, _PEAK_PCT)))
    _finish(proc, produced=bool(luma), what="video")
    times = np.arange(len(luma), dtype=np.float64) / fps
    return times, np.asarray(luma, dtype=np.float64), np.asarray(peak, dtype=np.float64)


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
    scale_w: int = 256,
    scale_h: int = 144,
) -> Signals:
    """Profile one file into brightness and loudness time series."""
    duration = probe_duration(path)
    v_times, v_luma, v_peak = _sample_video(path, video_fps, scale_w, scale_h)
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
        v_peak=v_peak,
    )
