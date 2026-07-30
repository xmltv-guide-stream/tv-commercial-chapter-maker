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
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

import numpy as np


class FfmpegNotFound(RuntimeError):
    pass


class ProbeCancelled(Exception):
    """Raised when a caller's cancel Event fires mid-decode (see probe_file).
    Deliberately NOT a RuntimeError so the CPU-fallback path doesn't swallow it."""


def _cancelled(cancel) -> bool:
    return cancel is not None and cancel.is_set()


def _app_dir() -> str | None:
    """Directory to look for sibling tools next to a packaged (PyInstaller) exe."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return None


def find_tool(name: str) -> str | None:
    """Locate an external tool: PATH first, then alongside a frozen exe (so a
    user can just drop ffmpeg.exe / ffprobe.exe / mkvpropedit.exe next to the
    downloaded program). Returns the full path, or None."""
    found = shutil.which(name)
    if found:
        return found
    d = _app_dir()
    if d:
        for cand in (os.path.join(d, name + ".exe"), os.path.join(d, name),
                     os.path.join(d, "ffmpeg", name + ".exe"),
                     os.path.join(d, "bin", name + ".exe")):
            if os.path.isfile(cand):
                return cand
    return None


def _require(tool: str) -> str:
    path = find_tool(tool)
    if not path:
        raise FfmpegNotFound(
            f"'{tool}' was not found. Install ffmpeg (which includes ffmpeg and "
            f"ffprobe) from https://ffmpeg.org/download.html and add it to your "
            f"PATH — or place ffmpeg.exe and ffprobe.exe next to this program."
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
    # fraction of the frame occupied by a detected persistent overlay (channel
    # logo / bug) that was excluded from the peak. 0.0 = none detected.
    logo_frac: float = 0.0

    def as_dict(self) -> dict:
        d = {
            "duration": self.duration,
            "v_times": self.v_times.tolist(),
            "v_luma": self.v_luma.tolist(),
            "a_times": self.a_times.tolist(),
            "a_rms": self.a_rms.tolist(),
            "has_audio": self.has_audio,
            "logo_frac": self.logo_frac,
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
            logo_frac=float(d.get("logo_frac", 0.0)),
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
    proc = subprocess.Popen(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Drain stderr continuously on a background thread. Otherwise a chatty
    # decoder (e.g. a problematic AVI spewing warnings) can fill the ~64KB stderr
    # pipe buffer and block ffmpeg's writes while we're blocked reading stdout —
    # a classic deadlock that looks like the analyze "hanging" on one file. The
    # text is kept so _finish can still report it on failure.
    proc._stderr_buf: list[bytes] = []

    def _drain(p=proc):
        try:
            while True:
                chunk = p.stderr.read(65536)
                if not chunk:
                    break
                p._stderr_buf.append(chunk)
        except Exception:
            pass

    th = threading.Thread(target=_drain, daemon=True)
    th.start()
    proc._stderr_thread = th
    return proc


def _drain_stderr(proc) -> bytes:
    th = getattr(proc, "_stderr_thread", None)
    if th is not None:
        th.join(timeout=2.0)
    return b"".join(getattr(proc, "_stderr_buf", []))


def _finish(proc, produced: bool, what: str) -> None:
    ret = proc.wait()
    err = _drain_stderr(proc)
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

# --- persistent-overlay (channel logo / bug) detection --------------------- #
# A broadcast/DSR source often has a station logo burned into a fixed spot. It
# stays lit even when the picture fades to black, so its bright pixels keep the
# per-frame peak high and the blank guard wrongly concludes the file never goes
# dark. We find it from the file's keyframes: a pixel that stays bright across
# *almost all* of them is a static overlay, not scene content (which dips dark
# at some point over a whole episode). We use a low PERCENTILE per pixel rather
# than the strict minimum, so a black cold-open/outro or a few dark scenes over
# the logo don't hide it (one dark frame would defeat a pure minimum). Those
# pixels are then excluded from the peak so a fade-behind-a-logo reads dark.
_LOGO_MIN_LEVEL = 48.0    # a pixel bright even at its low percentile = persistent
_LOGO_PCTL = 10.0         # "bright in >=90% of keyframes" (tolerates intro/outro black)
_LOGO_MAX_FRAC = 0.05     # if "persistent bright" exceeds this, it's not a logo
_LOGO_MIN_KEYFRAMES = 8   # too few keyframes -> not enough evidence, skip
_LOGO_MAX_KEYFRAMES = 1500  # cap frames held in memory for the percentile

# --- recurring CORNER overlay (rating bug / brief network bug) -------------- #
# A TV rating bug ("TV-14 D") or a brief post-break network bug isn't persistent,
# so the logo test above misses it — but it flashes up near a fixed EDGE/corner on
# the *fade* frames after each break. Keyframes are too sparse to catch those
# brief fades, so this is detected during the dense video pass (_sample_video):
# among the DARK frames, pixels in the outer EDGE band that light up in >= 2
# distinct dark *segments* (i.e. at >= 2 breaks) are excluded from the peak.
# Two safety valves keep false positives near-zero: (a) only the outer edge band
# is eligible, so the central region — where real scene content lives — is never
# masked; (b) requiring >= 2 separate dark segments means a single dark scene
# with a bright edge won't trip it (only something recurring, i.e. an overlay).
_LOGO_EDGE_FRAC = 0.25         # eligible band = within 25% of any edge (central 50% safe)
_LOGO_DARK_FRAME = 24.0        # a frame whose mean is below this = a fade/near-blank
_LOGO_RECUR_BRIGHT = 64.0      # a pixel this bright counts as "lit"
_LOGO_RECUR_SEGMENTS = 2       # lit in >= this many distinct dark segments = overlay
_LOGO_RECUR_MIN_PIXELS = 12    # ignore tiny/noise clusters
_LOGO_RECUR_MAX_CANDIDATES = 4000  # cap dark bright-peak frames held for re-scoring


def _edge_region_mask(scale_w: int, scale_h: int, frac: float) -> np.ndarray:
    """Flat bool mask, True within `frac` of any edge (top/bottom/left/right
    bands). The central region is left out, so scene content there is never
    masked — that's the false-positive safety valve for overlay detection."""
    cw = max(1, int(round(scale_w * frac)))
    ch = max(1, int(round(scale_h * frac)))
    m = np.zeros((scale_h, scale_w), dtype=bool)
    m[:ch, :] = True
    m[scale_h - ch:, :] = True
    m[:, :cw] = True
    m[:, scale_w - cw:] = True
    return m.ravel()


def _logo_mask(path: str, scale_w: int, scale_h: int, hwaccel: str | None = None, cancel=None):
    """Return (flat bool mask of overlay pixels, fraction) or (None, 0.0).

    Decodes only keyframes (`-skip_frame nokey`) so this extra pass is cheap.
    Never raises for decode issues — a failure just means "no logo detected",
    leaving peak computation exactly as it was."""
    frame_bytes = scale_w * scale_h
    hw = ["-hwaccel", hwaccel] if hwaccel else []
    proc = _popen([
        "-skip_frame", "nokey", *hw, "-i", path, "-map", "0:v:0",
        "-an", "-sn", "-dn", "-vsync", "0",
        "-vf", f"scale={scale_w}:{scale_h}", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    frames: list[np.ndarray] = []
    try:
        while len(frames) < _LOGO_MAX_KEYFRAMES:
            if _cancelled(cancel):
                raise ProbeCancelled()
            buf = _read_exact(proc.stdout, frame_bytes)
            if len(buf) < frame_bytes:
                break
            frames.append(np.frombuffer(buf, dtype=np.uint8))
    except ProbeCancelled:
        raise
    except Exception:
        frames = []
    finally:
        try:
            proc.kill()  # we may have stopped early at the cap
        except Exception:
            pass
        try:
            proc.wait()
        except Exception:
            pass
        _drain_stderr(proc)  # let the background reader finish (owns proc.stderr)
    if len(frames) < _LOGO_MIN_KEYFRAMES:
        return None, 0.0
    # persistent overlay only: bright even at its low percentile => bright in
    # >= (100-pctl)% of keyframes (an always-on channel logo). Transient rating
    # bugs are handled in _sample_video from the dense pass.
    lo = np.percentile(np.stack(frames), _LOGO_PCTL, axis=0)
    thresh = max(_LOGO_MIN_LEVEL, float(np.median(lo)) + 40.0)
    mask = lo >= thresh
    frac = float(mask.mean())
    if frac <= 0.0 or frac > _LOGO_MAX_FRAC:
        return None, 0.0
    return mask, frac


def _sample_video(path: str, fps: float, scale_w: int, scale_h: int,
                  logo_mask=None, hwaccel: str | None = None, cancel=None,
                  detect_corner_overlay: bool = True):
    """Return (times, luma, peak, extra_overlay_frac) from raw grayscale frames.

    luma = frame mean (scale-invariant, so cheap to compute at any size); peak =
    the frame's high-percentile brightness. A blank frame is dark in both; a
    scene on a black background — even a very dark one with only faint detail —
    has a low mean but a peak that rises above the file's darkest frames, which
    is how detection avoids marking dark-scene frames as breaks.

    `logo_mask` (persistent-overlay pixels) is excluded from the peak. While
    streaming, we also look for a RECURRING CORNER overlay (a rating bug that
    flashes on the fades after each break): corner pixels lit across >= 2 distinct
    dark segments are excluded from the peak of those dark frames, so a fade
    behind a rating bug still reads blank. `extra_overlay_frac` reports how much
    of the frame that recurring overlay covered (0.0 if none)."""
    frame_bytes = scale_w * scale_h
    keep = ~logo_mask if logo_mask is not None else None
    hw = ["-hwaccel", hwaccel] if hwaccel else []
    proc = _popen([
        *hw, "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={scale_w}:{scale_h}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    luma: list[float] = []
    peak: list[float] = []
    # recurring edge-overlay accumulation (segment-counted so it means ">=N breaks")
    region = _edge_region_mask(scale_w, scale_h, _LOGO_EDGE_FRAC) if detect_corner_overlay else None
    seg_count = np.zeros(frame_bytes, dtype=np.int32) if region is not None else None
    seg_lit = np.zeros(frame_bytes, dtype=bool) if region is not None else None
    in_dark = False
    candidates: list = []   # (index, px copy) for dark frames that have a bright peak
    while True:
        if _cancelled(cancel):
            proc.kill()
            _drain_stderr(proc)
            raise ProbeCancelled()
        buf = _read_exact(proc.stdout, frame_bytes)
        if len(buf) < frame_bytes:
            break  # clean EOF (ignore any partial trailing frame)
        px = np.frombuffer(buf, dtype=np.uint8)
        m = float(px.mean())
        luma.append(m)
        pk = float(np.percentile(px[keep] if keep is not None else px, _PEAK_PCT))
        peak.append(pk)
        if region is not None:
            if m < _LOGO_DARK_FRAME:
                if not in_dark:
                    in_dark = True
                    seg_lit[:] = False
                np.logical_or(seg_lit, px >= _LOGO_RECUR_BRIGHT, out=seg_lit)
                if pk > _LOGO_RECUR_BRIGHT and len(candidates) < _LOGO_RECUR_MAX_CANDIDATES:
                    candidates.append((len(luma) - 1, px.copy()))
            elif in_dark:
                in_dark = False
                seg_count += seg_lit
    _finish(proc, produced=bool(luma), what="video")
    if region is not None and in_dark:
        seg_count += seg_lit

    times = np.arange(len(luma), dtype=np.float64) / fps
    peak_arr = np.asarray(peak, dtype=np.float64)
    extra_frac = 0.0
    if region is not None:
        recurring = region & (seg_count >= _LOGO_RECUR_SEGMENTS)
        if int(np.count_nonzero(recurring)) >= _LOGO_RECUR_MIN_PIXELS:
            extra_frac = float(recurring.mean())
            excl = recurring | logo_mask if logo_mask is not None else recurring
            keep2 = ~excl
            for idx, px in candidates:      # re-score just the bug-lit dark frames
                peak_arr[idx] = float(np.percentile(px[keep2], _PEAK_PCT))
    return times, np.asarray(luma, dtype=np.float64), peak_arr, extra_frac


def _sample_audio_rms(path: str, window: float, sr: int = 8000, cancel=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, rms dBFS) from raw mono float PCM, in `window`-sec bins."""
    win = max(1, int(sr * window))
    win_bytes = win * 4  # float32
    proc = _popen([
        "-i", path, "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ])
    rms: list[float] = []
    while True:
        if _cancelled(cancel):
            proc.kill()
            _drain_stderr(proc)
            raise ProbeCancelled()
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
    logo_detect: bool = True,
    hwaccel: str | None = None,
    cancel=None,
) -> Signals:
    """Profile one file into brightness and loudness time series.

    `hwaccel` (e.g. 'cuda', 'qsv', 'd3d11va', 'auto') offloads video decoding to
    a GPU/hardware decoder; the main video pass falls back to CPU if it fails.
    `cancel` is an optional threading.Event — when set mid-decode the running
    ffmpeg is killed and ProbeCancelled is raised, so a GUI can stop promptly."""
    duration = probe_duration(path)
    logo_mask, logo_frac = (None, 0.0)
    if logo_detect:
        try:
            logo_mask, logo_frac = _logo_mask(path, scale_w, scale_h, hwaccel=hwaccel, cancel=cancel)
        except (FfmpegNotFound, ProbeCancelled):
            raise
        except Exception:
            logo_mask, logo_frac = None, 0.0   # never let logo detection break profiling
    try:
        v_times, v_luma, v_peak, recur_frac = _sample_video(
            path, video_fps, scale_w, scale_h, logo_mask, hwaccel=hwaccel, cancel=cancel)
    except RuntimeError:
        if hwaccel:   # hardware decode failed -> retry on the CPU
            v_times, v_luma, v_peak, recur_frac = _sample_video(
                path, video_fps, scale_w, scale_h, logo_mask, hwaccel=None, cancel=cancel)
        else:
            raise
    logo_frac = float(min(1.0, logo_frac + recur_frac))   # persistent + recurring-corner
    has_audio = _has_audio_stream(path)
    if has_audio:
        a_times, a_rms = _sample_audio_rms(path, audio_window, cancel=cancel)
    else:
        a_times, a_rms = np.empty(0), np.empty(0)
    return Signals(
        duration=duration or (float(v_times[-1]) if v_times.size else 0.0),
        v_times=v_times, v_luma=v_luma,
        a_times=a_times, a_rms=a_rms,
        has_audio=has_audio,
        v_peak=v_peak,
        logo_frac=logo_frac,
    )
