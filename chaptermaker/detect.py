"""Adaptive commercial-break detection.

The key difference from typical black/silence detectors: thresholds are derived
*per file* from that file's own distribution of brightness and loudness, rather
than fixed absolute values. A VHS capture whose "black" sits at luma ~40 and
whose "silence" floor is -55 dBFS (tape hiss) still works, because we anchor to
the file's own darkest/quietest level and look for excursions toward it.

Tuning knobs (all exposed on the CLI):
  sensitivity     global multiplier on how far from the floor still counts
  black_margin    luma units above the file's black floor still counted "black"
  quiet_margin    dB above the file's quiet floor still counted "quiet"
  quiet_floor_trim  fraction of the quietest audio samples ignored when setting
                  the quiet floor (audio only), so a silent outro / a few
                  digital-silence samples don't pin the threshold absurdly low
  blank_guard     require a dark frame to be dark at its *peak* too, so a scene
                  on a black background (low average, bright objects) is not
                  mistaken for a blank/fade frame
  bright_margin   how far a frame's peak may rise above the file's darkest-peak
                  floor and still count as "blank"
  peak_cap        absolute peak ceiling; a frame brighter than this at its peak
                  is never "blank", so a file that never truly goes dark yields
                  no dark breaks rather than marking its dimmest scenes
  mode            'and'  -> a break needs BOTH dark and quiet (fewest false +)
                  'or'   -> either dark OR quiet
  near_miss       in 'and' mode, also accept a point where one signal is deep
                  past its floor and the other only *just* misses its threshold
                  (rescues fades that hit true-quiet but not-quite-black, or
                  vice-versa — common on VHS where interior fades stop at grey)
  min_duration    a candidate must persist at least this long (seconds)
  min_gap         drop a break if the last kept chapter was <this ago (seconds)
  skip_start      never place a detected break before this time (seconds)
  skip_end        never place a detected break within this of the end (seconds)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .probe import Signals


@dataclass
class DetectConfig:
    sensitivity: float = 1.0
    black_margin: float = 14.0      # luma units (0..255)
    quiet_margin: float = 8.0       # dB
    quiet_floor_trim: float = 0.01  # ignore this fraction of quietest audio
                                    # samples when setting the floor, so a silent
                                    # outro / digital-silence doesn't pin it low
    blank_guard: bool = True        # a dark frame must ALSO be dark at its peak
    bright_margin: float = 32.0     # how far a frame's peak may rise above the
                                    # darkest-peak floor and still count "blank"
    peak_cap: float = 100.0         # absolute ceiling: a frame whose peak exceeds
                                    # this has real bright content and is NEVER
                                    # blank, whatever the adaptive floor says — so
                                    # a file with no true-dark frames yields no
                                    # dark breaks instead of marking dim scenes
    mode: str = "and"               # 'and' | 'or'
    near_miss: bool = True          # in 'and' mode, rescue deep-one-signal + barely-missed-other
    near_deep: float = 0.5          # frac of band from floor that counts as "deep past floor"
    near_tol: float = 0.25          # frac of band past threshold still counted as "nearly met"
    ignore_audio: bool = False      # detect on darkness alone (video-only)
    audio_fallback: bool = True     # if AND+audio finds too few, retry video-only
    min_duration: float = 0.30      # seconds a break must last
    min_gap: float = 45.0           # min seconds since the last kept chapter
    skip_start: float = 0.0         # suppress detected breaks before this time
    skip_end: float = 0.0           # suppress detected breaks within this of the end
    grid_step: float = 0.10         # resampling resolution (seconds)
    add_intro: bool = True          # always put a chapter at 00:00
    mark_at: str = "mid"            # 'start' | 'mid' | 'end' of the dark/quiet run
    # --- auto-escalation to reach a minimum chapter count ---
    min_chapters: int = 0           # 0 = off; else loosen thresholds until met
    min_duration_floor: float = 0.05  # shortest gap escalation will accept (sec)
    escalate_attempts: int = 12     # max loosening rounds before giving up
    max_chapters: int | None = None


@dataclass
class Break:
    time: float          # chapter position (seconds)
    start: float         # start of the dark/quiet run
    end: float           # end of the dark/quiet run
    duration: float
    min_luma: float
    min_rms: float
    score: float         # rough confidence, higher = stronger break
    is_intro: bool = False


@dataclass
class Profile:
    """Summary of a file's derived thresholds — useful for review/debugging."""

    luma_floor: float
    black_thresh: float
    rms_floor: float
    quiet_thresh: float
    has_audio: bool
    peak_floor: float = float("nan")     # file's darkest per-frame peak
    peak_thresh: float | None = None     # peak at/below this = "blank" (None = guard off)
    extras: dict = field(default_factory=dict)


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if times.size == 0:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, times, values, left=values[0], right=values[-1])


# How long a dark/quiet level must persist to count as the file's "floor".
# This rejects a lone noise-dark frame while still capturing genuinely brief
# (~0.1s) breaks, which a percentile-based floor would miss entirely.
_FLOOR_SUSTAIN = 0.10   # seconds
# If the gap between the floor and the file's typical level is smaller than
# this, there's no reliable dark/quiet excursion to key on -> that signal is
# treated as uninformative (detection falls back to the other signal).
_MIN_SPREAD_LUMA = 8.0  # luma units (0..255)
_MIN_SPREAD_DB = 5.0    # dB


def _sustained_floor(values: np.ndarray, times: np.ndarray, sustain: float,
                     trim_frac: float = 0.0) -> float:
    """The lowest level held for at least `sustain` seconds (robust near-min).

    `trim_frac` discards that fraction of the most-extreme (lowest) samples
    before taking the floor. Use it ONLY for audio: a silent outro or a few
    digital-silence samples sit far below the content's real quiet level and,
    left in, pin the floor (and the derived threshold) absurdly low. Luma must
    NOT be trimmed — its darkest frames are the actual breaks we're hunting and
    are often well under 1% of the file, so trimming would erase them."""
    n = values.size
    if n == 0:
        return float("nan")
    span = float(times[-1] - times[0]) if times.size > 1 else 1.0
    rate = n / span if span > 0 else n
    k_sustain = int(round(sustain * rate))
    k_trim = int(round(trim_frac * n))
    k = max(1, min(max(k_sustain, k_trim), n))
    return float(np.partition(values, k - 1)[k - 1])


def _threshold(values, times, margin, sensitivity, min_spread, trim_frac=0.0):
    """Adaptive threshold anchored to the file's own floor, clamped so it can
    never reach the file's typical level (so flat content is never flagged).
    Returns -inf when the signal has no usable excursion."""
    floor = _sustained_floor(values, times, _FLOOR_SUSTAIN, trim_frac)
    typical = float(np.median(values))
    spread = typical - floor
    if spread < min_spread:
        return floor, float("-inf")
    return floor, floor + min(margin * sensitivity, 0.6 * spread)


def build_profile(sig: Signals, cfg: DetectConfig) -> Profile:
    if sig.v_luma.size:
        luma_floor, black_thresh = _threshold(
            sig.v_luma, sig.v_times, cfg.black_margin, cfg.sensitivity, _MIN_SPREAD_LUMA)
    else:
        luma_floor, black_thresh = 0.0, float("-inf")

    if sig.has_audio and sig.a_rms.size:
        rms_floor, quiet_thresh = _threshold(
            sig.a_rms, sig.a_times, cfg.quiet_margin, cfg.sensitivity, _MIN_SPREAD_DB,
            cfg.quiet_floor_trim)
    else:
        rms_floor, quiet_thresh = float("nan"), float("nan")

    # Peak (blank) guard: anchor to the file's *darkest* per-frame peak. During a
    # real blank/fade the brightest region is dark too, so the peak dips to this
    # floor; a scene on black keeps a bright peak and sits well above it. The
    # threshold is a flat floor+margin (not clamped to a fraction of the spread
    # like luma/audio) because we specifically want to allow the peak to rise a
    # fixed amount above pure-black without letting real content through.
    peak_floor, peak_thresh = float("nan"), None
    if cfg.blank_guard and getattr(sig, "v_peak", None) is not None and sig.v_peak.size:
        peak_floor = _sustained_floor(sig.v_peak, sig.v_times, _FLOOR_SUSTAIN)
        # Clamp to an absolute ceiling: the adaptive part alone would drift up on
        # a file with no true-dark frames (high darkest-peak floor) — and inflate
        # under --min-chapters escalation — until bright scenes slip through. The
        # cap means a frame with a genuinely bright region is never "blank", and
        # if even the darkest frame's peak sits above the cap, nothing qualifies.
        peak_thresh = min(peak_floor + cfg.bright_margin * cfg.sensitivity, cfg.peak_cap)

    return Profile(
        luma_floor=luma_floor,
        black_thresh=black_thresh,
        rms_floor=rms_floor,
        quiet_thresh=quiet_thresh,
        has_audio=sig.has_audio,
        peak_floor=peak_floor,
        peak_thresh=peak_thresh,
        extras={
            "luma_median": float(np.median(sig.v_luma)) if sig.v_luma.size else None,
            "rms_median": float(np.median(sig.a_rms)) if sig.a_rms.size else None,
        },
    )


def _combined_mask(luma_g, rms_g, prof, cfg, black_ok, quiet_ok, peak_g=None):
    """Build the per-grid-point break mask, plus the raw dark/quiet masks.

    In 'and' mode with near-miss enabled, a point also qualifies when one signal
    is *deep* past its own floor while the other only just misses its threshold.
    This rescues real breaks that a hard AND drops on a hair — e.g. a fade that
    hits the file's quiet floor but bottoms out at dark-grey (luma just over the
    black line), which is exactly how interior fades look on many VHS rips.

    Blank guard: when a per-frame peak signal is available, a frame is "dark"
    only if its peak is also low — so a busy scene on a black background (low
    average luma, bright objects) is not mistaken for a blank/fade frame.
    Returns (combined, black_mask, quiet_mask)."""
    shape = luma_g.shape
    black_mask = (luma_g <= prof.black_thresh) if black_ok else np.zeros(shape, bool)
    if (black_ok and cfg.blank_guard and peak_g is not None
            and prof.peak_thresh is not None and np.isfinite(prof.peak_thresh)):
        black_mask = black_mask & (peak_g <= prof.peak_thresh)
    quiet_mask = (rms_g <= prof.quiet_thresh) if quiet_ok else np.zeros(shape, bool)

    if black_ok and quiet_ok:
        if cfg.mode == "and":
            combined = black_mask & quiet_mask
            if cfg.near_miss:
                band_l = prof.black_thresh - prof.luma_floor
                band_a = prof.quiet_thresh - prof.rms_floor
                very_dark = luma_g <= prof.luma_floor + cfg.near_deep * band_l
                very_quiet = rms_g <= prof.rms_floor + cfg.near_deep * band_a
                nearly_dark = luma_g <= prof.black_thresh + cfg.near_tol * band_l
                nearly_quiet = rms_g <= prof.quiet_thresh + cfg.near_tol * band_a
                combined = combined | (very_dark & nearly_quiet) | (very_quiet & nearly_dark)
        else:
            combined = black_mask | quiet_mask
    elif black_ok:
        combined = black_mask
    elif quiet_ok:
        combined = quiet_mask
    else:
        combined = np.zeros(shape, bool)
    return combined, black_mask, quiet_mask


def detect(sig: Signals, cfg: DetectConfig) -> tuple[list[Break], Profile]:
    prof = build_profile(sig, cfg)
    duration = sig.duration or (float(sig.v_times[-1]) if sig.v_times.size else 0.0)
    if duration <= 0:
        return [], prof

    grid = np.arange(0.0, duration, cfg.grid_step)
    if grid.size == 0:
        return [], prof

    # A signal is "usable" only if build_profile found a real excursion in it
    # (finite threshold). This lets detection degrade gracefully to video-only
    # or audio-only when one signal is flat/uninformative — common on VHS where
    # the audio never truly quiets.
    black_ok = np.isfinite(prof.black_thresh)
    quiet_ok = (not cfg.ignore_audio and prof.has_audio
                and sig.a_rms.size > 0 and np.isfinite(prof.quiet_thresh))

    luma_g = _resample(sig.v_times, sig.v_luma, grid)
    if quiet_ok:
        rms_g = _resample(sig.a_times, sig.a_rms, grid)
    else:
        rms_g = np.full(grid.shape, np.nan)
    peak_g = (_resample(sig.v_times, sig.v_peak, grid)
              if getattr(sig, "v_peak", None) is not None and sig.v_peak.size else None)

    combined, black_mask, quiet_mask = _combined_mask(
        luma_g, rms_g, prof, cfg, black_ok, quiet_ok, peak_g)

    breaks = _runs_to_breaks(grid, combined, luma_g, rms_g, cfg)

    # Suppress breaks inside the lead-in / trailing windows (e.g. no chapters
    # in the first 300s of a show, or in the closing credits).
    if cfg.skip_start > 0 or cfg.skip_end > 0:
        end_cut = duration - cfg.skip_end if cfg.skip_end > 0 else duration
        breaks = [b for b in breaks if b.time >= cfg.skip_start and b.time <= end_cut]

    # Enforce minimum spacing: once a chapter is placed, drop any later
    # candidate that falls within min_gap seconds of it.
    breaks = _enforce_min_gap(breaks, cfg.min_gap)

    if cfg.add_intro:
        if not breaks or breaks[0].time > 1.0:
            breaks.insert(0, Break(
                time=0.0, start=0.0, end=0.0, duration=0.0,
                min_luma=float("nan"), min_rms=float("nan"),
                score=0.0, is_intro=True,
            ))

    if cfg.max_chapters is not None and len(breaks) > cfg.max_chapters:
        # Keep the intro + strongest breaks, then re-sort by time.
        intro = [b for b in breaks if b.is_intro]
        rest = sorted((b for b in breaks if not b.is_intro),
                      key=lambda b: b.score, reverse=True)
        keep = intro + rest[: max(0, cfg.max_chapters - len(intro))]
        breaks = sorted(keep, key=lambda b: b.time)

    return breaks, prof


def count_breaks(breaks: list[Break]) -> int:
    """Number of *detected* breaks, excluding the forced 00:00 intro marker."""
    return sum(1 for b in breaks if not b.is_intro)


def detect_to_target(sig: Signals, cfg: DetectConfig):
    """Run detection with automatic recovery when the first pass finds too few
    breaks. In order:

      1. Detect as configured (default: dark AND quiet).
      2. Audio fallback — if too few breaks were found *and* audio was the
         limiting factor (AND-mode with usable audio), retry video-only. On
         messy VHS/DVR sources the audio during a fade-to-black is often not
         quiet, so requiring it drops real breaks.
      3. Escalation — if a --min-chapters target is still unmet, loosen the
         thresholds (raise sensitivity) and shorten the minimum break duration
         toward min_duration_floor, retrying until met or attempts run out.
         If audio was the limiter (AND-mode, usable audio, but the video has a
         real dark excursion), escalation runs *video-only* — otherwise it would
         just crank sensitivity while still demanding a silence that never comes.

    Returns (breaks, profile, trace). Each trace entry is a dict with a "kind"
    of "fallback" or "escalate".
    """
    breaks, prof = detect(sig, cfg)
    trace: list[dict] = []
    # We always want at least one break; a --min-chapters target raises the bar.
    need = cfg.min_chapters if cfg.min_chapters > 0 else 1
    work = cfg

    # --- step 2: audio fallback -------------------------------------------
    audio_was_used = (not cfg.ignore_audio and prof.has_audio
                      and np.isfinite(prof.quiet_thresh))
    if (cfg.audio_fallback and cfg.mode == "and" and audio_was_used
            and count_breaks(breaks) < need):
        va_cfg = replace(work, ignore_audio=True)
        va_breaks, va_prof = detect(sig, va_cfg)
        if count_breaks(va_breaks) > count_breaks(breaks):
            work, breaks, prof = va_cfg, va_breaks, va_prof
            trace.append({"kind": "fallback", "found": count_breaks(breaks)})

    # --- step 3: --min-chapters escalation --------------------------------
    if cfg.min_chapters <= 0 or count_breaks(breaks) >= cfg.min_chapters:
        return breaks, prof, trace

    # If audio was the limiter — AND-mode with usable audio that still left us
    # short, while the video *does* have a real dark excursion — escalate
    # video-only. Cranking sensitivity while still requiring a silence that never
    # comes (music over the fades, or a digital-silent outro that pins the quiet
    # floor absurdly low) just spins uselessly to the attempt cap.
    if (not work.ignore_audio and cfg.mode == "and"
            and audio_was_used and np.isfinite(prof.black_thresh)):
        work = replace(work, ignore_audio=True)
        breaks, prof = detect(sig, work)
        trace.append({"kind": "fallback", "found": count_breaks(breaks)})
        if count_breaks(breaks) >= cfg.min_chapters:
            return breaks, prof, trace

    for attempt in range(1, cfg.escalate_attempts + 1):
        work = replace(
            work,
            sensitivity=work.sensitivity * 1.25,
            min_duration=max(cfg.min_duration_floor, work.min_duration * 0.6),
        )
        breaks, prof = detect(sig, work)
        found = count_breaks(breaks)
        trace.append({
            "kind": "escalate",
            "attempt": attempt,
            "sensitivity": round(work.sensitivity, 3),
            "min_duration": round(work.min_duration, 3),
            "found": found,
        })
        if found >= cfg.min_chapters:
            break
    return breaks, prof, trace


def _runs_to_breaks(grid, mask, luma_g, rms_g, cfg) -> list[Break]:
    breaks: list[Break] = []
    n = mask.size
    i = 0
    step = cfg.grid_step
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        start_t = float(grid[i])
        end_t = float(grid[j - 1]) + step
        dur = end_t - start_t
        if dur >= cfg.min_duration:
            seg_luma = luma_g[i:j]
            seg_rms = rms_g[i:j]
            min_luma = float(np.nanmin(seg_luma)) if seg_luma.size else float("nan")
            min_rms = float(np.nanmin(seg_rms)) if seg_rms.size and not np.all(np.isnan(seg_rms)) else float("nan")
            if cfg.mark_at == "mid":
                t = (start_t + end_t) / 2.0
            elif cfg.mark_at == "end":
                t = end_t
            else:
                t = start_t
            breaks.append(Break(
                time=t, start=start_t, end=end_t, duration=dur,
                min_luma=min_luma, min_rms=min_rms,
                score=dur,  # longer dark/quiet gap -> stronger signal
            ))
        i = j
    return breaks


def _enforce_min_gap(breaks: list[Break], min_gap: float) -> list[Break]:
    """Keep the earliest break, then reject any later break that falls within
    `min_gap` seconds of the last one we kept (anchored on the kept chapter,
    not on the rejected candidates)."""
    if not breaks or min_gap <= 0:
        return sorted(breaks, key=lambda b: b.time)
    breaks = sorted(breaks, key=lambda b: b.time)
    kept = [breaks[0]]
    for b in breaks[1:]:
        if b.time - kept[-1].time >= min_gap:
            kept.append(b)
        # else: last kept chapter was < min_gap ago -> skip this one
    return kept
