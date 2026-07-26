"""Command-line interface: scan a folder of videos and mark commercial breaks."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .chapters import (MkvpropeditNotFound, embed_chapters, to_ogm,
                       write_sidecars)
from .detect import (Break, DetectConfig, _FLOOR_SUSTAIN, _MIN_SPREAD_DB,
                     _MIN_SPREAD_LUMA, _combined_mask, _resample,
                     _sustained_floor, build_profile, count_breaks,
                     detect_to_target)
from .probe import FfmpegNotFound, probe_file
from .profilecache import ProfileCache

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".ts", ".m2ts", ".mpg", ".mpeg", ".vob", ".wmv", ".mov"}


def _fmt_clock(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def find_videos(root: Path, recursive: bool, exts: set[str]) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)


def build_config(args) -> DetectConfig:
    return DetectConfig(
        sensitivity=args.sensitivity,
        black_margin=args.black_margin,
        quiet_margin=args.quiet_margin,
        mode=args.mode,
        quiet_floor_trim=args.quiet_floor_trim,
        blank_guard=args.blank_guard,
        bright_margin=args.bright_margin,
        peak_cap=args.peak_cap,
        near_miss=args.near_miss,
        near_deep=args.near_deep,
        near_tol=args.near_tol,
        ignore_audio=args.ignore_audio,
        audio_fallback=args.audio_fallback,
        min_duration=args.min_duration,
        min_gap=args.min_gap,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        add_intro=not args.no_intro,
        mark_at=args.mark_at,
        max_chapters=args.max_chapters,
        min_chapters=args.min_chapters,
        min_duration_floor=args.min_duration_floor,
    )


def _get_signals(path: Path, cache: ProfileCache, args):
    sig = None if args.reprofile else cache.get(str(path))
    if sig is None:
        if args.verbose:
            print(f"  profiling (decoding)…", flush=True)
        sig = probe_file(str(path), video_fps=args.video_fps, audio_window=args.audio_window)
        cache.put(str(path), sig)
    elif args.verbose:
        print("  using cached profile", flush=True)
    return sig


def diagnose_file(path: Path, cache: ProfileCache, cfg: DetectConfig, args, root: Path) -> None:
    """Print a detailed brightness/loudness report for one file, to explain why
    breaks are (or aren't) being detected. Writes nothing."""
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    sig = _get_signals(path, cache, args)
    duration = sig.duration or (float(sig.v_times[-1]) if sig.v_times.size else 0.0)
    prof = build_profile(sig, cfg)

    print(f"\n=== DIAGNOSE: {rel} ===")
    print(f"duration={_fmt_clock(duration)}  mode={cfg.mode}  sensitivity={cfg.sensitivity}  "
          f"min_duration={cfg.min_duration}s  min_gap={cfg.min_gap}s  skip_start={cfg.skip_start}s")

    def stats(arr):
        return (f"min={np.min(arr):.1f} p1={np.percentile(arr,1):.1f} "
                f"p5={np.percentile(arr,5):.1f} median={np.median(arr):.1f} max={np.max(arr):.1f}")

    # --- video ---
    L = sig.v_luma
    if L.size:
        vspan = float(sig.v_times[-1] - sig.v_times[0]) if sig.v_times.size > 1 else 1.0
        vfps = L.size / vspan if vspan > 0 else L.size
        floor = _sustained_floor(L, sig.v_times, _FLOOR_SUSTAIN)
        spread = float(np.median(L)) - floor
        bt = prof.black_thresh
        print(f"\nLUMA (0=black..255)  samples={L.size} (~{vfps:.1f}/s)")
        print(f"  {stats(L)}")
        print(f"  sustained_floor={floor:.1f}  spread_to_median={spread:.1f} "
              f"(gate needs >= {_MIN_SPREAD_LUMA})")
        print(f"  black_thresh = {'OFF (no dark excursion)' if not np.isfinite(bt) else f'{bt:.1f}'}")
        if prof.peak_thresh is not None:
            capped = prof.peak_thresh >= cfg.peak_cap - 1e-9
            capnote = f" [hit absolute cap {cfg.peak_cap:.0f}]" if capped else ""
            print(f"  peak(blank-guard): darkest-peak floor={prof.peak_floor:.1f}  "
                  f"peak_thresh={prof.peak_thresh:.1f} (a dark frame must peak <= this){capnote}")
            if prof.peak_floor > cfg.peak_cap:
                print(f"    note: even the darkest frame's peak ({prof.peak_floor:.1f}) exceeds the cap "
                      f"-> no frame is 'blank' -> no dark breaks (file never truly goes dark)")
        elif cfg.blank_guard:
            print("  peak(blank-guard): no peak data (legacy cache) — re-run with --reprofile to enable")
    else:
        print("\nLUMA: no video samples!")

    # --- audio ---
    R = sig.a_rms
    if sig.has_audio and R.size:
        raw_floor = _sustained_floor(R, sig.a_times, _FLOOR_SUSTAIN)
        floor = _sustained_floor(R, sig.a_times, _FLOOR_SUSTAIN, cfg.quiet_floor_trim)
        spread = float(np.median(R)) - floor
        qt = prof.quiet_thresh
        print(f"\nAUDIO RMS (dBFS)  samples={R.size}")
        print(f"  {stats(R)}")
        trimnote = (f"  (raw min-floor {raw_floor:.1f}dB trimmed {cfg.quiet_floor_trim:.1%} "
                    f"-> ignores silent-outro/digital-silence)" if cfg.quiet_floor_trim > 0
                    and abs(floor - raw_floor) > 0.05 else "")
        print(f"  quiet_floor={floor:.1f}dB  spread_to_median={spread:.1f}dB "
              f"(gate needs >= {_MIN_SPREAD_DB}){trimnote}")
        print(f"  quiet_thresh = {'OFF (no quiet excursion)' if not np.isfinite(qt) else f'{qt:.1f}dB'}")
    else:
        print("\nAUDIO: none / not usable -> detection is video-only")

    # --- grid masks: how much qualifies, and under which rule ---
    grid = np.arange(0.0, duration, cfg.grid_step)
    luma_g = _resample(sig.v_times, sig.v_luma, grid)
    black_ok = np.isfinite(prof.black_thresh)
    quiet_ok = sig.has_audio and R.size > 0 and np.isfinite(prof.quiet_thresh)
    if quiet_ok:
        rms_g = _resample(sig.a_times, sig.a_rms, grid)
    else:
        rms_g = np.full(grid.shape, np.nan)
    peak_g = (_resample(sig.v_times, sig.v_peak, grid)
              if getattr(sig, "v_peak", None) is not None and sig.v_peak.size else None)
    # raw dark (mean only), to show what the blank guard is rejecting
    raw_dark = (luma_g <= prof.black_thresh) if black_ok else np.zeros(grid.shape, bool)
    combined_mask, black_mask, quiet_mask = _combined_mask(
        luma_g, rms_g, prof, cfg, black_ok, quiet_ok, peak_g)
    pct = lambda m: 100.0 * np.count_nonzero(m) / max(1, m.size)
    nm = ""
    if cfg.near_miss and cfg.mode == "and" and black_ok and quiet_ok:
        rescued = np.count_nonzero(combined_mask & ~(black_mask & quiet_mask))
        nm = f"  near-miss rescues={pct(combined_mask & ~(black_mask & quiet_mask)):.2f}%"
    bg = ""
    if peak_g is not None and prof.peak_thresh is not None:
        blocked = np.count_nonzero(raw_dark & ~black_mask)
        bg = f"  blank-guard rejects={pct(raw_dark & ~black_mask):.2f}% (dark-average but bright peak)"
    print(f"\nGRID (@{cfg.grid_step}s)  dark={pct(black_mask):.2f}%  quiet={pct(quiet_mask):.2f}%  "
          f"dark&quiet={pct(black_mask & quiet_mask):.2f}%  dark|quiet={pct(black_mask | quiet_mask):.2f}%"
          f"  kept={pct(combined_mask):.2f}%{nm}{bg}")

    # --- the darkest moments, and what the audio is doing there ---
    have_peak = peak_g is not None
    pkhdr = f"  {'peak':>5}" if have_peak else ""
    print("\nDarkest moments (spaced >=2s apart):")
    print(f"  {'time':>8}  {'luma':>6}{pkhdr}  {'rms(dB)':>8}  dark? quiet? keep?")
    kept_t: list[float] = []
    for i in np.argsort(luma_g):
        t = float(grid[i])
        if any(abs(t - kt) < 2.0 for kt in kept_t):
            continue
        kept_t.append(t)
        d = "yes" if black_mask[i] else "no "
        q = "yes" if quiet_ok and quiet_mask[i] else "no "
        k = "yes" if combined_mask[i] else "no "
        raw_and = black_mask[i] and quiet_mask[i]
        if have_peak and raw_dark[i] and not black_mask[i]:
            flag = " <-blank-guard (has content)"
        elif combined_mask[i] and not raw_and:
            flag = " <-near-miss"
        else:
            flag = ""
        rv = f"{rms_g[i]:.1f}" if quiet_ok else "n/a"
        pk = f"  {peak_g[i]:5.1f}" if have_peak else ""
        print(f"  {_fmt_clock(t):>8}  {luma_g[i]:6.1f}{pk}  {rv:>8}   {d}   {q}   {k}{flag}")
        if len(kept_t) >= 15:
            break

    # --- interpretation hint ---
    breaks, _, _ = detect_to_target(sig, cfg)
    print(f"\nWith current settings -> {count_breaks(breaks)} breaks detected.")
    if cfg.mode == "and" and black_ok and quiet_ok:
        dark_not_quiet = np.count_nonzero(black_mask & ~quiet_mask)
        if dark_not_quiet > np.count_nonzero(black_mask & quiet_mask):
            print("  Hint: many dark moments are NOT quiet — 'and' mode is dropping real breaks.")
            print("        The audio is an unreliable signal here; try --ignore-audio "
                  "(video-only), or --mode or.")
    if black_ok and np.count_nonzero(black_mask) and count_breaks(breaks) == 0:
        print("  Hint: dark moments exist but none survived. Try lowering --min-duration "
              "(e.g. 0.1) or --min-gap, or raise --sensitivity.")
    print("=== END DIAGNOSE ===\n")


def process_file(path: Path, cache: ProfileCache, cfg: DetectConfig, args, root: Path) -> int:
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    # 1. Profile (from cache if available) -----------------------------------
    sig = _get_signals(path, cache, args)

    # 2. Detect (auto-loosening if a minimum chapter count is requested) ------
    breaks, prof, trace = detect_to_target(sig, cfg)

    fell_back = any(t.get("kind") == "fallback" for t in trace)
    escalations = [t for t in trace if t.get("kind") == "escalate"]

    notes = []
    if fell_back:
        notes.append("audio unreliable — fell back to video-only")
    if cfg.min_chapters > 0 and escalations:
        found = count_breaks(breaks)
        last = escalations[-1]
        if found >= cfg.min_chapters:
            notes.append(f"loosened to sensitivity {last['sensitivity']}, "
                         f"min-duration {last['min_duration']}s")
        else:
            notes.append(f"! only found {found}/{cfg.min_chapters} after "
                         f"{len(escalations)} loosening rounds")
    note = ("  (" + "; ".join(notes) + ")") if notes else ""

    audio_ignored = cfg.ignore_audio or fell_back
    b_str = f"{prof.black_thresh:.1f}luma" if math.isfinite(prof.black_thresh) else "off"
    q_str = ("ignored" if audio_ignored
             else (f"{prof.quiet_thresh:.1f}dB" if math.isfinite(prof.quiet_thresh) else "off"))
    print(f"{rel}  [{_fmt_clock(sig.duration)}]  "
          f"black<= {b_str}  quiet<= {q_str}  "
          f"-> {len(breaks)} chapters{note}")

    if trace and args.verbose:
        for t in trace:
            if t.get("kind") == "fallback":
                print(f"      fell back to video-only (audio ignored) -> {t['found']} breaks")
            else:
                print(f"      escalate #{t['attempt']}: sensitivity={t['sensitivity']} "
                      f"min-duration={t['min_duration']}s -> {t['found']} breaks")

    if args.verbose or args.dry_run:
        for i, b in enumerate(breaks, 1):
            tag = "intro" if b.is_intro else f"{b.duration:.1f}s dark/quiet"
            print(f"    {i:>3}. {_fmt_clock(b.time):>8}  ({tag})")

    if args.dry_run:
        return len(breaks)

    # 3. Write sidecars ------------------------------------------------------
    paths = write_sidecars(str(path), breaks)
    if args.verbose:
        print(f"    wrote {Path(paths['xml']).name} + {Path(paths['ogm']).name}")

    # 4. Optionally embed ----------------------------------------------------
    if args.embed and path.suffix.lower() == ".mkv":
        prior_entry = cache.entry_path(str(path))  # keyed on the pre-embed stat
        try:
            embed_chapters(str(path), paths["xml"])
            # Embedding rewrites the MKV -> mtime/size change would invalidate the
            # cache key. Re-store the (unchanged) signals under the new stat so
            # re-runs stay warm instead of re-decoding every embedded file.
            cache.reput(str(path), sig, prior_entry)
            if args.verbose:
                print("    embedded chapters into MKV")
        except MkvpropeditNotFound as e:
            print(f"    ! {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover - surfacing mkvpropedit errors
            print(f"    ! failed to embed: {e}", file=sys.stderr)
    elif args.embed and path.suffix.lower() != ".mkv":
        print("    (embed skipped: not an MKV — sidecars written)")

    return len(breaks)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chaptermark",
        description="Adaptively insert chapter markers at likely commercial "
                    "breaks in video files (built for messy VHS/DVR rips).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", nargs="?", default=None,
                   help="Folder to scan for video files (optional when launching --gui)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--gui", action="store_true",
                   help="Launch the graphical tuner instead of scanning. Any other options passed pre-populate the controls; nothing runs until you click Analyze / Save in the window")

    scan = p.add_argument_group("scanning")
    scan.add_argument("-r", "--recursive", dest="recursive", action="store_true", default=True,
                      help="Recurse into subfolders, e.g. a show folder with season subfolders (default)")
    scan.add_argument("--no-recursive", dest="recursive", action="store_false",
                      help="Only scan the top-level folder, ignore subfolders")
    scan.add_argument("--ext", action="append", default=None,
                      help="Extra file extension to include (repeatable), e.g. --ext .mkv")

    tune = p.add_argument_group("detection tuning")
    tune.add_argument("--mode", choices=["and", "or"], default="and",
                      help="'and' = dark AND quiet; 'or' = dark OR quiet")
    tune.add_argument("--no-near-miss", dest="near_miss", action="store_false", default=True,
                      help="Disable near-miss rescue: normally, in 'and' mode, a point where one signal is deep past its floor and the other only just misses its threshold still counts (catches fades that hit true-quiet but only dark-grey, or vice-versa)")
    tune.add_argument("--near-deep", type=float, default=0.5,
                      help="Near-miss: fraction of a signal's floor-to-threshold band, measured from the floor, that counts as 'deep past floor' (lower = stricter)")
    tune.add_argument("--near-tol", type=float, default=0.25,
                      help="Near-miss: fraction of the band past the threshold the other signal may sit and still count as 'nearly met' (higher = more forgiving)")
    tune.add_argument("--ignore-audio", action="store_true",
                      help="Detect on darkness alone (video-only). Best when audio is an unreliable break signal — common on VHS/DVR fade-to-black transitions")
    tune.add_argument("--no-audio-fallback", dest="audio_fallback", action="store_false",
                      help="Disable the automatic retry: normally, if 'and' mode finds no breaks, the tool falls back to video-only")
    tune.add_argument("--sensitivity", type=float, default=1.0,
                      help="Global multiplier on how far from the file's floor still counts (higher = more breaks)")
    tune.add_argument("--black-margin", type=float, default=14.0,
                      help="Luma units above the file's darkest level still counted as 'black'")
    tune.add_argument("--no-blank-guard", dest="blank_guard", action="store_false", default=True,
                      help="Disable the blank guard: normally a frame must be dark at its brightest region too, so a scene on a black background (bright objects, dark background) isn't mistaken for a blank/fade frame")
    tune.add_argument("--bright-margin", type=float, default=32.0,
                      help="Blank guard: how far a frame's peak brightness may rise above the file's darkest-peak level and still count as 'blank' (lower = stricter about rejecting on-black content)")
    tune.add_argument("--peak-cap", type=float, default=100.0,
                      help="Blank guard: absolute peak-brightness ceiling (0..255). A frame brighter than this at its peak is never treated as blank, so a file with no truly dark frames gets no dark breaks instead of marking its dimmest scenes")
    tune.add_argument("--quiet-margin", type=float, default=8.0,
                      help="dB above the file's quietest level still counted as 'silent'")
    tune.add_argument("--quiet-floor-trim", type=float, default=0.01,
                      help="Ignore this fraction of the quietest audio samples when setting the quiet floor, so a silent outro or a few digital-silence samples don't drag the threshold unrealistically low (0 = off; audio only)")
    tune.add_argument("--min-duration", type=float, default=0.30,
                      help="Minimum seconds a dark/quiet gap must last to count")
    tune.add_argument("--min-gap", "--min-spacing", dest="min_gap", type=float, default=45.0,
                      help="Don't insert a chapter if the last kept one was fewer than this many seconds ago")
    tune.add_argument("--skip-start", "--no-chapters-before", dest="skip_start", type=float, default=0.0,
                      help="Suppress detected breaks before this time, e.g. 300 = no chapters in the first 5 min (the 00:00 marker is unaffected; use --no-intro to drop it too)")
    tune.add_argument("--skip-end", "--no-chapters-after-before-end", dest="skip_end", type=float, default=0.0,
                      help="Suppress detected breaks within this many seconds of the end (e.g. skip end credits)")
    tune.add_argument("--mark-at", choices=["start", "mid", "end"], default="mid",
                      help="Place the marker at the start/middle/end of the dark-quiet gap")
    tune.add_argument("--max-chapters", type=int, default=None,
                      help="Cap the number of chapters (keeps strongest)")
    tune.add_argument("--min-chapters", type=int, default=0,
                      help="Minimum breaks to find. If fewer are found, automatically loosen thresholds and shorten min-duration (retrying) until met")
    tune.add_argument("--min-duration-floor", type=float, default=0.05,
                      help="Shortest dark/quiet gap auto-escalation will accept (seconds) — lower this to catch very brief breaks")
    tune.add_argument("--no-intro", action="store_true",
                      help="Do not force a chapter at 00:00")

    sampling = p.add_argument_group("sampling (advanced)")
    sampling.add_argument("--video-fps", type=float, default=12.0,
                          help="Frames per second sampled for brightness analysis (higher catches briefer breaks; decode cost is nearly unchanged)")
    sampling.add_argument("--audio-window", type=float, default=0.1,
                          help="Audio RMS window length in seconds")

    out = p.add_argument_group("output")
    out.add_argument("--embed", action="store_true",
                     help="Embed chapters into the MKV in place (needs mkvpropedit)")
    out.add_argument("--dry-run", action="store_true",
                     help="Analyze and print proposed chapters without writing anything")
    out.add_argument("--reprofile", action="store_true",
                     help="Ignore cached profiles and re-decode")
    out.add_argument("--clear-cache", action="store_true",
                     help="Delete cached profiles for this folder and exit")
    out.add_argument("--diagnose", action="store_true",
                     help="Print a detailed brightness/loudness report for each file (to troubleshoot why breaks are/aren't found) and exit without writing")
    out.add_argument("-v", "--verbose", action="store_true", help="Per-chapter detail")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # GUI mode: hand off to the (optional) PySide6 tuner. Imported lazily so the
    # CLI has no GUI dependency and its behaviour is entirely unchanged.
    if args.gui:
        try:
            from . import gui
        except ImportError as e:
            print(f"The GUI needs PySide6, which isn't installed.\n"
                  f"  Install it with:  pip install PySide6\n  ({e})", file=sys.stderr)
            return 4
        start = Path(args.folder).expanduser() if args.folder else None
        return gui.launch(start, build_config(args), args)

    if args.folder is None:
        print("error: a folder is required (or pass --gui to open the tuner)", file=sys.stderr)
        return 2
    target = Path(args.folder).expanduser()
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        return 2

    # Accept either a folder or a single file (handy for --diagnose on one file).
    single_file = target.is_file()
    root = target.parent if single_file else target
    cache = ProfileCache(base=str(root))

    if args.clear_cache:
        n = cache.clear()
        print(f"Cleared {n} cached profile(s).")
        return 0

    exts = set(VIDEO_EXTS)
    if args.ext:
        exts |= {e if e.startswith(".") else "." + e for e in args.ext}

    if single_file:
        videos = [target]
    else:
        videos = find_videos(root, args.recursive, exts)
    if not videos:
        print(f"No video files found in {target}")
        return 0

    cfg = build_config(args)

    # Diagnose mode: print detailed reports and exit, writing nothing.
    if args.diagnose:
        for path in videos:
            try:
                diagnose_file(path, cache, cfg, args, root)
            except FfmpegNotFound as e:
                print(f"\n{e}", file=sys.stderr)
                return 3
            except Exception as e:
                print(f"  ! error diagnosing {path.name}: {e}", file=sys.stderr)
        return 0

    print(f"Scanning {len(videos)} file(s) in {target}\n")

    total = 0
    for i, path in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] ", end="")
        try:
            total += process_file(path, cache, cfg, args, root)
        except FfmpegNotFound as e:
            print(f"\n{e}", file=sys.stderr)
            return 3
        except KeyboardInterrupt:
            print("\nInterrupted — cached profiles are kept; re-run to resume.", file=sys.stderr)
            return 130
        except Exception as e:  # keep going on a bad file
            print(f"  ! error on {path.name}: {e}", file=sys.stderr)

    print(f"\nDone. {total} chapter markers across {len(videos)} file(s).")
    if not args.embed and not args.dry_run:
        print("Sidecar .chapters.xml/.txt written next to each video. "
              "Re-run with --embed to write them into the MKVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
