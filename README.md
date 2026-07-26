# chaptermark — adaptive commercial-break chapter marker

Scans a folder of video files and inserts chapter markers where commercial
breaks likely are. Built specifically for **messy sources like VHS and DVR
recordings**, where the usual fixed black-level / silence detectors fail
because the picture never goes truly black and the audio never goes truly
silent (tape hiss, noisy capture).

## How it's different

Instead of asking you to pick a black level or a volume threshold, it
**profiles each file individually**: it finds that file's own *blackest black*
and *quietest quiet*, then looks for moments that dip toward those floors. A
tape whose "black" sits at luma 33 and whose "silence" floor is −50 dBFS still
works, because the thresholds are anchored to the file itself, not to absolute
values a clean broadcast would hit.

It also **caches each file's profile** (the expensive ffmpeg decode pass), so
if a batch is interrupted, re-running resumes instantly and re-tuning the
detection is near-instant — no re-decoding.

## Requirements

- **Python 3.9+**
- **ffmpeg / ffprobe** on your PATH — https://ffmpeg.org/ (the decode + analysis engine)
- **mkvpropedit** (from **MKVToolNix**, https://mkvtoolnix.download/) — only needed
  for `--embed`, to write chapters into the MKV in place. Without it, sidecar
  chapter files are still written.

## Install

```bash
# From the repo root — installs the `chaptermark` command and its numpy dependency:
pip install .

# Or, for development (editable install):
pip install -e .

# To also install the optional graphical tuner (PySide6):
pip install ".[gui]"
```

You can then run `chaptermark ...` from anywhere. If you'd rather not install,
run it in place with `python -m chaptermaker ...` after `pip install numpy`.

## Quick start

```bash
# Analyze a folder and see what it proposes — writes nothing:
chaptermark "D:\Tapes" --dry-run -v

# Write sidecar chapter files (.chapters.xml + .chapters.txt) next to each video:
chaptermark "D:\Tapes"

# Also embed the chapters directly into each .mkv (needs mkvpropedit):
chaptermark "D:\Tapes" --embed

# Point it at a whole show — it recurses into season subfolders by default:
chaptermark "D:\Shows\My Show"

# Only scan the top-level folder, ignore subfolders:
chaptermark "D:\Tapes" --no-recursive
```

## Graphical tuner (`--gui`)

If you installed the optional GUI (`pip install ".[gui]"`), add `--gui` to open a
visual tuner instead of running the scan:

```bash
chaptermark "D:\Shows\My Show" --gui
```

The window lists every file next to a timeline showing exactly where chapters
would land. Every command-line option is a control — sliders for thresholds,
checkboxes for on/off flags — and any options you pass on the command line
pre-populate them. The recommended workflow:

1. **Open a folder.** It reads the existing `.chapterprofiles` cache and draws
   the timelines immediately. It does **not** decode anything automatically.
   Profiles saved by an older version (before the format changed) still preview,
   but are shown in muted colours with an *"outdated"* tag and an amber filename —
   click **Analyze** to refresh them to the current sampling (the blank-guard /
   peak features stay off on a file until it's re-analyzed).
2. **Analyze.** For files with no cached profile yet (shown as *not analyzed*),
   click **Analyze** to run the one-time ffmpeg decode pass. This is the only
   slow step, and it runs in the background with a progress bar.
3. **Tune live.** Drag any slider or toggle any box — every file's timeline
   re-computes and redraws instantly, because detection runs on the cached
   signals (no re-decoding). Green tick = the 00:00 intro marker; orange tick =
   a detected break; gridlines every 5 minutes.
4. **Save / Write.** Writes the sidecar `.chapters.xml`/`.txt` files with the
   current settings. If the **Embed** checkbox is ticked, it also embeds the
   chapters into each `.mkv` in place (needs mkvpropedit).

Changing a **Sampling** control (video fps / audio window) requires a re-decode,
so the Analyze button will say *"Analyze (sampling changed)"* and re-profile on
the next run. The GUI never modifies files until you press Save.

> The examples below use the installed `chaptermark` command; `python -m chaptermaker`
> works identically if you didn't install.

Every run writes two sidecar files per video (per your setup, "both"):
`name.chapters.txt` (simple OGM format, hand-editable) and
`name.chapters.xml` (Matroska XML — what `--embed` feeds to mkvpropedit).

## Tuning (per run)

Nothing here is a fixed absolute threshold — every knob is relative to the
file's own floor, so the same settings travel across differently-degraded tapes.

| Flag | What it does | Default |
|------|--------------|---------|
| `--mode {and,or}` | `and` = a break must be **dark AND quiet** (fewest false positives); `or` = dark OR quiet (catches more) | `and` |
| `--ignore-audio` | Detect on **darkness alone** (video-only). Best when audio is an unreliable break signal — e.g. fade-to-black transitions where music/ambience is still playing | off |
| `--no-audio-fallback` | Disable the automatic video-only retry (see below) | off |
| `--sensitivity N` | Global multiplier on how far from the floor still counts. Higher = more breaks | `1.0` |
| `--black-margin N` | Luma units above the file's darkest level still counted as "black" | `14` |
| `--quiet-margin N` | dB above the file's quietest level still counted as "silent" | `8` |
| `--min-duration S` | A dark/quiet gap must last at least this long (seconds) | `0.30` |
| `--min-gap S` (`--min-spacing`) | Don't insert a chapter if the last kept one was fewer than S seconds ago | `45` |
| `--skip-start S` (`--no-chapters-before`) | No detected breaks before S seconds (e.g. `300` = skip the first 5 min). The 00:00 marker is unaffected — add `--no-intro` to drop it too | `0` |
| `--skip-end S` | No detected breaks within S seconds of the end (e.g. skip end credits) | `0` |
| `--mark-at {start,mid,end}` | Put the marker at the start/middle/end of the gap | `start` |
| `--max-chapters N` | Cap the number of chapters (keeps the strongest) | none |
| `--min-chapters N` | Minimum breaks to find. If fewer are found, automatically loosen thresholds **and** shorten `--min-duration` (down to `--min-duration-floor`), retrying until met or attempts run out | `0` (off) |
| `--min-duration-floor S` | Shortest gap the auto-escalation will accept — lower to catch very brief breaks (~0.1s) | `0.05` |
| `--no-intro` | Don't force a chapter at 00:00 | off |

Advanced sampling: `--video-fps` (brightness sample rate, default 8) and
`--audio-window` (audio RMS window seconds, default 0.1). Lower fps = faster
scans on huge files.

### Typical tuning workflow

1. Run with `--dry-run -v` and eyeball the proposed times.
2. Too few breaks? Raise `--sensitivity` (e.g. `1.3`) or switch `--mode or`.
3. Too many / spurious ones? Lower `--sensitivity`, raise `--min-duration`,
   or raise `--min-gap`.
4. Know roughly how many breaks a show has but the tape's breaks are very
   short/shallow? Set `--min-chapters N` and let it auto-loosen until it finds
   them (use `-v` to see what it settled on).
5. Happy with it? Drop `--dry-run` (add `--embed` to write into the MKVs).

The detector adapts to each file's own darkest/quietest *sustained* level, and
if one signal is flat (e.g. a VHS whose audio never quiets) it automatically
falls back to detecting on the other signal alone rather than finding nothing.

**Automatic audio fallback.** By default detection requires a break to be both
dark and quiet (`--mode and`). If that finds no breaks (or fewer than
`--min-chapters`) and audio was the limiting factor, the tool automatically
retries **video-only** — because on VHS/DVR fade-to-black transitions the audio
often isn't quiet, so requiring it would miss real breaks. You'll see
`fell back to video-only` in the output when this happens. Disable it with
`--no-audio-fallback`, or force video-only from the start with `--ignore-audio`.

Because profiles are cached, steps 1–3 don't re-decode anything — only the
first pass on each file is slow.

## Troubleshooting: breaks not detected

Point `--diagnose` at a single file you *know* has breaks to see the raw numbers
behind the decision (it writes nothing):

```bash
python -m chaptermaker "D:\Shows\My Show\S01E07.mkv" --diagnose
```

It prints the file's luma/audio distributions, the adaptive thresholds it chose,
what fraction of the file qualifies as dark / quiet / both, and a table of the
darkest moments flagged `dark?`/`quiet?`. Common findings:

- **Darkest moments show `dark? yes / quiet? no`** — the black frames are real but
  the audio there isn't quiet (common: fade-to-black with music still playing, or
  loud commercials right up to the cut). This also happens when a few near-silent
  spots drag `quiet_thresh` unrealistically low. Fix: `--ignore-audio` to detect on
  darkness alone (best for VHS/DVR), or `--mode or`.
- **`black_thresh = OFF`** — no reliable dark excursion was found. The break black
  may be very brief; raise `--video-fps` (e.g. 20) and/or `--reprofile`.
- **Dark moments exist but 0 breaks** — they're shorter than `--min-duration`;
  lower it (e.g. `0.1`) or use `--min-chapters N` to auto-loosen.

## Cache

Profiles live in a `.chapterprofiles/` folder inside the scanned directory,
keyed by file path + size + modification time (replace/re-capture a file and it
re-profiles automatically). Clear them with:

```bash
python -m chaptermaker "D:\Tapes" --clear-cache
```

## Notes & limitations

- Works on any format ffmpeg reads (`.mkv .mp4 .avi .ts .mpg .vob` …), but
  `--embed` only applies to `.mkv`; other formats get sidecars only.
- Detection is heuristic — always review with `--dry-run -v` on a new tape
  before trusting a big batch. The `.chapters.txt` sidecar is easy to hand-edit.
- A uniformly dark/quiet clip may be flagged as one long "break"; that's
  expected for degenerate inputs and harmless for real recordings.

## Project layout

```
chaptermaker/
  probe.py         ffmpeg/ffprobe -> per-file brightness & loudness time series
  profilecache.py  gzip-JSON cache of profiles (resume-safe)
  detect.py        adaptive thresholds + break-region logic (the core)
  chapters.py      OGM/XML sidecar writers + mkvpropedit embedding
  cli.py           argparse CLI / batch runner
```
