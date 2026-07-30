"""Persistent cache of probed signals so interrupted runs resume for free.

The expensive part of this tool is the ffmpeg decode pass in probe.py. Once we
have a file's brightness/loudness time series, we never need to decode it again
-- re-running with different detection thresholds is pure numpy on cached
arrays. That's the whole point of caching the *profile* rather than the result.

A cache entry is keyed by the file's absolute path + size + mtime, so if you
replace or re-capture a file the profile is transparently recomputed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

from .probe import Signals

CACHE_DIRNAME = ".chapterprofiles"

# Bump whenever the *meaning* of a cached signal changes (sampling resolution,
# a new/redefined array, etc.) so stale-format entries are transparently
# recomputed instead of silently mixing old numbers with new detection logic.
#   1: original luma+audio
#   2: added per-frame peak (95th @128x72)
#   3: peak now 99.9th @256x144 so faint highlights survive downscale
#   4: peak excludes a detected persistent overlay (channel logo/bug)
#   5: logo detected via low-percentile (was min) so a black intro/outro or a
#      few dark scenes over the logo no longer hide it
#   6: peak also excludes recurring corner overlays (rating bugs / brief bugs)
#   7: recurring-corner detection moved to the dense video pass + segment-counted
#      (keyframe-only was too sparse to catch brief fade bugs)
#   8: overlay-eligible region widened from tight corner boxes to a 25% edge band
#      (rating bugs are often inset from the exact corner)
#   9: dilate the detected overlay + lower the "lit" threshold, to cover a bug
#      that's brighter/bigger on one frame than its recurring core
#  10: per-frame overlay exclusion also sweeps the dim anti-aliased halo (exclude
#      everything above the dark-frame level inside the overlay neighborhood)
SIGNALS_VERSION = 10


def _key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class ProfileCache:
    def __init__(self, cache_dir: str | os.PathLike | None = None, base: str | None = None):
        # By default the cache lives alongside the videos being scanned.
        if cache_dir is None:
            root = base or os.getcwd()
            cache_dir = os.path.join(root, CACHE_DIRNAME)
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._src_index: dict[str, Path] | None = None

    def _entry_path(self, path: str) -> Path:
        return self.dir / f"{_key(path)}.json.gz"

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _peek_source(self, entry: Path) -> str | None:
        """Read just the "source" path from an entry without decoding the big
        signal arrays (source is written before them). Used to re-associate an
        entry whose stat-key no longer matches the file (e.g. after embedding
        changed its mtime)."""
        try:
            with gzip.open(entry, "rt", encoding="utf-8") as fh:
                head = fh.read(4096)
        except OSError:
            return None
        m = re.search(r'"source"\s*:\s*"((?:[^"\\]|\\.)*)"', head)
        if not m:
            return None
        try:
            return json.loads('"' + m.group(1) + '"')
        except ValueError:
            return None

    def _source_index(self) -> dict[str, Path]:
        """Map normalized source path -> entry file, built once and cached."""
        if self._src_index is None:
            idx: dict[str, Path] = {}
            for f in self.dir.glob("*.json.gz"):
                src = self._peek_source(f)
                if src:
                    idx[self._norm(src)] = f
            self._src_index = idx
        return self._src_index

    def entry_path(self, path: str) -> Path:
        """Public accessor for a file's cache entry path (keyed on current stat)."""
        return self._entry_path(path)

    def _load_raw(self, path: str) -> tuple[Signals | None, str]:
        """Load an entry regardless of version. Returns (signals, status) where
        status is 'missing', 'current', or 'stale' (loadable but an older
        format). Corrupt/unreadable entries come back as (None, 'missing')."""
        try:
            p = self._entry_path(path)   # stats the file to build its key
        except OSError:
            return None, "missing"       # file vanished -> nothing to load
        if not p.exists():
            return None, "missing"
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            sig = Signals.from_dict(data["signals"])
        except (OSError, KeyError, ValueError):
            return None, "missing"  # corrupt/partial -> treat as absent
        status = "current" if data.get("version") == SIGNALS_VERSION else "stale"
        return sig, status

    def get(self, path: str) -> Signals | None:
        """Return signals only if the cached profile is the current format.
        Stale/missing -> None, so callers (the CLI) transparently re-decode."""
        sig, status = self._load_raw(path)
        return sig if status == "current" else None

    def resolve_entry(self, path: str) -> Path | None:
        """The on-disk entry currently representing this file — the exact
        stat-key entry if present, else one matched by stored source path.
        Call *before* modifying the file (e.g. embedding) to capture it."""
        try:
            p = self._entry_path(path)
        except OSError:
            p = None
        if p is not None and p.exists():
            return p
        return self._source_index().get(self._norm(path))

    def rekey(self, old_entry, path: str) -> None:
        """Rename an existing entry to the file's *current* stat-key. Used after
        WE changed the file's mtime by embedding chapters: the audio/video is
        unchanged, so the (possibly stale-format) profile is still valid — this
        keeps it findable by exact key instead of orphaning it, preserving its
        stored contents and version (so an outdated profile stays outdated)."""
        if not old_entry:
            return
        try:
            new = self._entry_path(path)
        except OSError:
            return
        old = Path(old_entry)
        if old.exists() and old != new:
            try:
                os.replace(old, new)
                self._src_index = None
            except OSError:
                pass

    def get_any(self, path: str) -> tuple[Signals | None, str]:
        """Load even a stale-format profile (for previewing), with its status.
        Used by the GUI so old profiles still draw a timeline while being clearly
        flagged as outdated and re-analyzable.

        If the exact stat-key entry is missing, fall back to matching by the
        source path recorded inside each entry — this recovers profiles that
        were orphaned when the file's mtime/size changed (an old build, or a
        drive whose reported mtime drifts). The recovered profile reports its
        REAL format version: a current-format profile stays 'current' (only its
        stat drifted, not its data), and only a genuinely old format is 'stale'.
        Outdated must mean old format, never merely 'located by source match'."""
        sig, status = self._load_raw(path)
        if status != "missing":
            return sig, status
        entry = self._source_index().get(self._norm(path))
        if entry is None:
            return None, "missing"
        try:
            with gzip.open(entry, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            status = "current" if data.get("version") == SIGNALS_VERSION else "stale"
            return Signals.from_dict(data["signals"]), status
        except (OSError, KeyError, ValueError):
            return None, "missing"

    def put(self, path: str, signals: Signals) -> None:
        p = self._entry_path(path)
        tmp = p.with_suffix(".tmp")
        payload = {"version": SIGNALS_VERSION,
                   "source": os.path.abspath(path), "signals": signals.as_dict()}
        # Write to a temp file then rename so an interrupted write never leaves
        # a half-valid cache entry. A cache failure is non-fatal — the profile
        # was already computed for this run; we just lose the speedup next time.
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            tmp.replace(p)
            self._src_index = None  # a new entry exists; rebuild index on demand
        except OSError as e:
            print(f"  ! could not cache profile: {e}")

    def reput(self, path: str, signals: Signals, prior_entry: Path | None = None) -> None:
        """Re-store signals under the file's *current* stat, dropping a now-stale
        prior entry. Used after embedding chapters rewrites the MKV in place
        (changing its mtime/size) — without this the key would miss on the next
        run and force a needless re-decode of an already-processed file."""
        self.put(path, signals)
        if prior_entry is not None:
            current = self._entry_path(path)
            if prior_entry != current and prior_entry.exists():
                try:
                    prior_entry.unlink()
                except OSError:
                    pass

    def clear(self) -> int:
        n = 0
        for f in self.dir.glob("*.json.gz"):
            try:
                f.unlink()
                n += 1
            except OSError as e:
                print(f"  ! could not delete {f.name}: {e}")
        return n
