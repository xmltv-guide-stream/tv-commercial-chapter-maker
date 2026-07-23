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
from pathlib import Path

from .probe import Signals

CACHE_DIRNAME = ".chapterprofiles"


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

    def _entry_path(self, path: str) -> Path:
        return self.dir / f"{_key(path)}.json.gz"

    def get(self, path: str) -> Signals | None:
        p = self._entry_path(path)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            return Signals.from_dict(data["signals"])
        except (OSError, KeyError, ValueError):
            return None  # corrupt/partial cache -> just recompute

    def put(self, path: str, signals: Signals) -> None:
        p = self._entry_path(path)
        tmp = p.with_suffix(".tmp")
        payload = {"source": os.path.abspath(path), "signals": signals.as_dict()}
        # Write to a temp file then rename so an interrupted write never leaves
        # a half-valid cache entry. A cache failure is non-fatal — the profile
        # was already computed for this run; we just lose the speedup next time.
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            tmp.replace(p)
        except OSError as e:
            print(f"  ! could not cache profile: {e}")

    def clear(self) -> int:
        n = 0
        for f in self.dir.glob("*.json.gz"):
            try:
                f.unlink()
                n += 1
            except OSError as e:
                print(f"  ! could not delete {f.name}: {e}")
        return n
