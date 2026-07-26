"""Optional PySide6 GUI for tuning detection and previewing chapter placement.

Launched via ``chaptermark <folder> --gui``. This module is imported lazily by
the CLI, so the command-line tool keeps zero GUI dependencies and its behaviour
is entirely unchanged when the GUI isn't used.

Design goals (see cli.py's --gui handoff):
  * Every CLI option is a control — sliders for numeric thresholds, checkboxes
    for on/off flags, combos for choices — pre-populated from whatever flags
    were passed on the command line.
  * Nothing decodes automatically. On opening a folder we only *read* the
    existing .chapterprofiles cache (instant) and draw a per-file timeline of
    where chapters would land. Files with no cached profile show "not analyzed".
  * "Analyze" runs the (expensive) ffmpeg decode pass, in a background thread,
    only for files missing from the cache (or all files if sampling changed).
  * Because detection is pure numpy over the cached signals, every slider move
    re-detects across all loaded files in milliseconds and redraws live —
    without touching disk.
  * "Save / Write" applies the current settings to disk (sidecars always; embed
    into the MKV only if the Embed checkbox is ticked), also in a thread.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .chapters import MkvpropeditNotFound, embed_chapters, write_sidecars
from .cli import VIDEO_EXTS, find_videos
from .detect import DetectConfig, count_breaks, detect_to_target
from .probe import probe_file
from .profilecache import ProfileCache


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _decimals(step: float) -> int:
    txt = f"{step:.6f}".rstrip("0").rstrip(".")
    return len(txt.split(".")[1]) if "." in txt else 0


class FloatControl(QWidget):
    """A labelled slider + spinbox kept in sync, emitting `changed` on any edit."""

    changed = Signal()

    def __init__(self, label, lo, hi, step, value, integer=False, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.step, self.integer = lo, hi, step, integer
        self._n = max(1, int(round((hi - lo) / step)))
        self._guard = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        name = QLabel(label)
        name.setMinimumWidth(118)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._n)
        if integer:
            self.spin = QSpinBox()
            self.spin.setRange(int(lo), int(hi))
            self.spin.setSingleStep(max(1, int(step)))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(lo, hi)
            self.spin.setSingleStep(step)
            self.spin.setDecimals(_decimals(step))
        self.spin.setMaximumWidth(88)

        lay.addWidget(name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)

        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_slider(self, v):
        return max(0, min(self._n, int(round((v - self.lo) / self.step))))

    def value(self):
        v = self.spin.value()
        return int(v) if self.integer else float(v)

    def set_value(self, v):
        self._guard = True
        self.spin.setValue(v)
        self.slider.setValue(self._to_slider(v))
        self._guard = False

    def _from_slider(self, s):
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(self.lo + s * self.step)
        self._guard = False
        self.changed.emit()

    def _from_spin(self, v):
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(self._to_slider(v))
        self._guard = False
        self.changed.emit()


class TimelineView(QWidget):
    """A horizontal bar for one file: minor gridlines every 5 min, plus a tick
    at each detected chapter (green = 00:00 intro, orange = detected break)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 0.0
        self.marks: list[tuple[float, bool]] | None = None
        self.stale = False
        self.placeholder = "not analyzed"
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def set_not_analyzed(self):
        self.duration = 0.0
        self.marks = None
        self.stale = False
        self.setToolTip("")
        self.update()

    def set_data(self, duration, marks, stale=False):
        self.duration = float(duration or 0.0)
        self.marks = list(marks)
        self.stale = stale
        if stale:
            self.setToolTip("Preview from an OUTDATED profile — click Analyze to "
                            "refresh (blank-guard/peak features are off until you do)")
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(1, 1, -2, -2)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(28, 30, 34))
        p.drawRoundedRect(r, 4, 4)

        if self.duration <= 0 or self.marks is None:
            p.setPen(QColor(120, 122, 128))
            p.drawText(r, Qt.AlignCenter, self.placeholder)
            return

        x0, y0, w, h = r.left(), r.top(), r.width(), r.height()
        p.setPen(QPen(QColor(52, 55, 61), 1))
        t = 300.0
        while t < self.duration:
            x = x0 + w * (t / self.duration)
            p.drawLine(int(x), y0 + 3, int(x), y0 + h - 3)
            t += 300.0

        # Stale profiles preview in muted colours; current ones are bright.
        intro_c = QColor(70, 120, 84) if self.stale else QColor(90, 180, 110)
        break_c = QColor(150, 110, 70) if self.stale else QColor(240, 150, 70)
        for tm, is_intro in self.marks:
            x = x0 + w * (tm / max(self.duration, 1e-9))
            p.setPen(QPen(intro_c if is_intro else break_c, 2))
            p.drawLine(int(x), y0 + 1, int(x), y0 + h - 1)

        if self.stale:
            p.setPen(QColor(210, 170, 90))
            p.drawText(r.adjusted(0, 0, -4, 0), Qt.AlignRight | Qt.AlignVCenter, "outdated")

    def mouseMoveEvent(self, e):
        if self.duration > 0 and self.marks is not None:
            w = max(1, self.width() - 3)
            frac = (e.position().x() - 1) / w
            t = max(0.0, min(self.duration, frac * self.duration))
            near = None
            for tm, _ in self.marks:
                if near is None or abs(tm - t) < abs(near - t):
                    near = tm
            if near is not None and abs(near - t) / max(self.duration, 1e-9) * w < 6:
                self.setToolTip(f"chapter @ {_hms(near)}")
            else:
                self.setToolTip(_hms(t))
        super().mouseMoveEvent(e)


# --------------------------------------------------------------------------- #
#  background workers
# --------------------------------------------------------------------------- #

class ProbeWorker(QThread):
    """Loads cached signals and/or decodes files, off the UI thread.

    mode: 'cache'     -> only read the cache; missing files stay not-analyzed
          'analyze'   -> read cache; decode any file that's missing
          'reprofile' -> decode every file, ignoring the cache
    """

    progress = Signal(int, int, str)   # done, total, current name
    fileReady = Signal(str, object, str)  # path, Signals | None, status
    failed = Signal(str, str)          # path, message
    finishedAll = Signal()

    def __init__(self, paths, cache, video_fps, audio_window, mode):
        super().__init__()
        self.paths = list(paths)
        self.cache = cache
        self.video_fps = video_fps
        self.audio_window = audio_window
        self.mode = mode
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        n = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._stop:
                break
            self.progress.emit(i, n, os.path.basename(path))
            try:
                if self.mode == "reprofile":
                    sig, status = None, "missing"
                else:
                    sig, status = self.cache.get_any(path)
                # 'analyze' refreshes anything not current (missing OR stale);
                # 'reprofile' refreshes everything; 'cache' only reads.
                need = (self.mode == "reprofile"
                        or (self.mode == "analyze" and status != "current"))
                if need:
                    sig = probe_file(path, video_fps=self.video_fps,
                                     audio_window=self.audio_window)
                    self.cache.put(path, sig)
                    status = "current"
                self.fileReady.emit(path, sig, status)
            except Exception as e:  # keep going on a bad file
                self.failed.emit(path, str(e))
        self.progress.emit(n, n, "")
        self.finishedAll.emit()


class SaveWorker(QThread):
    """Writes sidecars (and optionally embeds) for every loaded file."""

    progress = Signal(int, int, str)
    failed = Signal(str, str)
    finishedAll = Signal(int, bool)    # count written, mkvpropedit_missing

    def __init__(self, items, cfg, embed, cache):
        super().__init__()
        self.items = list(items)       # list[(path, Signals, status)]
        self.cfg = cfg
        self.embed = embed
        self.cache = cache

    def run(self):
        n = len(self.items)
        written = 0
        mkv_missing = False
        for i, (path, sig, status) in enumerate(self.items):
            self.progress.emit(i, n, os.path.basename(path))
            try:
                breaks, _, _ = detect_to_target(sig, self.cfg)
                paths = write_sidecars(path, breaks)
                if self.embed and path.lower().endswith(".mkv"):
                    try:
                        # Capture the entry BEFORE embedding changes the mtime.
                        prior = self.cache.resolve_entry(path)
                        embed_chapters(path, paths["xml"])
                        # Embedding doesn't change the audio/video, so carry the
                        # profile forward to the new stat-key rather than orphan
                        # it. Current profiles are rewritten fresh; a stale one is
                        # just re-keyed so it stays valid but still flagged old.
                        if status == "current":
                            self.cache.reput(path, sig, prior)
                        else:
                            self.cache.rekey(prior, path)
                    except MkvpropeditNotFound:
                        mkv_missing = True
                written += 1
            except Exception as e:
                self.failed.emit(path, str(e))
        self.progress.emit(n, n, "")
        self.finishedAll.emit(written, mkv_missing)


# --------------------------------------------------------------------------- #
#  main window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    def __init__(self, start_folder, cfg: DetectConfig, args):
        super().__init__()
        self.setWindowTitle("chaptermaker — chapter tuner")
        self.resize(1180, 720)
        self.args = args
        self.cfg0 = cfg
        self.cache: ProfileCache | None = None
        self.folder: str | None = None
        self.exts = set(VIDEO_EXTS)
        if getattr(args, "ext", None):
            self.exts |= {e if e.startswith(".") else "." + e for e in args.ext}
        self.signals: dict[str, object] = {}   # path -> Signals | None
        self.file_status: dict[str, str] = {}  # path -> missing|current|stale
        self.rows: dict[str, dict] = {}         # path -> {name,count,timeline}
        self.floats: dict[str, FloatControl] = {}
        self.checks: dict[str, QCheckBox] = {}
        self.combos: dict[str, QComboBox] = {}
        self._worker: QThread | None = None
        self._sampling_dirty = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._recompute_all)

        self._build_ui()
        self._populate_from(cfg, args)

        if start_folder and os.path.isdir(str(start_folder)):
            self._set_folder(str(start_folder))

    # -- UI construction --------------------------------------------------- #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left: controls in a scroll area
        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(8, 8, 8, 8)

        g = self._group("Detection", cl)
        self._combo(g, "mode", "mode", ["and", "or"])
        self._float(g, "sensitivity", "sensitivity", 0.1, 5.0, 0.05)
        self._float(g, "black_margin", "black margin", 0.0, 64.0, 0.5)
        self._float(g, "grid_step", "grid step (s)", 0.05, 0.5, 0.05)
        self._combo(g, "mark_at", "mark at", ["start", "mid", "end"])
        self._check(g, "add_intro", "add 00:00 intro marker")
        self._check(g, "near_miss", "near-miss rescue")
        self._float(g, "near_deep", "near: deep frac", 0.0, 1.0, 0.05)
        self._float(g, "near_tol", "near: tol frac", 0.0, 1.0, 0.05)

        g = self._group("Audio", cl)
        self._check(g, "ignore_audio", "ignore audio (video-only)")
        self._check(g, "audio_fallback", "auto fall back to video-only")
        self._float(g, "quiet_margin", "quiet margin (dB)", 0.0, 30.0, 0.5)
        self._float(g, "quiet_floor_trim", "quiet-floor trim", 0.0, 0.1, 0.005)

        g = self._group("Blank guard", cl)
        self._check(g, "blank_guard", "reject bright-content frames")
        self._float(g, "bright_margin", "bright margin", 0.0, 128.0, 1.0)
        self._float(g, "peak_cap", "peak cap (abs)", 0.0, 255.0, 1.0)

        g = self._group("Timing / spacing", cl)
        self._float(g, "min_duration", "min duration (s)", 0.05, 5.0, 0.05)
        self._float(g, "min_gap", "min gap (s)", 0.0, 600.0, 5.0)
        self._float(g, "skip_start", "skip start (s)", 0.0, 900.0, 5.0)
        self._float(g, "skip_end", "skip end (s)", 0.0, 900.0, 5.0)
        self._float(g, "max_chapters", "max chapters (0=off)", 0, 50, 1, integer=True)

        g = self._group("Escalation (--min-chapters)", cl)
        self._float(g, "min_chapters", "min chapters (0=off)", 0, 20, 1, integer=True)
        self._float(g, "min_duration_floor", "min-dur floor (s)", 0.01, 1.0, 0.01)

        g = self._group("Sampling — needs re-analyze", cl)
        self._float(g, "video_fps", "video fps", 1.0, 30.0, 1.0, live=False)
        self._float(g, "audio_window", "audio window (s)", 0.02, 0.5, 0.01, live=False)

        g = self._group("Output", cl)
        self._check(g, "embed", "embed chapters into MKV on Save", live=False)
        self._check(g, "recursive", "scan subfolders", live=False,
                    on_toggle=self._rescan)

        cl.addStretch(1)

        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(controls)
        left.setMinimumWidth(390)
        left.setMaximumWidth(430)
        root.addWidget(left)

        # right: toolbar + file list + status
        right = QVBoxLayout()
        root.addLayout(right, 1)

        bar = QHBoxLayout()
        self.folder_label = QLabel("(no folder selected)")
        self.folder_label.setStyleSheet("color:#bbb;")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._analyze)
        self.reprofile_btn = QPushButton("Re-analyze all")
        self.reprofile_btn.clicked.connect(lambda: self._analyze(reprofile=True))
        self.save_btn = QPushButton("Save / Write")
        self.save_btn.clicked.connect(self._save)
        bar.addWidget(self.folder_label, 1)
        bar.addWidget(browse)
        bar.addWidget(self.analyze_btn)
        bar.addWidget(self.reprofile_btn)
        bar.addWidget(self.save_btn)
        right.addLayout(bar)

        legend = QLabel("green = 00:00 intro    orange = detected break    "
                        "gridlines every 5 min")
        legend.setStyleSheet("color:#888; padding:2px 0;")
        right.addWidget(legend)

        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(4, 4, 4, 4)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.list_container)
        right.addWidget(scroll, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#aaa;")
        right.addWidget(self.status)

    def _group(self, title, parent_layout) -> QVBoxLayout:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setSpacing(3)
        parent_layout.addWidget(box)
        return lay

    def _float(self, layout, key, label, lo, hi, step, value=None,
               integer=False, live=True):
        c = FloatControl(label, lo, hi, step, value if value is not None else lo, integer)
        self.floats[key] = c
        c.changed.connect(self._on_live_change if live else self._on_sampling_change)
        layout.addWidget(c)

    def _check(self, layout, key, label, live=True, on_toggle=None):
        cb = QCheckBox(label)
        self.checks[key] = cb
        if on_toggle is not None:
            cb.toggled.connect(lambda _=None, f=on_toggle: f())
        elif live:
            cb.toggled.connect(self._on_live_change)
        else:
            cb.toggled.connect(self._on_sampling_change)
        layout.addWidget(cb)

    def _combo(self, layout, key, label, options):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        name = QLabel(label)
        name.setMinimumWidth(118)
        cb = QComboBox()
        cb.addItems(options)
        self.combos[key] = cb
        cb.currentTextChanged.connect(self._on_live_change)
        rl.addWidget(name)
        rl.addWidget(cb, 1)
        layout.addWidget(row)

    # -- populate controls from an existing config/args -------------------- #

    def _populate_from(self, cfg: DetectConfig, args):
        fv = {
            "sensitivity": cfg.sensitivity, "black_margin": cfg.black_margin,
            "grid_step": cfg.grid_step, "near_deep": cfg.near_deep,
            "near_tol": cfg.near_tol, "quiet_margin": cfg.quiet_margin,
            "quiet_floor_trim": cfg.quiet_floor_trim, "bright_margin": cfg.bright_margin,
            "peak_cap": cfg.peak_cap, "min_duration": cfg.min_duration,
            "min_gap": cfg.min_gap, "skip_start": cfg.skip_start,
            "skip_end": cfg.skip_end, "max_chapters": cfg.max_chapters or 0,
            "min_chapters": cfg.min_chapters, "min_duration_floor": cfg.min_duration_floor,
            "video_fps": getattr(args, "video_fps", 12.0),
            "audio_window": getattr(args, "audio_window", 0.1),
        }
        for k, v in fv.items():
            if k in self.floats:
                self.floats[k].set_value(v)
        for k, v in {
            "add_intro": cfg.add_intro, "near_miss": cfg.near_miss,
            "ignore_audio": cfg.ignore_audio, "audio_fallback": cfg.audio_fallback,
            "blank_guard": cfg.blank_guard, "embed": getattr(args, "embed", False),
            "recursive": getattr(args, "recursive", True),
        }.items():
            if k in self.checks:
                self.checks[k].setChecked(bool(v))
        self.combos["mode"].setCurrentText(cfg.mode)
        self.combos["mark_at"].setCurrentText(cfg.mark_at)

    def _cfg(self) -> DetectConfig:
        f = lambda k: self.floats[k].value()
        b = lambda k: self.checks[k].isChecked()
        maxc = int(f("max_chapters"))
        return DetectConfig(
            sensitivity=f("sensitivity"), black_margin=f("black_margin"),
            quiet_margin=f("quiet_margin"), quiet_floor_trim=f("quiet_floor_trim"),
            blank_guard=b("blank_guard"), bright_margin=f("bright_margin"),
            peak_cap=f("peak_cap"), mode=self.combos["mode"].currentText(),
            near_miss=b("near_miss"), near_deep=f("near_deep"), near_tol=f("near_tol"),
            ignore_audio=b("ignore_audio"), audio_fallback=b("audio_fallback"),
            min_duration=f("min_duration"), min_gap=f("min_gap"),
            skip_start=f("skip_start"), skip_end=f("skip_end"),
            grid_step=f("grid_step"), add_intro=b("add_intro"),
            mark_at=self.combos["mark_at"].currentText(),
            min_chapters=int(f("min_chapters")),
            min_duration_floor=f("min_duration_floor"),
            max_chapters=None if maxc <= 0 else maxc,
        )

    # -- folder / file list ------------------------------------------------ #

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select a folder of videos",
                                             self.folder or "")
        if d:
            self._set_folder(d)

    def _rescan(self):
        if self.folder:
            self._set_folder(self.folder)

    def _set_folder(self, folder):
        self.folder = folder
        self.cache = ProfileCache(base=folder)
        self.folder_label.setText(folder)
        recursive = self.checks["recursive"].isChecked()
        from pathlib import Path
        files = [str(p) for p in find_videos(Path(folder), recursive, self.exts)]
        self._build_rows(files)
        self.signals = {p: None for p in files}
        self.file_status = {p: "missing" for p in files}
        if not files:
            self.status.setText("No video files found in this folder.")
            return
        # read existing cache only (no decoding) so timelines draw immediately
        self._start_worker(files, mode="cache")

    def _build_rows(self, files):
        # clear existing rows (keep the trailing stretch)
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.rows.clear()
        for path in files:
            row = QFrame()
            row.setFrameShape(QFrame.StyledPanel)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(6, 3, 6, 3)
            name = QLabel(os.path.basename(path))
            name.setMinimumWidth(230)
            name.setMaximumWidth(230)
            name.setWordWrap(False)
            name.setToolTip(path)
            name.setTextInteractionFlags(Qt.TextSelectableByMouse)
            count = QLabel("–")
            count.setMinimumWidth(28)
            count.setAlignment(Qt.AlignCenter)
            count.setToolTip("chapters detected")
            timeline = TimelineView()
            rl.addWidget(name)
            rl.addWidget(count)
            rl.addWidget(timeline, 1)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self.rows[path] = {"name": name, "count": count, "timeline": timeline}

    # -- change handling / live recompute ---------------------------------- #

    def _on_live_change(self, *_):
        self._debounce.start()

    def _on_sampling_change(self, *_):
        self._sampling_dirty = True
        self.analyze_btn.setText("Analyze (sampling changed)")

    def _recompute_all(self):
        cfg = self._cfg()
        for path, sig in self.signals.items():
            self._update_row(path, sig, cfg)
        self._update_status()

    def _update_row(self, path, sig, cfg=None):
        row = self.rows.get(path)
        if row is None:
            return
        stale = self.file_status.get(path) == "stale"
        # amber filename for outdated profiles, plain otherwise
        row["name"].setStyleSheet("color:#d2aa5a;" if stale else "")
        if sig is None:
            row["timeline"].set_not_analyzed()
            row["count"].setText("–")
            return
        cfg = cfg or self._cfg()
        try:
            breaks, _, _ = detect_to_target(sig, cfg)
        except Exception as e:
            row["timeline"].placeholder = f"error: {e}"[:60]
            row["timeline"].set_not_analyzed()
            return
        marks = [(b.time, b.is_intro) for b in breaks]
        row["timeline"].set_data(sig.duration, marks, stale=stale)
        row["count"].setText(str(count_breaks(breaks)))

    def _update_status(self):
        total = len(self.signals)
        current = sum(1 for s in self.file_status.values() if s == "current")
        stale = sum(1 for s in self.file_status.values() if s == "stale")
        parts = [f"{current}/{total} analyzed"]
        if stale:
            parts.append(f"{stale} outdated (preview only — click Analyze to refresh)")
        if current == 0 and stale == 0:
            parts.append("click Analyze to decode")
        elif current or stale:
            parts.append("adjust sliders to preview live")
        self.status.setText("  ·  ".join(parts))

    # -- workers ----------------------------------------------------------- #

    def _busy(self, on):
        for b in (self.analyze_btn, self.reprofile_btn, self.save_btn):
            b.setEnabled(not on)
        self.progress.setVisible(on)

    def _start_worker(self, files, mode):
        if self._worker is not None:
            return
        w = ProbeWorker(files, self.cache,
                        self.floats["video_fps"].value(),
                        self.floats["audio_window"].value(), mode)
        self._worker = w
        w.progress.connect(self._on_progress)
        w.fileReady.connect(self._on_file_ready)
        w.failed.connect(self._on_file_failed)
        w.finishedAll.connect(self._on_worker_done)
        self._busy(mode != "cache")
        self.progress.setVisible(True)
        w.start()

    def _analyze(self, reprofile=False):
        if not self.folder or self._worker is not None:
            return
        files = list(self.signals.keys())
        if not files:
            return
        mode = "reprofile" if (reprofile or self._sampling_dirty) else "analyze"
        self._start_worker(files, mode=mode)

    def _on_progress(self, done, total, name):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        if name:
            self.status.setText(f"{done}/{total}  ·  {name}")

    def _on_file_ready(self, path, sig, status):
        self.signals[path] = sig
        self.file_status[path] = status
        self._update_row(path, sig)

    def _on_file_failed(self, path, msg):
        row = self.rows.get(path)
        if row:
            row["timeline"].placeholder = f"failed: {msg}"[:60]
            row["timeline"].set_not_analyzed()

    def _on_worker_done(self):
        self._worker = None
        self._sampling_dirty = False
        self.analyze_btn.setText("Analyze")
        self._busy(False)
        self.progress.setVisible(False)
        self._recompute_all()

    # -- save -------------------------------------------------------------- #

    def _save(self):
        if self._worker is not None:
            return
        items = [(p, s, self.file_status.get(p, "current"))
                 for p, s in self.signals.items() if s is not None]
        if not items:
            self.status.setText("Nothing to save — analyze some files first.")
            return
        embed = self.checks["embed"].isChecked()
        w = SaveWorker(items, self._cfg(), embed, self.cache)
        self._worker = w
        w.progress.connect(self._on_progress)
        w.failed.connect(self._on_file_failed)
        w.finishedAll.connect(self._on_save_done)
        self._busy(True)
        self.progress.setVisible(True)
        w.start()

    def _on_save_done(self, written, mkv_missing):
        self._worker = None
        self._busy(False)
        self.progress.setVisible(False)
        msg = f"Wrote chapters for {written} file(s)."
        if mkv_missing:
            msg += "  (mkvpropedit not found — sidecars written, MKVs not embedded)"
        self.status.setText(msg)

    def closeEvent(self, e):
        if self._worker is not None:
            if hasattr(self._worker, "stop"):
                self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(e)


def launch(start_folder, cfg: DetectConfig, args) -> int:
    """Entry point used by `cli.main()` when --gui is passed."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = MainWindow(start_folder, cfg, args)
    win.show()
    return app.exec()
