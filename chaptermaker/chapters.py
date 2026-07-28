"""Write detected breaks out as chapter files, and optionally embed into MKV.

Two sidecar formats are produced (both human-readable, both accepted by
mkvpropedit / mkvmerge / most players):

  * OGM simple  (<name>.chapters.txt) -- trivially editable by hand
  * Matroska XML (<name>.chapters.xml) -- what we feed to mkvpropedit

Embedding uses `mkvpropedit --chapters`, which rewrites only the chapter
element in place -- no re-encode, no remux, near-instant even on huge files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from .detect import Break
from .probe import find_tool


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:09.6f}"


def chapter_titles(breaks: list[Break]) -> list[str]:
    titles = []
    n = 1
    for b in breaks:
        if b.is_intro:
            titles.append("Start")
        else:
            titles.append(f"Segment {n}")
        n += 1 if not b.is_intro else 0
    # Ensure at least generic numbering if intro absent
    if not titles:
        return titles
    # Renumber segments sequentially regardless of intro presence
    seg = 1
    out = []
    for b in breaks:
        if b.is_intro:
            out.append("Start")
        else:
            out.append(f"Segment {seg}")
            seg += 1
    return out


def to_ogm(breaks: list[Break]) -> str:
    lines = []
    titles = chapter_titles(breaks)
    for i, (b, title) in enumerate(zip(breaks, titles), start=1):
        ts = _fmt_ts(b.time)
        lines.append(f"CHAPTER{i:02d}={ts}")
        lines.append(f"CHAPTER{i:02d}NAME={title}")
    return "\n".join(lines) + "\n"


def to_xml(breaks: list[Break]) -> str:
    titles = chapter_titles(breaks)
    atoms = []
    for b, title in zip(breaks, titles):
        atoms.append(
            "    <ChapterAtom>\n"
            f"      <ChapterTimeStart>{_fmt_ts(b.time)}</ChapterTimeStart>\n"
            "      <ChapterDisplay>\n"
            f"        <ChapterString>{escape(title)}</ChapterString>\n"
            "        <ChapterLanguage>eng</ChapterLanguage>\n"
            "      </ChapterDisplay>\n"
            "    </ChapterAtom>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE Chapters SYSTEM \"matroskachapters.dtd\">\n"
        "<Chapters>\n"
        "  <EditionEntry>\n"
        + "\n".join(atoms) + "\n"
        "  </EditionEntry>\n"
        "</Chapters>\n"
    )


def write_sidecars(video_path: str, breaks: list[Break]) -> dict[str, str]:
    p = Path(video_path)
    txt = p.with_suffix("").with_suffix(".chapters.txt")
    xml = p.with_suffix("").with_suffix(".chapters.xml")
    txt.write_text(to_ogm(breaks), encoding="utf-8")
    xml.write_text(to_xml(breaks), encoding="utf-8")
    return {"ogm": str(txt), "xml": str(xml)}


class MkvpropeditNotFound(RuntimeError):
    pass


def embed_chapters(video_path: str, xml_path: str) -> None:
    """Embed chapters into the MKV in place via mkvpropedit (no re-encode)."""
    tool = find_tool("mkvpropedit")
    if not tool:
        raise MkvpropeditNotFound(
            "mkvpropedit not found. Install MKVToolNix "
            "(https://mkvtoolnix.download/) — or place mkvpropedit.exe next to "
            "this program — to embed chapters into MKV files. "
            "The sidecar .chapters.xml/.txt files were still written."
        )
    subprocess.run(
        [tool, video_path, "--chapters", xml_path],
        capture_output=True, text=True, check=True,
    )
