"""Entry point for the packaged (PyInstaller) Windows executable.

Double-clicking the built `chaptermark.exe` runs this with no arguments, which
opens the graphical tuner. Running it from a terminal with arguments behaves as
the normal command-line tool (`chaptermark.exe "D:\\Show" --embed`, etc.).
"""

import sys

from chaptermaker.cli import main

if __name__ == "__main__":
    sys.exit(main())
