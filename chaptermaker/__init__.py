"""chaptermaker — adaptive commercial-break chapter marking for messy video (VHS/DVR) rips.

The core idea: instead of a fixed black/silence threshold (which fails on VHS
because the picture never goes truly black and the audio never goes truly
silent due to tape hiss), we *profile each file individually* to find its own
"blackest black" and "quietest quiet", then set thresholds relative to that.
"""

__version__ = "0.1.3"
