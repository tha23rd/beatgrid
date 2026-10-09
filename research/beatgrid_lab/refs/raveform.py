"""Reference grids from Raveform (Kim, Kim, Kim & Nam, TISMIR 2026; labels CC BY 4.0).

1,423 electronic tracks gridded in Rekordbox and corrected by three experts. Beats come as
structures/beats/<index>.<youtube id>.beat.csv with columns time, downbeat (position in
the bar, 1-4), section; track metadata (genre, 8 folds, Rekordbox tempo markers) is in
structures/segments.json. Audio is the YouTube video of that id.
"""

import csv
import json
from pathlib import Path

import numpy as np

from ..grid import Grid


def read_beats(path: Path) -> Grid:
    """Every beat as a marker, numbered so that position 1 in the bar is a multiple of 4."""
    times, numbers = [], []
    number = None
    previous = None
    with open(path) as f:
        for row in csv.DictReader(f):
            t, pos = float(row["time"]), int(row["downbeat"])
            if number is None:
                number = pos - 1
            else:
                # Advance one beat, or jump to the next bar when the position resets
                # (a short bar in an edit leaves a gap in the numbering, not an error).
                step = pos - previous if pos > previous else (4 - previous) + pos
                number += step
            times.append(t)
            numbers.append(number)
            previous = pos
    numbers = np.array(numbers, float)
    numbers -= 4 * np.floor(numbers[0] / 4)
    return Grid(numbers, np.array(times))


def read_dataset(root: Path) -> list[dict]:
    structures = root / "structures"
    tracks = []
    for t in json.load(open(structures / "segments.json")):
        beats = structures / "beats" / f"{t['key']}.beat.csv"
        if not beats.exists():
            continue
        tracks.append({
            "youtube_id": t["id"],
            "key": t["key"],
            "title": t["title"],
            "genre": t["genre"],
            "fold": t["fold"],
            "duration": t["duration"],
            "tempo_markers": len(t["tempos"]),
            "reference": read_beats(beats),
        })
    return tracks
