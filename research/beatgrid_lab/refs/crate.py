"""Reference grids from a SQLite database of DJ-approved grids (`crate.sqlite`).

The database comes from an earlier grid-correction effort and is not distributed; any
database with the same two tables can be imported:

  tracks(id, kind, engine_id, path, title, artist, genre, bpm)
  grids(track_id, status, provenance, human_approved, beats_json, downbeats_json,
        tempo_engine, engine_agrees)

`grids` holds a chosen grid per track as explicit beat and downbeat times (JSON lists of
seconds, on ffmpeg's trimmed timeline) with a status. We take the usable ones — `trusted`
(the fit agreed with Engine DJ's analysis; nobody necessarily listened) and `repaired` — and
keep `human_approved` apart: those grids were heard and accepted by a DJ, the strongest
references there are.

Tracks are `fixture` (tracks from an Engine DJ library, with an Engine id) or `reference`
(tracks with no Engine entry, e.g. store copies of Raveform tracks).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..grid import Grid


def to_grid(beats: list[float], downbeats: list[float]) -> Grid:
    """Every beat as a marker; downbeats land on multiples of four."""
    beats = np.asarray(beats, float)
    down = np.zeros(len(beats), bool)
    for d in downbeats:
        i = int(np.argmin(np.abs(beats - d)))
        if abs(beats[i] - d) < 1e-3:
            down[i] = True
    numbers = np.arange(len(beats), dtype=float)
    first = int(np.flatnonzero(down)[0]) if down.any() else 0
    numbers -= first  # first downbeat is number 0
    # A bar that is not four beats long (an edit) shifts the count; re-anchor on each downbeat.
    for i in np.flatnonzero(down):
        drift = numbers[i] % 4
        if drift:
            numbers[i:] += 4 - drift
    return Grid(numbers, beats)


def read_crate(db_path: Path) -> list[dict]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """SELECT g.track_id, g.status, g.provenance, g.human_approved, g.beats_json, g.downbeats_json,
                  g.tempo_engine, g.engine_agrees, t.kind, t.engine_id, t.path, t.title, t.artist, t.genre, t.bpm
           FROM grids g JOIN tracks t ON t.id = g.track_id
           WHERE t.kind IN ('fixture', 'reference') AND g.status IN ('trusted', 'repaired')"""
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "crate_id": r["track_id"],
            "kind": r["kind"],
            "engine_id": r["engine_id"],
            "path": r["path"],
            "title": r["title"],
            "artist": r["artist"],
            "genre": r["genre"],
            "engine_bpm": r["bpm"],
            "status": r["status"],
            "provenance": r["provenance"],
            "human_approved": bool(r["human_approved"]),
            "engine_agrees": r["engine_agrees"],
            "reference": to_grid(json.loads(r["beats_json"]), json.loads(r["downbeats_json"])),
        })
    return out
