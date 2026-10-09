"""Reference grids from an Engine DJ library (Engine Library/Database2/m.db).

PerformanceData.beatData holds two grids: Engine's own analysis ("default") and the one
the DJ edited ("adjusted"); they are equal until someone touches the grid. Layout, as
documented by libdjinterop: a 4-byte big-endian length, then zlib data holding
sample rate (f64 BE), sample count (f64 BE), a set flag (u8), and for each grid a marker
count (i64 BE) followed by markers of {sample offset f64 LE, beat number i64 LE,
beats to next marker i32 LE, unknown i32 LE}. Times are on the decoder-trimmed timeline.
"""

import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np

from ..grid import Grid

AUDIO_EXTENSIONS = (".mp3", ".flac", ".wav", ".m4a", ".aiff", ".aif", ".ogg")


def decode_beat_data(blob: bytes) -> tuple[float, float, list, list]:
    """Returns (sample_rate, sample_count, default_markers, adjusted_markers);
    markers are (sample_offset, beat_number) pairs."""
    expected = struct.unpack_from(">I", blob)[0]
    raw = zlib.decompress(blob[4:])
    if len(raw) != expected:
        raise ValueError("beat data length does not match its header")
    sample_rate, sample_count, _ = struct.unpack_from(">ddB", raw)
    offset = 17
    grids = []
    for _ in range(2):
        (count,) = struct.unpack_from(">q", raw, offset)
        offset += 8
        markers = []
        for _ in range(count):
            sample, number, _, _ = struct.unpack_from("<dqii", raw, offset)
            offset += 24
            markers.append((sample, number))
        grids.append(markers)
    return sample_rate, sample_count, grids[0], grids[1]


def to_grid(markers: list, sample_rate: float) -> Grid:
    return Grid(
        np.array([m[1] for m in markers], dtype=float),
        np.array([m[0] / sample_rate for m in markers], dtype=float),
    )


def same_markers(a: list, b: list) -> bool:
    return len(a) == len(b) and all(abs(x[0] - y[0]) < 1 and x[1] == y[1] for x, y in zip(a, b))


def read_library(db_path: Path) -> list[dict]:
    """Every track with a saved grid, read-only."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute(
        """SELECT t.id, t.path, t.artist, t.title, t.genre, t.isPlayed, t.timeLastPlayed,
                  t.isBeatGridLocked, p.beatData
           FROM Track t JOIN PerformanceData p ON p.trackId = t.id
           WHERE p.beatData IS NOT NULL"""
    ).fetchall()
    tracks = []
    for tid, path, artist, title, genre, played, last_played, locked, blob in rows:
        try:
            rate, count, default, adjusted = decode_beat_data(blob)
        except (ValueError, struct.error, zlib.error):
            continue
        if len(adjusted) < 2 or len(default) < 2:
            continue
        tracks.append({
            "track_id": tid,
            "path": path,
            "artist": artist,
            "title": title,
            "genre": genre,
            "played": bool(played) or last_played is not None,
            "locked": bool(locked),
            "corrected": not same_markers(default, adjusted),
            "duration": count / rate,
            "reference": to_grid(adjusted, rate),
            "engine_default": to_grid(default, rate),
        })
    return tracks
