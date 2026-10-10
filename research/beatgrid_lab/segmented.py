"""A grid whose tempo changes: constant-tempo segments, numbered on across the joins. The
reference for the Rust crate's `SegmentedGrid` (crates/beatgrid/src/grid.rs); the two are
kept equal.

Each segment starts at a beat (`start`, seconds), holds a tempo (`bpm`) from there, and says
where that beat falls in its bar (`beat_in_bar`, 1-4; 1 is a downbeat). A segment's beats are
start + j * period for j >= 0, up to the last one at least half a period before the next
segment starts; the first segment also extends back before its start and the last one on
forever, so a one-segment grid is a constant grid. Beat numbers run on without gaps across
segments, 0 being the first downbeat at or after time 0 (as for a constant grid); bars count
on too, a new bar starting at every beat whose position in the bar does not follow on from
the beat before (normally a downbeat).
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    bpm: float
    beat_in_bar: int = 1

    @property
    def period(self) -> float:
        return 60.0 / self.bpm


class Segmented:
    def __init__(self, segments: list[Segment], end: float | None = None):
        if not segments:
            raise ValueError("a segmented grid needs at least one segment")
        for s in segments:
            if not (math.isfinite(s.start) and math.isfinite(s.bpm) and s.bpm > 0 and 1 <= s.beat_in_bar <= 4):
                raise ValueError(f"bad segment {s}")
        for a, b in zip(segments, segments[1:]):
            if not b.start - a.start >= a.period / 2:
                raise ValueError("segments must start at least half a beat after the one before")
        self.segments = list(segments)
        self.end = end  # length of the track, when known (for `bpm`)
        n = len(segments)
        # Beats in each segment; the last one is open-ended.
        self.counts = [int(math.floor((b.start - a.period / 2 - a.start) / a.period)) + 1
                       for a, b in zip(segments, segments[1:])] + [None]
        # Provisional numbering from the first segment's first beat, then shifted so that the
        # first downbeat at or after time 0 is beat 0.
        first = [0]
        for c in self.counts[:-1]:
            first.append(first[-1] + c)
        self._first = first
        zero = self._first_downbeat_number()
        self._first = [f - zero for f in first]
        # Bars: position in the bar of each segment's first beat is given; the bar number of
        # each segment's first beat follows from the segment before.
        bars = [0] * n
        for i in range(1, n):
            last_pos = (segments[i - 1].beat_in_bar - 1 + self.counts[i - 1] - 1)
            last_bar = bars[i - 1] + last_pos // 4
            bib_last = last_pos % 4 + 1
            bars[i] = last_bar + (1 if segments[i].beat_in_bar <= bib_last else 0)
        self._bars = bars
        # Shift so beat 0 is in bar 1.
        i0, j0 = self._locate(0)
        bar0 = bars[i0] + (segments[i0].beat_in_bar - 1 + j0) // 4
        self._bars = [b - bar0 + 1 for b in bars]

    # --- numbering ---------------------------------------------------------------------
    def _locate(self, number: int) -> tuple[int, int]:
        """(segment, beat within it) of an integer beat number (first segment extends back)."""
        i = 0
        for k in range(len(self.segments)):
            if self._first[k] <= number:
                i = k
        return i, number - self._first[i]

    def _first_downbeat_number(self) -> int:
        """The number, in the current numbering, of the first downbeat at or after time 0."""
        n = int(math.ceil(self._number(0.0) - 1e-9)) - 1
        while True:
            i, j = self._locate(n)
            if self._time(n) >= 0.0 and (self.segments[i].beat_in_bar - 1 + j) % 4 == 0:
                return n
            n += 1

    def _last_time(self, i: int) -> float:
        s = self.segments[i]
        return s.start + (self.counts[i] - 1) * s.period

    def _number(self, t: float) -> float:
        segs = self.segments
        i = 0
        for k in range(len(segs)):
            if segs[k].start <= t:
                i = k
        s = segs[i]
        if i == len(segs) - 1 or t <= self._last_time(i):
            return self._first[i] + (t - s.start) / s.period
        last = self._last_time(i)
        return self._first[i] + self.counts[i] - 1 + (t - last) / (segs[i + 1].start - last)

    def _time(self, n: float) -> float:
        segs = self.segments
        i = 0
        for k in range(len(segs)):
            if self._first[k] <= n:
                i = k
        s = segs[i]
        if i == len(segs) - 1 or n <= self._first[i] + self.counts[i] - 1:
            return s.start + (n - self._first[i]) * s.period
        last = self._last_time(i)
        return last + (n - (self._first[i] + self.counts[i] - 1)) * (segs[i + 1].start - last)

    # --- public ------------------------------------------------------------------------
    def number_at(self, times) -> np.ndarray:
        return np.array([self._number(float(t)) for t in np.atleast_1d(times)])

    def time_of(self, numbers) -> np.ndarray:
        return np.array([self._time(float(n)) for n in np.atleast_1d(numbers)])

    def segment_at(self, t: float) -> int:
        i = 0
        for k, s in enumerate(self.segments):
            if s.start <= t:
                i = k
        return i

    def bpm_at(self, t: float) -> float:
        return self.segments[self.segment_at(t)].bpm

    def first_beat_number(self, i: int) -> int:
        return self._first[i]

    def beat_list(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(index, time, bar, beat_in_bar) of every beat with start <= time < end."""
        if not start < end:
            e = np.array([], int)
            return e, np.array([]), e, e
        lo = int(math.ceil(self._number(start) - 1e-9)) - 1
        idx, times, bars, pos = [], [], [], []
        n = lo
        while True:
            t = self._time(n)
            if t >= end:
                break
            if t >= start:
                i, j = self._locate(n)
                p = self.segments[i].beat_in_bar - 1 + j
                idx.append(n)
                times.append(t)
                bars.append(self._bars[i] + p // 4)
                pos.append(p % 4 + 1)
            n += 1
        return np.array(idx, int), np.array(times), np.array(bars, int), np.array(pos, int)

    def beats(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """(numbers, times) in [start, end] for the metrics: numbers are 4 * (bar - 1) +
        beat_in_bar - 1, so downbeats are multiples of four, as on a marker grid."""
        _, t, bar, pos = self.beat_list(start, end + 1e-9)
        return (4 * (bar - 1) + pos - 1).astype(float), t

    @property
    def bpm(self) -> float:
        """The tempo of the longest segment (the last one runs to `end`, or 60 s)."""
        segs = self.segments
        lengths = [b.start - a.start for a, b in zip(segs, segs[1:])]
        lengths.append((self.end - segs[-1].start) if self.end is not None else 60.0)
        return segs[int(np.argmax(lengths))].bpm

    @property
    def is_constant(self) -> bool:
        return len(self.segments) == 1

    @classmethod
    def constant(cls, bpm: float, downbeat: float, end: float | None = None) -> "Segmented":
        period = 60.0 / bpm
        return cls([Segment(downbeat % (4 * period), bpm, 1)], end)

    def as_json(self) -> list[dict]:
        out = []
        for i, s in enumerate(self.segments):
            out.append({"start": s.start, "bpm": s.bpm, "beat_in_bar": s.beat_in_bar,
                        "beat": self._first[i], "bar": self._bars[i]})
        return out
