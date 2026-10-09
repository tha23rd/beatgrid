"""A beat grid as DJ software stores it: anchor markers (beat number, time) joined by
straight lines, extended past the ends at the first and last segment's tempo.

Beat numbers count from an arbitrary origin; a beat whose number is a multiple of
four is a downbeat. A constant-tempo grid is two markers.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    numbers: np.ndarray  # beat number of each marker, strictly increasing
    times: np.ndarray  # seconds, strictly increasing

    def __post_init__(self):
        if len(self.numbers) < 2:
            raise ValueError("a grid needs at least two markers")
        if np.any(np.diff(self.numbers) <= 0) or np.any(np.diff(self.times) <= 0):
            raise ValueError("grid markers must increase")

    @classmethod
    def constant(cls, bpm: float, downbeat: float) -> "Grid":
        """A grid at one tempo with a downbeat at `downbeat` seconds (any bar will do)."""
        period = 60.0 / bpm
        return cls(np.array([0.0, 4.0]), np.array([downbeat, downbeat + 4 * period]))

    @property
    def bpm(self) -> float:
        """Average tempo across the markers."""
        return 60.0 * (self.numbers[-1] - self.numbers[0]) / (self.times[-1] - self.times[0])

    @property
    def is_constant(self) -> bool:
        tempos = 60.0 * np.diff(self.numbers) / np.diff(self.times)
        return bool(np.ptp(tempos) < 1e-3)

    def time_of(self, numbers) -> np.ndarray:
        n = np.atleast_1d(np.asarray(numbers, dtype=float))
        out = np.interp(n, self.numbers, self.times)
        first = (self.times[1] - self.times[0]) / (self.numbers[1] - self.numbers[0])
        last = (self.times[-1] - self.times[-2]) / (self.numbers[-1] - self.numbers[-2])
        before, after = n < self.numbers[0], n > self.numbers[-1]
        out[before] = self.times[0] + (n[before] - self.numbers[0]) * first
        out[after] = self.times[-1] + (n[after] - self.numbers[-1]) * last
        return out

    def number_at(self, times) -> np.ndarray:
        """Inverse of time_of: fractional beat number at each time."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        out = np.interp(t, self.times, self.numbers)
        first = (self.numbers[1] - self.numbers[0]) / (self.times[1] - self.times[0])
        last = (self.numbers[-1] - self.numbers[-2]) / (self.times[-1] - self.times[-2])
        before, after = t < self.times[0], t > self.times[-1]
        out[before] = self.numbers[0] + (t[before] - self.times[0]) * first
        out[after] = self.numbers[-1] + (t[after] - self.times[-1]) * last
        return out

    def beats(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """Integer beat numbers and their times within [start, end]."""
        lo, hi = self.number_at([start, end])
        n = np.arange(np.ceil(lo - 1e-9), np.floor(hi + 1e-9) + 1)
        return n, self.time_of(n)

    def shifted(self, seconds: float) -> "Grid":
        return Grid(self.numbers, self.times + seconds)

    def renumbered(self, offset: int) -> "Grid":
        """Same beats, bar lines moved by `offset` beats."""
        return Grid(self.numbers + offset, self.times)

    def doubled(self) -> "Grid":
        """Twice the tempo: a beat between every pair."""
        return Grid(self.numbers * 2, self.times)

    def halved(self, keep_odd: bool = False) -> "Grid":
        """Half the tempo, keeping the even (or odd) beats."""
        return Grid((self.numbers - (1 if keep_odd else 0)) / 2, self.times)
