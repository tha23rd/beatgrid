"""Grids from other systems, for comparison."""

from collections import Counter

import numpy as np

from .activations import Activations
from .grid import Grid


def _peaks(x: np.ndarray, fps: float) -> np.ndarray:
    """Beat This!'s own 'minimal' postprocessing: maxima within ±70 ms above 0.5."""
    neighbourhood = np.max([np.roll(x, k) for k in range(-3, 4)], axis=0)
    frames = np.flatnonzero((x > 0.5) & (x >= neighbourhood))
    if len(frames):  # drop adjacent duplicates
        frames = frames[np.concatenate([[True], np.diff(frames) > 1])]
    return frames / fps


def beat_this_raw(act: Activations) -> Grid | None:
    """The network's beats as-is, numbered so its downbeats land on multiples of four."""
    beats, downs = _peaks(act.beat, act.fps), _peaks(act.downbeat, act.fps)
    if len(beats) < 4:
        return None
    ioi = np.diff(beats)
    typical = np.median(ioi)
    numbers = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(ioi / typical)))])
    votes = Counter(int(numbers[np.argmin(np.abs(beats - d))]) % 4 for d in downs)
    phase = votes.most_common(1)[0][0] if votes else 0
    return Grid(numbers - phase, beats)
