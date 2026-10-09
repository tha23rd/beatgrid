"""Audit reference grids against their own audio.

A reference is only as good as the person or program that made it. This checks a grid
using nothing but the grid and the audio, never another gridder's answer: split the music
into segments, fold the high-resolution flux around the reference's beats in each, and
find where the attack sits. On a right grid the attack stays put; on a grid at a slightly
wrong tempo, or one bent where the music is straight, it walks.

  drift_ms    how far the attack moves end to end on a robust (Theil–Sen) line through the
              segments — a reference at the wrong tempo
  spread_ms   scatter about that line (median absolute deviation)
  verdict     "steady": drift <= STEADY_MS and at most one segment off the line (a section
                with a different attack, e.g. a breakdown, is not a grid error)
              "drifts": the tempo is wrong
              "uneven": several sections disagree — a bent grid or a track that changes
              "unclear": too few segments with a clear attack to say
"""

import numpy as np

from .grid import Grid
from .pipeline import _peak, fold

SEGMENTS = 8
SEARCH_S = 0.06
MIN_PROMINENCE = 1.5
STEADY_MS = 8.0


def attack_positions(grid: Grid, flux: np.ndarray, dt: float, t_first: float, start: float, end: float) -> list[tuple[float, float]]:
    """(attack offset in ms relative to the grid, prominence) for each segment of [start, end]."""
    out = []
    edges = np.linspace(start, end, SEGMENTS + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        _, beats = grid.beats(lo, hi)
        if len(beats) < 8:
            continue
        # Fold over just this segment's beats: a local grid at the local tempo.
        bpm = 60.0 / np.median(np.diff(beats))
        i0, i1 = max(int((lo - t_first) / dt), 0), max(int((hi - t_first) / dt), 0)
        segment = np.zeros_like(flux)
        segment[i0:i1] = flux[i0:i1]
        folded = fold(segment, dt, t_first, bpm, beats[0], SEARCH_S)
        i, height = _peak(folded)
        prominence = height / (np.median(folded[folded > 0]) + 1e-12) if np.any(folded > 0) else 0.0
        out.append(((i - (len(folded) - 1) / 2) * dt * 1000, float(prominence)))
    return out


def audit(grid: Grid, flux: np.ndarray, dt: float, t_first: float, start: float, end: float) -> dict:
    positions = attack_positions(grid, flux, dt, t_first, start, end)
    clear = [(i, p) for i, (p, prom) in enumerate(positions) if prom >= MIN_PROMINENCE]
    result = {"segments": positions}
    if len(clear) < 3:
        return {**result, "verdict": "unclear", "drift_ms": None, "spread_ms": None, "attack_ms": None}
    x = np.array([i for i, _ in clear], float)
    y = np.array([p for _, p in clear])
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
    slope = float(np.median(slopes))
    residual = y - slope * x
    residual -= np.median(residual)
    spread = float(np.median(np.abs(residual)))
    outliers = int(np.sum(np.abs(residual) > max(5.0, 3 * spread)))
    drift = abs(slope) * (SEGMENTS - 1)
    if drift > STEADY_MS:
        verdict = "drifts"
    elif outliers > 1 or spread > 3.0:
        verdict = "uneven"
    else:
        verdict = "steady"
    return {**result, "verdict": verdict, "drift_ms": float(drift), "spread_ms": spread,
            "outliers": outliers, "attack_ms": float(np.median(y))}
