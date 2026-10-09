"""EXPERIMENTAL, not used by the pipeline. Tracks whose tempo changes: transition edits
(124 -> 150), mashups of songs at slightly different tempos, and old recordings that drift.

State (experiments/sections_try.py): sections are decided jointly over 16 s windows by
Viterbi over tempo states in one octave band (staying free, drifting cheap, jumping costly),
which no longer splits steady tracks and finds the change on transition edits (124 -> 150,
116.7 -> 160). A sectioned grid is kept only if it sits on the attacks 15% better than one
tempo. Open: the per-section tempo refit on the flux can lock to the wrong tempo (117.0 for
116.7 on one track), and references for these tracks disagree with the audio about where the
change is — they need ears.

The single-tempo fit stays the answer unless it clearly fails. When it does, the track is cut
into overlapping windows, each fitted on its own the same way the whole track is (a broad
tempo from the network, refined on the attacks), and consecutive windows that agree are joined
into sections. A section at one tempo is refitted as a whole and snapped like any track; one
whose tempo keeps moving follows its windows. The grid is then markers at section boundaries,
numbered so bar 1 of each section is a downbeat.
"""

from dataclasses import dataclass

import numpy as np

from .activations import Activations
from .grid import Grid
from .pipeline import (Params, _circular_peak_height, attack_offset, bar_scores, coarse_bpm, coherence_at,
                       fit_tempo, fold_period, phase_of, tempo_by_flux)

WINDOW_S = 16.0
HOP_S = 8.0
SAME_TEMPO = 0.4  # BPM: windows closer than this belong to one section


@dataclass
class Window:
    start: float
    end: float
    bpm: float
    beat: float  # time of a beat near the window centre (on the attack)
    strength: float  # coherence of the activation in the window


def _masked(act: Activations, lo: float, hi: float) -> Activations:
    keep = (act.times >= lo) & (act.times < hi)
    return Activations(np.where(keep, act.beat, 0.0), np.where(keep, act.downbeat, 0.0), act.fps)


def _flux_slice(flux: np.ndarray, dt: float, t_first: float, lo: float, hi: float) -> tuple[np.ndarray, float]:
    """The flux between lo and hi, and the time of its first value."""
    i0 = min(max(int((lo - t_first) / dt), 0), len(flux))
    i1 = min(max(int((hi - t_first) / dt), i0), len(flux))
    return flux[i0:i1], t_first + i0 * dt


STEP = 0.25  # BPM between tempo states
SWITCH_COST = 8.0  # log units: what a jump between tempos costs
DRIFT_COST = 0.5  # moving one state (0.25 BPM) to follow a drift


def salience(act: Activations, lo: float, hi: float, bpms: np.ndarray) -> np.ndarray:
    """Coherence of the beat activation in [lo, hi) with a pulse train at each tempo."""
    keep = (act.times >= lo) & (act.times < hi) & (act.beat > 1e-3)
    t, a = act.times[keep], act.beat[keep]
    if a.sum() <= 0:
        return np.zeros(len(bpms))
    phase = 2j * np.pi * np.outer(bpms / 60.0, t)
    return np.abs((a * np.exp(-phase)).sum(axis=1)) / a.sum()


def tempo_path(act: Activations, start: float, end: float, centre_bpm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Most likely tempo per window (Viterbi), within one octave band around the track's tempo.
    Returns (window starts, tempo per window, salience strength per window)."""
    bpms = np.arange(centre_bpm / np.sqrt(2), centre_bpm * np.sqrt(2), STEP)
    starts = np.arange(start, max(end - WINDOW_S / 2, start + 1e-9), HOP_S)
    emit = np.array([salience(act, lo, lo + WINDOW_S, bpms) for lo in starts])
    strength = emit.max(axis=1)
    loge = np.log(emit + 1e-3)
    n, k = loge.shape
    dist = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])
    trans = np.where(dist == 0, 0.0, np.where(dist == 1, -DRIFT_COST, -SWITCH_COST))
    score = loge[0].copy()
    back = np.zeros((n, k), int)
    for i in range(1, n):
        cand = score[:, None] + trans
        back[i] = np.argmax(cand, axis=0)
        # Quiet windows (breakdowns) carry no evidence: let them follow their neighbours.
        score = cand[back[i], np.arange(k)] + (loge[i] if strength[i] > 0.3 else 0.0)
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return starts, bpms[path], strength


def attack_fit(grid: Grid, flux: np.ndarray, dt: float, t_first: float, start: float, end: float) -> float:
    """How well a grid sits on the attacks: mean flux within ±8 ms of its beats, against the
    track's median flux."""
    _, times = grid.beats(start, end)
    idx = np.round((times - t_first) / dt).astype(int)
    half = int(round(0.008 / dt))
    vals = [flux[max(i - half, 0) : i + half + 1].max() for i in idx if 0 <= i < len(flux)]
    return float(np.mean(vals) / (np.median(flux) + 1e-12)) if vals else 0.0


def sections(starts: np.ndarray, path: np.ndarray, min_windows: int = 4) -> list[tuple[float, float, float]]:
    """Runs of the tempo path that hold a tempo (within drift) for at least `min_windows`."""
    runs = []
    i = 0
    while i < len(path):
        j = i
        while j + 1 < len(path) and abs(path[j + 1] - path[i]) <= 2 * STEP:
            j += 1
        runs.append([starts[i], starts[j] + WINDOW_S, float(np.median(path[i : j + 1])), j - i + 1])
        i = j + 1
    # Absorb short runs into a neighbour: a fill or breakdown, not a tempo.
    out = []
    for r in runs:
        if r[3] < min_windows and out:
            out[-1][1] = r[1]
        else:
            out.append(r)
    if len(out) > 1 and out[0][3] < min_windows:
        out[1][0] = out[0][0]
        out.pop(0)
    return [(a, b, bpm) for a, b, bpm, _ in out]


def fit_sections(track, act: Activations, params: Params, single: Grid | None = None) -> tuple[Grid | None, dict]:
    """A multi-section grid, or None when one tempo describes the track (or fits as well)."""
    flux, dt, t_first = track.flux()
    start, end = getattr(track, "music_span", (0.0, act.times[-1]))
    centre = single.bpm if single is not None else (coarse_bpm(act, params) or 120.0)
    starts, path, strength = tempo_path(act, start, end, centre)
    found = sections(starts, path)
    diag = {"sections": [(round(a, 1), round(b, 1), round(bpm, 2)) for a, b, bpm in found]}
    if len(found) < 2:
        return None, diag
    groups = [[Window(a, b, bpm, 0.0, 1.0)] for a, b, bpm in found]
    # Sections overlap by a window; split each overlap in the middle.
    bounds = [0.0] + [(groups[i][-1].end + groups[i + 1][0].start) / 2 for i in range(len(groups) - 1)] + [end]
    numbers, times = [], []
    number = 0.0
    for gi, g in enumerate(groups):
        lo, hi = bounds[gi], bounds[gi + 1]
        if gi == len(groups) - 1:
            hi = end + 4 * 60.0 / g[-1].bpm
        bpm = float(np.median([w.bpm for w in g]))
        # Refit the section as a whole on the attacks, and snap like any track.
        a = _masked(act, lo, hi)
        f, tf = _flux_slice(flux, dt, t_first, lo, hi)
        p = Params(**{**params.__dict__, "flux_span": 0.3})
        bpm, height, height_at = tempo_by_flux(a, f, dt, tf, bpm, p)
        target = round(bpm * 2) / 2
        if abs(target - bpm) < params.snap_max_bpm and height_at(target) >= height * (1 - params.snap_flux_tolerance):
            bpm = target
        period = 60.0 / bpm
        t0 = phase_of(coherence_at(a, bpm), period)
        shift, _ = attack_offset(f, dt, tf, bpm, t0, params)
        t0 += shift + params.offset_ms / 1000
        rotation = int(np.argmax(bar_scores(a, bpm, t0)))
        downbeat = t0 + rotation * period
        # This section starts at its first bar line after the previous section's last beat.
        after = times[-1] + 0.5 * period if times else lo
        k0 = np.ceil((max(after, lo if not times else after) - downbeat) / period / 4) * 4
        if not times:
            k0 = np.ceil((lo - downbeat) / period / 4) * 4 - 4  # extend one bar before the start
        first = downbeat + k0 * period
        if numbers:
            number = np.ceil((numbers[-1] + 1) / 4) * 4
        count = int(np.floor((hi - downbeat) / period) - k0)
        if count < 4:
            continue
        numbers += [number, number + count]
        times += [first, first + count * period]
        number += count
    if len(numbers) < 4:
        return None, diag
    keep = np.concatenate([[True], (np.diff(numbers) > 0) & (np.diff(times) > 0)])
    multi = Grid(np.array(numbers)[keep], np.array(times)[keep])
    if single is not None:
        fit_single, fit_multi = attack_fit(single, flux, dt, t_first, start, end), attack_fit(multi, flux, dt, t_first, start, end)
        diag.update(fit_single=round(fit_single, 3), fit_multi=round(fit_multi, 3))
        if fit_multi < fit_single * 1.15:
            return None, diag
    return multi, diag
