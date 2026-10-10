"""The reference pipeline: neural beat activations in, a DJ grid out.

A DJ grid for a produced track is a handful of numbers — tempo, where the beats fall, which
beat is the one — so after the network this is model fitting with strong priors, not beat
tracking:

1. coarse tempo from the network's beat peaks
2. tempo and phase from the coherence of the whole beat activation with a pulse train,
   which uses every frame and so is not limited to the network's 20 ms frames
3. refine the tempo on the audio itself: the high-resolution flux folds into its sharpest
   peak at exactly the right tempo (a tempo 0.02 BPM off smears it by tens of ms over a
   track), then snap to an exact whole or half BPM if the audio agrees (tracks are made in
   a DAW)
4. measure how far stretches of the track bend away from one tempo, and follow them if asked
5. place the grid on the attack: fold a high-resolution (sub-millisecond) spectral flux of
   the audio at the beat period around the whole track and take its peak. Reference grids
   sit on this broadband click, not on the kick's low-frequency body (measured:
   experiments/placement.py)
6. choose the bar phase from the downbeat activation folded at the bar period
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .activations import Activations
from .grid import Grid


@dataclass
class Params:
    min_bpm: float = 60.0
    max_bpm: float = 200.0
    # Tempo search around the coarse estimate.
    search_span: float = 0.03
    # How much coherence an exact DAW tempo may give up against the best free tempo and
    # still be preferred, and how far from it it may be.
    snap: bool = True
    snap_tolerance: float = 0.02
    snap_max_bpm: float = 0.3
    snap_steps: tuple = (1.0, 0.5)
    # Stretches of this many beats are checked against the single tempo.
    bend_window: int = 32
    bend_ms: float = 25.0
    follow_bends: bool = False
    # Refine the tempo by flux fold sharpness within ±flux_span BPM of the activation fit;
    # an exact DAW tempo may lose this much fold height and still be preferred. Falls back
    # to the activation when the flux has no clear attack.
    tempo_by_flux: bool = True
    flux_span: float = 0.2
    flux_step: float = 0.002
    snap_flux_tolerance: float = 0.1
    # Below this much drift across the track (ms, either end vs the middle) two tempos are
    # indistinguishable by ear, so the exact DAW tempo is taken without asking the audio.
    snap_free_drift_ms: float = 3.0
    flux_min_prominence: float = 1.5
    # Grid placement: "flux" = peak of the folded high-resolution flux; "activation" = the
    # network's phase. `offset_ms` is the convention of where on that landmark a DJ's grid
    # line goes, calibrated on references.
    placement: str = "flux"
    # Calibrated on the Engine dev split (experiments/placement.py): DJ grid lines sit ~2 ms
    # after the folded flux peak.
    offset_ms: float = 2.0
    attack_search_ms: float = 50.0
    # Run the network on only this many 30-second windows spread over the music (None: the
    # whole track). The flux still covers the whole track; the network only has to supply
    # a rough tempo, phase and bar.
    network_windows: int | None = None
    # Bar 1 by several pooled votes (bars.py) instead of the network's downbeats alone;
    # weights per vote, and whether a prior grid (Engine's) may vote.
    # Measured (experiments/bar_votes.py) with the supervised student: its downbeats alone pick
    # bar 1 on 88-93% of Raveform; pooled 2:1:1 with arrangement changes and high-frequency
    # bursts on the one, 95.5-96.7% on held-out and dev halves. Backbeat adds nothing; an
    # Engine prior is right on only 68% of the tracks a DJ had to correct.
    bar_votes: bool = True
    bar_w_network: float = 2.0
    bar_w_backbeat: float = 0.0
    bar_w_changes: float = 1.0
    bar_w_fills: float = 1.0
    bar_w_prior: float = 0.0
    # Which network supplies the activations: the distilled, expert-supervised student by
    # default ("final0" is Beat This! itself, ~20x slower and a little less accurate here).
    network: str = "student-v4"
    # Octave convention. None keeps the network's octave. Otherwise the tempo is folded
    # into [lo, hi) — except that a track with an octave inside `octave_half_range` (drum &
    # bass at 165–185, say) is counted at half of it, as many DJs do.
    octave_range: tuple | None = None
    octave_half_range: tuple | None = None
    # Tempo changes (sections.py): "constant" keeps one tempo; "variable" and "auto" split the
    # grid into constant-tempo segments where the track changes tempo, "auto" only on strong
    # evidence. The seg_* knobs tune the search.
    tempo: str = "constant"
    seg_window_s: float = 16.0  # windows the local tempo is measured in
    seg_hop_s: float = 4.0
    seg_step: float = 0.005  # between tempo states (fraction)
    seg_drift_cost: float = 0.5  # log units: moving one tempo state between windows
    seg_switch_cost: float = 8.0  # jumping further
    seg_quiet: float = 0.3  # windows whose coherence peaks below this carry no evidence
    seg_anchor_min: float = 0.3  # an anchor needs this much coherence at its window's tempo
    seg_sigma_ms: float = 4.0  # anchor noise for the line costs
    seg_clip: float = 3.0  # anchors further off than this many sigmas cost no more
    seg_min_anchors: int = 4  # a segment spans at least this many anchors
    seg_search_span: float = 0.01  # tempo search around a segment's line (fraction)
    seg_join_s: float = 8.0  # a change is placed within this far of where the lines meet
    # What a segmentation must show to be taken (defaults: "auto"; TEMPO_MODES has "variable").
    seg_penalty: float = 24.0  # cost of an extra line through the anchors
    seg_min_tempo_change: float = 0.001  # (fraction) a join with less is no change...
    seg_min_jump: float = 0.0  # ... unless a same-tempo phase jump of at least this (beats; 0: never)
    seg_min_bars: float = 16.0  # every segment holds at least this many bars
    seg_min_apart: int = 64  # beats where the segmented grid is more than 10 ms from the constant one
    seg_margin: float = 0.25  # how much stronger the attacks on those beats must be under it
    seg_min_evidence: float = 2.5  # ... and how strong, over the track's median flux


@dataclass
class Result:
    grid: Grid | None
    diagnostics: dict = field(default_factory=dict)
    # The grid in constant-tempo segments when the tempo changes (params.tempo "variable" or
    # "auto"); None when one tempo holds.
    segments: "object | None" = None


def coarse_bpm(act: Activations, params: Params) -> float | None:
    """Tempo from the spacing of the network's beat peaks."""
    b = act.beat
    neighbourhood = np.max([np.roll(b, k) for k in range(-3, 4)], axis=0)
    frames = np.flatnonzero((b > 0.5) & (b >= neighbourhood))
    if len(frames) < 8:
        return None
    ioi = np.diff(frames) / act.fps
    typical = np.median(ioi)
    regular = ioi[np.abs(ioi / typical - 1) < 0.15]
    bpm = 60.0 / (np.mean(regular) if len(regular) else typical)
    return float(bpm) if params.min_bpm / 2 <= bpm <= params.max_bpm * 2 else None


def coherence(act: Activations, periods: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """sum a(t) e^{-2πi t/P} / sum a(t) for each period P. Its magnitude says how well a
    pulse train of that period fits the activation (1 = all of it on the train); its angle
    says where the train's pulses fall."""
    a = act.beat if weights is None else act.beat * weights
    keep = a > 1e-3
    t, a = act.times[keep], a[keep]
    total = a.sum()
    out = np.zeros(len(periods), complex)
    if total <= 0:
        return out
    for i in range(0, len(periods), 64):  # chunked to bound memory
        p = periods[i : i + 64, None]
        out[i : i + 64] = (a * np.exp(-2j * np.pi * t / p)).sum(axis=1) / total
    return out


def coherence_at(act: Activations, bpm: float, weights: np.ndarray | None = None) -> complex:
    return complex(coherence(act, np.array([60.0 / bpm]), weights)[0])


def fit_tempo(act: Activations, bpm0: float, params: Params) -> tuple[float, complex]:
    """Coherence peak near bpm0, refined to 0.0005 BPM."""
    bpms = np.arange(bpm0 * (1 - params.search_span), bpm0 * (1 + params.search_span), 0.01)
    best = bpms[np.argmax(np.abs(coherence(act, 60.0 / bpms)))]
    for step in (0.002, 0.0005):
        bpms = np.arange(best - 10 * step, best + 10.5 * step, step)
        best = bpms[np.argmax(np.abs(coherence(act, 60.0 / bpms)))]
    return float(best), coherence_at(act, best)


def snap_tempo(act: Activations, bpm: float, c_best: complex, params: Params) -> tuple[float, complex, bool]:
    """Prefer an exact whole (then half) BPM if it fits nearly as well as the free fit."""
    for step in params.snap_steps:
        target = round(bpm / step) * step
        if abs(target - bpm) > params.snap_max_bpm:
            continue
        c = coherence_at(act, target)
        if abs(c) >= abs(c_best) * (1 - params.snap_tolerance):
            return float(target), c, True
    return bpm, c_best, False


def phase_of(c: complex, period: float) -> float:
    """Time of a beat (modulo the period) implied by a coherence value."""
    return float((-np.angle(c) / (2 * np.pi) * period) % period)


def wrap(x: np.ndarray | float, period: float):
    return (np.asarray(x) + period / 2) % period - period / 2


def bend(act: Activations, bpm: float, t0: float, window_beats: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Offsets of stretches of the track from the single-tempo grid.
    Returns (worst deviation from the median in ms, stretch centres, offsets in seconds)."""
    period = 60.0 / bpm
    span = window_beats * period
    centres, offsets = [], []
    for start in np.arange(0.0, max(act.times[-1] - span / 2, 0.0), span / 2):
        w = ((act.times >= start) & (act.times < start + span)).astype(float)
        c = coherence_at(act, bpm, w)
        if abs(c) < 0.3:
            continue  # breakdowns, silence: no opinion
        centres.append(start + span / 2)
        offsets.append(float(wrap(phase_of(c, period) - t0, period)))
    centres, offsets = np.array(centres), np.array(offsets)
    if len(offsets) < 3:
        return 0.0, centres, offsets
    return float(np.max(np.abs(offsets - np.median(offsets))) * 1000), centres, offsets


FLUX_WINDOW = 512
FLUX_HOP = 32


def onset_flux(samples: np.ndarray, rate: int, n: int = FLUX_WINDOW, hop: int = FLUX_HOP) -> tuple[np.ndarray, float, float]:
    """Log-magnitude spectral flux at sub-millisecond hops (32 samples). Returns
    (flux, seconds per value, time of the first value). Computed in blocks to bound memory."""
    x = samples.astype(np.float32)
    window = np.hanning(n).astype(np.float32)
    frames = (len(x) - n) // hop + 1
    if frames < 2:
        return np.zeros(1, np.float32), hop / rate, 0.0
    out = np.empty(frames - 1, np.float32)
    previous = None
    block = 8192
    for b in range(0, frames, block):
        idx = (np.arange(b, min(b + block, frames)) * hop)[:, None] + np.arange(n)[None, :]
        mag = np.log1p(100 * np.abs(np.fft.rfft(x[idx] * window, axis=1)))
        if previous is not None:
            mag = np.vstack([previous, mag])
        rise = np.maximum(np.diff(mag, axis=0), 0).sum(axis=1)
        start = b - 1 if previous is not None else 0
        out[start : start + len(rise)] = rise
        previous = mag[-1:]
    # Value i is the change from frame i to i+1; place it between their centres.
    return out, hop / rate, (n / 2 + hop / 2) / rate


def fold(flux: np.ndarray, dt: float, t_first: float, bpm: float, t0: float, half_s: float) -> np.ndarray:
    """Mean of the flux in a ±half_s window around every beat of a grid at bpm through t0."""
    period = 60.0 / bpm
    duration = t_first + len(flux) * dt
    k = np.arange(np.ceil(-t0 / period), np.floor((duration - t0) / period) + 1)
    centres = np.round((t0 + k * period - t_first) / dt).astype(int)
    half = int(round(half_s / dt))
    idx = centres[:, None] + np.arange(-half, half + 1)[None, :]
    idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(flux))]
    if len(idx) == 0:
        return np.zeros(2 * half + 1)
    return flux[idx].mean(axis=0)


def _smooth(folded: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Light Gaussian smoothing (sigma in samples of the fold) so a peak a couple of
    samples wide is measured by its shape rather than its single highest sample."""
    x = np.arange(-4 * int(np.ceil(sigma)), 4 * int(np.ceil(sigma)) + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return np.convolve(folded, kernel / kernel.sum(), mode="same")


def _peak(folded: np.ndarray) -> tuple[float, float]:
    """(fractional index, height) of the maximum of the smoothed fold, parabolically refined."""
    folded = _smooth(folded)
    i = int(np.argmax(folded))
    if 0 < i < len(folded) - 1:
        a, b, c = folded[i - 1], folded[i], folded[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return i + 0.5 * (a - c) / denom, float(b - 0.125 * (a - c) ** 2 / denom)
    return float(i), float(folded[i])


def attack_offset(flux: np.ndarray, dt: float, t_first: float, bpm: float, t0: float, params: Params) -> tuple[float, float]:
    """Fold the flux around every beat of the grid and find its peak.
    Returns (shift in seconds that puts the grid on it, prominence of the peak)."""
    half_s = params.attack_search_ms / 1000
    folded = fold(flux, dt, t_first, bpm, t0, half_s)
    i, height = _peak(folded)
    prominence = float(height / (np.median(folded) + 1e-12))
    return (i - (len(folded) - 1) / 2) * dt, prominence


def second_attack(flux: np.ndarray, dt: float, t_first: float, bpm: float, t0: float, params: Params) -> float:
    """How strong the runner-up peak of the folded attack is, as a fraction of the main one
    (both measured above the fold's median); 0 when there is none. A peak counts when it is
    more than 5 ms from the main one and the fold dips at least 20% below it in between: a
    second attack (a clap or a click a few ms off the kick, say) that a DJ might put the grid
    on instead. Folded over the same window as `attack_offset`."""
    folded = fold(flux.astype(np.float64), dt, t_first, bpm, t0, params.attack_search_ms / 1000)
    s = _smooth(folded)
    floor = float(np.median(folded))
    i = int(np.argmax(s))
    main = s[i] - floor
    if not main > 0:
        return 0.0
    gap = int(round(0.005 / dt))
    best = 0.0
    for j in range(1, len(s) - 1):
        if abs(j - i) <= gap or s[j] < s[j - 1] or s[j] < s[j + 1]:
            continue
        height = s[j] - floor
        lo, hi = min(i, j), max(i, j)
        if height > best and s[lo : hi + 1].min() - floor < 0.8 * height:
            best = height
    return float(best / main)


# Self-audit: does the attack stay put under the grid from one end of the track to the other?
# The check `lab audit` runs on reference grids (audit.py), here over the whole decoded track.
STRETCHES = 8
STRETCH_SEARCH_S = 0.06
STRETCH_MIN_PROMINENCE = 1.5


def self_audit(flux: np.ndarray, dt: float, t_first: float, bpm: float, t0: float) -> tuple[float | None, int]:
    """Fold the flux around the grid's beats in each of 8 equal stretches of the track and find
    where the attack sits. Returns (how far it walks end to end in ms, on a Theil–Sen line
    through the stretches with a clear attack, or None when fewer than 3 have one; how many
    stretches have a clear attack). A grid at a slightly wrong tempo walks off the attacks;
    a right one stays put."""
    period = 60.0 / bpm
    duration = t_first + len(flux) * dt
    half = int(round(STRETCH_SEARCH_S / dt))
    xs, ys = [], []
    for s in range(STRETCHES):
        lo, hi = duration * s / STRETCHES, duration * (s + 1) / STRETCHES
        k = np.arange(np.ceil((lo - t0) / period), np.ceil((hi - t0) / period))
        centres = np.round((t0 + k * period - t_first) / dt).astype(int)
        idx = centres[:, None] + np.arange(-half, half + 1)[None, :]
        idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(flux))]
        if len(idx) < 8:
            continue
        folded = flux[idx].astype(np.float64).mean(axis=0)
        i, height = _peak(folded)
        if height / (np.median(folded) + 1e-12) < STRETCH_MIN_PROMINENCE:
            continue
        xs.append(s)
        ys.append((i - half) * dt * 1000)
    if len(xs) < 3:
        return None, len(xs)
    slopes = [(ys[b] - ys[a]) / (xs[b] - xs[a]) for a in range(len(xs)) for b in range(a + 1, len(xs))]
    return float(abs(np.median(slopes)) * (STRETCHES - 1)), len(xs)


def fold_period(flux: np.ndarray, dt: float, t_first: float, bpm: float) -> np.ndarray:
    """Mean flux by phase within the beat, over a whole period (one bin per flux sample)."""
    period = 60.0 / bpm
    bins = int(np.ceil(period / dt))
    phase = np.mod(t_first + np.arange(len(flux)) * dt, period)
    idx = np.minimum((phase / dt).astype(int), bins - 1)
    total = np.bincount(idx, weights=flux, minlength=bins)
    count = np.bincount(idx, minlength=bins)
    return total / np.maximum(count, 1)


def _circular_peak_height(folded: np.ndarray) -> float:
    pad = 8
    wrapped = np.concatenate([folded[-pad:], folded, folded[:pad]])
    return float(_smooth(wrapped)[pad:-pad].max())


def tempo_by_flux(act: Activations, flux: np.ndarray, dt: float, t_first: float, bpm: float, params: Params):
    """Tempo at which the flux folds into the sharpest peak over a whole beat period,
    searched around `bpm`. Any transient that repeats every beat (kick, hat, snare) sharpens
    at the right tempo, so this needs no phase from the network. Returns (bpm, height,
    height_at) where height_at scores any other tempo the same way."""

    def height_at(b: float) -> float:
        return _circular_peak_height(fold_period(flux, dt, t_first, b))

    candidates = np.arange(bpm - params.flux_span, bpm + params.flux_span + params.flux_step / 2, params.flux_step)
    heights = np.array([height_at(b) for b in candidates])
    i = int(np.argmax(heights))
    best = float(candidates[i])
    if 0 < i < len(heights) - 1:
        a, b, c = heights[i - 1], heights[i], heights[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            best += 0.5 * (a - c) / denom * params.flux_step
    return best, height_at(best), height_at


def bar_scores(act: Activations, bpm: float, t0: float) -> np.ndarray:
    """Mean downbeat activation at each of the four bar phases (beat index mod 4 from t0)."""
    period = 60.0 / bpm
    k = np.arange(np.ceil(-t0 / period), np.floor((act.times[-1] - t0) / period) + 1).astype(int)
    frames = (t0 + k * period) * act.fps
    near = np.clip(np.round(frames).astype(int)[:, None] + np.arange(-1, 2)[None, :], 0, len(act.downbeat) - 1)
    strength = act.downbeat[near].max(axis=1)
    return np.array([strength[np.mod(k, 4) == r].mean() if np.any(np.mod(k, 4) == r) else 0.0 for r in range(4)])


def bar_phase(act: Activations, bpm: float, t0: float) -> tuple[int, float]:
    """Which beat index modulo 4 (counting from t0) carries the downbeat, and the margin."""
    period = 60.0 / bpm
    k = np.arange(np.ceil(-t0 / period), np.floor((act.times[-1] - t0) / period) + 1).astype(int)
    frames = (t0 + k * period) * act.fps
    # The strongest downbeat response within a frame either side of each beat.
    near = np.clip(np.round(frames).astype(int)[:, None] + np.arange(-1, 2)[None, :], 0, len(act.downbeat) - 1)
    strength = act.downbeat[near].max(axis=1)
    scores = np.array([strength[np.mod(k, 4) == r].mean() if np.any(np.mod(k, 4) == r) else 0.0 for r in range(4)])
    order = np.argsort(scores)[::-1]
    margin = float((scores[order[0]] - scores[order[1]]) / (scores[order[0]] + 1e-9))
    return int(order[0]), margin


NO_ATTACK = "no clear attack to place the grid on"
WEAK_PULSE = "weak or changing pulse"
TEMPO_CHANGE = "tempo may change"
BAR_DISAGREE = "bar 1 unclear: the downbeat cues disagree"
OCTAVE_HALVED = "octave chosen by the profile: the audio's pulse is twice as fast"
OCTAVE_DOUBLED = "octave chosen by the profile: the audio's pulse is half as fast"
TEMPO_SEGMENTS = "tempo changes: check the grid where it changes"


def review_reasons(diag: dict) -> list[str]:
    """Why a DJ should look at this grid before trusting it (empty: no reason found).
    Measured with experiments/review_flags.py; what each reason catches and costs, and the
    signals that were tried and not kept (a second attack, a grid that walks off the attacks,
    a narrow bar-1 margin), are in docs/results.md. Those signals stay in the diagnostics
    (attack_second, stretch_drift_ms, snap_drift_ms, bar_margin). A grid in segments gets one
    reason, to check the joins: the others describe a single-tempo grid."""
    if diag.get("segmented"):
        return [TEMPO_SEGMENTS]
    reasons = []
    # No usable attack: the audio could not refine the tempo, or under the final grid fewer
    # than 3 of the track's 8 stretches show one.
    if diag.get("tempo_source") != "flux" or ("clear_stretches" in diag and diag.get("stretch_drift_ms") is None):
        reasons.append(NO_ATTACK)
    if diag.get("coherence_final", 1.0) < 0.5:
        reasons.append(WEAK_PULSE)
    if diag.get("bend_ms", 0.0) > 40:
        reasons.append(TEMPO_CHANGE)
    votes = diag.get("bar_votes") or {}
    if "network" in votes and "changes" in votes and int(np.argmax(votes["network"])) != int(np.argmax(votes["changes"])):
        reasons.append(BAR_DISAGREE)
    # The profile, not the audio, chose the octave: a convention DJs differ on.
    if "bpm_grid" in diag and diag["bpm_grid"] != diag["bpm"]:
        reasons.append(OCTAVE_HALVED if diag["bpm_grid"] < diag["bpm"] else OCTAVE_DOUBLED)
    return reasons


# The knobs each tempo mode sets (on top of the defaults, which are "auto"'s).
TEMPO_MODES = {
    "constant": {"tempo": "constant"},
    "auto": {"tempo": "auto"},
    "variable": {"tempo": "variable", "seg_penalty": 12.0, "seg_min_tempo_change": 0.0005, "seg_min_jump": 0.1,
                 "seg_min_bars": 4.0, "seg_min_apart": 16, "seg_margin": 0.05, "seg_min_evidence": 1.5},
}

PROFILES = {
    # Drum & bass in half time, everything else between 100 and 200.
    "edm": {"octave_range": (100.0, 200.0), "octave_half_range": (165.0, 185.0)},
}


def apply_octave(bpm: float, params: Params) -> float:
    if params.octave_half_range:
        lo, hi = params.octave_half_range
        for candidate in (bpm / 2, bpm, bpm * 2):
            if lo <= candidate < hi:
                return candidate / 2
    if params.octave_range:
        lo, hi = params.octave_range
        while bpm < lo:
            bpm *= 2
        while bpm >= hi:
            bpm /= 2
    return bpm


NETWORK_WINDOW_S = 30.0


def network_windows(start: float, end: float, n: int) -> list[tuple[float, float]]:
    """n windows of NETWORK_WINDOW_S evenly spread over [start, end] (merged if they touch)."""
    span = end - start
    if n * NETWORK_WINDOW_S >= span:
        return [(start, end)]
    centres = start + span * (np.arange(n) + 0.5) / n
    return [(c - NETWORK_WINDOW_S / 2, c + NETWORK_WINDOW_S / 2) for c in centres]


def masked(act: Activations, windows: list[tuple[float, float]]) -> Activations:
    """The activations as if the network had only seen these windows."""
    keep = np.zeros(len(act.beat), bool)
    for lo, hi in windows:
        keep |= (act.times >= lo) & (act.times < hi)
    return Activations(np.where(keep, act.beat, 0.0), np.where(keep, act.downbeat, 0.0), act.fps)


def run(track, params: Params = Params(), prior: Grid | None = None) -> Result:
    """Grid a `Track` (anything with .activations and .flux()). `prior` is another grid for
    the same audio (Engine's, say) that may vote on bar 1 when bar_votes is on."""
    act: Activations = track.activations_for(params.network) if hasattr(track, "activations_for") else track.activations
    if params.network_windows:
        start, end = getattr(track, "music_span", (0.0, act.times[-1]))
        act = masked(act, network_windows(start, end, params.network_windows))
    diag: dict = {}
    bpm0 = coarse_bpm(act, params)
    if bpm0 is None:
        return Result(None, {"reason": "no pulse"})
    bpm, c = fit_tempo(act, bpm0, params)
    diag.update(bpm_coarse=bpm0, bpm_activation=bpm, coherence=abs(c))
    snapped, tempo_source = False, "activation"
    flux = None
    if params.placement == "flux" or params.tempo_by_flux:
        flux, dt, t_first = track.flux()
    if flux is not None and params.tempo_by_flux:
        # Judge whether the audio has a clear attack at the best tempo the search finds:
        # at the network's slightly-off tempo the fold is smeared and looks weak.
        flux_bpm, height, height_at = tempo_by_flux(act, flux, dt, t_first, bpm, params)
        folded = fold_period(flux, dt, t_first, flux_bpm)
        prominence = _circular_peak_height(folded) / (np.median(folded) + 1e-12)
        diag["flux_prominence"] = float(prominence)
        if prominence >= params.flux_min_prominence:
            bpm = flux_bpm
            tempo_source = "flux"
            diag.update(bpm_flux=bpm, flux_height=height)
            if params.snap:
                half_track = (len(flux) * dt) / 2
                for step in params.snap_steps:
                    target = round(bpm / step) * step
                    if abs(target - bpm) > params.snap_max_bpm:
                        continue
                    drift_ms = half_track * abs(target - bpm) / bpm * 1000
                    if drift_ms <= params.snap_free_drift_ms or height_at(target) >= height * (1 - params.snap_flux_tolerance):
                        # How far the exact tempo walks from the measured one, end to end.
                        diag["snap_drift_ms"] = 2 * drift_ms
                        bpm, snapped = float(target), True
                        break
    if tempo_source == "activation" and params.snap:
        bpm, c, snapped = snap_tempo(act, bpm, c, params)
    c = coherence_at(act, bpm)
    diag.update(bpm=bpm, snapped=snapped, tempo_source=tempo_source, coherence_final=abs(c))
    period = 60.0 / bpm
    t0 = phase_of(c, period)

    worst, centres, offsets = bend(act, bpm, t0, params.bend_window)
    diag["bend_ms"] = worst

    if params.placement == "flux":
        shift, prominence = attack_offset(flux, dt, t_first, bpm, t0, params)
        second = second_attack(flux, dt, t_first, bpm, t0, params)
        diag.update(attack_shift_ms=shift * 1000, attack_prominence=prominence, attack_second=second)
        t0 += shift
    t0 += params.offset_ms / 1000
    if flux is not None:
        drift, clear = self_audit(flux, dt, t_first, bpm, t0)
        diag.update(stretch_drift_ms=drift, clear_stretches=clear)

    phase, margin = bar_phase(act, bpm, t0)
    if params.bar_votes and hasattr(track, "band_energy"):
        from .bars import Weights, decide

        weights = Weights(params.bar_w_network, params.bar_w_backbeat, params.bar_w_changes,
                          params.bar_w_fills, params.bar_w_prior)
        decision = decide(track, act, bpm, t0, weights, prior)
        diag.update(bar_network=int(phase), bar_votes=decision.votes)
        phase, margin = decision.rotation, decision.margin
    diag.update(bar_margin=margin)
    out_bpm = apply_octave(bpm, params)
    diag["bpm_grid"] = out_bpm
    diag["review"] = review_reasons(diag)
    downbeat = t0 + phase * period
    segments = None
    if params.tempo in ("variable", "auto"):
        from .sections import fit as fit_sections, with_octave

        found, diag["sections"] = fit_sections(track, act, params, bpm, downbeat)
        if found is not None:
            segments = with_octave(found, params)
            diag["segmented"] = True
            diag["review"] = review_reasons(diag)
    if params.follow_bends and worst > params.bend_ms and len(offsets) >= 3:
        # Markers at each stretch's centre, at the offset that stretch measured.
        k = np.round((centres - downbeat) / period)
        times = downbeat + k * period + wrap(offsets - np.median(offsets), period)
        order = np.argsort(times)
        numbers, times = k[order], times[order]
        keep = np.concatenate([[True], (np.diff(numbers) > 0) & (np.diff(times) > 0)])
        if keep.sum() >= 2:
            grid = Grid(numbers[keep], times[keep])
            return Result(grid if out_bpm == bpm else Grid.constant(out_bpm, downbeat), diag)
    return Result(Grid.constant(out_bpm, downbeat), diag, segments)
