"""Grids for tracks whose tempo changes: transition edits (124 -> 150), mashups, tracks that
change tempo between sections. The reference for crates/beatgrid/src/segments.rs.

The constant-tempo fit (pipeline.run) stays the answer unless the track clearly does not hold
one tempo. To find out:

1. **Tempo path.** The beat activation's coherence with a pulse train is measured in 16 s
   windows every 4 s, over tempos within an octave band around the track's tempo, and the
   most likely tempo per window decoded jointly (Viterbi: staying free, drifting a step cheap,
   jumping costly). Breakdowns carry no evidence and follow their neighbours.
2. **Anchors.** In each window, the phase of that coherence at the window's tempo puts a beat
   near the window's centre. Consecutive clear anchors are numbered by how many beats lie
   between them, so the anchors trace beat time against beat number.
3. **Segments.** One tempo is a straight line through the anchors; a tempo change bends it,
   a phase jump steps it. The anchors are split into the fewest lines that fit them (dynamic
   programming over every split, robust least squares, a fixed cost per extra segment).
4. **Each segment is fitted like a whole track** on its own stretch: tempo refined on the
   audio and snapped to an exact whole or half BPM when the audio agrees, beats placed on the
   folded attack, bar 1 from the pooled votes over that stretch. Neighbours that come out on
   one grid are merged.
5. **Joins.** Each change is put on the downbeat of the new segment, within a window of the
   rough change, where the old grid's beats before it and the new grid's beats after it sit
   best on the attacks.
6. **Gate.** The segments replace the constant grid only where the two disagree by more than
   10 ms for long enough, and there the segmented grid's beats sit clearly better on the
   attacks than the constant grid's. `auto` asks for more of both than `variable`.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .activations import Activations
from .pipeline import (Params, _circular_peak_height, apply_octave, attack_offset, coherence_at, fit_tempo,
                       fold_period, phase_of, tempo_by_flux)
from .segmented import Segment, Segmented

@dataclass
class Fit:
    lo: float
    hi: float
    bpm: float
    downbeat: float
    diag: dict = field(default_factory=dict)


def masked(act: Activations, lo: float, hi: float) -> Activations:
    keep = (act.times >= lo) & (act.times < hi)
    return Activations(np.where(keep, act.beat, 0.0), np.where(keep, act.downbeat, 0.0), act.fps)


def flux_slice(flux: np.ndarray, dt: float, t_first: float, lo: float, hi: float) -> tuple[np.ndarray, float]:
    """The flux values whose time lies in [lo, hi), and the time of the first of them."""
    i0 = min(max(int(math.ceil((lo - t_first) / dt)), 0), len(flux))
    i1 = min(max(int(math.ceil((hi - t_first) / dt)), i0), len(flux))
    return flux[i0:i1], t_first + i0 * dt


# --- 1. tempo path ---------------------------------------------------------------------------

def tempo_states(centre: float, p: Params) -> np.ndarray:
    """Tempos within an octave band around `centre`, log-spaced by `seg_step`."""
    k = int(math.floor(math.log(math.sqrt(2.0)) / math.log1p(p.seg_step)))
    return centre * np.exp(np.arange(-k, k + 1) * math.log1p(p.seg_step))


def window_starts(start: float, end: float, p: Params) -> np.ndarray:
    n = max(int(math.floor((end - start - p.seg_window_s) / p.seg_hop_s)) + 1, 1)
    return start + np.arange(n) * p.seg_hop_s


def salience(act: Activations, lo: float, hi: float, bpms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coherence of the beat activation in [lo, hi) with a pulse train at each tempo: its
    magnitude, and the complex value."""
    keep = (act.times >= lo) & (act.times < hi) & (act.beat > 1e-3)
    t, a = act.times[keep], act.beat[keep].astype(np.float64)
    if a.sum() <= 0:
        return np.zeros(len(bpms)), np.zeros(len(bpms), complex)
    c = (a[None, :] * np.exp(-2j * np.pi * np.outer(bpms / 60.0, t))).sum(axis=1) / a.sum()
    return np.abs(c), c


def tempo_path(act: Activations, starts: np.ndarray, bpms: np.ndarray, p: Params):
    """Most likely tempo state per window. Returns (state per window, coherence per window and
    state, strength per window)."""
    sal, coh = zip(*(salience(act, lo, lo + p.seg_window_s, bpms) for lo in starts))
    sal, coh = np.array(sal), np.array(coh)
    strength = sal.max(axis=1)
    loge = np.log(sal + 1e-3)
    n, k = loge.shape
    dist = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])
    trans = np.where(dist == 0, 0.0, np.where(dist == 1, -p.seg_drift_cost, -p.seg_switch_cost))
    score = loge[0] if strength[0] >= p.seg_quiet else np.zeros(k)
    back = np.zeros((n, k), int)
    for i in range(1, n):
        cand = score[:, None] + trans
        back[i] = np.argmax(cand, axis=0)
        # Quiet windows (breakdowns) carry no evidence: they follow their neighbours.
        score = cand[back[i], np.arange(k)] + (loge[i] if strength[i] >= p.seg_quiet else 0.0)
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, coh, strength


# --- 2. anchors ------------------------------------------------------------------------------

@dataclass
class Anchors:
    number: np.ndarray  # beat number (from the first anchor)
    time: np.ndarray  # a beat near the window's centre (s)
    weight: np.ndarray
    window: np.ndarray  # which window


def anchors(starts: np.ndarray, bpms: np.ndarray, path: np.ndarray, coh: np.ndarray, p: Params) -> Anchors:
    nums, times, weights, wins = [], [], [], []
    for w, lo in enumerate(starts):
        c = coh[w, path[w]]
        if abs(c) < p.seg_anchor_min:
            continue
        period = 60.0 / bpms[path[w]]
        centre = lo + p.seg_window_s / 2
        phase = phase_of(c, period)
        t = phase + period * round((centre - phase) / period)
        if times:
            gap = 0.5 * (period + 60.0 / bpms[path[wins[-1]]])
            nums.append(nums[-1] + round((t - times[-1]) / gap))
        else:
            nums.append(0)
        times.append(t)
        weights.append(abs(c))
        wins.append(w)
    return Anchors(np.array(nums, float), np.array(times), np.array(weights), np.array(wins, int))


# --- 3. segments -----------------------------------------------------------------------------

def line_cost(n: np.ndarray, t: np.ndarray, w: np.ndarray, p: Params) -> tuple[float, float, float]:
    """Robust straight line t = a + b n through weighted anchors: least squares, again without
    the anchors more than `seg_clip` sigmas off, and the clipped cost. Returns (cost, a, b)."""
    sigma = p.seg_sigma_ms / 1000

    def fit(mask):
        ww = w[mask]
        sw, sn, st = ww.sum(), (ww * n[mask]).sum(), (ww * t[mask]).sum()
        snn, snt = (ww * n[mask] ** 2).sum(), (ww * n[mask] * t[mask]).sum()
        den = sw * snn - sn * sn
        if den <= 0:
            return st / sw, 0.0
        b = (sw * snt - sn * st) / den
        return (st - b * sn) / sw, b

    a, b = fit(np.ones(len(n), bool))
    r = (t - a - b * n) / sigma
    inl = np.abs(r) <= p.seg_clip
    if inl.sum() >= 2 and not inl.all():
        a, b = fit(inl)
        r = (t - a - b * n) / sigma
    return float((w * np.minimum(r * r, p.seg_clip**2)).sum()), a, b


def partition(an: Anchors, penalty: float, p: Params) -> list[tuple[int, int]]:
    """Split the anchors into runs [i, j) that each hold one line, minimising the summed cost
    plus `penalty` per extra run; runs hold at least `seg_min_anchors` anchors."""
    m = len(an.time)
    least = p.seg_min_anchors
    if m < 2 * least:
        return [(0, m)]
    cost = np.full((m + 1, m + 1), np.inf)
    for i in range(m):
        for j in range(i + least, m + 1):
            cost[i, j] = line_cost(an.number[i:j], an.time[i:j], an.weight[i:j], p)[0]
    best = np.full(m + 1, np.inf)
    prev = np.zeros(m + 1, int)
    best[0] = -penalty
    for j in range(least, m + 1):
        for i in range(0, j - least + 1):
            if not np.isfinite(best[i]):
                continue
            v = best[i] + cost[i, j] + penalty
            if v < best[j]:
                best[j], prev[j] = v, i
    runs = []
    j = m
    while j > 0:
        runs.append((int(prev[j]), j))
        j = prev[j]
    return runs[::-1]


# --- 4. fit each segment ---------------------------------------------------------------------

def fit_stretch(track, act: Activations, flux, dt, t_first, lo: float, hi: float, bpm0: float, p: Params) -> Fit:
    """The constant-tempo fit of pipeline.run, on [lo, hi) of the track only."""
    from .bars import Weights, decide

    a = masked(act, lo, hi)
    f, tf = flux_slice(flux, dt, t_first, lo, hi)
    bpm, c0 = fit_tempo(a, bpm0, Params(**{**p.__dict__, "search_span": p.seg_search_span}))
    diag = {"bpm_activation": bpm}
    source = "activation"
    flux_bpm, height, height_at = tempo_by_flux(a, f, dt, tf, bpm, p)
    folded = fold_period(f, dt, tf, flux_bpm)
    prominence = _circular_peak_height(folded) / (np.median(folded) + 1e-12)
    snapped = False
    if prominence >= p.flux_min_prominence:
        bpm, source = flux_bpm, "flux"
        if p.snap:
            half = (hi - lo) / 2
            for step in p.snap_steps:
                target = round(bpm / step) * step
                if abs(target - bpm) > p.snap_max_bpm:
                    continue
                drift_ms = half * abs(target - bpm) / bpm * 1000
                if drift_ms <= p.snap_free_drift_ms or height_at(target) >= height * (1 - p.snap_flux_tolerance):
                    bpm, snapped = float(target), True
                    break
    if source == "activation" and p.snap:
        for step in p.snap_steps:
            target = round(bpm / step) * step
            if abs(target - bpm) > p.snap_max_bpm:
                continue
            if abs(coherence_at(a, target)) >= abs(c0) * (1 - p.snap_tolerance):
                bpm, snapped = float(target), True
                break
    period = 60.0 / bpm
    t0 = phase_of(coherence_at(a, bpm), period)
    shift, _ = attack_offset(f, dt, tf, bpm, t0, p)
    t0 += shift + p.offset_ms / 1000
    weights = Weights(p.bar_w_network, p.bar_w_backbeat, p.bar_w_changes, p.bar_w_fills, p.bar_w_prior)
    bar = decide(track, a, bpm, t0, weights, span=(lo, hi))
    diag.update(bpm=bpm, snapped=snapped, tempo_source=source, flux_prominence=float(prominence))
    return Fit(lo, hi, bpm, t0 + bar.rotation * period, diag)


def same_grid(x: Fit, y: Fit, at: float) -> bool:
    """Whether two fits are one grid: the same tempo, and beats within 3 ms of each other
    where they meet."""
    if abs(x.bpm - y.bpm) > 1e-6 * x.bpm:
        return False
    period = 60.0 / x.bpm
    near = lambda f: f.downbeat + period * round((at - f.downbeat) / period)
    d = (near(x) - near(y) + period / 2) % period - period / 2
    return abs(d) <= 0.003


# --- 5. joins --------------------------------------------------------------------------------

def attack_evidence(flux: np.ndarray, dt: float, t_first: float, times: np.ndarray) -> np.ndarray:
    """The flux's peak in the 10 ms before each grid line (where the attack sits: grid lines go
    2 ms after it)."""
    lo = np.round((times - 0.008 - t_first) / dt).astype(int)
    hi = np.round((times + 0.002 - t_first) / dt).astype(int)
    out = np.zeros(len(times))
    for k, (a, b) in enumerate(zip(lo, hi)):
        a, b = max(a, 0), min(b + 1, len(flux))
        out[k] = float(flux[a:b].max()) if b > a else 0.0
    return out


def grid_times(f: Fit, lo: float, hi: float) -> np.ndarray:
    """Beats of a fit's grid in [lo, hi)."""
    period = 60.0 / f.bpm
    k = np.arange(math.ceil((lo - f.downbeat) / period), math.ceil((hi - f.downbeat) / period))
    return f.downbeat + k * period


def join(a: Fit, b: Fit, rough: float, flux, dt, t_first, p: Params) -> float:
    """Where the grid changes from `a` to `b`: the downbeat of b, within `seg_join_s` of the
    rough change, that puts a's beats before it and b's from it best on the attacks."""
    pa, pb = 60.0 / a.bpm, 60.0 / b.bpm
    lo, hi = rough - p.seg_join_s, rough + p.seg_join_s
    ta = grid_times(a, lo - 4 * pa, hi + 4 * pa)
    tb = grid_times(b, lo - 4 * pb, hi + 4 * pb)
    ea, eb = attack_evidence(flux, dt, t_first, ta), attack_evidence(flux, dt, t_first, tb)
    k = np.arange(math.ceil((lo - b.downbeat) / (4 * pb)), math.floor((hi - b.downbeat) / (4 * pb)) + 1)
    candidates = b.downbeat + 4 * pb * k
    best, best_score = rough, -np.inf
    for d in candidates:
        s = ea[ta <= d - pa / 2].sum() + eb[tb >= d - 1e-9].sum()
        if s > best_score:
            best, best_score = float(d), s
    return best


# --- 6. gate ---------------------------------------------------------------------------------

def gate(constant: Segmented, segmented: Segmented, flux, dt, t_first, start: float, end: float,
         p: Params) -> tuple[bool, dict]:
    """Whether the segmented grid should replace the constant one: on the beats where the two
    differ by more than 10 ms (at least `seg_min_apart` of them), the segmented grid's beats
    must sit on clear attacks (`seg_min_evidence` times the median flux), stronger than the
    constant grid's there by `seg_margin`."""
    _, ts = segmented.beats(start, end)
    _, tc = constant.beats(start - 2.0, end + 2.0)
    if len(ts) < 2 or len(tc) < 2:
        return False, {}
    idx = np.clip(np.searchsorted(tc, ts), 1, len(tc) - 1)
    nearest = np.where(np.abs(tc[idx - 1] - ts) <= np.abs(tc[idx] - ts), tc[idx - 1], tc[idx])
    apart = np.abs(nearest - ts) > 0.010
    diag = {"apart_beats": int(apart.sum())}
    if apart.sum() == 0:
        return False, diag
    # The constant grid's beats in the same stretches.
    _, tc_in = constant.beats(start, end)
    js = np.clip(np.searchsorted(ts, tc_in), 1, len(ts) - 1)
    near_s = np.where(np.abs(ts[js - 1] - tc_in) <= np.abs(ts[js] - tc_in), js - 1, js)
    c_apart = apart[near_s] & (np.abs(ts[near_s] - tc_in) > 0.010)
    floor = float(np.median(flux.astype(np.float64))) + 1e-12
    es = float(attack_evidence(flux, dt, t_first, ts[apart]).mean()) / floor
    ec = float(attack_evidence(flux, dt, t_first, tc_in[c_apart]).mean()) / floor if c_apart.any() else 0.0
    diag.update(evidence_segmented=es, evidence_constant=ec)
    ok = apart.sum() >= p.seg_min_apart and es >= p.seg_min_evidence and es >= ec * (1 + p.seg_margin)
    return bool(ok), diag


def phase_jump(a: Fit, at: float) -> float:
    """Where a beat at `at` falls on a's grid: (fraction of a beat, -0.5 to 0.5)."""
    period = 60.0 / a.bpm
    x = (at - a.downbeat) / period
    return float(x - round(x))


def offending(fits: list[Fit], changes: list[float], start: float, end: float, p: Params) -> int | None:
    """The first join (index into `changes`) that does not count: no tempo change worth the
    name (and no phase jump that counts), or a segment shorter than `seg_min_bars` either side
    of it. None when every join counts."""
    edges = [start] + changes + [end]
    for k, (a, b) in enumerate(zip(fits, fits[1:])):
        change = abs(b.bpm / a.bpm - 1)
        if change < p.seg_min_tempo_change:
            jump = abs(phase_jump(a, changes[k]))
            if p.seg_min_jump <= 0 or jump < p.seg_min_jump:
                return k
    for k, f in enumerate(fits):
        bars = (edges[k + 1] - edges[k]) * f.bpm / 240.0
        if bars < p.seg_min_bars:
            if k == 0:
                return 0
            if k == len(fits) - 1:
                return k - 1
            # Merge into the neighbour closer in tempo.
            left, right = abs(f.bpm / fits[k - 1].bpm - 1), abs(fits[k + 1].bpm / f.bpm - 1)
            return k - 1 if left <= right else k
    return None


# --- all together ----------------------------------------------------------------------------

def fit(track, act: Activations, p: Params, bpm: float, downbeat: float) -> tuple[Segmented | None, dict]:
    """A segmented grid (before the octave convention) for a track whose constant fit is
    `bpm` through `downbeat`, or None when one tempo should stay. The seg_* knobs of `p` say
    how hard to look (pipeline.TEMPO_MODES). The diagnostics hold the candidate even when it is
    not taken."""
    flux, dt, t_first = track.flux()
    duration = track.duration if hasattr(track, "duration") else float(act.times[-1])
    diag: dict = {}
    if duration < 2 * p.seg_window_s:
        return None, diag
    bpms = tempo_states(bpm, p)
    starts = window_starts(0.0, duration, p)
    path, coh, strength = tempo_path(act, starts, bpms, p)
    an = anchors(starts, bpms, path, coh, p)
    runs = partition(an, p.seg_penalty, p)
    diag["runs"] = len(runs)
    if len(runs) < 2:
        return None, diag
    # Where the beat is clear: from the first anchor's window to the last one's.
    start = max(an.time[0] - p.seg_window_s / 2, 0.0)
    end = min(an.time[-1] + p.seg_window_s / 2, duration)
    # Rough changes: halfway between the last anchor of one run and the first of the next.
    rough = [0.5 * (an.time[j - 1] + an.time[j]) for _, j in runs[:-1]]
    edges = [0.0] + rough + [duration]
    fits = []
    for r, (i, j) in enumerate(runs):
        lo = edges[r] + (p.seg_hop_s if r > 0 else 0.0)
        hi = edges[r + 1] - (p.seg_hop_s if r < len(runs) - 1 else 0.0)
        _, _, slope = line_cost(an.number[i:j], an.time[i:j], an.weight[i:j], p)
        fits.append(fit_stretch(track, act, flux, dt, t_first, lo, hi, 60.0 / slope, p))
    # Neighbours on one grid are one segment.
    merged, kept = [fits[0]], []
    for f, at in zip(fits[1:], rough):
        if same_grid(merged[-1], f, at):
            prev = merged.pop()
            merged.append(fit_stretch(track, act, flux, dt, t_first, prev.lo, f.hi, prev.bpm, p))
        else:
            merged.append(f)
            kept.append(at)
    rough = kept
    # Joins this mode does not count are merged too, one at a time.
    while len(merged) > 1:
        changes = [join(a, b, r, flux, dt, t_first, p) for a, b, r in zip(merged, merged[1:], rough)]
        k = offending(merged, changes, start, end, p)
        if k is None:
            break
        a, b = merged[k], merged[k + 1]
        seed = a.bpm if a.hi - a.lo >= b.hi - b.lo else b.bpm
        merged[k : k + 2] = [fit_stretch(track, act, flux, dt, t_first, a.lo, b.hi, seed, p)]
        del rough[k]
    diag["segments_fitted"] = len(merged)
    if len(merged) < 2:
        return None, diag
    first = merged[0]
    period = 60.0 / first.bpm
    segs = [Segment(first.downbeat % (4 * period), first.bpm, 1)]
    for f, at in zip(merged[1:], changes):
        if at - segs[-1].start < 4 * segs[-1].period - 1e-9:  # a segment holds at least a bar
            continue
        segs.append(Segment(at, f.bpm, 1))
    if len(segs) < 2:
        return None, diag
    segmented = Segmented(segs, duration)
    ok, g = gate(Segmented.constant(bpm, downbeat, duration), segmented, flux, dt, t_first, start, end, p)
    diag.update(g)
    diag["candidate"] = segmented.as_json()
    diag["segments"] = [(round(s.start, 3), round(s.bpm, 3)) for s in segs]
    return (segmented if ok else None), diag


def with_octave(segmented: Segmented, p: Params) -> Segmented:
    """The octave convention, per segment (each keeps its first downbeat)."""
    return Segmented([Segment(s.start, apply_octave(s.bpm, p), s.beat_in_bar) for s in segmented.segments],
                     segmented.end)
