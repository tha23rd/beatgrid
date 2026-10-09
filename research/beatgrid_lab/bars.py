"""Where is the one? Bar rotation by several independent votes.

No single cue is reliable — the network's downbeat output slips a whole beat on repetitive
techno, a kick on every beat carries no bar information, and a kick-energy vote is chance-
level — but several weak, independent cues pooled together are.

Each vote scores the four rotations r (r = beat index mod 4, counted from the grid's t0,
that is bar 1). Votes are turned into probabilities and pooled log-linearly with weights,
so a vote with nothing to say (flat) costs nothing and a confident one counts.

  network    the downbeat activation folded at the bar period
  backbeat   snare/clap onsets on beats 2 and 4 (separates 1/3 from 2/4)
  changes    the biggest beat-to-beat steps in level and brightness: drops, breakdowns and
             returns land on bar 1
  fills      bursts of high-frequency onsets (crashes, new hat patterns) on a bar line
  prior      another grid's bar lines (Engine's, when a library has one)
"""

from dataclasses import dataclass, field

import numpy as np

# 12 log-spaced bands from 40 Hz to 16 kHz (Track.band_energy): edges 40, 66, 109, 179, 295,
# 486, 801, 1320, 2175, 3585, 5907, 9735, 16000 Hz.
SNARE_BANDS = [3, 4, 7, 8]  # 179-486 Hz body, 1.3-3.6 kHz crack
CLICK_BANDS = [8, 9, 10]    # 2-10 kHz: hats, rolls, risers


@dataclass
class Weights:
    network: float = 1.0
    backbeat: float = 1.5
    changes: float = 2.0
    fills: float = 0.75
    prior: float = 0.5


@dataclass
class Decision:
    rotation: int
    margin: float  # best minus runner-up pooled log score
    votes: dict = field(default_factory=dict)


def beat_frames(bpm: float, t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    period = 60.0 / bpm
    k = np.arange(np.ceil(-t0 / period), np.floor((duration - t0) / period) + 1).astype(int)
    return k, t0 + k * period


def onsets(energy: np.ndarray) -> np.ndarray:
    """Positive frame-to-frame change of log band energy (frames x bands)."""
    d = np.diff(energy, axis=0, prepend=energy[:1])
    return np.maximum(d, 0.0)


def at_beats(values: np.ndarray, dt: float, times: np.ndarray, spread: int = 1) -> np.ndarray:
    """Max of `values` (frames x ...) within ±spread frames of each time."""
    frame = np.round(times / dt - 1).astype(int)  # band frames are centred at (i + 1) * dt
    idx = np.clip(frame[:, None] + np.arange(-spread, spread + 1)[None, :], 0, len(values) - 1)
    return values[idx].max(axis=1)


def per_beat(values: np.ndarray, dt: float, starts: np.ndarray) -> np.ndarray:
    """Mean of `values` (frames x bands) over each beat interval."""
    centres = (np.arange(len(values)) + 1) * dt
    which = np.searchsorted(starts, centres) - 1
    ok = (which >= 0) & (which < len(starts))
    sums = np.zeros((len(starts),) + values.shape[1:])
    counts = np.zeros(len(starts))
    np.add.at(sums, which[ok], values[ok])
    np.add.at(counts, which[ok], 1)
    return sums / np.maximum(counts, 1).reshape((-1,) + (1,) * (values.ndim - 1))


def fold4(values: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.array([values[np.mod(k, 4) == r].mean() if np.any(np.mod(k, 4) == r) else 0.0 for r in range(4)])


def vote_network(act, bpm: float, t0: float) -> np.ndarray:
    k, times = beat_frames(bpm, t0, act.times[-1])
    frames = np.clip(np.round(times * act.fps).astype(int)[:, None] + np.arange(-1, 2)[None, :], 0, len(act.downbeat) - 1)
    return fold4(act.downbeat[frames].max(axis=1), k)


def vote_backbeat(energy: np.ndarray, dt: float, bpm: float, t0: float, active: np.ndarray) -> np.ndarray:
    """Rotation r puts snares on r+1 and r+3."""
    k, times = beat_frames(bpm, t0, len(energy) * dt)
    snare = at_beats(onsets(energy)[:, SNARE_BANDS].sum(axis=1), dt, times)
    s = fold4(np.where(active[: len(k)], snare, 0.0), k) + 1e-9
    return np.array([(s[(r + 1) % 4] + s[(r + 3) % 4]) / (s[r] + s[(r + 2) % 4]) for r in range(4)])


def vote_changes(energy: np.ndarray, dt: float, bpm: float, t0: float, top: int = 12) -> np.ndarray:
    """The `top` biggest beat-to-beat steps in loudness and brightness, by the beat they land
    on; a rotation scores the steps it puts on bar 1."""
    k, starts = beat_frames(bpm, t0, len(energy) * dt)
    if len(k) < 32:
        return np.ones(4)
    beats = per_beat(energy, dt, starts)
    level = np.log(np.exp(beats).sum(axis=1) + 1e-9)
    weights = np.exp(beats)
    centroid = (weights * np.arange(beats.shape[1])).sum(axis=1) / (weights.sum(axis=1) + 1e-9)
    z = lambda x: (x - np.median(x)) / (np.median(np.abs(x - np.median(x))) + 1e-9)
    step = np.abs(np.diff(z(level), prepend=z(level)[0])) + np.abs(np.diff(z(centroid), prepend=z(centroid)[0]))
    # Steps two beats apart describe the same event; keep the larger.
    order = np.argsort(step)[::-1]
    chosen = []
    for i in order:
        if all(abs(i - j) > 2 for j in chosen):
            chosen.append(i)
        if len(chosen) == top:
            break
    votes = np.zeros(4)
    for i in chosen:
        votes[np.mod(k[i], 4)] += step[i]
    return votes + 1e-3 * votes.sum()


def vote_fills(energy: np.ndarray, dt: float, bpm: float, t0: float, top: int = 16) -> np.ndarray:
    """High-frequency onset bursts — crashes and new hat patterns — by the beat they land on;
    a rotation scores bursts on its bar 1. (Read the other way, as fills on the beat before
    the one, they were worse than chance on Raveform: 18%.)"""
    k, starts = beat_frames(bpm, t0, len(energy) * dt)
    if len(k) < 32:
        return np.ones(4)
    density = per_beat(onsets(energy)[:, CLICK_BANDS].sum(axis=1)[:, None], dt, starts)[:, 0]
    local = density - np.convolve(density, np.ones(16) / 16, mode="same")
    votes = np.zeros(4)
    for i in np.argsort(local)[::-1][:top]:
        votes[np.mod(k[i], 4)] += max(local[i], 0.0)
    return votes + 1e-3 * (votes.sum() + 1e-9)


def vote_prior(prior, bpm: float, t0: float, duration: float) -> np.ndarray:
    """Another grid's downbeats, if it agrees on the tempo."""
    if prior is None or abs(prior.bpm / bpm - 1) > 0.01:
        return np.ones(4)
    k, times = beat_frames(bpm, t0, duration)
    n = np.round(prior.number_at(times))
    on = np.mod(n, 4) == 0
    return fold4(on.astype(float), k) + 0.05


def decide(track, act, bpm: float, t0: float, weights: Weights = Weights(), prior=None) -> Decision:
    energy, dt = track.band_energy()
    duration = len(energy) * dt
    k, times = beat_frames(bpm, t0, duration)
    # Energetic beats only for the pattern votes: breakdowns have no backbeat to read.
    level = at_beats(np.log(np.exp(energy).sum(axis=1)), dt, times, spread=2)
    active = level > np.percentile(level, 40)
    votes = {
        "network": vote_network(act, bpm, t0),
        "backbeat": vote_backbeat(energy, dt, bpm, t0, active),
        "changes": vote_changes(energy, dt, bpm, t0),
        "fills": vote_fills(energy, dt, bpm, t0),
        "prior": vote_prior(prior, bpm, t0, duration),
    }
    pooled = np.zeros(4)
    for name, v in votes.items():
        p = np.maximum(v, 1e-9) / np.maximum(v, 1e-9).sum()
        pooled += getattr(weights, name) * np.log(p)
    order = np.argsort(pooled)[::-1]
    return Decision(int(order[0]), float(pooled[order[0]] - pooled[order[1]]),
                    {n: (v / (v.sum() + 1e-12)).round(3).tolist() for n, v in votes.items()})
