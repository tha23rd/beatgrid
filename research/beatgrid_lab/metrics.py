"""DJ-grade grid metrics.

Research beat tracking scores a beat as right if it lands within ±70 ms. A DJ needs the
grid within a few milliseconds over the whole track, at the tempo they count in, with bar
lines on the one. So each track is scored on:

- tempo octave: the ratio of detected to reference tempo (1x, 2x, 0.5x, other)
- bpm_error: absolute tempo difference, at the reference octave
- offsets: for every reference beat in the music, signed distance to the nearest detected
  beat (detected minus reference), summarised as median/p90 |offset| and bias
- within_5/10/20: share of reference beats with a detected beat that close
- bar_ok: the detected downbeats fall on reference downbeats
- usable_10 / usable_20: right octave, ≥90% of beats within 10/20 ms, and bar_ok —
  the grid could go on a stick without being touched

"octave-free" variants re-run the scoring after bringing a 2x/0.5x grid to the reference
tempo, separating "found the pulse" from "counted it the way this DJ counts".
"""

from dataclasses import asdict, dataclass

import numpy as np

from .grid import Grid

USABLE_SHARE = 0.9


@dataclass
class Score:
    octave: str
    bpm: float
    ref_bpm: float
    bpm_error: float
    median_abs_ms: float
    p90_abs_ms: float
    bias_ms: float
    within_5: float
    within_10: float
    within_20: float
    bar_ok: bool
    usable_10: bool
    usable_20: bool
    beats_scored: int

    def as_dict(self) -> dict:
        return asdict(self)


def octave_of(ratio: float) -> str:
    for name, target in (("1x", 1.0), ("2x", 2.0), ("0.5x", 0.5), ("1.5x", 1.5), ("0.67x", 2 / 3)):
        if abs(ratio / target - 1) < 0.04:
            return name
    return "other"


def score(detected: Grid | None, reference: Grid, start: float, end: float) -> Score:
    """Score `detected` against `reference` over the reference beats in [start, end]."""
    ref_n, ref_t = reference.beats(start, end)
    if detected is None or len(ref_t) == 0:
        return Score("none", 0.0, reference.bpm, np.inf, np.inf, np.inf, np.nan,
                     0.0, 0.0, 0.0, False, False, False, len(ref_t))
    octave = octave_of(detected.bpm / reference.bpm)
    det_n, det_t = detected.beats(start - 1.0, end + 1.0)
    if len(det_t) == 0:
        return Score(octave, detected.bpm, reference.bpm, np.inf, np.inf, np.inf, np.nan,
                     0.0, 0.0, 0.0, False, False, False, len(ref_t))
    idx = np.clip(np.searchsorted(det_t, ref_t), 1, len(det_t) - 1)
    left, right = det_t[idx - 1], det_t[idx]
    nearest = np.where(np.abs(left - ref_t) <= np.abs(right - ref_t), idx - 1, idx)
    offsets = det_t[nearest] - ref_t
    absolute = np.abs(offsets) * 1000
    close = absolute <= 70
    # A bar line is right when the detected beat nearest each reference downbeat is
    # itself a downbeat (and close enough to count as the same beat).
    down = (np.mod(ref_n, 4) == 0) & close
    bar_ok = bool(down.any()) and np.mean(np.mod(det_n[nearest[down]], 4) == 0) > 0.5
    w10, w20 = float(np.mean(absolute <= 10)), float(np.mean(absolute <= 20))
    ok = octave == "1x" and bar_ok
    return Score(
        octave=octave,
        bpm=detected.bpm,
        ref_bpm=reference.bpm,
        bpm_error=abs(detected.bpm - reference.bpm) if octave == "1x" else np.inf,
        median_abs_ms=float(np.median(absolute)),
        p90_abs_ms=float(np.percentile(absolute, 90)),
        bias_ms=float(np.median(offsets[close]) * 1000) if close.any() else np.nan,
        within_5=float(np.mean(absolute <= 5)),
        within_10=w10,
        within_20=w20,
        bar_ok=bar_ok,
        usable_10=ok and w10 >= USABLE_SHARE,
        usable_20=ok and w20 >= USABLE_SHARE,
        beats_scored=len(ref_t),
    )


def at_reference_octave(detected: Grid, reference: Grid, start: float, end: float) -> Grid:
    """Bring a double- or half-tempo grid to the reference tempo, choosing whichever of
    the possible halvings (or the doubling) lines up with the reference best."""
    octave = octave_of(detected.bpm / reference.bpm)
    if octave == "2x":
        candidates = [detected.halved(False), detected.halved(True)]
        candidates += [c.renumbered(2) for c in candidates]
    elif octave == "0.5x":
        doubled = detected.doubled()
        candidates = [doubled, doubled.renumbered(2)]
    else:
        return detected
    return max(candidates, key=lambda g: (lambda s: (s.bar_ok, s.within_20))(score(g, reference, start, end)))


def music_span(samples: np.ndarray, rate: int, floor_db: float = 30.0) -> tuple[float, float]:
    """First and last moment the track is within `floor_db` of its loudest 100 ms, so
    silence and fade tails (where there is no beat to find) are not scored."""
    hop = rate // 10
    n = len(samples) // hop
    if n == 0:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(samples[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = np.where(db > max(db.max() - floor_db, -45.0))[0]
    if len(loud) == 0:
        return 0.0, 0.0
    return loud[0] * hop / rate, (loud[-1] + 1) * hop / rate
