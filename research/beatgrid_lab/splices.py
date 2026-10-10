"""Synthetic tempo changes: sections of constant-tempo reference tracks joined end to end, the
way a transition edit or a mashup is made, with the change points and each section's grid
known exactly from the source references.

A splice is a recipe, not a file: its audio is rebuilt from the source tracks whenever it is
needed (only the features computed from it are cached), so no audio is copied. Sources are
references whose grid is one constant tempo, holds steady on its own audio (`lab audit`) and
sits on the attack. A splice built from dev sources is a dev splice, and from test sources a
test splice, so tuning on dev never sees a test track.

Kinds of join, by the tempo ratio between neighbouring sections:

  large    6-39% (124 -> 150), different tracks at their own tempo
  medium   1.5-6%, different tracks at their own tempo
  small    0.2-1.5%: the second track resampled to that ratio (a pitch change of at most a
           few tenths of a semitone)
  jump     the same tempo, but the next section cut a fraction of a beat late: the phase jumps
  same     the same tempo and phase, bar lines continuing: one grid, which must not be split

Each join is a cut (a 20 ms equal-power crossfade, against clicks) or a one-bar crossfade
centred on the change. The change point is where the next section's first beat lands.

  uv run lab splices --dev 240 --test 120      # write DATA/refs/splices.jsonl and cache features
"""

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from . import audio, refs
from .config import CACHE, DATA, REFS
from .grid import Grid
from .metrics import music_span
from .track import Track

RATE = 44100
CUT_FADE_S = 0.02
KINDS = {"large": 0.35, "medium": 0.2, "small": 0.2, "jump": 0.1, "same": 0.15}
SOURCE_SETS = [("engine", ("corrected", "accepted")), ("crate", ("approved", "trusted")), ("raveform", ("expert",))]


@dataclass
class Source:
    id: str
    audio: Path
    grid: Grid
    split: str
    bpm: float
    span: tuple[float, float]


def constant_reference(ref) -> bool:
    """One tempo: every beat within 1 ms of a straight line through them."""
    n, t = ref.grid.numbers, ref.grid.times
    if len(n) == 2:
        return True
    a = np.vstack([n, np.ones_like(n)]).T
    residual = t - a @ np.linalg.lstsq(a, t, rcond=None)[0]
    return float(np.max(np.abs(residual))) < 0.0015


def sources() -> list[Source]:
    out = []
    for name, kinds in SOURCE_SETS:
        audit = json.loads((REFS / f"{name}.audit.json").read_text())
        for r in refs.load(name):
            a = audit.get(r.id, {})
            if r.kind not in kinds or a.get("verdict") != "steady" or a.get("attack_ms") is None:
                continue
            if abs(a["attack_ms"] + 2.5) > 7.0 or not constant_reference(r):
                continue
            info = CACHE / "info" / f"{Track(r.audio).key}.json"
            if not info.exists():
                continue
            i = json.loads(info.read_text())
            out.append(Source(r.id, r.audio, r.grid, r.split, float(r.grid.bpm), (i["music_start"], i["music_end"])))
    return out


def _downbeats(src: Source, lo: float, hi: float) -> np.ndarray:
    n, t = src.grid.beats(lo, hi)
    return t[np.mod(n, 4) == 0]


def _pick_pair(rng, pool: list[Source], kind: str, a: Source, tempo: float) -> tuple[Source, float] | None:
    """A next source after `a` (playing at `tempo`) and the rate to play it at, for this kind
    of join."""
    ratio = lambda b: max(b.bpm / tempo, tempo / b.bpm)
    if kind == "large":
        cand = [(b, 1.0) for b in pool if b.id != a.id and 1.06 <= ratio(b) <= 1.39]
    elif kind == "medium":
        cand = [(b, 1.0) for b in pool if b.id != a.id and 1.015 <= ratio(b) < 1.06]
    elif kind == "small":
        cand = []
        for b in pool:
            if b.id == a.id or ratio(b) > 1.03:
                continue
            target = tempo * (1 + rng.choice([-1, 1]) * rng.uniform(0.002, 0.015))
            cand.append((b, target / b.bpm))
    else:  # jump, same: the same tempo, exactly
        cand = [(b, tempo / b.bpm) for b in pool if b.id != a.id and ratio(b) <= 1.03]
    if not cand:
        return None
    return cand[rng.integers(len(cand))]


def build(pool: list[Source], count: int, split: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    names = list(KINDS)
    weights = np.array([KINDS[k] for k in names])
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        kind = str(rng.choice(names, p=weights / weights.sum()))
        sections = 2 if rng.random() < 0.6 else 3
        xfade = "cut" if rng.random() < 0.6 else "xfade"
        chain = [(pool[rng.integers(len(pool))], 1.0)]
        for _ in range(sections - 1):
            nxt = _pick_pair(rng, pool, kind, chain[-1][0], chain[-1][0].bpm * chain[-1][1])
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) < sections:
            continue
        recipe = _recipe(rng, chain, kind, xfade)
        if recipe is None:
            continue
        recipe.update(id=f"splice:{split}:{len(out):04d}", split=split, kind=kind, transition=xfade)
        out.append(recipe)
    return out


def _recipe(rng, chain, kind: str, xfade: str) -> dict | None:
    parts = []
    at = 0.0
    for k, (src, rate) in enumerate(chain):
        first, last = k == 0, k == len(chain) - 1
        lo, hi = src.span
        bars = int(rng.integers(24, 65)) if first or last else int(rng.integers(16, 49))
        downs = _downbeats(src, lo + 1.0, hi - 1.0)
        bar_s = 4 * 60.0 / src.bpm
        usable = downs[downs + bars * bar_s <= hi - 1.0]
        if len(usable) == 0:
            return None
        if first and rng.random() < 0.5:
            start = 0.0  # keep the intro
            end = float(downs[np.argmin(np.abs(downs - (downs[0] + bars * bar_s)))])
            if end <= downs[0]:
                return None
        else:
            start = float(usable[rng.integers(len(usable))])
            end = start + bars * bar_s
            # Snap the end to the reference's downbeat.
            end = float(downs[np.argmin(np.abs(downs - end))])
        phase = 0.0
        if kind == "jump" and not first:
            phase = float(rng.uniform(0.2, 0.8))  # cut this far into the beat
            start += phase * 60.0 / src.bpm
        if last and rng.random() < 0.5:
            end = float(src.span[1])  # run out to the end of the music
        parts.append({"ref": src.id, "from": start, "to": end, "rate": float(rate), "at": at, "phase": phase})
        at += (end - start) / rate
    return {"parts": parts, "xfade_bars": 1 if xfade == "xfade" else 0}


@lru_cache(maxsize=8)
def _source_audio(path: str) -> tuple[np.ndarray, int]:
    x, rate = audio.load(Path(path))
    if rate != RATE:
        import soxr

        x = soxr.resample(x, rate, RATE).astype(np.float32)
    return x, RATE


class Splice:
    """A splice recipe with its audio (rebuilt on demand) and its true grid."""

    def __init__(self, record: dict, by_id: dict):
        self.record = record
        self.id = record["id"]
        self.split = record["split"]
        self.kind = record["kind"]
        self.transition = record["transition"]
        self.sources = [by_id[p["ref"]] for p in record["parts"]]

    @property
    def key(self) -> str:
        blob = json.dumps({k: self.record[k] for k in ("parts", "xfade_bars")}, sort_keys=True)
        return "splice-" + hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _xfade_s(self, k: int) -> float:
        """Crossfade (output seconds) at the start of part k."""
        if self.record["xfade_bars"]:
            src, p = self.sources[k], self.record["parts"][k]
            return 4 * 60.0 / (src.bpm * p["rate"])
        return CUT_FADE_S

    def render(self) -> np.ndarray:
        parts = self.record["parts"]
        total = parts[-1]["at"] + (parts[-1]["to"] - parts[-1]["from"]) / parts[-1]["rate"]
        out = np.zeros(int(np.ceil((total + 1.0) * RATE)), np.float32)
        for k, (p, src) in enumerate(zip(parts, self.sources)):
            x, _ = _source_audio(str(src.audio))
            fade_in = self._xfade_s(k) if k > 0 else 0.0
            fade_out = self._xfade_s(k + 1) if k + 1 < len(parts) else 0.0
            # Source seconds to take either side, at this part's rate.
            s0 = max(p["from"] - fade_in / 2 * p["rate"], 0.0)
            s1 = min(p["to"] + fade_out / 2 * p["rate"], len(x) / RATE)
            seg = x[int(round(s0 * RATE)) : int(round(s1 * RATE))].astype(np.float64)
            if p["rate"] != 1.0:
                import soxr

                seg = soxr.resample(seg, RATE, RATE / p["rate"], quality="VHQ")
            o0 = p["at"] - (p["from"] - s0) / p["rate"]
            t = o0 + np.arange(len(seg)) / RATE
            gain = np.ones(len(seg))
            if fade_in > 0:
                r = np.clip((t - (p["at"] - fade_in / 2)) / fade_in, 0, 1)
                gain *= np.sin(r * np.pi / 2)
            if fade_out > 0:
                end_at = p["at"] + (p["to"] - p["from"]) / p["rate"]
                r = np.clip((t - (end_at - fade_out / 2)) / fade_out, 0, 1)
                gain *= np.cos(r * np.pi / 2)
            # Level-match the sections, as an edit would.
            body = seg[(gain > 0.99)]
            rms = np.sqrt(np.mean(body**2)) if len(body) else 1.0
            seg = seg * gain * (0.1 / max(rms, 1e-4))
            i0 = int(round(o0 * RATE))
            if i0 < 0:
                seg, i0 = seg[-i0:], 0
            n = min(len(seg), len(out) - i0)
            out[i0 : i0 + n] += seg[:n].astype(np.float32)
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.99:
            out *= 0.99 / peak
        end = int(np.ceil(total * RATE))
        return out[:end]

    @cached_property
    def truth(self) -> tuple[Grid, list[dict]]:
        """The true grid (every beat a marker; downbeats on multiples of four) and its
        sections [{"start": time of the section's first beat, "bpm", "change": time of the
        join that opens it (None for the first)}]. Neighbouring parts on the same grid (the
        `same` kind) are one section."""
        numbers, times, sections = [], [], []
        for k, (p, src) in enumerate(zip(self.record["parts"], self.sources)):
            n, t = src.grid.beats(p["from"] - 1e-6, p["to"] - 1e-6)
            keep = (t >= p["from"] - 1e-6) & (t < p["to"] - 1e-6)
            n, t = n[keep], t[keep]
            out_t = p["at"] + (t - p["from"]) / p["rate"]
            bpm = src.bpm * p["rate"]
            if numbers:
                offset = 4 * np.ceil((numbers[-1] + 1 - n[0]) / 4)
                n = n + offset
                if self.kind == "same":
                    # One grid across the join: number on from the section before.
                    n = n - n[0] + numbers[-1] + 1
            numbers.extend(n.tolist())
            times.extend(out_t.tolist())
            if self.kind == "same" and sections:
                continue
            sections.append({"start": float(out_t[0]), "bpm": float(bpm), "change": float(p["at"]) if k else None,
                             "end": float(p["at"] + (p["to"] - p["from"]) / p["rate"])})
        if self.kind == "same":
            last = self.record["parts"][-1]
            sections[0]["end"] = float(last["at"] + (last["to"] - last["from"]) / last["rate"])
        return Grid(np.array(numbers, float), np.array(times)), sections


class SpliceTrack(Track):
    """A Track whose audio is a splice rendered in memory; features cached by recipe."""

    def __init__(self, splice: Splice):
        self.splice = splice
        self.path = Path(splice.id)
        self.key = splice.key

    @cached_property
    def _decoded(self) -> tuple[np.ndarray, int]:
        return self.splice.render(), RATE

    def activations_for(self, model: str):
        from . import activations

        path = CACHE / f"beatthis-{model}" / f"{self.key}.npz"
        if path.exists():
            d = np.load(path)
            return activations.Activations(d["beat"], d["downbeat"], float(d["fps"]))
        act = activations.compute(self.samples, self.rate, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, beat=act.beat, downbeat=act.downbeat, fps=act.fps)
        return act

    @property
    def activations(self):
        from .activations import MODEL

        return self.activations_for(MODEL)


PATH = REFS / "splices.jsonl"


def save(records: list[dict]) -> Path:
    REFS.mkdir(parents=True, exist_ok=True)
    with open(PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return PATH


def load() -> list[Splice]:
    by_id = {s.id: s for s in sources()}
    return [Splice(json.loads(line), by_id) for line in open(PATH)]
