"""How good is each bar-1 vote on its own, and what weights pool them best?

For every reference track where the pipeline's tempo is right, find the true rotation (which
of the pipeline's beats the reference calls bar 1), then score each vote's pick and the
pooled pick. Weights are fitted on one split and reported on the other.

  uv run python experiments/bar_votes.py --set raveform --fit dev --report test
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from beatgrid_lab import refs
from beatgrid_lab.bars import Weights, decide
from beatgrid_lab.pipeline import Params, run
from beatgrid_lab.track import Track

NAMES = ["network", "backbeat", "changes", "fills", "prior"]


def analyse(job):
    ref, use_prior, network = job
    track = Track(ref.audio)
    result = run(track, Params(network=network))
    grid = result.grid
    if grid is None or abs(grid.bpm / ref.grid.bpm - 1) > 0.002:
        return None
    period = 60.0 / grid.bpm
    t0 = float(grid.time_of(0.0)[0])  # a downbeat of the pipeline's grid: rotation 0
    start, end = track.music_span
    n, times = ref.grid.beats(start, end)
    down = times[np.mod(n, 4) == 0]
    # Only beats that line up (the reference may be off by a whole beat elsewhere).
    k = np.round((down - t0) / period)
    close = np.abs(down - (t0 + k * period)) < 0.06
    if close.sum() < 8:
        return None
    truth = int(np.bincount(np.mod(k[close], 4).astype(int), minlength=4).argmax())
    d = decide(track, track.activations_for(network), grid.bpm, t0, Weights(), ref.baselines.get("engine") if use_prior else None)
    return {"id": ref.id, "truth": truth, "votes": {k: np.array(v) for k, v in d.votes.items()}}


def pooled_pick(votes: dict, w: dict) -> int:
    s = np.zeros(4)
    for name, v in votes.items():
        p = np.maximum(v, 1e-9) / np.maximum(v, 1e-9).sum()
        s += w.get(name, 0.0) * np.log(p)
    return int(np.argmax(s))


def accuracy(rows, w):
    return np.mean([pooled_pick(r["votes"], w) == r["truth"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="raveform")
    ap.add_argument("--fit", default="dev")
    ap.add_argument("--report", default="test")
    ap.add_argument("--kind")
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--network", default="final0")
    args = ap.parse_args()
    tracks = [r for r in refs.load(args.set) if not args.kind or r.kind == args.kind]
    with ProcessPoolExecutor(12) as pool:
        rows = [r for r in pool.map(analyse, [(t, args.prior, args.network) for t in tracks]) if r]
    split = {t.id: t.split for t in tracks}
    fit = [r for r in rows if split[r["id"]] == args.fit]
    report = [r for r in rows if split[r["id"]] == args.report]
    print(f"{len(rows)} tracks with the right tempo; fit on {len(fit)} ({args.fit}), report on {len(report)} ({args.report})\n")
    names = [n for n in NAMES if n != "prior" or args.prior]
    print("each vote alone (argmax == truth; 'mod 2' = right up to a two-beat shift):")
    for name in names:
        picks = [int(np.argmax(r["votes"][name])) for r in rows]
        flat = np.mean([np.ptp(r["votes"][name]) < 1e-6 for r in rows])
        ok = np.mean([p == r["truth"] for p, r in zip(picks, rows)])
        ok2 = np.mean([p % 2 == r["truth"] % 2 for p, r in zip(picks, rows)])
        print(f"  {name:9s} {ok:5.0%}   mod 2 {ok2:5.0%}   (no opinion on {flat:4.0%})")
    initial = {"network": 1.0, "backbeat": 1.5, "changes": 2.0, "fills": 0.75, "prior": 0.5}  # bars.Weights()
    print(f"\nnetwork only: fit {accuracy(fit, {'network': 1}):.0%}  report {accuracy(report, {'network': 1}):.0%}")
    print(f"initial weights: fit {accuracy(fit, initial):.0%}  report {accuracy(report, initial):.0%}")
    grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    best = max((dict(zip(names, ws)) for ws in itertools.product(grid, repeat=len(names)) if ws[0] > 0),
               key=lambda w: (accuracy(fit, w), -sum(w.values())))
    print(f"fitted weights {best}: fit {accuracy(fit, best):.0%}  report {accuracy(report, best):.0%}")


if __name__ == "__main__":
    main()
