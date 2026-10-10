"""How well do the review reasons pick out the grids a DJ would have to fix?

Runs the pipeline on the reference sets (each in the octave profile its accuracy is quoted
with), labels every grid that is not usable@10 against its reference with the kind of
failure, and reports which share of those the review reasons catch (recall) and which share
of the usable grids they flag anyway (false flags): 0.0.1's reasons (recomputed from the
same diagnostics) against the current ones. Failure kinds, in order of precedence:

  variable   the reference's beat times are not one constant tempo (a transition edit, a
             mashup, a wobbling hand-made grid): no constant grid can match it
  octave     tempo off by an octave or another ratio
  drift      the right octave, but the tempo differs enough to walk >10 ms over the music
  phase      the same tempo, the beats off by a constant amount
  bar        the beats are right, bar 1 is not
  nogrid     no grid at all

  uv run python experiments/review_flags.py --split dev --changes --candidates
  uv run python experiments/review_flags.py --split test --changes
  uv run python experiments/review_flags.py --split all --accepted --changes   # never used for tuning

--changes scores each change to the reasons that was tried, alone on top of 0.0.1's, and the
set tuned on the dev split; --candidates scores further signals that were not kept.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from beatgrid_lab import refs
from beatgrid_lab.config import RUNS
from beatgrid_lab.metrics import score
from beatgrid_lab.pipeline import PROFILES, Params, _smooth, coherence_at, fold, phase_of, run, wrap
from beatgrid_lab.track import Track

# (reference set, kind, profile): the profiles the README's accuracy table quotes.
SETS = [("engine", "corrected", "edm"), ("crate", "approved", "edm"), ("crate", "trusted", "edm"),
        ("raveform", "expert", None)]
KINDS = ["variable", "octave", "drift", "phase", "bar", "nogrid"]


def constant_share(grid, start: float, end: float, within: float = 0.010) -> float:
    """Share of the reference's beats (in order, numbering ignored) within `within` seconds of
    the best constant tempo through them."""
    _, t = grid.beats(start, end)
    if len(t) < 8:
        return 1.0
    k = np.arange(len(t), dtype=float)
    a = np.vstack([k, np.ones_like(k)]).T
    residual = t - a @ np.linalg.lstsq(a, t, rcond=None)[0]
    return float(np.mean(np.abs(residual - np.median(residual)) <= within))


def failure(s, ref_grid, grid, start: float, end: float, ms: int = 10) -> str:
    """What is wrong with a grid at a tolerance of `ms` (10 or 20): "ok" when it is usable."""
    if s.usable_10 if ms == 10 else s.usable_20:
        return "ok"
    if constant_share(ref_grid, start, end, ms / 1000) < 0.9:
        return "variable"
    if grid is None:
        return "nogrid"
    if s.octave != "1x":
        return "octave"
    if abs(grid.bpm - ref_grid.bpm) / ref_grid.bpm * (end - start) * 1000 > ms:
        return "drift"
    if (s.within_10 if ms == 10 else s.within_20) < 0.9:
        return "phase"
    return "bar"


def disagree(d: dict) -> bool:
    v = d.get("bar_votes") or {}
    return "network" in v and int(np.argmax(v["network"])) != int(np.argmax(v["changes"]))


def old_flag(d: dict) -> bool:
    """Whether beatgrid 0.0.1 gave any review reason, from the same diagnostics."""
    if "bpm" not in d:
        return True
    return d.get("tempo_source") != "flux" or d["coherence_final"] < 0.5 or d["bend_ms"] > 40 or disagree(d)


# The changes that were tried, each a predicate on the diagnostics, with the thresholds tuned on
# the dev split. The first two were kept (they are in pipeline.review_reasons).
CHANGES = {
    "kept: no clear attack in 6+ of 8 stretches": lambda d: d.get("stretch_drift_ms") is None,
    "kept: octave chosen by the profile": lambda d: d["bpm_grid"] != d["bpm"],
    "second attack > 0.4 of the first": lambda d: d["attack_second"] > 0.4,
    "grid walks > 10 ms off the attacks": lambda d: (d.get("stretch_drift_ms") or 0) > 10 or d.get("snap_drift_ms", 0) > 10,
    "bar-1 margin < 0.5": lambda d: d["bar_margin"] < 0.5,
}
# Replaces 0.0.1's "the downbeat cues disagree" rather than adding to it.
TIGHTER_BAR = "cues disagree only when the margin is < 4"


def tighter_bar(d: dict) -> bool:
    return disagree(d) and d["bar_margin"] < 4.0


def old_without_bar(d: dict) -> bool:
    if "bpm" not in d:
        return True
    return d.get("tempo_source") != "flux" or d["coherence_final"] < 0.5 or d["bend_ms"] > 40


def dev_tuned(d: dict) -> bool:
    """The set tuned on the dev split and scored once on the held-out test: every change."""
    if "bpm" not in d:
        return True
    return old_without_bar(d) or tighter_bar(d) or any(f(d) for f in CHANGES.values())


def candidates(track, params, d: dict, grid) -> dict:
    """Signals that were tried and not kept (see docs/results.md)."""
    act = track.activations_for(params.network)
    flux, dt, t_first = track.flux()
    bpm = d["bpm"]
    period = 60.0 / bpm
    t0 = float(grid.time_of(0.0)[0])
    k = np.arange(np.ceil(-t0 / period), np.floor((act.times[-1] - t0) / period) + 1).astype(int)

    def strength(offset):
        frames = np.round((t0 + offset + k * period) * act.fps).astype(int)[:, None] + np.arange(-1, 2)[None, :]
        return act.beat[np.clip(frames, 0, len(act.beat) - 1)].max(axis=1)

    on, off = strength(0.0), strength(period / 2)
    even, odd = on[k % 2 == 0].mean(), on[k % 2 == 1].mean()
    folded = fold(flux.astype(np.float64), dt, t_first, bpm, t0, 0.05)
    s = _smooth(folded)
    i = int(np.argmax(s))
    above = s - np.median(folded) >= (s[i] - np.median(folded)) / 2
    lo, hi = i, i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(s) - 1 and above[hi + 1]:
        hi += 1
    return {
        # network beat activation between the beats / on them: double time plausible
        "offbeat_ratio": float(off.mean() / (on.mean() + 1e-9)),
        # weaker / stronger of alternate beats: half time plausible
        "alternate_ratio": float(min(even, odd) / (max(even, odd) + 1e-9)),
        "fold_width_ms": float((hi - lo + 1) * dt * 1000),
        "network_vs_attack_ms": float(wrap(t0 - phase_of(coherence_at(act, bpm), period), period) * 1000),
    }


def analyse(job) -> dict:
    ref, set_name, profile, with_candidates = job
    params = Params(**PROFILES[profile]) if profile else Params()
    track = Track(ref.audio)
    start, end = track.music_span
    result = run(track, params)
    s = score(result.grid, ref.grid, start, end)
    d = result.diagnostics
    row = {"id": ref.id, "set": set_name, "split": ref.split,
           "fail": failure(s, ref.grid, result.grid, start, end), "fail_20": failure(s, ref.grid, result.grid, start, end, 20),
           "before": old_flag(d), "review": d.get("review", ["no steady beat to grid"]) if "bpm" in d else ["no steady beat to grid"],
           "usable_10": s.usable_10, "usable_20": s.usable_20, "octave": s.octave, "bpm": s.bpm, "ref_bpm": s.ref_bpm,
           "diagnostics": {k: v for k, v in d.items() if k != "review"}}
    if with_candidates and result.grid is not None:
        row["candidates"] = candidates(track, params, d, result.grid)
    return row


def rate(flags: np.ndarray, mask: np.ndarray) -> str:
    n = int(mask.sum())
    return f"{flags[mask].mean():4.0%} ({int(flags[mask].sum())}/{n})" if n else "  -"


def table(rows: list[dict], audit: dict, key: str = "fail") -> None:
    fail = np.array([r[key] for r in rows])
    bad = fail != "ok"
    before = np.array([r["before"] for r in rows])
    after = np.array([bool(r["review"]) for r in rows])
    print(f"{'':22s} {'0.0.1':>14s} {'now':>14s}")
    print(f"{'recall (wrong grids)':22s} {rate(before, bad):>14s} {rate(after, bad):>14s}")
    print(f"{'false flags (usable)':22s} {rate(before, ~bad):>14s} {rate(after, ~bad):>14s}")
    for kind in KINDS:
        m = fail == kind
        if m.any():
            print(f"{'  ' + kind:22s} {rate(before, m):>14s} {rate(after, m):>14s}")
    for name in sorted({r["set"] for r in rows}):
        m = np.array([r["set"] == name for r in rows])
        print(f"  {name:20s} recall {rate(before, m & bad)} -> {rate(after, m & bad)}   "
              f"false {rate(before, m & ~bad)} -> {rate(after, m & ~bad)}")
    steady = np.array([audit.get(r["id"]) == "steady" for r in rows])
    print(f"  {'audit-steady refs':20s} recall {rate(before, steady & bad)} -> {rate(after, steady & bad)}   "
          f"false {rate(before, steady & ~bad)} -> {rate(after, steady & ~bad)}")


def reasons(rows: list[dict]) -> None:
    bad = np.array([r["fail"] != "ok" for r in rows])
    names = sorted({x for r in rows for x in r["review"]})
    print(f"\n{'reason (now)':66s} {'catches':>8s} {'alone':>6s} {'false':>6s}")
    for name in names:
        has = np.array([name in r["review"] for r in rows])
        alone = np.array([r["review"] == [name] for r in rows])
        print(f"{name:66s} {int(has[bad].sum()):8d} {int(alone[bad].sum()):6d} {int(has[~bad].sum()):6d}")


def changes(rows: list[dict]) -> None:
    bad = np.array([r["fail"] != "ok" for r in rows])
    gridded = np.array(["bpm" in r["diagnostics"] for r in rows])
    before = np.array([r["before"] for r in rows])
    print(f"\n{'change, alone on top of 0.0.1':46s} {'+catches':>9s} {'+false':>7s}")
    for name, f in CHANGES.items():
        on = before | np.array([g and f(r["diagnostics"]) for g, r in zip(gridded, rows)])
        print(f"{name:46s} {int(on[bad].sum() - before[bad].sum()):+9d} {int(on[~bad].sum() - before[~bad].sum()):+7d}")
    on = np.array([not g or old_without_bar(r["diagnostics"]) or tighter_bar(r["diagnostics"]) for g, r in zip(gridded, rows)])
    print(f"{TIGHTER_BAR:46s} {int(on[bad].sum() - before[bad].sum()):+9d} {int(on[~bad].sum() - before[~bad].sum()):+7d}")
    tuned = np.array([dev_tuned(r["diagnostics"]) for r in rows])
    print(f"\nset tuned on dev (all of the above): recall {rate(tuned, bad)}, false flags {rate(tuned, ~bad)}")
    fail = np.array([r["fail"] for r in rows])
    print("  by kind: " + ", ".join(f"{k} {rate(tuned, fail == k)}" for k in KINDS if (fail == k).any()))


def candidate_report(rows: list[dict]) -> None:
    rows = [r for r in rows if "candidates" in r]
    bad = np.array([r["fail"] != "ok" for r in rows])
    flagged = np.array([bool(r["review"]) for r in rows])
    c = lambda name: np.array([r["candidates"][name] for r in rows])
    fast = np.array([r["diagnostics"]["bpm"] >= 140 for r in rows])
    slow = np.array([r["diagnostics"]["bpm"] < 100 for r in rows])
    tests = {
        "half time plausible (>=140 BPM, alternate beats < 0.8)": fast & (c("alternate_ratio") < 0.8),
        "double time plausible (<100 BPM, offbeats > 0.4)": slow & (c("offbeat_ratio") > 0.4),
        "broad attack (fold > 8 ms wide)": c("fold_width_ms") > 8,
        "network phase >12 ms from the usual offset": np.abs(c("network_vs_attack_ms") + 9) > 12,
    }
    print(f"\n{'candidate, on top of the reasons now':58s} {'+catches':>9s} {'+false':>7s}")
    for name, f in tests.items():
        new = f & ~flagged
        print(f"{name:58s} {int(new[bad].sum()):+9d} {int(new[~bad].sum()):+7d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--accepted", action="store_true",
                    help="score on Engine's own grids that a DJ played unedited instead (not used for tuning)")
    ap.add_argument("--changes", action="store_true", help="score each change that was tried, and the dev-tuned set")
    ap.add_argument("--candidates", action="store_true", help="also measure further signals that were not kept")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    sets = [("engine", "accepted", None)] if args.accepted else SETS
    jobs = [(r, f"{name} {kind}", profile, args.candidates)
            for name, kind, profile in sets for r in refs.load(name)
            if r.kind == kind and (args.split == "all" or r.split == args.split)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(analyse, jobs, chunksize=1))
    audit = {}
    for name in {name for name, *_ in sets}:
        path = refs.REFS / f"{name}.audit.json"
        if path.exists():
            audit.update({k: v["verdict"] for k, v in json.loads(path.read_text()).items()})
    print(f"split={args.split}: {len(rows)} tracks, {sum(r['fail'] != 'ok' for r in rows)} not usable@10\n")
    table(rows, audit)
    reasons(rows)
    if args.changes:
        changes(rows)
    if args.candidates:
        candidate_report(rows)
    print(f"\nthe same at 20 ms ({sum(r['fail_20'] != 'ok' for r in rows)} not usable@20):")
    table(rows, audit, "fail_20")
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"review-flags-{'accepted-' if args.accepted else ''}{args.split}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nper-track rows: {out}")


if __name__ == "__main__":
    main()
