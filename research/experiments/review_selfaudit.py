"""Do a self-audit of beatgrid's own grid and disagreement between the bar votes predict a
grid a DJ would have to fix?

  uv run python experiments/review_selfaudit.py runs/cmp-student-v1-*-all.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from beatgrid_lab import refs
from beatgrid_lab.audit import audit
from beatgrid_lab.pipeline import Params, run
from beatgrid_lab.track import Track

sys.path.insert(0, "experiments")
from review_signals import auc  # noqa: E402


def features(job):
    ref, network = job
    track = Track(ref.audio)
    result = run(track, Params(network=network))
    if result.grid is None:
        return ref.id, None
    flux, dt, t_first = track.flux()
    start, end = track.music_span
    a = audit(result.grid, flux, dt, t_first, start, end)
    votes = result.diagnostics.get("bar_votes", {})
    disagree = float(np.argmax(votes.get("network", [1, 0, 0, 0])) != np.argmax(votes.get("changes", [1, 0, 0, 0])))
    return ref.id, {"self_drift": a["drift_ms"] if a["drift_ms"] is not None else 50.0,
                    "self_spread": a["spread_ms"] if a["spread_ms"] is not None else 10.0,
                    "self_unclear": float(a["verdict"] == "unclear"),
                    "self_not_steady": float(a["verdict"] != "steady"),
                    "votes_disagree": disagree}


def main():
    labels, sets = {}, {}
    for p in sys.argv[1:]:
        for r in json.load(open(p))["rows"]:
            if "bpm" in r["diagnostics"]:
                labels[r["id"]] = not r["pipeline_octfree"]["usable_20"]
    tracks = [r for s in ("engine", "crate", "raveform") for r in refs.load(s) if r.id in labels]
    with ProcessPoolExecutor(12) as pool:
        feats = {k: v for k, v in pool.map(features, [(r, "student-v1") for r in tracks]) if v}
    ids = list(feats)
    bad = np.array([labels[i] for i in ids])
    for name in next(iter(feats.values())):
        print(f"  {name:18s} AUC {auc([feats[i][name] for i in ids], bad):.3f}")
    flag = np.array([feats[i]["self_not_steady"] or feats[i]["votes_disagree"] for i in ids], bool)
    print(f"\nflag = self-audit not steady OR votes disagree: catches {flag[bad].mean():.0%} of bad, flags {flag[~bad].mean():.0%} of good")
    for th in (4, 6, 8, 12):
        f = np.array([feats[i]["self_drift"] > th or feats[i]["self_unclear"] or feats[i]["votes_disagree"] for i in ids], bool)
        print(f"flag = self drift > {th} ms OR unclear OR votes disagree: catches {f[bad].mean():.0%}, flags {f[~bad].mean():.0%} of good")
    json.dump({i: {**feats[i], "bad": bool(labels[i])} for i in ids}, open("runs/review-features.json", "w"))


if __name__ == "__main__":
    main()
