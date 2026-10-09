"""Which diagnostics predict a grid a DJ would have to fix?

Pools the per-track diagnostics from evaluation runs, labels each track by whether the grid
was usable at 20 ms (octave-free: octave is a convention, not an error), and reports how
well each signal separates good from bad (AUC), plus what the current review flag catches.

  uv run python experiments/review_signals.py runs/cmp-student-v1-*-all.json
"""

import json
import sys

import numpy as np


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


FEATURES = {
    "low coherence": lambda d: -d["coherence_final"],
    "low flux prominence": lambda d: -d.get("flux_prominence", 0.0),
    "low attack prominence": lambda d: -d["attack_prominence"],
    "bend_ms": lambda d: d["bend_ms"],
    "not snapped": lambda d: 0.0 if d["snapped"] else 1.0,
    "tempo from activation": lambda d: 0.0 if d["tempo_source"] == "flux" else 1.0,
    "low bar margin": lambda d: -d["bar_margin"],
    "|flux - activation bpm|": lambda d: abs(d.get("bpm_flux", d["bpm"]) - d["bpm_activation"]),
}


def load(paths):
    rows = []
    for p in paths:
        for r in json.load(open(p))["rows"]:
            d = r["diagnostics"]
            if "bpm" in d:
                rows.append((r["split"], not r["pipeline_octfree"]["usable_20"], d))
    return rows


def main():
    rows = load(sys.argv[1:])
    bad = np.array([b for _, b, _ in rows])
    print(f"{len(rows)} tracks; {bad.sum()} not usable @20 ms (octave-free)\n")
    for name, f in FEATURES.items():
        print(f"  {name:26s} AUC {auc([f(d) for *_, d in rows], bad):.3f}")
    flagged = np.array([bool(d.get("review")) for *_, d in rows])
    print(f"\ncurrent review flag: catches {flagged[bad].mean():.0%} of bad grids, flags {flagged[~bad].mean():.0%} of good ones")


if __name__ == "__main__":
    main()
