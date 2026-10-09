"""Can the high-resolution flux arbitrate tempos the network cannot tell apart?

For tracks where the pipeline's tempo differs slightly from the reference, measure how
sharply the flux folds at each tempo: attack position in each quarter of the track (a wrong
tempo makes it walk) and the height of the whole-track fold.

  uv run python experiments/tempo_by_flux.py runs/<run>.json
"""

import json
import sys

import numpy as np

from beatgrid_lab import refs
from beatgrid_lab.pipeline import Params, attack_offset
from beatgrid_lab.track import Track


def walk(flux, dt, t_first, bpm, t0, duration):
    """Attack offset (ms) in each quarter of the track, relative to a grid at bpm/t0."""
    out = []
    for q in range(4):
        lo, hi = duration * q / 4, duration * (q + 1) / 4
        i0, i1 = int((lo - t_first) / dt), int((hi - t_first) / dt)
        seg = np.zeros_like(flux)
        seg[max(i0, 0) : max(i1, 0)] = flux[max(i0, 0) : max(i1, 0)]
        shift, prom = attack_offset(seg, dt, t_first, bpm, t0, Params(attack_search_ms=60))
        out.append((shift * 1000, prom))
    return out


def main():
    rows = json.load(open(sys.argv[1]))["rows"]
    by_id = {r.id: r for r in refs.load("engine")}
    for row in rows:
        p = row["pipeline"]
        if p["octave"] != "1x" or p["bpm_error"] < 0.02:
            continue
        ref = by_id[row["id"]]
        track = Track(ref.audio)
        flux, dt, t_first = track.flux()
        mid = track.duration / 2
        n = np.round(ref.grid.number_at(mid)[0])
        t_ref = float(ref.grid.time_of(n)[0])
        print(f"{row['id']:12s} {row['kind'][:4]} {row['title'][:50]}")
        for label, bpm in (("ref", ref.grid.bpm), ("ours", p["bpm"])):
            # Anchor both grids on the reference's beat in the middle of the track.
            w = walk(flux, dt, t_first, bpm, t_ref, track.duration)
            shifts = [s for s, _ in w]
            print(f"   {label:4s} {bpm:8.3f}  attack per quarter (ms): " + " ".join(f"{s:+6.1f}" for s in shifts)
                  + f"   walk {max(shifts) - min(shifts):5.1f} ms   prominence " + " ".join(f"{pr:4.1f}" for _, pr in w))


if __name__ == "__main__":
    main()
