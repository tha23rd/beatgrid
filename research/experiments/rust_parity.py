"""Does the Rust core produce the same grids as the research pipeline?

Feeds both the same network activations (exported from the cache), runs the Rust CLI on the
audio, and compares tempo, beat placement and bar 1 per track; then scores the Rust grids
against the references like any other gridder.

  uv run python experiments/rust_parity.py BINARY --set engine [--profile edm]
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from beatgrid_lab import refs
from beatgrid_lab.config import DATA
from beatgrid_lab.grid import Grid
from beatgrid_lab.metrics import score
from beatgrid_lab.pipeline import PROFILES, Params, run
from beatgrid_lab.track import Track


def python_grid(job):
    ref, profile, network = job
    params = Params(**PROFILES[profile]) if profile else Params()
    params.network = network
    result = run(Track(ref.audio), params)
    return ref.id, result.grid, result.diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--set", default="engine")
    ap.add_argument("--profile")
    ap.add_argument("--network", help="compare the Rust build's own network against this Python checkpoint "
                                      "(default: feed both the cached Beat This! activations)")
    args = ap.parse_args()
    tracks = refs.load(args.set)
    act_dir = DATA / "parity" / "act"
    act_dir.mkdir(parents=True, exist_ok=True)
    for r in tracks if not args.network else []:
        a = Track(r.audio).activations
        np.concatenate([a.beat, a.downbeat]).astype("<f4").tofile(act_dir / f"{r.audio.stem}.act")
    feed = [] if args.network else ["--activations-dir", str(act_dir)]
    t = time.perf_counter()
    out = subprocess.run([args.binary, "analyze", "--profile", args.profile or "none", *feed,
                          *[str(r.audio) for r in tracks]], capture_output=True, text=True, check=True)
    print(f"rust: {len(tracks)} tracks in {time.perf_counter() - t:.1f}s")
    rust = {Path(d["file"]).stem: d for d in map(json.loads, out.stdout.splitlines())}
    with ProcessPoolExecutor(12) as pool:
        py = {rid: (g, d) for rid, g, d in pool.map(python_grid, [(r, args.profile, args.network or "final0") for r in tracks])}

    same_bpm = same_bar = 0
    offsets, rows = [], []
    for r in tracks:
        rd = rust[r.audio.stem]
        if "error" in rd:
            print(f"{r.id}: rust error {rd['error']}")
            continue
        g_py, _ = py[r.id]
        rg = rd["analysis"]["grid"]
        if rg is None or g_py is None:
            print(f"{r.id}: grid python={g_py is not None} rust={rg is not None}")
            continue
        g_rs = Grid.constant(rg["bpm"], rg["downbeat"])
        same_bpm += abs(g_rs.bpm - g_py.bpm) < 1e-6
        period = 60 / g_py.bpm
        mid = Track(r.audio).duration / 2
        n = np.round(g_py.number_at(mid)[0])
        t_py = g_py.time_of(n)[0]
        k = np.round(g_rs.number_at(t_py)[0])
        t_rs = g_rs.time_of(k)[0]
        offsets.append((t_rs - t_py) * 1000)
        same_bar += (k % 4 == 0) == (n % 4 == 0)
        start, end = Track(r.audio).music_span
        rows.append(score(g_rs, r.grid, start, end))
    off = np.abs(offsets)
    n = len(offsets)
    print(f"{n} tracks: same BPM {same_bpm}/{n}; same bar 1 {same_bar}/{n}; beat placement |Δ| median {np.median(off):.3f} ms, "
          f"max {off.max():.3f} ms, >1 ms on {np.sum(off > 1)}")
    print(f"rust grids vs references: usable@10 {np.mean([s.usable_10 for s in rows]):.0%}, usable@20 {np.mean([s.usable_20 for s in rows]):.0%}")


if __name__ == "__main__":
    main()
