"""Does the Rust core produce the same grids as the research pipeline?

Feeds both the same network activations (exported from the cache), runs the Rust CLI on the
audio, and compares tempo, beat placement, bar 1 and the review reasons per track; then
scores the Rust grids against the references like any other gridder.

  uv run python experiments/rust_parity.py BINARY --set engine [--profile edm] [--feed student-v4]
"""

import argparse
import json
import subprocess
import tempfile
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
                                      "(default: feed both the same cached activations)")
    ap.add_argument("--feed", default="final0", help="the cached activations fed to both (default: Beat This! final0)")
    args = ap.parse_args()
    tracks = refs.load(args.set)
    act_dir = DATA / "parity" / ("act" if args.feed == "final0" else f"act-{args.feed}")
    act_dir.mkdir(parents=True, exist_ok=True)
    for r in tracks if not args.network else []:
        a = Track(r.audio).activations_for(args.feed)
        np.concatenate([a.beat, a.downbeat]).astype("<f4").tofile(act_dir / f"{r.audio.stem}.act")
    feed = [] if args.network else ["--activations-dir", str(act_dir)]
    # Formats the Rust decoder does not read (YouTube's Opus in WebM): hand it ffmpeg's mono
    # decode as a float WAV, the very samples the Python side analyses.
    wav_dir = tempfile.TemporaryDirectory(dir=DATA / "parity")
    paths = []
    for r in tracks:
        if r.audio.suffix.lower() in (".webm", ".opus"):
            wav = Path(wav_dir.name) / f"{r.audio.stem}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(r.audio), "-map", "0:a:0", "-ac", "1",
                            "-c:a", "pcm_f32le", str(wav)], check=True)
            paths.append(str(wav))
        else:
            paths.append(str(r.audio))
    t = time.perf_counter()
    out = subprocess.run([args.binary, "analyze", "--profile", args.profile or "none", *feed, *paths],
                         capture_output=True, text=True)  # a file that fails is reported per track
    wav_dir.cleanup()
    print(f"rust: {len(tracks)} tracks in {time.perf_counter() - t:.1f}s")
    rust = {Path(d["file"]).stem: d for d in map(json.loads, out.stdout.splitlines())}
    with ProcessPoolExecutor(12) as pool:
        py = {rid: (g, d) for rid, g, d in pool.map(python_grid, [(r, args.profile, args.network or args.feed) for r in tracks])}

    same_review = compared = 0
    for r in tracks:
        rd = rust[r.audio.stem]
        if "error" in rd:
            continue
        compared += 1
        py_review = py[r.id][1].get("review", ["no steady beat to grid"]) if py[r.id][0] is not None else ["no steady beat to grid"]
        if rd["review"] == py_review:
            same_review += 1
        else:
            print(f"{r.id}: review python={py_review} rust={rd['review']}")
    print(f"same review reasons {same_review}/{compared}")
    # The evidence behind the reasons, where both sides measured it.
    for key in ("attack_second", "stretch_drift_ms", "snap_drift_ms", "bar_margin"):
        deltas, mismatched = [], 0
        for r in tracks:
            rd, (g_py, d_py) = rust[r.audio.stem], py[r.id]
            d_rs = (rd.get("analysis") or {}).get("diagnostics")
            if g_py is None or not d_rs:
                continue
            a, b = d_py.get(key, 0.0), d_rs.get(key)
            if (a is None) != (b is None):
                mismatched += 1
            elif a is not None:
                deltas.append(abs(a - b))
        print(f"  {key}: max |python - rust| {max(deltas, default=0.0):.2g}" + (f", {mismatched} measured on one side only" if mismatched else ""))

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
