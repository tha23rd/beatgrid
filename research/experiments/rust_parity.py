"""Does the Rust core produce the same grids as the research pipeline?

Feeds both the same network activations (exported from the cache), runs the Rust CLI on the
audio, and compares tempo, beat placement, bar 1 and the review reasons per track; then
scores the Rust grids against the references like any other gridder. With --tempo auto or
variable it also compares the segments (how many, where each starts, its tempo); --set
splices renders the synthetic tempo changes to temporary WAV files for the Rust side.

  uv run python experiments/rust_parity.py BINARY --set engine [--profile edm] [--feed student-v4]
  uv run python experiments/rust_parity.py BINARY --set splices --split dev --tempo auto --feed student-v4
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
from beatgrid_lab.pipeline import PROFILES, TEMPO_MODES, Params, run
from beatgrid_lab.splices import RATE, Splice, SpliceTrack
from beatgrid_lab.track import Track


def track_of(ref):
    if isinstance(ref, Splice):
        return SpliceTrack(ref)
    return Track(ref.audio)


def python_grid(job):
    ref, profile, network, tempo = job
    params = Params(**{**(PROFILES[profile] if profile else {}), **TEMPO_MODES[tempo]})
    params.network = network
    result = run(track_of(ref), params)
    segments = result.segments.as_json() if result.segments is not None else None
    return ref.id, result.grid, result.diagnostics, segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--set", default="engine")
    ap.add_argument("--profile")
    ap.add_argument("--network", help="compare the Rust build's own network against this Python checkpoint "
                                      "(default: feed both the same cached activations)")
    ap.add_argument("--feed", default="final0", help="the cached activations fed to both (default: Beat This! final0)")
    ap.add_argument("--tempo", default="constant", choices=["constant", "auto", "variable"])
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    args = ap.parse_args()
    if args.set == "splices":
        from beatgrid_lab import splices

        tracks = splices.load()
    else:
        tracks = refs.load(args.set)
    tracks = [r for r in tracks if args.split == "all" or r.split == args.split]
    stem = lambda r: r.id.replace(":", "_") if isinstance(r, Splice) else r.audio.stem
    act_dir = DATA / "parity" / ("act" if args.feed == "final0" else f"act-{args.feed}")
    act_dir.mkdir(parents=True, exist_ok=True)
    for r in tracks if not args.network else []:
        a = track_of(r).activations_for(args.feed)
        np.concatenate([a.beat, a.downbeat]).astype("<f4").tofile(act_dir / f"{stem(r)}.act")
    feed = [] if args.network else ["--activations-dir", str(act_dir)]
    # Formats the Rust decoder does not read (YouTube's Opus in WebM): hand it ffmpeg's mono
    # decode as a float WAV, the very samples the Python side analyses.
    wav_dir = tempfile.TemporaryDirectory(dir=DATA / "parity")
    paths = []
    for r in tracks:
        if isinstance(r, Splice):
            import soundfile

            wav = Path(wav_dir.name) / f"{stem(r)}.wav"
            soundfile.write(wav, track_of(r).samples, RATE, subtype="FLOAT")
            paths.append(str(wav))
        elif r.audio.suffix.lower() in (".webm", ".opus"):
            wav = Path(wav_dir.name) / f"{r.audio.stem}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(r.audio), "-map", "0:a:0", "-ac", "1",
                            "-c:a", "pcm_f32le", str(wav)], check=True)
            paths.append(str(wav))
        else:
            paths.append(str(r.audio))
    t = time.perf_counter()
    out = subprocess.run([args.binary, "analyze", "--profile", args.profile or "none", "--tempo", args.tempo, *feed, *paths],
                         capture_output=True, text=True)  # a file that fails is reported per track
    wav_dir.cleanup()
    print(f"rust: {len(tracks)} tracks in {time.perf_counter() - t:.1f}s")
    rust = {Path(d["file"]).stem: d for d in map(json.loads, out.stdout.splitlines())}
    with ProcessPoolExecutor(12) as pool:
        py = {rid: (g, d, segs) for rid, g, d, segs in
              pool.map(python_grid, [(r, args.profile, args.network or args.feed, args.tempo) for r in tracks])}

    same_review = compared = 0
    for r in tracks:
        rd = rust[stem(r)]
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
            rd, (g_py, d_py, _) = rust[stem(r)], py[r.id]
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
        rd = rust[stem(r)]
        if "error" in rd:
            print(f"{r.id}: rust error {rd['error']}")
            continue
        g_py, _, _ = py[r.id]
        rg = rd["analysis"]["grid"]
        if rg is None or g_py is None:
            print(f"{r.id}: grid python={g_py is not None} rust={rg is not None}")
            continue
        g_rs = Grid.constant(rg["bpm"], rg["downbeat"])
        same_bpm += abs(g_rs.bpm - g_py.bpm) < 1e-6
        period = 60 / g_py.bpm
        mid = track_of(r).duration / 2
        n = np.round(g_py.number_at(mid)[0])
        t_py = g_py.time_of(n)[0]
        k = np.round(g_rs.number_at(t_py)[0])
        t_rs = g_rs.time_of(k)[0]
        offsets.append((t_rs - t_py) * 1000)
        same_bar += (k % 4 == 0) == (n % 4 == 0)
        start, end = track_of(r).music_span
        rows.append(score(g_rs, r.truth[0] if isinstance(r, Splice) else r.grid, start, end))
    off = np.abs(offsets)
    n = len(offsets)
    print(f"{n} tracks: same BPM {same_bpm}/{n}; same bar 1 {same_bar}/{n}; beat placement |Δ| median {np.median(off):.3f} ms, "
          f"max {off.max():.3f} ms, >1 ms on {np.sum(off > 1)}")
    print(f"rust grids vs references: usable@10 {np.mean([s.usable_10 for s in rows]):.0%}, usable@20 {np.mean([s.usable_20 for s in rows]):.0%}")
    if args.tempo != "constant":
        compare_segments(tracks, rust, py, stem)


def compare_segments(tracks, rust, py, stem) -> None:
    """The segmented grids: the same number of segments, each starting at the same time (to
    1 ms) at the same tempo."""
    same, both_segmented, differ = 0, 0, []
    starts = []
    for r in tracks:
        rd = rust[stem(r)]
        if "error" in rd or rd["analysis"]["segments"] is None:
            continue
        rs = rd["analysis"]["segments"]
        ps = py[r.id][2]
        ps = ps if ps is not None else [{"start": s["start"], "bpm": s["bpm"]} for s in rs[:1]]  # constant: one segment
        if len(rs) > 1 or len(ps) > 1:
            both_segmented += 1
        ok = len(rs) == len(ps) and all(abs(a["bpm"] - b["bpm"]) < 1e-6 for a, b in zip(rs, ps))
        if ok and len(rs) > 1:
            d = [abs(a["start"] - b["start"]) * 1000 for a, b in zip(rs[1:], ps[1:])]
            starts.extend(d)
            ok = max(d) < 1.0
        same += ok
        if not ok:
            differ.append((r.id, [(round(x["start"], 3), round(x["bpm"], 3)) for x in ps],
                           [(round(x["start"], 3), round(x["bpm"], 3)) for x in rs]))
    n = sum(1 for r in tracks if "error" not in rust[stem(r)])
    print(f"segments: the same on {same}/{n} tracks ({both_segmented} segmented on either side); "
          f"change points |python - rust| max {max(starts, default=0.0):.3f} ms")
    for rid, a, b in differ:
        print(f"  {rid}: python {a}  rust {b}")


if __name__ == "__main__":
    main()
