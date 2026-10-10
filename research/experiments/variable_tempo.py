"""Variable-tempo grids: how well the segmented fit (sections.py) finds tempo changes, and that
it leaves steady tracks alone.

Sets:
  splices   the synthetic transition edits (`lab splices`): change points and each section's
            grid known exactly
  real      DJ-corrected Engine grids with several markers (transition edits, mashups, hand-
            followed grids) and any other DJ or expert grid that is not one constant tempo
  steady    the constant references the README quotes (DJ-corrected and machine-fitted
            `edm`, ear-approved `edm`, Raveform default): nothing may change, and every
            segmented grid there is a false segmentation

Metrics, for each tempo mode (constant, auto, variable):
  usable@10/@20 over the whole track (every section at the right octave, >=90% of the
  reference's beats within 10/20 ms, bar 1 right) and per section; change points found within
  one bar of the true one (precision, recall); each section's tempo against the truth.

  uv run python experiments/variable_tempo.py --split dev [--sets splices,real,steady] [-p seg_hop_s=4]
"""

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from beatgrid_lab import refs
from beatgrid_lab.config import RUNS
from beatgrid_lab.grid import Grid
from beatgrid_lab.metrics import score
from beatgrid_lab.pipeline import PROFILES, TEMPO_MODES, Params, run
from beatgrid_lab.sections import fit as fit_sections, with_octave
from beatgrid_lab.segmented import Segmented
from beatgrid_lab.track import Track
from review_flags import constant_share

MODES = ["constant", "auto", "variable"]
STEADY = [("engine", "corrected", "edm"), ("crate", "approved", "edm"), ("crate", "trusted", "edm"),
          ("raveform", "expert", None)]


def marker_sections(grid: Grid, end: float, tolerance: float = 0.003, least: int = 16) -> list[dict]:
    """Sections of a reference grid read from its beats: the local tempo (median of 9 beat
    intervals) starts a new section when it leaves the current section's tempo by more than
    `tolerance` for 8 beats in a row (a DJ's small adjustments within a section are one
    section); runs of fewer than `least` beats (the odd intervals of a transition) join the
    section before."""
    from scipy.ndimage import median_filter

    _, t = grid.beats(0.0, end)
    if len(t) < 2 * least:
        return [{"start": float(t[0]) if len(t) else 0.0, "bpm": float(grid.bpm), "change": None, "end": end}]
    local = 60.0 / median_filter(np.diff(t), size=9, mode="nearest")
    runs, i0, off = [], 0, 0
    for i in range(1, len(local)):
        ref = np.median(local[i0:i])
        off = off + 1 if abs(local[i] / ref - 1) > tolerance else 0
        if off == 8:
            runs.append((i0, i - 7))
            i0, off = i - 7, 0
    runs.append((i0, len(local)))
    merged = []
    for a, b in runs:
        if merged and b - a < least:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < least:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    out = []
    for k, (a, b) in enumerate(merged):
        bpm = float(np.median(local[a:b]))
        out.append({"start": float(t[a]), "bpm": float(bpm), "change": float(t[a]) if k else None,
                    "end": float(t[merged[k + 1][0]]) if k + 1 < len(merged) else end})
    return out


def score_sections(detected, truth: Grid, sections: list[dict], start: float, end: float) -> dict:
    """Whole-track and per-section scores of a detected grid (Grid or Segmented) against a
    reference with known sections."""
    ref_n, ref_t = truth.beats(start, end)
    det_n, det_t = detected.beats(start - 1.0, end + 1.0) if detected is not None else (np.array([]), np.array([]))
    out = {"sections": []}
    if len(det_t) < 2 or len(ref_t) == 0:
        return {**out, "usable_10": False, "usable_20": False, "within_10": 0.0, "within_20": 0.0, "bar_ok": False}
    idx = np.clip(np.searchsorted(det_t, ref_t), 1, len(det_t) - 1)
    nearest = np.where(np.abs(det_t[idx - 1] - ref_t) <= np.abs(det_t[idx] - ref_t), idx - 1, idx)
    absolute = np.abs(det_t[nearest] - ref_t) * 1000
    down = np.mod(ref_n, 4) == 0
    det_down = np.mod(det_n[nearest], 4) == 0
    octave_ok = True
    for s in sections:
        m = (ref_t >= max(s["start"], start)) & (ref_t < min(s["end"], end))
        mid = 0.5 * (max(s["start"], start) + min(s["end"], end))
        bpm = detected.bpm_at(mid) if isinstance(detected, Segmented) else detected.bpm
        ratio = bpm / s["bpm"]
        ok = abs(ratio - 1) < 0.04
        octave_ok &= ok
        md = m & down & (absolute <= 70)
        bar = bool(md.any()) and float(np.mean(det_down[md])) > 0.5
        w10 = float(np.mean(absolute[m] <= 10)) if m.any() else 1.0
        w20 = float(np.mean(absolute[m] <= 20)) if m.any() else 1.0
        out["sections"].append({"bpm": s["bpm"], "detected_bpm": bpm, "bpm_error": abs(bpm - s["bpm"]) if ok else np.inf,
                                "octave_ok": ok, "within_10": w10, "within_20": w20, "bar_ok": bar,
                                "usable_10": ok and bar and w10 >= 0.9, "usable_20": ok and bar and w20 >= 0.9,
                                "beats": int(m.sum())})
    md = down & (absolute <= 70)
    bar_ok = bool(md.any()) and float(np.mean(det_down[md])) > 0.5
    w10, w20 = float(np.mean(absolute <= 10)), float(np.mean(absolute <= 20))
    return {**out, "within_10": w10, "within_20": w20, "bar_ok": bar_ok,
            "usable_10": octave_ok and bar_ok and w10 >= 0.9, "usable_20": octave_ok and bar_ok and w20 >= 0.9}


def changes_found(detected, sections: list[dict]) -> dict:
    """Detected change points against the true ones, matched one to one within one bar."""
    truth = [(s["change"], 240.0 / s["bpm"]) for s in sections if s.get("change") is not None]
    found = [s.start for s in detected.segments[1:]] if isinstance(detected, Segmented) else []
    pairs = sorted(((abs(f - t), i, j) for i, (t, bar) in enumerate(truth) for j, f in enumerate(found)
                    if abs(f - t) <= bar + 1e-6))
    used_t, used_f, errors = set(), set(), []
    for d, i, j in pairs:
        if i in used_t or j in used_f:
            continue
        used_t.add(i)
        used_f.add(j)
        errors.append(d / (truth[i][1] / 4))  # in beats
    return {"true": len(truth), "found": len(found), "hits": len(used_t), "error_beats": errors}


def constant_run(track, params: Params, cache_key: str):
    """pipeline.run in constant mode, cached on disk (it does not depend on the seg_* knobs)."""
    import pickle

    path = RUNS / "constant-cache" / f"{cache_key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = run(track, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def evaluate(job) -> dict:
    kind, item, profile, overrides, modes = job
    if kind == "splice":
        from beatgrid_lab.splices import SpliceTrack

        track = SpliceTrack(item)
        truth, sections = item.truth
        rid, split, meta = item.id, item.split, {"kind": item.kind, "transition": item.transition}
    else:
        track = Track(item.audio)
        truth = item.grid
        rid, split = item.id, item.split
        sections = marker_sections(truth, track.duration)
        meta = {"set": kind}
    start, end = track.music_span
    row = {"id": rid, "split": split, **meta, "sections_true": len(sections), "modes": {},
           "ref_constant": bool(constant_share(truth, start, end) >= 0.9 and len(sections) == 1)}
    base = dict(PROFILES[profile]) if profile else {}
    params = Params(**{**base, **overrides, "tempo": "constant"})
    result = constant_run(track, params, f"{track.key}-{profile or 'default'}")

    def scores(detected) -> dict:
        if kind in ("splice", "real"):
            out = score_sections(detected, truth, sections, start, end)
            out["changes"] = changes_found(detected, sections)
            return out
        s = score(detected, truth, start, end)
        return {"usable_10": s.usable_10, "usable_20": s.usable_20, "bar_ok": s.bar_ok, "octave": s.octave,
                "within_10": s.within_10}

    for mode in modes:
        segments, sdiag = None, None
        if mode != "constant" and result.grid is not None:
            mp = Params(**{**params.__dict__, **TEMPO_MODES[mode], **overrides})
            found, sdiag = fit_sections(track, track.activations_for(mp.network), mp, result.diagnostics["bpm"],
                                        float(result.grid.time_of(0.0)[0]))
            segments = with_octave(found, params) if found is not None else None
        detected = segments if segments is not None else result.grid
        r = {"segments": len(segments.segments) if segments is not None else 1,
             "segment_list": segments.as_json() if segments is not None else None, "diag": sdiag,
             "bpm": float(result.grid.bpm) if result.grid is not None else None}
        r.update(scores(detected))
        if sdiag and "candidate" in sdiag and segments is None:
            from beatgrid_lab.segmented import Segment

            cand = Segmented([Segment(c["start"], c["bpm"], c["beat_in_bar"]) for c in sdiag["candidate"]], track.duration)
            r["candidate"] = scores(with_octave(cand, params))
        row["modes"][mode] = r
    return row


def rust_rows(binary: str, jobs: list, modes: list[str], overrides: dict) -> list[dict]:
    """The same evaluation with the grids from the Rust CLI, fed the cached activations (the
    splices rendered once to WAV files under DATA/parity, removed by hand when done)."""
    import subprocess

    from beatgrid_lab.config import DATA

    act_dir = DATA / "parity" / "act-student-v4"
    wav_dir = DATA / "parity" / "splices-wav"
    act_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(12) as pool:
        prepared = list(pool.map(prepare_rust_input, [(kind, item, str(act_dir), str(wav_dir)) for kind, item, *_ in jobs]))
    paths = {(kind, item.id): (path, profile) for (kind, item, profile, *_), path in zip(jobs, prepared)}
    out = {}
    params = [f"--param={k}={v}" for k, v in overrides.items()]
    for mode in modes:
        for profile in sorted({pr or "none" for _, pr in paths.values()}):
            files = [f for f, pr in paths.values() if (pr or "none") == profile]
            res = subprocess.run([binary, "analyze", "--format", "json", "--profile", profile, "--tempo", mode,
                                  "--activations-dir", str(act_dir), *params, *files], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                d = json.loads(line)
                out[(mode, d["file"])] = d
    with ProcessPoolExecutor(14) as pool:
        return list(pool.map(score_rust, [(job, {m: out.get((m, paths[(job[0], job[1].id)][0])) for m in modes})
                                          for job in jobs], chunksize=4))


def prepare_rust_input(args) -> str:
    """The file the Rust CLI reads for one job (a splice rendered to WAV, a WebM decoded to
    WAV), with its cached activations exported next to it."""
    import subprocess
    from pathlib import Path

    import soundfile

    from beatgrid_lab.config import DATA
    from beatgrid_lab.splices import RATE, SpliceTrack

    kind, item, act_dir, wav_dir = args
    if kind == "splice":
        stem = item.id.replace(":", "_")
        track = SpliceTrack(item)
        path = Path(wav_dir) / f"{stem}.wav"
        if not path.exists():
            soundfile.write(path, track.samples, RATE, subtype="FLOAT")
    else:
        track = Track(item.audio)
        stem = item.audio.stem
        path = item.audio
        if path.suffix.lower() in (".webm", ".opus"):
            wav = DATA / "parity" / "webm-wav" / f"{stem}.wav"
            if not wav.exists():
                wav.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-c:a",
                                "pcm_f32le", str(wav)], check=True)
            path = wav
    act = Path(act_dir) / f"{stem}.act"
    if not act.exists():
        a = track.activations_for("student-v4")
        np.concatenate([a.beat, a.downbeat]).astype("<f4").tofile(act)
    return str(path)


def score_rust(args) -> dict:
    from beatgrid_lab.segmented import Segment

    job, results = args
    kind, item, profile, overrides, modes = job
    if kind == "splice":
        from beatgrid_lab.splices import SpliceTrack

        track = SpliceTrack(item)
        truth, sections = item.truth
        rid, split, meta = item.id, item.split, {"kind": item.kind, "transition": item.transition}
    else:
        track = Track(item.audio)
        truth = item.grid
        rid, split = item.id, item.split
        sections = marker_sections(truth, track.duration)
        meta = {"set": kind}
    start, end = track.music_span
    row = {"id": rid, "split": split, **meta, "sections_true": len(sections), "modes": {},
           "ref_constant": bool(constant_share(truth, start, end) >= 0.9 and len(sections) == 1)}
    for mode in modes:
        d = results[mode]
        a = d["analysis"] if d and "analysis" in d else None
        grid = Grid.constant(a["grid"]["bpm"], a["grid"]["downbeat"]) if a and a["grid"] else None
        segs = a["segments"] if a else None
        detected = grid
        if segs and len(segs) > 1:
            detected = Segmented([Segment(x["start"], x["bpm"], x["beat_in_bar"]) for x in segs], track.duration)
        search = (a.get("diagnostics") or {}).get("segment_search") if a else None
        r = {"segments": len(segs) if segs else 1, "segment_list": segs if segs and len(segs) > 1 else None,
             "diag": search, "bpm": grid.bpm if grid is not None else None}
        if kind in ("splice", "real"):
            r.update(score_sections(detected, truth, sections, start, end))
            r["changes"] = changes_found(detected, sections)
        else:
            sc = score(detected, truth, start, end)
            r.update(usable_10=sc.usable_10, usable_20=sc.usable_20, bar_ok=sc.bar_ok, octave=sc.octave,
                     within_10=sc.within_10)
        row["modes"][mode] = r
    return row


def jobs_for(sets: list[str], split: str, overrides: dict, modes: list[str]) -> list:
    out = []
    if "splices" in sets:
        from beatgrid_lab import splices

        out += [("splice", s, None, overrides, modes) for s in splices.load() if split == "all" or s.split == split]
    if "real" in sets:
        for name in ("engine", "crate"):
            for r in refs.load(name):
                if r.kind not in ("corrected", "approved", "trusted"):
                    continue
                several = name == "engine" and len(r.grid.numbers) > 2
                start, end = Track(r.audio).music_span
                if several or constant_share(r.grid, start, end) < 0.9:
                    out.append(("real", r, "edm", overrides, modes))  # every split: these are never tuned on alone
    if "raveform-variable" in sets:
        for r in refs.load("raveform"):
            start, end = Track(r.audio).music_span
            if constant_share(r.grid, start, end) < 0.9 and (split == "all" or r.split == split):
                out.append(("raveform-variable", r, None, overrides, modes))
    if "accepted" in sets:  # Engine's own grids on tracks played unedited: never used for tuning
        out += [("engine accepted", r, None, overrides, modes) for r in refs.load("engine")
                if r.kind == "accepted" and (split == "all" or r.split == split)]
    if "steady" in sets:
        for name, kind, profile in STEADY:
            for r in refs.load(name):
                if r.kind != kind or not (split == "all" or r.split == split):
                    continue
                out.append((f"{name} {kind}", r, profile, overrides, modes))
    return out


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * np.mean(xs):3.0f}% ({int(np.sum(xs))}/{len(xs)})" if xs else "-"


def report(rows: list[dict], modes: list[str]) -> None:
    by_set = defaultdict(list)
    for r in rows:
        by_set[r.get("set") or "splices"].append(r)
    for name, sub in by_set.items():
        print(f"\n== {name} (n={len(sub)})")
        if name in ("splices", "real"):
            print(f"  {'mode':9s} {'usable@10':>14s} {'usable@20':>14s} {'sections@10':>15s} {'changes P':>14s} "
                  f"{'changes R':>14s} {'false splits':>13s} {'bpm err med':>11s}")
            for mode in modes:
                m = [r["modes"][mode] for r in sub]
                secs = [s for x in m for s in x["sections"]]
                ch = [x["changes"] for x in m]
                found, hits, true = sum(c["found"] for c in ch), sum(c["hits"] for c in ch), sum(c["true"] for c in ch)
                one = [x for x, r in zip(m, sub) if r["sections_true"] == 1]
                errs = [s["bpm_error"] for s in secs if np.isfinite(s["bpm_error"])]
                print(f"  {mode:9s} {pct(x['usable_10'] for x in m):>14s} {pct(x['usable_20'] for x in m):>14s} "
                      f"{pct(s['usable_10'] for s in secs):>15s} "
                      f"{(f'{hits / found:4.0%} ({hits}/{found})' if found else '-'):>14s} "
                      f"{(f'{hits / true:4.0%} ({hits}/{true})' if true else '-'):>14s} "
                      f"{pct(x['segments'] > 1 for x in one):>13s} {np.median(errs) if errs else float('nan'):11.3f}")
            if name == "splices":
                for key in ("kind", "transition"):
                    for v in sorted({r[key] for r in sub}):
                        part = [r for r in sub if r[key] == v]
                        print(f"    {key}={v:7s} n={len(part):3d}  " + "  ".join(
                            f"{mode} u10 {pct(r['modes'][mode]['usable_10'] for r in part)}" for mode in modes))
        else:
            print(f"  {'mode':9s} {'usable@10':>14s} {'usable@20':>14s} {'bar ok':>14s} {'segmented':>13s} "
                  f"{'segmented (const ref)':>22s} {'changed':>8s}")
            for mode in modes:
                m = [r["modes"][mode] for r in sub]
                c = [r["modes"][modes[0]] for r in sub]
                changed = sum((x["usable_10"], x["usable_20"], x["bar_ok"]) != (y["usable_10"], y["usable_20"], y["bar_ok"])
                              for x, y in zip(m, c))
                const_ref = [x["segments"] > 1 for x, r in zip(m, sub) if r["ref_constant"]]
                print(f"  {mode:9s} {pct(x['usable_10'] for x in m):>14s} {pct(x['usable_20'] for x in m):>14s} "
                      f"{pct(x['bar_ok'] for x in m):>14s} {pct(x['segments'] > 1 for x in m):>13s} "
                      f"{pct(const_ref):>22s} {changed:>8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--sets", default="splices,real,steady")
    ap.add_argument("-p", "--param", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--label", default="variable")
    ap.add_argument("--modes", default="constant,auto,variable")
    ap.add_argument("--rust", metavar="BINARY", help="grid with the Rust CLI (fed the cached activations) instead")
    args = ap.parse_args()
    overrides = {}
    for item in args.param:
        k, _, v = item.partition("=")
        overrides[k] = ast.literal_eval(v)
    modes = args.modes.split(",")
    jobs = jobs_for(args.sets.split(","), args.split, overrides, modes)
    if args.rust:
        rows = rust_rows(args.rust, jobs, modes, overrides)
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(evaluate, jobs, chunksize=1))
    report(rows, modes)
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.label}-{args.split}.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nper-track rows: {out}")


if __name__ == "__main__":
    main()
