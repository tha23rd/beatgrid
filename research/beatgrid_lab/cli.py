"""lab: build reference sets, cache activations, evaluate gridders.

  uv run lab import-engine                      # DATA/engine/m.db + DATA/audio/engine -> refs/engine.jsonl
  uv run lab cache --set engine                 # Beat This! activations for every track
  uv run lab audit --set engine                 # which reference grids hold steady on their audio
  uv run lab eval --set engine --split dev -p offset_ms=2
  uv run lab review                             # listen to grids in a browser (port 8765)
"""

import argparse
import ast
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import refs
from .config import DATA, RUNS
from .metrics import at_reference_octave, score
from .pipeline import Params, run
from .track import Track


def cmd_import_engine(args):
    from .refs.engine import AUDIO_EXTENSIONS, read_library

    db = Path(args.db or DATA / "engine" / "m.db")
    audio_dir = Path(args.audio or DATA / "audio" / "engine")
    files = {p.stem: p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS}
    records = []
    for t in read_library(db):
        audio = files.get(str(t["track_id"]))
        if audio is None or not (t["played"] or t["corrected"]):
            continue
        rid = f"engine:{t['track_id']}"
        records.append({
            "id": rid,
            "audio": str(audio.relative_to(DATA)),
            "grid": refs.grid_json(t["reference"]),
            "split": refs.split_of(rid),
            # corrected: the DJ edited Engine's grid, so it is checked by ear.
            # accepted: played as Engine analysed it, never edited — trusted, not verified.
            "kind": "corrected" if t["corrected"] else "accepted",
            "baselines": {"engine": refs.grid_json(t["engine_default"])},
            "meta": {k: t[k] for k in ("track_id", "artist", "title", "genre", "path", "played", "locked")},
        })
    path = refs.save(args.name, records)
    kinds = {(r["kind"], r["split"]) for r in records}
    counts = {f"{k}/{s}": sum(r["kind"] == k and r["split"] == s for r in records) for k, s in sorted(kinds)}
    print(f"{len(records)} tracks -> {path}  {counts}")


def cmd_import_raveform(args):
    from .refs.engine import AUDIO_EXTENSIONS
    from .refs.raveform import read_dataset

    root = Path(args.root or DATA / "raveform" / "raveform")
    audio_dir = Path(args.audio or DATA / "audio" / "raveform")
    files = {p.stem: p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS + (".webm", ".opus")}
    records = []
    for t in read_dataset(root):
        audio = files.get(t["youtube_id"])
        if audio is None:
            continue
        rid = f"raveform:{t['youtube_id']}"
        records.append({
            "id": rid,
            "audio": str(audio.relative_to(DATA)),
            "grid": refs.grid_json(t["reference"]),
            "split": refs.split_of(rid),
            "kind": "expert",
            "meta": {k: t[k] for k in ("youtube_id", "key", "title", "genre", "fold", "tempo_markers")},
        })
    path = refs.save(args.name, records)
    print(f"{len(records)} tracks with audio -> {path}")


def cmd_import_crate(args):
    from .refs.crate import read_crate
    from .refs.engine import read_library

    db = Path(args.db or DATA / "crate" / "data" / "crate.sqlite")
    engine = {t["track_id"]: t for t in read_library(DATA / "engine" / "m.db")}
    engine_audio = {p.stem: p for p in (DATA / "audio" / "engine").iterdir()}
    ref_audio = {p.stem: p for p in (DATA / "audio" / "crate-ref").iterdir()}
    records, missing = [], 0
    for t in read_crate(db):
        audio = engine_audio.get(str(t["engine_id"])) if t["kind"] == "fixture" else ref_audio.get(t["crate_id"])
        if audio is None:
            missing += 1
            continue
        rid = f"crate:{t['crate_id']}"
        baselines = {}
        if t["engine_id"] in engine:
            baselines["engine"] = refs.grid_json(engine[t["engine_id"]]["engine_default"])
        records.append({
            "id": rid,
            "audio": str(audio.relative_to(DATA)),
            "grid": refs.grid_json(t["reference"]),
            "split": refs.split_of(rid),
            # approved: a human listened and accepted this exact grid.
            # trusted: the database's fitted grid agreed with Engine (or repaired it) - unheard.
            "kind": "approved" if t["human_approved"] else "trusted",
            "baselines": baselines,
            "meta": {k: t[k] for k in ("crate_id", "kind", "engine_id", "title", "artist", "genre",
                                         "engine_bpm", "status", "provenance", "engine_agrees")},
        })
    path = refs.save(args.name, records)
    print(f"{len(records)} tracks -> {path} ({missing} without audio); "
          f"approved {sum(r['kind'] == 'approved' for r in records)}, trusted {sum(r['kind'] == 'trusted' for r in records)}")


def _cache_one(path: str) -> str:
    """Audio features only. Network activations are computed in the parent process: torch
    initialised there and then used in forked workers deadlocks (OpenMP after fork)."""
    t = Track(Path(path))
    t.flux()
    t.band_energy()
    t.music_span
    return path


def cmd_cache(args):
    tracks = refs.load(args.set)
    started = time.time()
    # Activations first, one at a time (the network is multi-threaded); then the
    # single-threaded parts in parallel.
    for i, r in enumerate(tracks, 1):
        Track(r.audio).activations_for(args.network)
        print(f"[{i}/{len(tracks)}] {r.id}  {time.time() - started:.0f}s", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, _ in enumerate(pool.map(_cache_one, [str(r.audio) for r in tracks]), 1):
            if i % 20 == 0 or i == len(tracks):
                print(f"features {i}/{len(tracks)}  {time.time() - started:.0f}s", flush=True)


def _audit_one(ref) -> tuple[str, dict]:
    from .audit import audit

    track = Track(ref.audio)
    flux, dt, t_first = track.flux()
    start, end = track.music_span
    result = audit(ref.grid, flux, dt, t_first, start, end)
    if "engine" in ref.baselines:
        result["engine_default"] = audit(ref.baselines["engine"], flux, dt, t_first, start, end)["verdict"]
    return ref.id, result


def cmd_audit(args):
    from collections import Counter

    tracks = refs.load(args.set)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(pool.map(_audit_one, tracks))
    path = refs.REFS / f"{args.set}.audit.json"
    path.write_text(json.dumps(results, indent=1))
    for kind in sorted({r.kind for r in tracks}):
        sub = [results[r.id]["verdict"] for r in tracks if r.kind == kind]
        print(f"{kind:10s} " + "  ".join(f"{v}: {n}" for v, n in sorted(Counter(sub).items())))
    print(f"-> {path}")


def _parse_params(items: list[str], profile: str | None = None) -> Params:
    from .pipeline import PROFILES

    values = dict(PROFILES[profile]) if profile else {}
    fields = {f.name for f in dataclasses.fields(Params)}
    for item in items or []:
        key, _, raw = item.partition("=")
        if key not in fields:
            sys.exit(f"unknown parameter {key!r}; known: {sorted(fields)}")
        try:
            values[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            values[key] = raw
    return Params(**values)


def _evaluate_one(job) -> dict:
    ref, params, engine_prior = job
    track = Track(ref.audio)
    start, end = track.music_span
    from .baselines import beat_this_raw

    grids = {"engine": ref.baselines.get("engine"), "beatthis": beat_this_raw(track.activations_for(params.network))}
    result = run(track, params, ref.baselines.get("engine") if engine_prior else None)
    grids["pipeline"] = result.grid
    row = {"id": ref.id, "kind": ref.kind, "split": ref.split,
           "title": f"{ref.meta.get('artist')} - {ref.meta.get('title')}", "diagnostics": result.diagnostics}
    for name, grid in grids.items():
        s = score(grid, ref.grid, start, end)
        row[name] = s.as_dict()
        if grid is not None and s.octave in ("2x", "0.5x"):
            row[name + "_octfree"] = score(at_reference_octave(grid, ref.grid, start, end), ref.grid, start, end).as_dict()
        else:
            row[name + "_octfree"] = s.as_dict()
    return row


def summarise(rows: list[dict], methods: list[str]) -> str:
    lines = []
    kinds = [k for k in ("corrected", "accepted") if any(r["kind"] == k for r in rows)]
    kinds += sorted({r["kind"] for r in rows} - set(kinds))
    for kind in kinds:
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        lines.append(f"\n{kind} (n={len(sub)})")
        lines.append(f"  {'method':12s} {'usable@10':>9s} {'usable@20':>9s} {'oct-free@20':>11s} {'tempo 1x':>8s} "
                     f"{'bar ok':>7s} {'≤10ms':>6s} {'bias ms':>8s} {'bpm err':>8s}")
        for m in methods:
            s = [r[m] for r in sub]
            of = [r[m + "_octfree"] for r in sub]
            one = [x for x in s if x["octave"] == "1x"]
            pct = lambda xs: f"{100 * np.mean(xs):.0f}%"
            lines.append(
                f"  {m:12s} {pct([x['usable_10'] for x in s]):>9s} {pct([x['usable_20'] for x in s]):>9s} "
                f"{pct([x['usable_20'] for x in of]):>11s} {pct([x['octave'] == '1x' for x in s]):>8s} "
                f"{pct([x['bar_ok'] for x in one]) if one else '-':>7s} "
                f"{np.median([x['within_10'] for x in s]):6.2f} "
                f"{np.nanmedian([x['bias_ms'] for x in one]) if one else float('nan'):+8.1f} "
                f"{np.median([x['bpm_error'] for x in one]) if one else float('nan'):8.3f}"
            )
    return "\n".join(lines)


def cmd_eval(args):
    params = _parse_params(args.param, args.profile)
    tracks = [r for r in refs.load(args.set) if args.split == "all" or r.split == args.split]
    if args.verdict:
        audit_path = refs.REFS / f"{args.set}.audit.json"
        if not audit_path.exists():
            sys.exit(f"no audit for {args.set}; run `lab audit --set {args.set}` first")
        verdicts = {k: v["verdict"] for k, v in json.loads(audit_path.read_text()).items()}
        tracks = [r for r in tracks if verdicts.get(r.id) in args.verdict.split(",")]
    if args.kind:
        tracks = [r for r in tracks if r.kind == args.kind]
    from .activations import _cache_path

    ready = [r for r in tracks if _cache_path(r.audio, params.network).exists()]
    if len(ready) < len(tracks):
        print(f"skipping {len(tracks) - len(ready)} tracks without cached activations (run `lab cache`)")
    tracks = ready
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(_evaluate_one, [(r, params, args.engine_prior) for r in tracks]))
    methods = ["engine", "beatthis", "pipeline"]
    print(f"set={args.set} split={args.split} params: " +
          ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(params).items() if v != getattr(Params(), k)))
    print(summarise(rows, methods))
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.label or 'run'}-{args.set}-{args.split}.json"
    out.write_text(json.dumps({"params": dataclasses.asdict(params), "rows": rows}, indent=1, default=float))
    print(f"\nper-track results: {out}")


def cmd_review(args):
    from .review.server import serve

    print(f"listening on http://{args.host}:{args.port}")
    serve(args.host, args.port)


def main():
    ap = argparse.ArgumentParser(prog="lab")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("import-engine", help="build a reference set from an Engine DJ library")
    p.add_argument("--db")
    p.add_argument("--audio")
    p.add_argument("--name", default="engine")
    p.set_defaults(func=cmd_import_engine)

    p = sub.add_parser("import-raveform", help="build a reference set from Raveform (needs the audio)")
    p.add_argument("--root")
    p.add_argument("--audio")
    p.add_argument("--name", default="raveform")
    p.set_defaults(func=cmd_import_raveform)

    p = sub.add_parser("import-crate", help="build a reference set from a SQLite database of DJ-approved grids (crate.sqlite)")
    p.add_argument("--db")
    p.add_argument("--name", default="crate")
    p.set_defaults(func=cmd_import_crate)

    p = sub.add_parser("cache", help="precompute activations, onset flux and spans")
    p.add_argument("--set", default="engine")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--network", default="final0", help="Beat This! checkpoint, e.g. small0")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("audit", help="check reference grids against their own audio")
    p.add_argument("--set", default="engine")
    p.add_argument("--jobs", type=int, default=12)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("eval", help="score baselines and the pipeline")
    p.add_argument("--set", default="engine")
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--kind", choices=["corrected", "accepted"])
    p.add_argument("--verdict", help="only references the audit judged e.g. 'steady' or 'steady,unclear'")
    p.add_argument("-p", "--param", action="append", help="pipeline parameter, e.g. -p offset_ms=2")
    p.add_argument("--profile", help="octave convention profile, e.g. 'edm'")
    p.add_argument("--engine-prior", action="store_true",
                   help="let the reference's Engine-default grid vote on bar 1 (meaningless where the reference IS Engine's grid)")
    p.add_argument("--label")
    p.add_argument("--jobs", type=int, default=12)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("review", help="web app to listen to grids and record verdicts")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_review)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
