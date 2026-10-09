"""Multi-section fitting on known tempo-changing tracks and constant-tempo controls."""
import sys
import time

from beatgrid_lab import refs
from beatgrid_lab.metrics import score
from beatgrid_lab.pipeline import Params, run
from beatgrid_lab.sections import fit_sections
from beatgrid_lab.track import Track

R = {r.id: r for s in ("engine", "crate") for r in refs.load(s)}
for rid in sys.argv[1:]:
    r = R[rid]
    t = Track(r.audio)
    p = Params()
    single = run(t, p)
    started = time.time()
    multi, d = fit_sections(t, t.activations, p, single.grid)
    took = time.time() - started
    start, end = t.music_span
    s1 = score(single.grid, r.grid, start, end)
    s2 = score(multi, r.grid, start, end) if multi is not None else None
    print(f"{rid:15s} {took:5.1f}s sections {d['sections']} fit {d.get('fit_single')} -> {d.get('fit_multi')}")
    print(f"   single: w10 {s1.within_10:.2f} w20 {s1.within_20:.2f} bar {s1.bar_ok}   multi: "
          + (f"w10 {s2.within_10:.2f} w20 {s2.within_20:.2f} bar {s2.bar_ok}" if s2 else "(one tempo)")
          + f"   | {r.meta.get('title', '')[:40]}", flush=True)
