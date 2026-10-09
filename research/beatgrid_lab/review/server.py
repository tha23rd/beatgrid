"""Listen to grids: a local web app for checking beatgrid's grids against references by ear.

  uv run lab review [--host 0.0.0.0] [--port 8765]

Audio is served as FLAC decoded by ffmpeg — the same trimmed timeline the grids live on — so
what plays lines up with what is drawn to the sample. Verdicts are appended to
DATA/verdicts.jsonl, one JSON object per judgement, newest last.
"""

import json
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import refs
from ..audit import audit
from ..baselines import beat_this_raw
from ..config import CACHE, DATA, REFS
from ..grid import Grid
from ..metrics import score
from ..pipeline import Params, run
from ..track import Track

STATIC = Path(__file__).parent / "static"
VERDICTS = DATA / "verdicts.jsonl"

# beatgrid's settings per reference set: the DJ's own library is shown in their octave
# convention (drum & bass in half time), Raveform in its annotators' (full time).
PROFILE_FOR_SET = {"engine": "edm", "crate": "edm", "library": "edm"}
# Per-set network overrides (the default is the shipping network, Params().network).
NETWORK_FOR_SET: dict[str, str] = {}


def params_for(set_name: str) -> Params:
    from ..pipeline import PROFILES

    profile = PROFILE_FOR_SET.get(set_name)
    params = Params(**PROFILES[profile]) if profile else Params()
    params.network = NETWORK_FOR_SET.get(set_name, params.network)
    return params

_sets: dict[str, dict[str, refs.Ref]] = {}
_summaries: dict[str, dict] = {}
_lock = threading.Lock()


def clean(x):
    """JSON-safe: numpy scalars to Python, non-finite floats to null."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def reply(data) -> JSONResponse:
    return JSONResponse(clean(data))


def ref_sets() -> list[str]:
    return sorted(p.stem for p in REFS.glob("*.jsonl"))


def get_set(name: str) -> dict[str, refs.Ref]:
    with _lock:
        if name not in _sets:
            _sets[name] = {r.id: r for r in refs.load(name)}
        return _sets[name]


def grids_for(set_name: str, ref: refs.Ref) -> tuple[dict[str, Grid | None], dict]:
    track = Track(ref.audio)
    grids: dict[str, Grid | None] = {"reference": ref.grid}
    result = run(track, params_for(set_name))
    grids["beatgrid"] = result.grid
    diagnostics = {"beatgrid": result.diagnostics}
    if "engine" in ref.baselines:
        grids["engine"] = ref.baselines["engine"]
    grids["beatthis-raw"] = beat_this_raw(track.activations_for("final0")) if _has_cache(track, "final0") else None
    return grids, diagnostics


def _has_cache(track: Track, model: str) -> bool:
    from ..activations import _cache_path

    return _cache_path(track.path, model).exists()


def _summarise(job) -> dict:
    set_name, ref = job
    track = Track(ref.audio)
    start, end = track.music_span
    grids, diagnostics = grids_for(set_name, ref)
    out = {"id": ref.id, "title": ref.meta.get("title") or ref.id, "artist": ref.meta.get("artist"),
           "kind": ref.kind, "split": ref.split, "genre": ref.meta.get("genre"),
           "ref_bpm": ref.grid.bpm, "scores": {}, "review": diagnostics["beatgrid"].get("review", [])}
    for name, grid in grids.items():
        if name != "reference":
            out["scores"][name] = score(grid, ref.grid, start, end).as_dict()
    return out


def summary(set_name: str) -> dict:
    """Per-track scores for a whole set, computed once per server run (in parallel)."""
    with _lock:
        if set_name in _summaries:
            return _summaries[set_name]
    tracks = list(get_set(set_name).values())
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(_summarise, [(set_name, r) for r in tracks]))
    audit_path = REFS / f"{set_name}.audit.json"
    audits = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for row in rows:
        row["audit"] = audits.get(row["id"], {}).get("verdict")
    result = {"set": set_name, "rows": rows, "computed": time.time()}
    with _lock:
        _summaries[set_name] = result
    return result


def latest_verdicts() -> dict:
    out = {}
    if VERDICTS.exists():
        for line in VERDICTS.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                out[f"{v['set']}/{v['id']}"] = v
    return out


def grid_json(grid: Grid | None, end: float) -> dict | None:
    if grid is None:
        return None
    numbers, times = grid.beats(0.0, end)
    return {"bpm": grid.bpm, "times": times.round(5).tolist(), "downbeat": (np.mod(numbers, 4) == 0).tolist(),
            "constant": grid.is_constant}


# ---------------------------------------------------------------- routes

async def api_sets(request: Request):
    return reply([{"name": s, "tracks": len(get_set(s))} for s in ref_sets()])


async def api_tracks(request: Request):
    set_name = request.query_params["set"]
    data = summary(set_name)
    verdicts = latest_verdicts()
    for row in data["rows"]:
        v = verdicts.get(f"{set_name}/{row['id']}")
        row["verdict"] = v["verdict"] if v else None
    return reply(data)


async def api_track(request: Request):
    set_name, rid = request.query_params["set"], request.query_params["id"]
    ref = get_set(set_name)[rid]
    track = Track(ref.audio)
    start, end = track.music_span
    grids, diagnostics = grids_for(set_name, ref)
    flux, dt, t_first = track.flux()
    v = latest_verdicts().get(f"{set_name}/{rid}")
    return reply({
        "id": rid, "set": set_name, "kind": ref.kind, "split": ref.split, "meta": ref.meta,
        "duration": track.duration, "music": [start, end],
        "grids": {k: grid_json(g, track.duration) for k, g in grids.items()},
        "scores": {k: score(g, ref.grid, start, end).as_dict() for k, g in grids.items() if k != "reference"},
        "diagnostics": diagnostics,
        "audit": audit(ref.grid, flux, dt, t_first, start, end),
        "verdict": v,
    })


def _listen_path(ref: refs.Ref) -> Path:
    track = Track(ref.audio)
    path = CACHE / "listen" / f"{track.key}.flac"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.flac")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ref.audio), "-map", "0:a:0",
                        "-c:a", "flac", "-sample_fmt", "s16", str(tmp)], check=True)
        tmp.rename(path)
    return path


async def api_audio(request: Request):
    ref = get_set(request.query_params["set"])[request.query_params["id"]]
    return FileResponse(_listen_path(ref), media_type="audio/flac")


def _bands(ref: refs.Ref) -> tuple[np.ndarray, float]:
    """Low (<200 Hz), mid, high (>2 kHz) RMS every 5 ms, for colouring the waveform."""
    track = Track(ref.audio)
    path = CACHE / "bands" / f"{track.key}.npy"
    hop_s = 0.005
    if path.exists():
        return np.load(path), hop_s
    x, rate = track.samples.astype(np.float64), track.rate
    hop = int(rate * hop_s)
    n = len(x) // hop
    out = np.empty((3, n), np.float32)
    filters = [butter(4, 200, "low", fs=rate, output="sos"),
               butter(4, [200, 2000], "bandpass", fs=rate, output="sos"),
               butter(4, 2000, "high", fs=rate, output="sos")]
    for i, sos in enumerate(filters):
        y = sosfiltfilt(sos, x)[: n * hop].reshape(n, hop)
        out[i] = np.sqrt(np.mean(y**2, axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out, hop_s


async def api_bands(request: Request):
    ref = get_set(request.query_params["set"])[request.query_params["id"]]
    bands, dt = _bands(ref)
    return Response(bands.astype("<f4").tobytes(), media_type="application/octet-stream",
                    headers={"X-Dt": str(dt), "X-Count": str(bands.shape[1])})


async def api_flux(request: Request):
    ref = get_set(request.query_params["set"])[request.query_params["id"]]
    flux, dt, t_first = Track(ref.audio).flux()
    return Response(np.asarray(flux, "<f4").tobytes(), media_type="application/octet-stream",
                    headers={"X-Dt": repr(dt), "X-T0": repr(t_first)})


async def api_verdict(request: Request):
    body = await request.json()
    entry = {"set": body["set"], "id": body["id"], "verdict": body["verdict"], "note": body.get("note", ""),
             "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "time": body.get("time")}
    with _lock, open(VERDICTS, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return reply(entry)


async def index(request: Request):
    return FileResponse(STATIC / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/sets", api_sets),
    Route("/api/tracks", api_tracks),
    Route("/api/track", api_track),
    Route("/api/audio", api_audio),
    Route("/api/bands", api_bands),
    Route("/api/flux", api_flux),
    Route("/api/verdict", api_verdict, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
])


def serve(host: str, port: int):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
