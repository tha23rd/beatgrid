"""Which landmark of the beat do reference grids sit on?

For each track where the pipeline's tempo matches the reference, fold an onset function at
the beat period around the pipeline's grid and take a landmark from the fold. A good
landmark sits at a *consistent* offset from the reference grid across tracks (a constant
offset is just a convention to calibrate); its spread is what is left as error.

  uv run python experiments/placement.py [--split dev] [--limit N]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, stft

from beatgrid_lab import refs
from beatgrid_lab.activations import _cache_path
from beatgrid_lab.pipeline import Params, coarse_bpm, fit_tempo, phase_of, snap_tempo, wrap
from beatgrid_lab.track import Track

SEARCH_MS = 50.0


def band_envelope(x, rate, lo, hi, smooth_ms=2.0, hop_s=0.0005):
    sos = butter(4, [lo, hi] if lo else hi, "bandpass" if lo else "low", fs=rate, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    hop = max(1, int(round(hop_s * rate)))
    win = max(1, int(round(smooth_ms / 1000 * rate)))
    env = np.convolve(env, np.ones(win) / win, mode="same")[::hop]
    return env, hop / rate


def flux(x, rate, n=512, hop=32):
    _, _, z = stft(x, fs=rate, nperseg=n, noverlap=n - hop, boundary=None, padded=False)
    mag = np.log1p(100 * np.abs(z))
    f = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
    # stft frame i is centred at (i * hop + n/2) samples; diff moves it half a hop on
    times = (np.arange(len(f)) + 1) * hop / rate + (n / 2 - hop / 2) / rate
    return f, times


def fold(curve, dt, t_curve0, bpm, t0, duration):
    period = 60.0 / bpm
    k = np.arange(np.ceil(-t0 / period), np.floor((duration - t0) / period) + 1)
    centres = np.round((t0 + k * period - t_curve0) / dt).astype(int)
    half = int(round(SEARCH_MS / 1000 / dt))
    idx = centres[:, None] + np.arange(-half, half + 1)[None, :]
    idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(curve))]
    return curve[idx].mean(axis=0), (np.arange(-half, half + 1)) * dt


def landmarks(folded, lags):
    """Peak; steepest rise; and where it first reaches half way from floor to peak."""
    i = int(np.argmax(folded))
    floor = np.percentile(folded, 10)
    half = floor + 0.5 * (folded[i] - floor)
    j = i
    while j > 0 and folded[j - 1] > half:
        j -= 1
    slope = np.diff(folded)
    k = int(np.argmax(slope[: max(i, 1)])) if i > 0 else 0
    return {"peak": lags[i], "half_rise": lags[j], "steepest": (lags[k] + lags[k + 1]) / 2}


def analyse(ref):
    track = Track(ref.audio)
    act = track.activations
    p = Params()
    bpm0 = coarse_bpm(act, p)
    if bpm0 is None:
        return None
    bpm, c = fit_tempo(act, bpm0, p)
    bpm, c, _ = snap_tempo(act, bpm, c, p)
    if abs(bpm - ref.grid.bpm) > 0.05:
        return None
    period = 60.0 / bpm
    t0 = phase_of(c, period)
    # Reference phase relative to our grid, at the middle of the track.
    mid = track.duration / 2
    n = np.round(ref.grid.number_at(mid)[0])
    ref_t = float(ref.grid.time_of(n)[0])
    ref_off = float(wrap(ref_t - t0, period))

    x, rate = track.samples.astype(np.float64), track.rate
    out = {"id": ref.id, "kind": ref.kind, "activation": -ref_off}
    curves = {
        "low_env": band_envelope(x, rate, 0, 150),
        "body_env": band_envelope(x, rate, 40, 250),
        "click_env": band_envelope(x, rate, 1500, 8000),
    }
    for name, (env, dt) in curves.items():
        logenv = np.log(env + 1e-5)
        rise = np.maximum(np.diff(logenv, prepend=logenv[0]), 0)
        fe, lags = fold(env, dt, 0.0, bpm, t0, track.duration)
        fr, _ = fold(rise, dt, 0.0, bpm, t0, track.duration)
        for lm, v in landmarks(fe, lags).items():
            out[f"{name}.{lm}"] = v - ref_off
        out[f"{name}.rise_peak"] = lags[int(np.argmax(fr))] - ref_off
    f, ft = flux(x, rate)
    ff, lags = fold(f, ft[1] - ft[0], ft[0], bpm, t0, track.duration)
    for lm, v in landmarks(ff, lags).items():
        out[f"flux.{lm}"] = v - ref_off
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tracks = [r for r in refs.load("engine") if r.split == args.split and _cache_path(r.audio).exists()]
    if args.limit:
        tracks = tracks[: args.limit]
    with ProcessPoolExecutor(12) as pool:
        rows = [r for r in pool.map(analyse, tracks) if r]
    print(f"{len(rows)} tracks with the right tempo ({args.split})\n")
    keys = [k for k in rows[0] if k not in ("id", "kind")]
    print(f"{'landmark':24s} {'median':>8s} {'IQR':>6s} {'≤3ms':>6s} {'≤5ms':>6s} {'≤10ms':>6s}   (offset from reference, ms)")
    stats = []
    for k in keys:
        v = np.array([r[k] for r in rows]) * 1000
        d = np.abs(v - np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        stats.append((np.mean(d <= 5), k, np.median(v), q3 - q1, np.mean(d <= 3), np.mean(d <= 10)))
    for share5, k, med, iqr, share3, share10 in sorted(stats, reverse=True):
        print(f"{k:24s} {med:+8.1f} {iqr:6.1f} {share3:6.0%} {share5:6.0%} {share10:6.0%}")


if __name__ == "__main__":
    main()
