"""Does the Rust decoder put samples on the same timeline as ffmpeg?

For files of each format: decode with `beatgrid decode` and with ffmpeg (both mono), find the
exact sample offset by cross-correlating the first 20 seconds, and report the offset, the
length difference and how different the aligned samples are.

  uv run python experiments/decode_parity.py BINARY FILE...
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from beatgrid_lab import audio


def rust_decode(binary: str, path: Path) -> np.ndarray:
    out = subprocess.run([binary, "decode", str(path)], capture_output=True, check=True)
    return np.frombuffer(out.stdout, "<f4")


def offset(a: np.ndarray, b: np.ndarray, rate: int, max_lag_s: float = 0.2) -> int:
    """Lag (samples) by which b is late relative to a."""
    n = min(len(a), len(b), rate * 20)
    lag = int(rate * max_lag_s)
    x, y = a[:n].astype(np.float64), b[:n].astype(np.float64)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y, size) * np.conj(np.fft.rfft(x, size)), size)
    c = np.concatenate([c[-lag:], c[: lag + 1]])
    return int(np.argmax(c)) - lag


def main():
    binary, files = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    by_ext = defaultdict(list)
    for p in files:
        ff, rate = audio.load(p)
        rs = rust_decode(binary, p)
        lag = offset(ff, rs, rate)
        a, b = (ff[lag:], rs) if lag < 0 else (ff, rs[lag:]) if lag > 0 else (ff, rs)
        a, b = (ff, rs[lag:]) if lag >= 0 else (ff[-lag:], rs)
        n = min(len(a), len(b))
        diff = float(np.max(np.abs(a[:n] - b[:n]))) if n else float("nan")
        by_ext[p.suffix.lower()].append(lag)
        print(f"{p.name:24s} {rate:6d} Hz  rust is {lag:+6d} samples ({lag / rate * 1000:+7.2f} ms) vs ffmpeg  "
              f"length {len(rs) - len(ff):+7d}  max |diff| aligned {diff:.2e}")
    print()
    for ext, lags in sorted(by_ext.items()):
        print(f"{ext:6s} n={len(lags):2d}  offsets {sorted(set(lags))}")


if __name__ == "__main__":
    main()
