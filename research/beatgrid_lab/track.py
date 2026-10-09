"""One audio file and everything derived from it, each piece computed once and cached."""

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from . import activations, audio
from .activations import Activations, _key
from .config import CACHE
from .metrics import music_span


class Track:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.key = _key(self.path)

    @cached_property
    def _decoded(self) -> tuple[np.ndarray, int]:
        return audio.load(self.path)

    @property
    def samples(self) -> np.ndarray:
        return self._decoded[0]

    @property
    def rate(self) -> int:
        return self._decoded[1]

    @cached_property
    def activations(self) -> Activations:
        return self.activations_for(activations.MODEL)

    def activations_for(self, model: str) -> Activations:
        if activations._cache_path(self.path, model).exists():
            return activations.load(self.path, model=model)
        return activations.load(self.path, self.samples, self.rate, model)

    def flux(self) -> tuple[np.ndarray, float, float]:
        from .pipeline import FLUX_HOP, FLUX_WINDOW, onset_flux

        path = CACHE / f"flux-{FLUX_WINDOW}-{FLUX_HOP}" / f"{self.key}.npz"
        if path.exists():
            d = np.load(path)
            return d["flux"], float(d["dt"]), float(d["t_first"])
        flux, dt, t_first = onset_flux(self.samples, self.rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, flux=flux, dt=dt, t_first=t_first)
        return flux, dt, t_first

    def band_energy(self) -> tuple[np.ndarray, float]:
        """Log energy in 12 log-spaced bands (40 Hz-16 kHz), frames of 2048 hop 1024.
        Returns (frames x 12, seconds per frame); frame i is centred at (i*1024+1024)/rate."""
        path = CACHE / "bands12" / f"{self.key}.npz"
        if path.exists():
            d = np.load(path)
            return d["e"], float(d["dt"])
        x, rate = self.samples.astype(np.float32), self.rate
        n, hop = 2048, 1024
        frames = (len(x) - n) // hop + 1
        window = np.hanning(n).astype(np.float32)
        edges = np.geomspace(40, 16000, 13)
        freqs = np.fft.rfftfreq(n, 1 / rate)
        band_of = np.clip(np.searchsorted(edges, freqs) - 1, -1, 11)
        out = np.zeros((max(frames, 0), 12), np.float32)
        for b in range(0, frames, 4096):
            idx = (np.arange(b, min(b + 4096, frames)) * hop)[:, None] + np.arange(n)[None, :]
            power = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
            for k in range(12):
                out[b : b + len(idx), k] = power[:, band_of == k].sum(axis=1)
        out = np.log(out + 1e-8)
        dt = hop / rate
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, e=out, dt=dt)
        return out, dt

    @cached_property
    def _info(self) -> dict:
        path = CACHE / "info" / f"{self.key}.json"
        if path.exists():
            return json.loads(path.read_text())
        start, end = music_span(self.samples, self.rate)
        info = {"duration": len(self.samples) / self.rate, "music_start": start, "music_end": end}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info))
        return info

    @property
    def duration(self) -> float:
        return self._info["duration"]

    @property
    def music_span(self) -> tuple[float, float]:
        return self._info["music_start"], self._info["music_end"]
