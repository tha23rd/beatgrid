"""Frame-wise beat and downbeat activations from Beat This!, cached on disk.

The network is the slow part (seconds per track on CPU); everything after it is cheap,
so the pipeline is iterated against cached activations.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio
from .config import CACHE

MODEL = "final0"  # default checkpoint; "small0" is the 2 M-parameter variant
FPS = 50  # Beat This! predicts at 50 frames per second (22.05 kHz, hop 441)

_models: dict = {}


@dataclass
class Activations:
    beat: np.ndarray  # probabilities, one per frame
    downbeat: np.ndarray
    fps: float = FPS

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.beat)) / self.fps


def _key(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:20]


def _cache_path(path: Path, model: str = MODEL) -> Path:
    return CACHE / f"beatthis-{model}" / f"{_key(path)}.npz"


def _student(name: str):
    """A distilled student from DATA/models/<name>.pt (see beatgrid_lab.student)."""
    import torch

    from .config import DATA
    from .student.model import Student

    checkpoint = torch.load(DATA / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    net = Student(**checkpoint["config"])
    net.load_state_dict(checkpoint["state_dict"])
    return net.eval()


def compute_student(samples: np.ndarray, rate: int, model: str) -> Activations:
    import soxr
    import torch
    from beat_this.preprocessing import LogMelSpect

    if model not in _models:
        _models[model] = (_student(model), LogMelSpect(device="cpu"))
    net, mel = _models[model]
    x = samples.astype(np.float32)
    if rate != 22050:
        x = soxr.resample(x, rate, 22050)
    with torch.inference_mode():
        logits = net(mel(torch.from_numpy(x))[None])[0]
    probs = torch.sigmoid(logits).numpy()
    return Activations(probs[:, 0].copy(), probs[:, 1].copy())


def compute(samples: np.ndarray, rate: int, model: str = MODEL) -> Activations:
    import torch
    from beat_this.inference import Audio2Frames

    if model.startswith("student"):
        return compute_student(samples, rate, model)
    if model not in _models:
        torch.set_num_threads(max(1, (torch.get_num_threads() * 3) // 4))
        _models[model] = Audio2Frames(checkpoint_path=model, device="cpu")
    with torch.inference_mode():
        beat, down = _models[model](samples.astype(np.float64), rate)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x.float().cpu().numpy()))
    return Activations(sigmoid(beat), sigmoid(down))


def load(path: Path, samples: np.ndarray | None = None, rate: int | None = None, model: str = MODEL) -> Activations:
    cached = _cache_path(path, model)
    if cached.exists():
        d = np.load(cached)
        return Activations(d["beat"], d["downbeat"], float(d["fps"]))
    if samples is None:
        samples, rate = audio.load(path)
    act = compute(samples, rate, model)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, beat=act.beat, downbeat=act.downbeat, fps=act.fps)
    return act
