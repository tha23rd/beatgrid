"""Reference sets: tracks with an audio file and a grid we trust.

Stored as JSONL under DATA/refs/<set>.jsonl, one track per line:
{"id", "audio" (relative to DATA), "grid": {"numbers", "times"}, "split": "dev"|"test",
 "kind": "corrected"|"accepted"|..., "baselines": {name: grid}, "meta": {...}}
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DATA, REFS
from ..grid import Grid


def split_of(track_id: str, test_share: float = 0.3) -> str:
    """Deterministic dev/test assignment, so a track never changes sides."""
    h = int(hashlib.sha1(track_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_share else "dev"


def grid_json(grid: Grid) -> dict:
    return {"numbers": grid.numbers.tolist(), "times": grid.times.tolist()}


def grid_from_json(d: dict) -> Grid:
    return Grid(np.asarray(d["numbers"], float), np.asarray(d["times"], float))


@dataclass
class Ref:
    id: str
    audio: Path
    grid: Grid
    split: str
    kind: str
    baselines: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def load(name: str) -> list[Ref]:
    refs = []
    for line in open(REFS / f"{name}.jsonl"):
        d = json.loads(line)
        refs.append(Ref(
            id=d["id"],
            audio=DATA / d["audio"],
            grid=grid_from_json(d["grid"]),
            split=d["split"],
            kind=d["kind"],
            baselines={k: grid_from_json(v) for k, v in d.get("baselines", {}).items()},
            meta=d.get("meta", {}),
        ))
    return refs


def save(name: str, records: list[dict]) -> Path:
    REFS.mkdir(parents=True, exist_ok=True)
    path = REFS / f"{name}.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path
