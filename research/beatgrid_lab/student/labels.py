"""Expert labels for supervised training: Raveform's beat grids as frame targets.

  python -m beatgrid_lab.student.labels TARGETS_DIR RAVEFORM_ROOT

For every prepared track (TARGETS_DIR/<youtube id>.npz) with a Raveform annotation, adds
`label_beat` and `label_down` (1 on the labelled frame, 0.5 either side — the labels are
grid times, the audio's attack can sit a frame off) and `label_ok`.

`label_ok` is false when the labelled beats do not line up with the beats the teacher hears
in this audio: a re-uploaded or re-edited video carries labels for a different recording
(another edit, mix or master of the same track). Those tracks are still used for
distillation, without labels.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

FPS = 50
MIN_AGREEMENT = 0.85


def frames_of(times: list[float], length: int) -> np.ndarray:
    target = np.zeros(length, np.float32)
    for t in times:
        f = int(round(t * FPS))
        for d, v in ((-1, 0.5), (0, 1.0), (1, 0.5)):
            if 0 <= f + d < length:
                target[f + d] = max(target[f + d], v)
    return target


def beat_agreement(teacher: np.ndarray, label_times: list[float], tolerance: int = 2) -> float:
    """F1 between the teacher's beat peaks and the labelled beats, within ±tolerance frames."""
    neighbourhood = np.max([np.roll(teacher, k) for k in range(-3, 4)], axis=0)
    peaks = np.flatnonzero((teacher > 0.5) & (teacher >= neighbourhood))
    labels = np.round(np.asarray(label_times) * FPS).astype(int)
    labels = labels[labels < len(teacher)]
    if len(peaks) == 0 or len(labels) == 0:
        return 0.0
    hit_l = np.array([np.min(np.abs(peaks - x)) <= tolerance for x in labels])
    hit_p = np.array([np.min(np.abs(labels - x)) <= tolerance for x in peaks])
    p, r = hit_p.mean(), hit_l.mean()
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def main():
    targets, root = Path(sys.argv[1]), Path(sys.argv[2])
    keys = {t["id"]: t["key"] for t in json.load(open(root / "structures" / "segments.json"))}
    ok = bad = skipped = 0
    for npz in sorted(targets.glob("*.npz")):
        if npz.name.endswith(".tmp.npz") or npz.stem not in keys:
            skipped += 1
            continue
        d = dict(np.load(npz))
        rows = list(csv.DictReader(open(root / "structures" / "beats" / f"{keys[npz.stem]}.beat.csv")))
        beats = [float(r["time"]) for r in rows]
        downs = [float(r["time"]) for r in rows if int(r["downbeat"]) == 1]
        n = len(d["beat"])
        agreement = beat_agreement(d["beat"].astype(np.float32), beats)
        d["label_beat"] = frames_of(beats, n).astype(np.float16)
        d["label_down"] = frames_of(downs, n).astype(np.float16)
        d["label_ok"] = np.array(agreement >= MIN_AGREEMENT)
        d["label_agreement"] = np.array(agreement, np.float32)
        tmp = npz.with_name(npz.stem + ".tmp.npz")
        np.savez(tmp, **d)
        tmp.rename(npz)
        ok += agreement >= MIN_AGREEMENT
        bad += agreement < MIN_AGREEMENT
    print(f"labelled {ok + bad} tracks: {ok} usable, {bad} whose labels do not match the audio; {skipped} without annotation")


if __name__ == "__main__":
    main()
