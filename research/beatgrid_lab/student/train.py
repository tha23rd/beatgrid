"""Distil the student from the teacher's probabilities.

  python -m beatgrid_lab.student.train TARGETS_DIR OUT.pt [--channels 32] [--steps 30000]

TARGETS_DIR holds prepare.py's .npz files. They are packed once into two memory-mapped
arrays so random crops are slices, not file reads. 5% of tracks (by name) are held out to
measure agreement with the teacher.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import Student, parameters


def held_out(name: str) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 20 == 0


def pack(targets: Path) -> dict:
    """Concatenate every track's spect (T×128) and targets (T×2) into memmaps."""
    files = sorted(f for f in targets.glob("*.npz") if not f.name.endswith(".tmp.npz"))
    index_path = targets / "pack.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["files"] == len(files):
            return index
    lengths, names = [], []
    for f in files:
        with np.load(f) as d:
            lengths.append(len(d["beat"]))
        names.append(f.stem)
    total = int(sum(lengths))
    spect = np.lib.format.open_memmap(targets / "pack-spect.npy", "w+", np.float16, (total, 128))
    probs = np.lib.format.open_memmap(targets / "pack-probs.npy", "w+", np.float16, (total, 2))
    # Expert labels where a track has usable ones (labels.py), NaN elsewhere.
    labels = np.lib.format.open_memmap(targets / "pack-labels.npy", "w+", np.float16, (total, 2))
    at = 0
    labelled = 0
    for f, n in zip(files, lengths):
        with np.load(f) as d:
            spect[at : at + n] = d["spect"][:n]
            probs[at : at + n, 0] = d["beat"][:n]
            probs[at : at + n, 1] = d["downbeat"][:n]
            if "label_ok" in d and bool(d["label_ok"]):
                labels[at : at + n, 0] = d["label_beat"][:n]
                labels[at : at + n, 1] = d["label_down"][:n]
                labelled += 1
            else:
                labels[at : at + n] = np.nan
        at += n
    spect.flush()
    probs.flush()
    labels.flush()
    offsets = np.concatenate([[0], np.cumsum(lengths)]).tolist()
    index = {"files": len(files), "names": names, "offsets": offsets, "labelled": labelled}
    index_path.write_text(json.dumps(index))
    return index


def peak_frames(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    neighbourhood = np.max([np.roll(p, k) for k in range(-3, 4)], axis=0)
    return np.flatnonzero((p > threshold) & (p >= neighbourhood))


def agreement(student: np.ndarray, teacher: np.ndarray, tolerance: int = 1) -> float:
    """F1 between the two sets of peaks, within ±tolerance frames."""
    s, t = peak_frames(student), peak_frames(teacher)
    if len(s) == 0 or len(t) == 0:
        return float(len(s) == len(t))
    hit_t = np.array([np.min(np.abs(s - x)) <= tolerance for x in t])
    hit_s = np.array([np.min(np.abs(t - x)) <= tolerance for x in s])
    precision, recall = hit_s.mean(), hit_t.mean()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def augment(spect: torch.Tensor) -> torch.Tensor:
    """Cheap robustness: a smooth random tilt/EQ across mel bands and a level change (both
    additive in the log-ish domain), plus a little noise."""
    b = spect.shape[0]
    bands = torch.linspace(-1, 1, spect.shape[-1], device=spect.device)
    tilt = torch.randn(b, 1, 1, device=spect.device) * 0.5 * bands
    bump = torch.randn(b, 1, 1, device=spect.device) * 0.5 * torch.exp(-((bands - torch.rand(b, 1, 1, device=spect.device) * 2 + 1) ** 2) / 0.1)
    level = torch.randn(b, 1, 1, device=spect.device) * 0.5
    noise = torch.randn_like(spect) * 0.05
    return torch.clamp(spect + tilt + bump + level + noise, min=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("out")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=3000, help="frames per training crop (50 fps)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--label-beat-weight", type=float, default=0.25)
    ap.add_argument("--label-down-weight", type=float, default=1.0)
    args = ap.parse_args()

    targets = Path(args.targets)
    index = pack(targets)
    spect = np.load(targets / "pack-spect.npy", mmap_mode="r")
    probs = np.load(targets / "pack-probs.npy", mmap_mode="r")
    labels = np.load(targets / "pack-labels.npy", mmap_mode="r") if (targets / "pack-labels.npy").exists() else None
    offsets = index["offsets"]
    tracks = [(name, offsets[i], offsets[i + 1]) for i, name in enumerate(index["names"])]
    train = [t for t in tracks if not held_out(t[0]) and t[2] - t[1] > args.crop]
    val = [t for t in tracks if held_out(t[0])]
    print(f"{len(train)} training tracks, {len(val)} held out; {offsets[-1] / 50 / 3600:.1f} h of audio; "
          f"{index.get('labelled', 0)} with expert labels", flush=True)

    model = Student(channels=args.channels, blocks=args.blocks).to(args.device)
    print(f"student: {parameters(model) / 1e3:.0f}k parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    lengths = np.array([e - s for _, s, e in train], float)
    weights = lengths / lengths.sum()  # sample tracks by length: every minute equally likely

    def batch():
        xs, ys, ls = [], [], []
        for i in rng.choice(len(train), size=args.batch, p=weights):
            _, s, e = train[i]
            a = int(rng.integers(s, e - args.crop))
            xs.append(np.asarray(spect[a : a + args.crop], np.float32))
            ys.append(np.asarray(probs[a : a + args.crop], np.float32))
            ls.append(np.asarray(labels[a : a + args.crop], np.float32) if labels is not None
                      else np.full((args.crop, 2), np.nan, np.float32))
        to = lambda v: torch.from_numpy(np.stack(v)).to(args.device)
        return to(xs), to(ys), to(ls)

    def label_loss(logits, lab):
        mask = ~torch.isnan(lab)
        if not mask.any():
            return logits.sum() * 0.0
        w = torch.tensor([args.label_beat_weight, args.label_down_weight], device=logits.device).expand_as(lab)
        loss = F.binary_cross_entropy_with_logits(logits[mask], lab[mask], weight=w[mask], reduction="sum")
        return loss / mask.sum()

    def evaluate() -> dict:
        model.eval()
        losses, beat_f1, down_f1, label_down_f1 = [], [], [], []
        with torch.inference_mode():
            for _, s, e in val:
                x = torch.from_numpy(np.asarray(spect[s:e], np.float32))[None].to(args.device)
                y = torch.from_numpy(np.asarray(probs[s:e], np.float32))[None].to(args.device)
                logits = model(x)
                losses.append(F.binary_cross_entropy_with_logits(logits, y).item())
                p = torch.sigmoid(logits)[0].cpu().numpy()
                t = y[0].cpu().numpy()
                beat_f1.append(agreement(p[:, 0], t[:, 0]))
                down_f1.append(agreement(p[:, 1], t[:, 1]))
                if labels is not None and not np.isnan(float(labels[s, 1])):
                    label_down_f1.append(agreement(p[:, 1], np.asarray(labels[s:e, 1], np.float32), tolerance=2))
        model.train()
        return {"loss": float(np.mean(losses)), "beat_f1": float(np.mean(beat_f1)), "downbeat_f1": float(np.mean(down_f1)),
                "label_downbeat_f1": float(np.mean(label_down_f1)) if label_down_f1 else None, "labelled_val": len(label_down_f1)}

    best = -math.inf
    started = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        x, y, lab = batch()
        logits = model(augment(x))
        loss = F.binary_cross_entropy_with_logits(logits, y) + label_loss(logits, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        schedule.step()
        running += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            expert = f" expert-downbeat {v['label_downbeat_f1']:.3f} (n={v['labelled_val']})" if v["label_downbeat_f1"] is not None else ""
            print(f"step {step} train {running / args.eval_every:.4f} val {v['loss']:.4f} "
                  f"beat-agree {v['beat_f1']:.3f} downbeat-agree {v['downbeat_f1']:.3f}{expert} "
                  f"lr {schedule.get_last_lr()[0]:.2e} {time.time() - started:.0f}s", flush=True)
            running = 0.0
            # Selected on agreement with the teacher's peaks, which is what the pipeline
            # uses; the loss bottoms out early while downbeat agreement keeps improving.
            # With expert labels in the held-out set, downbeats are judged against the experts.
            agree = v["beat_f1"] + (v["label_downbeat_f1"] if v["label_downbeat_f1"] is not None else v["downbeat_f1"])
            if agree > best:
                best = agree
                torch.save({"config": model.config, "state_dict": model.state_dict(), "step": step, "val": v}, args.out)
    print(f"best beat+downbeat agreement {best:.3f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
