"""Where does the time go? Stage-by-stage timing for one track, nothing cached.

  uv run python experiments/stage_timing.py <audio file> [--threads N] [--checkpoint final0|small0]
"""

import argparse
import time

import numpy as np
import soxr
import torch

from beatgrid_lab import audio
from beatgrid_lab.activations import Activations
from beatgrid_lab.pipeline import Params, onset_flux, run


class Timer:
    def __init__(self):
        self.rows = []

    def __call__(self, label):
        timer = self

        class Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.rows.append((label, time.perf_counter() - self.t))

        return Ctx()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--checkpoint", default="final0")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from beat_this.inference import Spect2Frames, load_model
    from beat_this.preprocessing import LogMelSpect

    t = Timer()
    with t("load model (once per process)"):
        frames_model = Spect2Frames(checkpoint_path=args.checkpoint, device="cpu")
        mel = LogMelSpect(device="cpu")
    with t("decode (ffmpeg)"):
        x, rate = audio.load(args.path)
    duration = len(x) / rate
    with t("resample to 22.05 kHz (soxr)"):
        y = soxr.resample(x.astype(np.float32), rate, 22050)
    with t("log-mel spectrogram"):
        spect = mel(torch.from_numpy(y))
    with t(f"network inference ({args.checkpoint}, {args.threads} threads)"):
        beat, down = frames_model(spect)
    params = sum(p.numel() for p in frames_model.model.parameters())
    act = Activations(torch.sigmoid(beat).numpy(), torch.sigmoid(down).numpy())
    with t("onset flux (512/32)"):
        flux = onset_flux(x, rate)

    class T:
        activations = act

        def flux(self):
            return flux

    with t("fitting (tempo, snap, placement, bar)"):
        result = run(T(), Params())
    total = sum(s for label, s in t.rows if "once" not in label)
    print(f"{args.path.split('/')[-1]}: {duration:.0f} s of audio at {rate} Hz; model {params / 1e6:.1f} M parameters")
    for label, s in t.rows:
        print(f"  {label:45s} {s:6.2f} s")
    print(f"  {'total per track (excluding model load)':45s} {total:6.2f} s   -> {duration / total:.0f}x real time")
    print(f"  grid: {result.grid.bpm:.3f} BPM")


if __name__ == "__main__":
    main()
