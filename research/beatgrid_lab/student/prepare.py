"""Teacher targets for distillation: for every audio file, the log-mel spectrogram the
student will see and Beat This!'s beat/downbeat probabilities for it.

  python -m beatgrid_lab.student.prepare AUDIO_DIR OUT_DIR [--device cuda] [--watch]

--watch keeps going as new files appear (e.g. while a download is still running).
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import soxr
import torch

AUDIO = (".mp3", ".flac", ".wav", ".m4a", ".webm", ".opus", ".ogg", ".aiff")


def decode_22k(path: Path) -> np.ndarray:
    """Mono float32 at 22.05 kHz, decoder-trimmed timeline (ffmpeg), resampled with soxr like
    Beat This! does."""
    rate = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                               "stream=sample_rate", "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, check=True).stdout.strip().rstrip(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, "<f4")
    return soxr.resample(x, rate, 22050) if rate != 22050 else x.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default="final0")
    ap.add_argument("--watch", action="store_true", help="keep processing new files until none arrive for 10 minutes")
    args = ap.parse_args()
    from beat_this.inference import Spect2Frames
    from beat_this.preprocessing import LogMelSpect

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    teacher = Spect2Frames(checkpoint_path=args.checkpoint, device=args.device)
    mel = LogMelSpect(device=args.device)
    idle_since = time.time()
    done = 0
    while True:
        todo = [p for p in sorted(Path(args.audio_dir).iterdir())
                if p.suffix.lower() in AUDIO and not (out / f"{p.stem}.npz").exists()]
        if not todo:
            if not args.watch or time.time() - idle_since > 600:
                break
            time.sleep(30)
            continue
        idle_since = time.time()
        for p in todo:
            started = time.time()
            try:
                x = decode_22k(p)
            except subprocess.CalledProcessError:
                print(f"skip {p.name}: cannot decode", flush=True)
                continue
            if len(x) < 22050 * 30:
                continue
            with torch.inference_mode():
                spect = mel(torch.from_numpy(x).to(args.device))
                beat, down = teacher(spect)
            # Written under a temporary name and renamed, so a reader never sees half a file.
            tmp = out / f"{p.stem}.tmp.npz"
            np.savez(tmp,
                     spect=spect.cpu().numpy().astype(np.float16),
                     beat=torch.sigmoid(beat).cpu().numpy().astype(np.float16),
                     downbeat=torch.sigmoid(down).cpu().numpy().astype(np.float16))
            tmp.rename(out / f"{p.stem}.npz")
            done += 1
            print(f"[{done}] {p.stem} {len(x) / 22050:.0f}s audio in {time.time() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
