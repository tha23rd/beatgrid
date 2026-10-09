"""Decoding. ffmpeg trims the MP3/AAC encoder delay, and that trimmed timeline is the one
DJ software stores grids on (measured against Engine DJ: adding the delay back shifts every
affected track by exactly its size)."""

import subprocess
from pathlib import Path

import numpy as np


def load(path: Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """Decode to float32 at the file's native rate. Returns (samples, sample_rate);
    samples are 1-D when mono, else (frames, channels)."""
    rate = int(
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().rstrip(",")
    )
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le"]
    if mono:
        cmd += ["-ac", "1"]
    raw = subprocess.run(cmd + ["-"], capture_output=True, check=True).stdout
    samples = np.frombuffer(raw, dtype="<f4")
    if not mono:
        channels = int(
            subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=channels", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=True,
            ).stdout.strip().rstrip(",")
        )
        samples = samples.reshape(-1, channels)
    return samples.copy(), rate
