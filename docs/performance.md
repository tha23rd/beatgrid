# Performance

Where the time went in the Python research pipeline when it ran Beat This! itself, and what
was done about it. These measurements are what led to the distilled network and the Rust
core; the last section is the outcome.

Measured 2026-10-08 on a 2019 MacBook Pro (Intel i9-9980HK, 8 cores / 16 threads, no GPU),
PyTorch 2.14 CPU, one 244-second 48 kHz MP3 (`research/experiments/stage_timing.py`).

| stage | Beat This! final0 (20 M params) | small0 (2 M params) |
|---|---|---|
| decode (ffmpeg) | 0.38 s | 0.39 s |
| resample + log-mel | 0.12 s | 0.11 s |
| network, whole track, 8 threads | **5.64 s** | 3.49 s |
| onset flux (numpy, 512/32) | 2.03 s | 1.92 s |
| fitting | 0.13 s | 0.16 s |

Per 30-second network chunk: final0 0.63 s (8 threads), 0.79 s (4), 1.14 s (2); small0 0.39 s,
0.47 s, 0.66 s.

## What does not help

- **ONNX Runtime** (traced export, opset 17): slower than PyTorch — 0.87 s/chunk fp32, 0.79 s
  int8 — and flat across thread counts. The traced graph loses PyTorch's fused attention;
  the cost is the front end's attention over the full 1500 × 128 spectrogram, not the
  parameter count (which is also why small0 is only 1.6× faster for 10× fewer parameters).

## What does

- **Run the network on part of the track.** The pipeline only needs rough tempo, phase and
  bar from the network; the flux does the precise work over the whole track. Four 30-second
  windows instead of the whole track cost ~1 track in 40 on the dev set.
- **small0** with four windows: about 1.6 s of network time per track, at a cost of 1–3
  points of usable@10 on the dev set (noisy at this size).
- **Flux in Rust** (realfft): the same 380k FFTs of 512 points should take ~0.2 s.
- **Throughput:** the network does not scale well past 4 threads, so a library run should
  analyse several tracks at once with a few threads each.

## The real fix: a small distilled network (done)

Because the pipeline no longer needs the network to be precise, a much smaller network can
supply the activations. It was distilled from Beat This! — the teacher's activations on
dance music are the training targets — and its downbeats were then also trained on
Raveform's expert grids. The shipped network is a 529k-parameter convolutional TCN
([model.md](model.md)); with it the whole Rust pipeline (decode, network, flux, fitting)
runs at about 0.5 s per track in batch on the same laptop, against ~6 s per track with
Beat This! in the research pipeline, and is level with or ahead of Beat This! on every
reference set ([results.md](results.md)).
