# Model card: beatgrid's beat and downbeat network

| | |
|---|---|
| file | `crates/beatgrid/models/student.safetensors` (2.1 MB, float32; embedded in the Rust library at build time) |
| name in its metadata | `student-v4` |
| size | 529,154 parameters |
| input | log-mel spectrogram, 128 bands, of mono audio at 22.05 kHz, 50 frames per second (hop 441) — the same input as Beat This! |
| output | per frame, a beat and a downbeat logit (sigmoid → probability) |
| teacher | Beat This! `final0` (Foscarin, Schlüter & Widmer, ISMIR 2024; code and weights MIT) |
| expert labels | Raveform (Kim, Kim, Kim & Nam, TISMIR 2026; labels CC BY 4.0) |
| training code | `research/beatgrid_lab/student/` |

## What it is for

beatgrid turns a produced track into a DJ grid: an exact tempo, every beat within a few
milliseconds of the attack, and bar 1 on the one. The network's job in that pipeline is
small: a rough pulse, phase and bar. Tempo is refined and every beat placed on the audio
itself (a sub-millisecond spectral flux), and bar 1 comes from the network's downbeats pooled
with other cues (`research/beatgrid_lab/bars.py`). So the network can be far smaller than
Beat This! as long as it gets the pulse and the downbeats right, and it runs on a CPU without
a deep-learning runtime.

It is not meant as a general-purpose beat tracker on its own: its 20 ms frames are not
precise enough for DJ grids without the fitting that follows.

## Architecture

A temporal convolutional network in the style of Böck & Davies (2019/2020):

- **front end:** three 2-D convolutions strided over frequency fold the 128 mel bands into a
  64-channel feature vector per frame (128 → 43 → 15 → 1 bands; ELU)
- **body:** 10 residual blocks of dilated 1-D convolutions over time, 64 channels, kernel 5;
  block *i* runs dilations 2^*i* and 2^(*i*+1) side by side and mixes them with a 1×1
  convolution. The receptive field is about 80 s either side of each frame (non-causal: the
  network looks ahead, as offline analysis can)
- **head:** a 1×1 convolution to two logits per frame (beat, downbeat)

Convolutions only: cheap on a CPU, and simple to port. The Rust crate runs it with its own
im2col + sgemm implementation (`crates/beatgrid/src/network.rs`);
`research/experiments/rust_parity.py --network student-v4` compares it end to end with the
PyTorch checkpoint.

## Training

1. **Distillation.** Beat This! `final0` was run on about 1,800 dance-music tracks (1,813)
   and its frame-wise beat and downbeat probabilities used as soft targets
   (`student/prepare.py`, binary cross-entropy against the teacher's probabilities). The
   audio, spectrograms and teacher outputs are not distributed.
2. **Expert supervision.** Raveform's expert-corrected beat grids were turned into frame
   targets (1 on the labelled frame, 0.5 either side) for the tracks where they match the
   audio: the teacher's beats and the labelled beats must agree with an F1 of at least 0.85
   within ±2 frames (`student/labels.py`). A re-uploaded or re-edited video carries labels
   for a different recording; those tracks were used for distillation only. 1,001 tracks
   passed. The label loss weighs downbeats at 1.0 and beats at 0.25, and is added to the
   distillation loss.
3. **Optimisation** (`student/train.py`; this network with `--channels 64 --blocks 10`). The
   script trains on random 60-second crops with AdamW and a one-cycle schedule, gradient
   clipping, and spectral augmentation (random tilt, a band boost or cut, level change, a
   little noise). 5% of tracks, chosen by a hash of their name, are held out; the checkpoint
   is chosen on beat agreement with the teacher plus downbeat agreement with the expert
   labels on those tracks. The shipped checkpoint is from step 20,000.

Held-out agreement at that step (F1 of peaks; ±1 frame against the teacher, ±2 frames
against the experts): beats vs teacher 0.90, downbeats vs experts 0.90 (45 labelled tracks),
downbeats vs teacher 0.70.

To reproduce, with the audio and Raveform's labels in hand:

```sh
cd research
scripts/setup-gpu-env.sh
PYTHONPATH=. .venv-gpu/bin/python -m beatgrid_lab.student.prepare AUDIO_DIR TARGETS_DIR
PYTHONPATH=. .venv-gpu/bin/python -m beatgrid_lab.student.labels TARGETS_DIR RAVEFORM_ROOT
PYTHONPATH=. .venv-gpu/bin/python -m beatgrid_lab.student.train TARGETS_DIR ../data/models/student-v4.pt --channels 64 --blocks 10
uv run --group dev python scripts/export_student.py student-v4
```

## How well it does

- **Downbeats alone** pick bar 1 on 91% of the 218 Raveform evaluation tracks, against 87% for
  Beat This! `final0`.
- **In the full pipeline** it is level with or ahead of Beat This! on every reference set
  (`usable@10`: 71-84% depending on the set), and the whole Rust pipeline runs at about
  0.5 s per track in batch on a 2019 laptop CPU, against ~6 s with Beat This!. The numbers,
  the references and what they can and cannot show are in [results.md](results.md).

The Raveform evaluation tracks were drawn from a list kept apart from the training lists;
one of the 218 also appears on the annotated training list.

## Limits

- **Dance music in 4/4.** It was trained and evaluated on electronic dance music. Live
  drumming, swing, rubato, odd meters and most non-dance music are untested and likely
  worse, and beatgrid's bar logic assumes four beats to the bar.
- **Octave is a convention, not something the network knows.** Drum & bass is 87 or 174,
  dubstep 70 or 140, depending on who you ask; beatgrid picks the octave by profile
  (`--profile edm`).
- **One tempo per track.** Tracks whose tempo changes (transition edits, mashups, drifting
  live recordings) get a single tempo and the review reason "tempo may change".
- **Downbeats are the weak part.** On about one Raveform track in ten the downbeat output
  alone is wrong (on repetitive techno it can slip a whole beat); the pipeline pools it with
  arrangement changes and high-frequency bursts, and flags tracks where those cues disagree.
- **Inherited conventions.** It imitates Beat This! and Raveform's annotators (Rekordbox grids
  corrected by experts), including their idea of where a beat and a bar start.

## Licence and credit

The weights ship with beatgrid under its licence (MIT or Apache-2.0). They were learned from
Beat This!'s outputs (code and weights MIT) and from Raveform's labels (CC BY 4.0, which asks
for attribution, given here).

- Beat This!: F. Foscarin, J. Schlüter and G. Widmer, ISMIR 2024 ·
  https://github.com/CPJKU/beat_this · https://arxiv.org/abs/2407.21658
- Raveform: Kim, Kim, Kim and Nam, Transactions of the International Society for Music
  Information Retrieval (TISMIR), 2026 · https://huggingface.co/datasets/taejunkim/raveform ·
  https://transactions.ismir.net/articles/10.5334/tismir.288
