# Changelog

## 0.0.2 (2026-10-09)

Review reasons. Grids, tempos and bar 1 are unchanged (every track of every reference set
scores the same as with 0.0.1).

- New: `octave chosen by the profile: the audio's pulse is twice as fast` (or `half as fast`),
  when the octave convention (`--profile edm`) moved the tempo off the octave the audio
  suggests: drum & bass counted in half time, a slow track doubled. DJs count these both ways.
- Changed: `no clear attack to place the grid on` is also given when, under the final grid,
  fewer than 3 of the track's 8 stretches show a clear attack.
- New diagnostics (in the CLI's JSON output and `Analysis::diagnostics`): `attack_second` (the
  runner-up attack as a fraction of the main one), `stretch_drift_ms` and `clear_stretches`
  (how far the attack walks under the grid from one end of the track to the other, measured in
  8 stretches), and `snap_drift_ms` (how far the exact whole or half BPM walks from the
  measured tempo). They are not review reasons on their own: tried as reasons, they flagged
  usable grids at least as often as wrong ones.
- Measured with `research/experiments/review_flags.py` against DJ-corrected, ear-approved,
  machine-fitted and Raveform grids: the reasons are given for 52% (dev) and 38% (test) of the
  grids that are not usable@10, against 46% and 38% for 0.0.1, at 22-24% false flags (0.0.1:
  21-23%). A larger set tuned on dev did not hold up on the held-out test; the numbers and what
  was tried are in [docs/results.md](docs/results.md#review-reasons-2026-10-09-002).

## 0.0.1 (2026-10-09)

First release: Rust library, CLI and Python bindings with the embedded `student-v4` network.
