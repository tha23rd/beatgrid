# Changelog

## 0.1.0 (2026-10-10)

Grids whose tempo changes. Constant grids are unchanged: with the default `constant` tempo
mode every track of every reference set scores as with 0.0.2.

- New: segmented grids. `TempoMode` (`--tempo constant|auto|variable` in the CLI, `tempo=` in
  Python, `Options::tempo` in Rust) lets a grid change tempo. `auto` splits a track into
  constant-tempo segments where it clearly changes tempo (transition edits, mashups) and
  keeps one tempo everywhere else; `variable` looks harder and also follows a phase jump at
  the same tempo. The default stays `constant`.
- New: `SegmentedGrid` and `Segment` (`start`, `bpm`, `beat_in_bar`), with `beats`,
  `beat_number_at`, `time_of_beat`, `first_beat`, `first_downbeat`, `bpm_at`, `segment_at`,
  `start_beat`, `start_bar` and `From<Grid>`. Beat numbers and bars count on across segments.
  `Analysis::segments` always holds the grid to use (one segment when the tempo holds);
  `Analysis::grid` is still the best single-tempo grid.
- New review reason: `tempo changes: check the grid where it changes`, in place of
  `tempo may change`, when the grid is in segments.
- CLI: the JSON has `segments` (each with the `beat` and `bar` it starts on), the CSV a
  `segments` column (`start@bpm` each, after `error`), the text table a line per further
  segment; `beatgrid beats --tempo auto` follows the segments. Python: `grid["segments"]`, and
  `grid["beats"]` follows them.
- Changed: `Options` has a second field, `tempo`; a struct literal needs it (or
  `..Options::default()`). `Params` has the `seg_*` knobs and `Params::with_tempo`.
- Measured with `research/experiments/variable_tempo.py` on a synthetic set of transition
  edits made from steady reference tracks, the DJ grids with several tempo markers, and the
  steady reference sets ([docs/results.md](docs/results.md#tempo-changes-2026-10-10-010)).
  On 150 held-out synthetic edits `auto` is usable@10 on 72% and `variable` on 85% (one
  tempo: 15%); `auto` segmented none of the 327 one-tempo references of the DJ-corrected,
  ear-approved, machine-fitted and Raveform sets, whose scores are unchanged, and 7 of
  Engine DJ's 108 own grids (6 of them already wrong at one tempo).

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
