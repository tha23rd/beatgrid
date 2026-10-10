# beatgrid

DJ-grade beat grids for any audio file: the exact tempo (snapped to the whole or half BPM a
track was produced at, when the audio agrees), every beat on the attack within a few
milliseconds of where DJs put grid lines, and bar 1 on the one. Research beat trackers count
a beat as right within ±70 ms; a grid you can mix on has to hold to about 10 ms for the
whole track, and that is what beatgrid is built and measured for. It is a Rust library with
a CLI and Python bindings, runs in about half a second per track on a laptop CPU with no ML
runtime, and returns plain numbers (a tempo, a downbeat, every beat), so it drops into any
app and is not tied to any DJ software's library format. Tracks whose tempo changes
(transition edits, mashups) can get a grid in constant-tempo segments, the way DJ software
stores a grid with several tempo markers.

> **Status:** pre-release (0.1.0; changes in [CHANGELOG.md](CHANGELOG.md)). Not yet published
> to crates.io or PyPI; install from source as below.

## Install

Everything builds from source with a Rust toolchain (1.87 or newer). The network weights
(2.1 MB) are compiled into the library, so there are no model files to ship.

```sh
# CLI
cargo install --git https://github.com/tha23rd/beatgrid beatgrid-cli
# or, from a clone
git clone https://github.com/tha23rd/beatgrid && cd beatgrid
cargo install --path crates/beatgrid-cli
```

```toml
# Rust library (Cargo.toml)
[dependencies]
beatgrid = { git = "https://github.com/tha23rd/beatgrid" }
```

```sh
# Python ≥ 3.10 (pip builds the extension with maturin, so Rust is needed)
pip install "git+https://github.com/tha23rd/beatgrid#subdirectory=crates/beatgrid-py"
```

## Quick start

### CLI

```console
$ beatgrid analyze track.flac other.mp3
FILE             BPM  DOWNBEAT  REVIEW
track.flac   127.000     1.889  ok
other.mp3    124.000     1.740  tempo may change

$ beatgrid analyze --format csv track.flac other.mp3
file,bpm,first_beat,first_downbeat,duration,review,error,segments
track.flac,127,0.472118,1.889441,355.778,,,1.889441@127
other.mp3,124,0.288236,1.739849,170.031,tempo may change,,1.739849@124

$ beatgrid analyze --tempo auto edit.mp3
FILE          BPM  DOWNBEAT  REVIEW
edit.mp3  124.000     0.022  tempo changes: check the grid where it changes
          150.000    59.886  from bar 32

$ beatgrid beats track.flac | head -6
time,bar,beat_in_bar
0.472118,0,2
0.944559,0,3
1.417000,0,4
1.889441,1,1
2.361882,1,2
```

- `beatgrid analyze FILES...` prints a table on a terminal and one JSON object per line when
  piped; `--format text|json|csv` chooses. The JSON carries the full analysis, diagnostics
  included. Files are gridded in parallel (`-j N` limits the threads); a file that fails is
  reported, the rest carry on, and the exit code is 1.
- `beatgrid beats FILE` lists every beat (`--format csv|json`).
- `--profile edm` on either counts drum & bass in half time and folds everything else into
  100–200 BPM, and says so in the review reasons when that moved a track off the octave the
  audio suggests. The default keeps the octave the audio suggests.
- `--tempo auto` on either splits the grid into constant-tempo segments where the track
  clearly changes tempo (a transition edit, a mashup) and keeps one tempo everywhere else;
  `--tempo variable` looks harder and also follows a phase jump at the same tempo, for tracks
  you know change. The default, `constant`, is one tempo per track. The segments are in the
  JSON (`segments`: `start`, `bpm`, `beat_in_bar`, and the `beat` and `bar` the segment
  starts on) and the CSV (`segments`: `start@bpm` each, space-separated); the text table
  shows each further segment on a line of its own, and `beatgrid beats` follows them.
- Reads MP3, FLAC, WAV, Ogg Vorbis, and AAC/ALAC in MP4.

### Rust

```rust,no_run
use beatgrid::{Analyzer, Options, Profile, TempoMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the network once; an Analyzer is Send + Sync and cheap to clone.
    let analyzer = Analyzer::with_options(Options { profile: Profile::Edm, tempo: TempoMode::Constant })?;
    let analysis = analyzer.analyze_file("track.flac")?;

    if let Some(grid) = &analysis.grid {
        println!("{} BPM, bar 1 at {:.3} s", grid.bpm, grid.first_downbeat());
        for beat in grid.beats(0.0, analysis.duration) {
            println!("{:.3} s  bar {} beat {}", beat.time, beat.bar, beat.beat_in_bar);
        }
    }
    if analysis.needs_review() {
        println!("worth checking by ear: {}", analysis.review.join("; "));
    }

    // A track whose tempo changes: constant-tempo segments where it clearly does.
    let analyzer = Analyzer::with_options(Options { tempo: TempoMode::Auto, ..Options::default() })?;
    let analysis = analyzer.analyze_file("edit.mp3")?;
    if let Some(grid) = &analysis.segments {
        for (i, segment) in grid.segments().iter().enumerate() {
            println!("from {:.3} s (bar {}): {} BPM", segment.start, grid.start_bar(i), segment.bpm);
        }
        for beat in grid.beats(0.0, analysis.duration) {
            println!("{:.3} s  bar {} beat {}", beat.time, beat.bar, beat.beat_in_bar);
        }
    }
    Ok(())
}
```

`Analyzer::new()` uses the default profile and one tempo per track. If you already have decoded audio, call
`analyzer.analyze_samples(&mono, sample_rate)`, mixing stereo to mono as (L + R) / √2 the way
the network was trained (`beatgrid::audio::downmix` does it). `analyze_with_activations`
takes beat and downbeat probabilities from your own network instead. API docs:
`cargo doc -p beatgrid --open`.

### Python

```python
import beatgrid

r = beatgrid.analyze("track.flac")              # profile="edm" for D&B in half time
grid = r["grid"]                                # None when there is no steady beat
if grid:
    print(grid["bpm"], grid["first_downbeat"])  # 127.0 1.889...
    bars = [t for t, down in zip(grid["beats"], grid["is_downbeat"]) if down]
print(r["review"])                              # [] when nothing looked doubtful

results = beatgrid.analyze_many(paths, profile="edm")  # in parallel, in the order given

r = beatgrid.analyze("edit.mp3", tempo="auto")  # segments where the tempo clearly changes
for s in r["grid"]["segments"]:                 # [{"start": 0.022, "bpm": 124.0, ...}, ...]
    print(s["start"], s["bpm"], s["bar"])       # "beats" and "is_downbeat" follow them
```

## What you get

| field | meaning |
|---|---|
| `bpm` | one constant tempo: a whole or half BPM (`127.0`) when the audio agrees, otherwise the measured value (`126.48688…`) |
| `downbeat` (Rust) | the time of a downbeat; it can be any downbeat of the grid, even one before the track starts |
| `first_beat` | the first beat at or after 0 s |
| `first_downbeat` | the first downbeat at or after 0 s: beat 1 of bar 1 |
| `period()` (Rust) | seconds per beat, `60 / bpm` |
| `beats(start, end)` | every beat in `[start, end)`, each with `time`, `index` (0 at the first downbeat, negative before it), `bar` (bar 1 starts at the first downbeat; a pickup before it is bar 0) and `beat_in_bar` (1–4, 1 = downbeat). In Python, `beats` holds the times and `is_downbeat` flags the ones |
| `beat_number_at(t)`, `time_of_beat(n)` (Rust) | convert between a time and a (fractional) beat number |
| `segments` | the grid to use, as constant-tempo segments (`SegmentedGrid` in Rust): several where the track changes tempo (`--tempo auto` or `variable`), otherwise one with the same beats as the constant grid. Each segment has `start` (the time of its first beat), `bpm`, and `beat_in_bar` (where that beat falls in its bar; beatgrid starts every segment on a downbeat); the JSON and Python also give the `beat` and `bar` it starts on. `beats`, `beat_number_at`, `time_of_beat`, `first_beat` and `first_downbeat` work the same on it, plus `bpm_at(t)` and `segment_at(t)` |
| `duration` | length of the decoded audio in seconds |
| `review` | reasons to check the grid by ear before trusting it; empty when nothing looked doubtful |

Times are seconds on the decoded timeline, with encoder delay and padding trimmed the way
ffmpeg trims them, which is the timeline DJ software stores grids on. A grid is 4/4 and
constant: one tempo and one phase, from which every beat follows. A segmented grid is several
of those one after another: segment *i* holds the beats from its `start` up to the last one
at least half a beat before the next segment starts, the first segment also runs back before
its start and the last one on to the end, and beat numbers and bars count on across the joins
(a bar cut short by an edit just ends early). `bpm`, `first_beat` and `first_downbeat` at the
top level of the CLI's and Python's output describe the best single-tempo grid (`grid`), which
is always there too.

Review reasons:

| reason | what it means |
|---|---|
| `no clear attack to place the grid on` | the audio's attacks do not line up into one sharp peak per beat, so the tempo comes from the network's 20 ms frames instead of the audio; or, under the final grid, fewer than 3 of the track's 8 stretches show a clear attack. Placement is less certain |
| `weak or changing pulse` | the network's beats fit a steady pulse at the final tempo poorly |
| `tempo may change` | some stretch of the track drifts more than 40 ms from the single tempo: a tempo change, a transition edit, or live playing. Try `--tempo auto` or `variable` |
| `tempo changes: check the grid where it changes` | the grid is in segments (`--tempo auto` or `variable`); the joins are where it is most likely to be off |
| `bar 1 unclear: the downbeat cues disagree` | the network's downbeats and the track's arrangement changes vote for different beats as the one |
| `octave chosen by the profile: the audio's pulse is twice as fast` (or `half as fast`) | the profile, not the audio, set the octave (drum & bass in half time, a slow track doubled under `edm`); DJs count these both ways |
| `no steady beat to grid` | no grid at all (silence, beatless audio) |

Against the grids DJs and experts made (DJ-corrected, ear-approved, machine-fitted and
Raveform), these reasons are given for 48% of the grids that are not usable@10 and for 23% of
the ones that are. Treat them as a listening queue, not a verdict: most of the disagreements
they miss are on the reference's side (a DJ grid that drifts against its own audio, a
download offset), and the signals tried for the rest (a second attack near the first, a grid
that walks off the attacks, a narrow bar-1 margin) flagged usable grids at least as often as
wrong ones. Those measurements are in the JSON output's `diagnostics` (`attack_second`,
`stretch_drift_ms`, `snap_drift_ms`, `bar_margin`); [docs/results.md](docs/results.md#review-reasons-2026-10-09-002)
has the numbers by failure kind.

## Accuracy

The headline metric is **usable@10**, scored per track against a reference grid over the
part of the track with music in it: right tempo octave, at least 90% of beats within 10 ms of
the reference, and bar 1 on the reference's bar 1. A track that passes needs no editing.
usable@20 is the same at 20 ms.

| reference | tracks | usable@10 | usable@20 | bar 1 | Engine DJ's own analysis (usable@10) |
|---|---|---|---|---|---|
| grids a DJ corrected in Engine DJ | 66 | 73% (76% with `edm`) | 74% | 93% | 27% |
| Engine DJ's analysis on tracks played unedited | 108 | 71% | 74% | 83% | (is the reference) |
| grids a DJ approved by ear (`edm`) | 26 | 73% | 81% | 96% | 42% |
| machine-fitted grids that agree with Engine DJ (`edm`) | 51 | 84% | 92% | 96% | 76% |
| [Raveform](https://mir-aidj.github.io/raveform/) expert-corrected grids | 218 | 75% | 84% | 96% (tempo 100%) | — |

On the same tracks, the DJ application's own analysis reaches 27–76%. The first four
reference sets come from one DJ's private Engine DJ library and cannot be redistributed;
Raveform's labels are public, so its row can be reproduced. Where each reference comes from,
how far it can be trusted, a whole-library comparison, and what was tried and did not work
are in [docs/results.md](docs/results.md).

**Tempo changes.** On 150 held-out synthetic transition edits (sections of the steady
references joined at a new tempo, change points known exactly), `--tempo auto` is usable@10
on 72% of tracks and `--tempo variable` on 85%, against 15% for one tempo; each segment's
tempo is within 0.01 BPM of the truth for 94-97% of sections. `auto` split none of the 327
one-tempo references in the DJ-corrected, ear-approved, machine-fitted and Raveform rows
(`variable`: 10), whose scores it leaves unchanged; it split 7 of Engine DJ's 108 own grids,
6 of them already wrong at one tempo. Real DJ edits and the details are in
[docs/results.md](docs/results.md#tempo-changes-2026-10-10-010).

## How it works

A DJ grid for produced music is a handful of numbers, so after a small network this is model
fitting rather than beat tracking:

1. **Network.** A 529k-parameter convolutional network gives beat and downbeat probabilities
   at 50 frames a second. It is distilled from Beat This!, its downbeats trained on
   expert-corrected grids, and runs in pure Rust ([docs/model.md](docs/model.md)).
2. **Tempo.** A coarse tempo from the coherence of those probabilities with a pulse train,
   then refined on the audio itself: a sub-millisecond spectral flux, folded over one beat
   period, has its sharpest peak at exactly the right tempo. The tempo snaps to the whole or
   half BPM the track was produced at when the audio agrees.
3. **Beats.** Every beat goes on the peak of that folded attack, plus 2 ms, which is where DJ
   grids sit.
4. **Bar 1.** Pooled votes for which beat is the one: the network's downbeats, the track's
   biggest arrangement changes, and high-frequency bursts.
5. **Octave and review.** The tempo octave comes from the profile, and the reasons a DJ might
   want to listen before trusting the grid are attached.
6. **Tempo changes** (`--tempo auto` or `variable`). The local tempo is measured in 16 s
   windows and decoded jointly across the track; a beat anchored in each window traces beat
   time against beat number, which is one straight line for one tempo. Where the anchors need
   more than one line, each stretch is fitted like a whole track (tempo on the audio and
   snapped, beats on the attack, its own bar 1), each change goes on the new segment's
   downbeat where the attacks agree, and the segments replace the constant grid only where
   they sit clearly better on the attacks.

The Python pipeline in `research/` is the reference implementation. Given the same network
output, the Rust core reproduces it: the same tempo, the same bar 1, beats within a
microsecond, and the same segments.

## Speed

The whole pipeline (decode, network, flux, fit) takes about 0.4–0.5 s per track in batch on
a 2019 MacBook Pro (8-core i9, no GPU); a batch of 64 tracks took 25 s. One file on its own
takes about 1.5 s of wall time, using every core. Beat This! itself needs about 6 s per track
on the same machine. `--tempo auto` costs about 12% more CPU time (472 against 420 CPU
seconds for 64 tracks; the same wall time in batch). Details are in
[docs/performance.md](docs/performance.md).

## Limits

- **Tempo changes are opt-in.** By default a track gets one tempo, and the review reason
  `tempo may change` when it does not hold one. `--tempo auto` gives transition edits and
  mashups a grid in constant-tempo segments and leaves steady tracks alone; it follows clear
  changes between sections, not gradual drift (live drumming, old recordings) or a phase jump
  at the same tempo, which `--tempo variable` also follows at a higher risk of splitting a
  track that does not change.
- **Octave is a convention.** Drum & bass is 87 or 174 BPM and dubstep 70 or 140, depending
  on who you ask. The default keeps what the audio suggests; use `--profile edm`
  (`Profile::Edm`, `profile="edm"`) for drum & bass in half time and 100–200 BPM for the rest.
- **4/4 dance music.** The network was trained and evaluated on electronic dance music, and
  the bar logic assumes four beats to the bar. Swing, rubato, odd meters and most non-dance
  music are untested.
- **No DJ-software writers.** beatgrid outputs numbers. Writing them into a DJ application's
  library (rekordbox XML, Serato, Traktor, Engine DJ) is not implemented here.

## Development

```sh
cargo test --workspace --exclude beatgrid-py
cargo clippy --workspace --exclude beatgrid-py
cd crates/beatgrid-py && maturin develop --release   # the Python bindings, into the active venv
```

`research/` is a Python workspace (uv) holding the reference pipeline, the evaluation harness
`lab`, a listening UI, and the network's training code:

```sh
cd research
uv sync
uv run lab import-engine    # references from an Engine DJ library: DATA/engine/m.db + DATA/audio/engine/<track id>.<ext>
uv run lab import-raveform  # Raveform labels + audio: DATA/raveform/raveform + DATA/audio/raveform/<youtube id>.<ext>
uv run lab cache            # network activations, onset flux (slow, once)
uv run lab audit            # which reference grids hold steady on their own audio
uv run lab eval --split dev
uv run lab splices          # synthetic transition edits from the steady references (no audio written)
uv run python experiments/variable_tempo.py --split dev   # tempo changes: splices, DJ grids, steady sets
uv run --group dev pytest -q
```

`DATA` is `./data` (git-ignored) or `$BEATGRID_DATA`. Tune on `dev`; `test` is held out.

### Listening to grids

```sh
cd research && uv run lab review    # then open http://localhost:8765
```

Pick a reference set and a track. The reference grid is drawn in the top half of the
waveform and beatgrid's in the bottom half, so any misalignment shows as a broken line; by
default the clicks put the reference in your left ear and beatgrid in your right. Verdicts
("which grid is right?") are appended to `DATA/verdicts.jsonl`.

| key | does |
|---|---|
| space | play / pause |
| ← / → | back / forward one beat (shift: one bar) |
| n / p | next / previous beat where the grids disagree |
| c | cycle click source |
| + / − | zoom |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise, any
contribution you submit for inclusion in this project shall be dual licensed as above,
without any additional terms or conditions.

The embedded network weights are released under the same terms; [NOTICE](NOTICE) lists what
they are derived from.

## Credits

- **[Beat This!](https://github.com/CPJKU/beat_this)** (F. Foscarin, J. Schlüter and
  G. Widmer, ISMIR 2024; code and weights MIT). beatgrid's network is distilled from its
  `final0` model, and the research pipeline uses it as a baseline.
- **[Raveform](https://mir-aidj.github.io/raveform/)** (Kim, Kim, Kim and Nam, TISMIR 2026;
  labels CC BY 4.0). Its expert-corrected grids supervised the network's downbeats and are
  one of the evaluation references.
- [symphonia](https://github.com/pdeljanov/Symphonia) (decoding),
  [realfft](https://github.com/HEnquist/realfft) (FFT),
  [rubato](https://github.com/HEnquist/rubato) (resampling) and
  [matrixmultiply](https://github.com/bluss/matrixmultiply) (the network's matrix products).
