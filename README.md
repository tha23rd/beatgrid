# beatgrid

DJ-grade beat grids for any audio file: the exact tempo (snapped to the whole or half BPM a
track was produced at, when the audio agrees), every beat on the attack within a few
milliseconds of where DJs put grid lines, and bar 1 on the one. Research beat trackers count
a beat as right within ±70 ms; a grid you can mix on has to hold to about 10 ms for the
whole track, and that is what beatgrid is built and measured for. It is a Rust library with
a CLI and Python bindings, runs in about half a second per track on a laptop CPU with no ML
runtime, and returns plain numbers (a tempo, a downbeat, every beat), so it drops into any
app and is not tied to any DJ software's library format.

> **Status:** pre-release (0.0.2; changes in [CHANGELOG.md](CHANGELOG.md)). Not yet published
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
file,bpm,first_beat,first_downbeat,duration,review,error
track.flac,127,0.472118,1.889441,355.778,,
other.mp3,124,0.288236,1.739849,170.031,tempo may change,

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
- Reads MP3, FLAC, WAV, Ogg Vorbis, and AAC/ALAC in MP4.

### Rust

```rust,no_run
use beatgrid::{Analyzer, Options, Profile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the network once; an Analyzer is Send + Sync and cheap to clone.
    let analyzer = Analyzer::with_options(Options { profile: Profile::Edm })?;
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
    Ok(())
}
```

`Analyzer::new()` uses the default profile. If you already have decoded audio, call
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
| `duration` | length of the decoded audio in seconds |
| `review` | reasons to check the grid by ear before trusting it; empty when nothing looked doubtful |

Times are seconds on the decoded timeline, with encoder delay and padding trimmed the way
ffmpeg trims them, which is the timeline DJ software stores grids on. A grid is 4/4 and
constant: one tempo and one phase, from which every beat follows.

Review reasons:

| reason | what it means |
|---|---|
| `no clear attack to place the grid on` | the audio's attacks do not line up into one sharp peak per beat, so the tempo comes from the network's 20 ms frames instead of the audio; or, under the final grid, fewer than 3 of the track's 8 stretches show a clear attack. Placement is less certain |
| `weak or changing pulse` | the network's beats fit a steady pulse at the final tempo poorly |
| `tempo may change` | some stretch of the track drifts more than 40 ms from the single tempo: a tempo change, a transition edit, or live playing |
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

The Python pipeline in `research/` is the reference implementation. Given the same network
output, the Rust core reproduces it: the same tempo, the same bar 1, beats within a
microsecond.

## Speed

The whole pipeline (decode, network, flux, fit) takes about 0.4–0.5 s per track in batch on
a 2019 MacBook Pro (8-core i9, no GPU); a batch of 64 tracks took 25 s. One file on its own
takes about 1.5 s of wall time, using every core. Beat This! itself needs about 6 s per track
on the same machine. Details are in [docs/performance.md](docs/performance.md).

## Limits

- **One tempo per track.** Tracks whose tempo changes (transition edits, mashups, live
  drumming) get one tempo and the review reason `tempo may change`.
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
