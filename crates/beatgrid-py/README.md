# beatgrid (Python)

DJ-grade beat grids: the exact tempo (snapped to the whole or half BPM a track was produced
at, when the audio agrees), every beat on the attack within a few milliseconds of where DJs
put grid lines, and bar 1 on the one. These are Python bindings to the Rust library: no
ML runtime, about half a second per track in batch on a laptop CPU.

## Install

Not on PyPI yet. Building from source needs a Rust toolchain (1.87 or newer):

```sh
pip install "git+https://github.com/tha23rd/beatgrid#subdirectory=crates/beatgrid-py"
```

The wheel is abi3, so one build works on CPython 3.10 and newer.

## Use

```python
import beatgrid

r = beatgrid.analyze("track.flac")              # or profile="edm"
grid = r["grid"]                                # None when there is no steady beat
if grid:
    print(grid["bpm"], grid["first_downbeat"])  # 127.0 1.889...
    bars = [t for t, down in zip(grid["beats"], grid["is_downbeat"]) if down]
print(r["review"])                              # [] when nothing looked doubtful

results = beatgrid.analyze_many(["a.mp3", "b.flac"], profile="edm")
```

`analyze(path, profile="none")` returns a dict:

| key | value |
|---|---|
| `grid` | `None` when there is no steady beat, otherwise a dict: |
| `grid["bpm"]` | one constant tempo; a whole or half BPM when the audio agrees |
| `grid["first_beat"]` | time (s) of the first beat at or after 0 s |
| `grid["first_downbeat"]` | time (s) of the first downbeat at or after 0 s: bar 1 |
| `grid["beats"]` | every beat time (s) from 0 to the end of the track |
| `grid["is_downbeat"]` | for each beat, whether it is beat 1 of a bar |
| `duration` | length of the decoded audio (s) |
| `review` | reasons to check the grid by ear (e.g. `"tempo may change"`, `"bar 1 unclear: the downbeat cues disagree"`); empty when nothing looked doubtful |
| `snapped`, `tempo_source` | whether the tempo was snapped to a whole/half BPM, and whether it came from the audio (`"flux"`) or the network (`"activation"`); present when there is a grid |

Profiles: `"none"` (or `"default"`) keeps the tempo octave the audio suggests; `"edm"` counts
drum & bass in half time and folds everything else into 100–200 BPM.

`analyze_many(paths, profile="none")` grids files in parallel (the GIL is released) and
returns a list in the same order; a file that fails gives `{"error": message}` instead of
raising.

Times are seconds on the decoded timeline with encoder delay trimmed the way ffmpeg trims
it. Formats: MP3, FLAC, WAV, Ogg Vorbis, AAC/ALAC (MP4).

More, including accuracy figures and limits: https://github.com/tha23rd/beatgrid

## License

MIT or Apache-2.0, at your option. The embedded network is distilled from Beat This! (MIT)
and trained in part on Raveform's labels (CC BY 4.0); see
[NOTICE](https://github.com/tha23rd/beatgrid/blob/main/NOTICE).
