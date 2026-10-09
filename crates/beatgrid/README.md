# beatgrid

DJ-grade beat grids: the exact tempo (snapped to the whole or half BPM a track was produced
at, when the audio agrees), every beat on the attack within a few milliseconds of where DJs
put grid lines, and bar 1 on the one.

This crate is the library. It decodes MP3, FLAC, WAV, Ogg Vorbis and AAC/ALAC (MP4) itself,
or takes mono samples from your own decoder, and runs a small built-in network (2.1 MB of
weights compiled in, no ML runtime) followed by a fit on the audio. It takes about half a
second per track in batch on a laptop CPU. The result is plain numbers (a tempo, a
downbeat, every beat), not tied to any DJ software's format.

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

`analysis.review` lists reasons to check a grid by ear (a tempo that may change, no clear
attack, an unclear bar 1); it is empty when nothing looked doubtful.

The CLI, Python bindings, accuracy figures, how it works and its limits are in the
[repository README](https://github.com/tha23rd/beatgrid#readme).

## License

MIT or Apache-2.0, at your option. The embedded network is distilled from Beat This! (MIT)
and trained in part on Raveform's labels (CC BY 4.0); see
[NOTICE](https://github.com/tha23rd/beatgrid/blob/main/NOTICE).
