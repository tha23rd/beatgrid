//! DJ-grade beat grids: the exact tempo, every beat on the attack, bar 1 on the one.
//!
//! beatgrid finds the grid a DJ would otherwise fix by hand: one constant tempo (snapped to
//! the whole or half BPM the track was produced at when the audio agrees), beats placed
//! within a few milliseconds of the attack, and bar 1 on the one. A small neural network
//! (built in; no files to ship) supplies beat and downbeat probabilities, and the grid is
//! fitted to them and to the audio itself.
//!
//! ```no_run
//! use beatgrid::Analyzer;
//!
//! // Load the network once and share the analyzer (it is Send + Sync and cheap to clone).
//! let analyzer = Analyzer::new()?;
//! let analysis = analyzer.analyze_file("track.mp3")?;
//! if let Some(grid) = &analysis.grid {
//!     println!("{:.3} BPM, bar 1 at {:.3} s", grid.bpm, grid.first_downbeat());
//!     for beat in grid.beats(0.0, analysis.duration) {
//!         println!("{:.3} s  bar {} beat {}", beat.time, beat.bar, beat.beat_in_bar);
//!     }
//! }
//! if analysis.needs_review() {
//!     println!("worth checking by ear: {}", analysis.review.join("; "));
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! - [`Analyzer::analyze_file`] decodes MP3, FLAC, WAV, Ogg Vorbis and AAC/ALAC (MP4) itself;
//!   [`Analyzer::analyze_samples`] takes mono samples from your own decoder.
//! - [`Options`] picks an octave convention ([`Profile`]); [`Params`] exposes every tuning
//!   knob for experiments.
//! - Times are seconds on the decoded timeline with encoder delay and padding trimmed the
//!   way ffmpeg trims them, which is the timeline DJ software stores grids on.
//!
//! The modules [`audio`] (decoding) and [`network`] (the activation network) are the
//! building blocks underneath, for callers who need them.

pub mod audio;
#[doc(hidden)]
pub mod bars;
#[doc(hidden)]
pub mod dsp;
#[doc(hidden)]
pub mod fit;
mod grid;
pub mod network;

use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};

pub use fit::{Diagnostics, Params};
pub use grid::{Beat, Grid};

/// Compiles the README's example with the doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// The review reason given when a track has no grid.
const NO_GRID: &str = "no steady beat to grid";

/// An octave convention: whether a track is counted at, say, 87 or 174 BPM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Keep the tempo octave the audio suggests.
    #[default]
    Default,
    /// The usual EDM convention: drum & bass (165-185 BPM) in half time, everything else
    /// folded into 100-200 BPM.
    Edm,
}

impl Profile {
    /// The name [`FromStr`] accepts: "default" or "edm".
    pub fn name(self) -> &'static str {
        match self {
            Profile::Default => "default",
            Profile::Edm => "edm",
        }
    }

    /// The fit parameters for this profile.
    pub fn params(self) -> Params {
        match self {
            Profile::Default => Params::default(),
            Profile::Edm => Params::default().edm_octaves(),
        }
    }
}

impl FromStr for Profile {
    type Err = anyhow::Error;

    /// "default" (or "none") and "edm", ignoring case.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "default" | "none" => Ok(Profile::Default),
            "edm" => Ok(Profile::Edm),
            _ => bail!("unknown profile {s:?} (use \"default\" or \"edm\")"),
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How an [`Analyzer`] grids tracks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// The octave convention.
    pub profile: Profile,
}

impl From<Profile> for Options {
    fn from(profile: Profile) -> Self {
        Options { profile }
    }
}

/// What beatgrid found in a track.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Analysis {
    /// The grid; `None` when the track has no steady beat to grid.
    pub grid: Option<Grid>,
    /// Length of the decoded audio in seconds.
    pub duration: f64,
    /// Reasons a person should check the grid by ear before trusting it (a tempo that may
    /// change, no clear attack, an unclear bar 1, no grid at all). Empty when nothing looked
    /// doubtful.
    pub review: Vec<String>,
    /// How the grid was found, for debugging; `None` when there is no grid.
    pub diagnostics: Option<Diagnostics>,
}

impl Analysis {
    /// Whether anything about the grid looked doubtful (`review` is not empty).
    pub fn needs_review(&self) -> bool {
        !self.review.is_empty()
    }
}

/// Grids tracks. Holds the network, loaded once; share one analyzer across threads (it is
/// `Send + Sync`) or clone it (clones share the network).
#[derive(Clone)]
pub struct Analyzer {
    network: Arc<network::Student>,
    params: Params,
}

impl Analyzer {
    /// The built-in network with default [`Options`].
    pub fn new() -> Result<Self> {
        Self::with_options(Options::default())
    }

    /// The built-in network with `options`.
    pub fn with_options(options: Options) -> Result<Self> {
        Self::with_params(options.profile.params())
    }

    /// The built-in network with hand-tuned fit parameters.
    pub fn with_params(params: Params) -> Result<Self> {
        Ok(Analyzer { network: Arc::new(network::Student::embedded()?), params })
    }

    /// Use `options` from now on (replacing any parameters set with
    /// [`set_params`](Self::set_params)).
    pub fn set_options(&mut self, options: Options) {
        self.params = options.profile.params();
    }

    /// Use hand-tuned fit parameters from now on.
    pub fn set_params(&mut self, params: Params) {
        self.params = params;
    }

    /// The fit parameters in use.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Decode and grid an audio file.
    pub fn analyze_file(&self, path: impl AsRef<Path>) -> Result<Analysis> {
        let audio = audio::decode(path.as_ref())?;
        self.analyze_samples(&audio.samples, audio.rate)
    }

    /// Grid mono samples at `sample_rate` Hz, nominally in [-1, 1].
    ///
    /// The network was trained on ffmpeg's mono downmix, so mix stereo to mono the same way:
    /// (L + R) / sqrt 2, not the average ([`audio::downmix`] does it). Times in the result
    /// count from the first sample, so trim encoder delay first if your decoder keeps it.
    pub fn analyze_samples(&self, samples: &[f32], sample_rate: u32) -> Result<Analysis> {
        ensure!(sample_rate > 0, "the sample rate must be positive");
        // The network and the audio features are independent: compute them side by side.
        let (activations, features) =
            rayon::join(|| self.activations(samples, sample_rate), || Features::of(samples, sample_rate));
        let (beat, downbeat) = activations?;
        Ok(self.fit(&features, &beat, &downbeat, network::FPS))
    }

    /// The network's beat and downbeat probabilities for mono samples, one per frame at
    /// [`network::FPS`] frames a second.
    pub fn activations(&self, samples: &[f32], sample_rate: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(sample_rate > 0, "the sample rate must be positive");
        let y = network::resample(samples, sample_rate)?;
        Ok(self.network.run(&network::log_mel(&y)))
    }

    /// Grid mono samples given beat and downbeat probabilities from elsewhere (another
    /// network, or a cache): one value per frame at `fps` frames a second, frame 0 at the
    /// first sample. The analyzer's own network is not run.
    pub fn analyze_with_activations(
        &self, samples: &[f32], sample_rate: u32, beat: &[f32], downbeat: &[f32], fps: f64,
    ) -> Result<Analysis> {
        ensure!(sample_rate > 0, "the sample rate must be positive");
        ensure!(fps.is_finite() && fps > 0.0, "the activation frame rate must be positive");
        ensure!(
            beat.len() == downbeat.len(),
            "{} beat activations but {} downbeat ones",
            beat.len(),
            downbeat.len()
        );
        Ok(self.fit(&Features::of(samples, sample_rate), beat, downbeat, fps))
    }

    fn fit(&self, features: &Features, beat: &[f32], downbeat: &[f32], fps: f64) -> Analysis {
        let act = fit::Activations { beat, downbeat, fps };
        let duration = features.duration;
        match fit::fit(&act, &features.flux, &features.bands, &self.params) {
            Some(f) => Analysis {
                grid: Some(f.grid),
                duration,
                review: f.review.iter().map(|r| r.to_string()).collect(),
                diagnostics: Some(f.diagnostics),
            },
            None => Analysis { grid: None, duration, review: vec![NO_GRID.to_string()], diagnostics: None },
        }
    }
}

/// What the fit measures on the audio itself.
struct Features {
    flux: dsp::Flux,
    bands: dsp::Bands,
    duration: f64,
}

impl Features {
    fn of(samples: &[f32], sample_rate: u32) -> Self {
        Features {
            flux: dsp::onset_flux(samples, sample_rate),
            bands: dsp::band_energy(samples, sample_rate),
            duration: samples.len() as f64 / sample_rate as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzer_is_shareable_across_threads() {
        fn shareable<T: Send + Sync + Clone>() {}
        shareable::<Analyzer>();
    }

    #[test]
    fn profiles_parse_by_name() {
        assert_eq!("none".parse::<Profile>().unwrap(), Profile::Default);
        assert_eq!("default".parse::<Profile>().unwrap(), Profile::Default);
        assert_eq!("EDM".parse::<Profile>().unwrap(), Profile::Edm);
        assert!("techno".parse::<Profile>().is_err());
        assert_eq!(Params::default().with_profile("edm").unwrap().octave_range, Profile::Edm.params().octave_range);
        assert!(Params::default().with_profile("techno").is_none());
    }

    #[test]
    fn silence_and_empty_input_have_no_grid() {
        let analyzer = Analyzer::new().unwrap();
        for samples in [Vec::new(), vec![0.0f32; 1000], vec![0.0f32; 44100 * 5]] {
            let a = analyzer.analyze_samples(&samples, 44100).unwrap();
            assert!(a.grid.is_none());
            assert!(a.needs_review());
        }
        assert!(analyzer.analyze_samples(&[0.0; 10], 0).is_err());
    }

    /// A drum machine: a kick on every beat, a snare on 2 and 4, a crash opening every
    /// 4-bar phrase, and hats joining after 8 bars.
    fn drums(bpm: f64, seconds: f64, rate: u32) -> Vec<f32> {
        let period = 60.0 / bpm;
        let mut noise = 12345u32;
        (0..(seconds * rate as f64) as usize)
            .map(|i| {
                let t = i as f64 / rate as f64;
                let beat = (t / period).floor() as i64;
                let since = t - beat as f64 * period;
                noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let white = noise as f64 / u32::MAX as f64 - 0.5;
                let kick = 0.8 * (-since * 30.0).exp() * (2.0 * std::f64::consts::PI * 55.0 * since).sin();
                let snare = if beat % 2 == 1 { 0.5 * white * (-since * 25.0).exp() } else { 0.0 };
                let off = since - period / 2.0;
                let hat = if beat >= 32 && off >= 0.0 { 0.2 * white * (-off * 40.0).exp() } else { 0.0 };
                let crash = if beat % 16 == 0 { 0.6 * white * (-since * 2.0).exp() } else { 0.0 };
                (kick + snare + hat + crash) as f32
            })
            .collect()
    }

    #[test]
    fn a_drum_machine_gets_its_tempo_beats_and_bars() {
        let analyzer = Analyzer::new().unwrap();
        let rate = 44100;
        let a = analyzer.analyze_samples(&drums(128.0, 60.0, rate), rate).unwrap();
        let grid = a.grid.expect("a grid");
        assert!((grid.bpm - 128.0).abs() < 1e-6, "bpm {}", grid.bpm);
        let near_zero = |t: f64, period: f64| t.min(period - t) * 1000.0;
        let beat_ms = near_zero(grid.first_beat(), grid.period());
        assert!(beat_ms < 10.0, "beats {beat_ms:.1} ms off the kicks");
        let bar_ms = near_zero(grid.first_downbeat(), 4.0 * grid.period());
        assert!(bar_ms < 10.0, "bar 1 is {bar_ms:.0} ms off the crash");
    }
}
