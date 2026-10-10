//! From activations and the onset flux to a grid: a port of research/beatgrid_lab/pipeline.py.
//!
//! 1. coarse tempo from the network's beat peaks
//! 2. tempo and phase from the coherence of the beat activation with a pulse train
//! 3. tempo refined on the audio (the flux folds sharpest at the right tempo), then snapped to
//!    an exact whole or half BPM when the drift is inaudible or the audio agrees
//! 4. the grid placed on the folded flux peak, plus the DJ-grid offset
//! 5. bar 1 from pooled votes (bars.rs); octave by profile

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::bars;
use crate::dsp::{Bands, Flux};
use crate::segments::{self, SegmentSearch};
use crate::{Grid, SegmentedGrid, TempoMode};

/// The network's beat and downbeat probabilities, one per frame at `fps` frames a second.
#[derive(Clone, Copy)]
pub struct Activations<'a> {
    pub beat: &'a [f32],
    pub downbeat: &'a [f32],
    pub fps: f64,
}

impl Activations<'_> {
    pub fn time(&self, i: usize) -> f64 {
        i as f64 / self.fps
    }
    pub fn last_time(&self) -> f64 {
        self.time(self.beat.len().saturating_sub(1))
    }
}

/// Tuning for the fit. The defaults are what the research pipeline measured best; most
/// callers only choose an octave convention ([`crate::Profile`]).
#[derive(Clone, Debug)]
pub struct Params {
    /// Tempo range (BPM) the coarse estimate may fall in, with an octave of slack each way.
    pub min_bpm: f64,
    /// See `min_bpm`.
    pub max_bpm: f64,
    /// Tempo search around the coarse estimate, as a fraction of it.
    pub search_span: f64,
    /// Snap to an exact whole or half BPM (the tempo a track was produced at) when the
    /// evidence allows.
    pub snap: bool,
    /// Coherence an exact tempo may give up against the best free tempo and still be taken
    /// (fraction), when the tempo comes from the network.
    pub snap_tolerance: f64,
    /// Furthest (BPM) an exact tempo may be from the measured one to be snapped to.
    pub snap_max_bpm: f64,
    /// The exact tempo steps to try, in order (1 and 0.5 BPM).
    pub snap_steps: Vec<f64>,
    /// Stretches of this many beats are checked against the single tempo (tempo changes).
    pub bend_window: usize,
    /// Tempo refinement on the audio searches ±this many BPM around the network's tempo.
    pub flux_span: f64,
    /// Step (BPM) of that search.
    pub flux_step: f64,
    /// Fold height an exact tempo may give up against the best free tempo and still be
    /// taken (fraction), when the tempo comes from the audio.
    pub snap_flux_tolerance: f64,
    /// Below this much drift (ms, the middle of the track against either end) two tempos are
    /// indistinguishable by ear, so the exact tempo is taken without asking the audio.
    pub snap_free_drift_ms: f64,
    /// The folded onset flux needs a peak this many times its median to refine the tempo;
    /// below it the network's tempo is used.
    pub flux_min_prominence: f64,
    /// Where on the folded attack a DJ's grid line goes (ms after its peak).
    pub offset_ms: f64,
    /// How far (ms) either side of the network's phase to look for the attack.
    pub attack_search_ms: f64,
    /// Weight of the network's downbeats in the bar-1 vote.
    pub bar_w_network: f64,
    /// Weight of arrangement changes (loudness and brightness steps) in the bar-1 vote.
    pub bar_w_changes: f64,
    /// Weight of high-frequency bursts (crashes, fills) in the bar-1 vote.
    pub bar_w_fills: f64,
    /// Octave convention: fold the tempo into `[lo, hi)` BPM. `None` keeps the octave the
    /// audio suggests.
    pub octave_range: Option<(f64, f64)>,
    /// A track with an octave inside `[lo, hi)` (drum & bass at 165-185, say) is counted at
    /// half of it, as many DJs do. Checked before `octave_range`.
    pub octave_half_range: Option<(f64, f64)>,
    /// Whether the grid may change tempo (segments.rs). The `seg_*` knobs tune the search.
    pub tempo: TempoMode,
    /// Windows (s) the local tempo is measured in.
    pub seg_window_s: f64,
    /// Hop (s) between those windows.
    pub seg_hop_s: f64,
    /// Step between tempo states (fraction).
    pub seg_step: f64,
    /// Cost (log units) of moving one tempo state between windows.
    pub seg_drift_cost: f64,
    /// Cost of jumping further.
    pub seg_switch_cost: f64,
    /// Windows whose coherence peaks below this carry no evidence.
    pub seg_quiet: f64,
    /// An anchor needs this much coherence at its window's tempo.
    pub seg_anchor_min: f64,
    /// Anchor noise (ms) for the line costs.
    pub seg_sigma_ms: f64,
    /// Anchors further off than this many sigmas cost no more.
    pub seg_clip: f64,
    /// A segment spans at least this many anchors.
    pub seg_min_anchors: usize,
    /// Tempo search (fraction) around a segment's line.
    pub seg_search_span: f64,
    /// A change is placed within this many seconds of where the lines meet.
    pub seg_join_s: f64,
    /// What a segmentation must show to be taken ([`Params::with_tempo`] sets these per mode).
    /// Cost of an extra line through the anchors.
    pub seg_penalty: f64,
    /// A join with a smaller tempo change (fraction) is no change...
    pub seg_min_tempo_change: f64,
    /// ... unless a same-tempo phase jump of at least this many beats (0: never).
    pub seg_min_jump: f64,
    /// Every segment holds at least this many bars.
    pub seg_min_bars: f64,
    /// Beats where the segmented grid must be more than 10 ms from the constant one.
    pub seg_min_apart: usize,
    /// How much stronger the attacks on those beats must be under the segmented grid...
    pub seg_margin: f64,
    /// ... and how strong, over the track's median flux.
    pub seg_min_evidence: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            min_bpm: 60.0,
            max_bpm: 200.0,
            search_span: 0.03,
            snap: true,
            snap_tolerance: 0.02,
            snap_max_bpm: 0.3,
            snap_steps: vec![1.0, 0.5],
            bend_window: 32,
            flux_span: 0.2,
            flux_step: 0.002,
            snap_flux_tolerance: 0.1,
            snap_free_drift_ms: 3.0,
            flux_min_prominence: 1.5,
            offset_ms: 2.0,
            attack_search_ms: 50.0,
            bar_w_network: 2.0,
            bar_w_changes: 1.0,
            bar_w_fills: 1.0,
            octave_range: None,
            octave_half_range: None,
            tempo: TempoMode::Constant,
            seg_window_s: 16.0,
            seg_hop_s: 4.0,
            seg_step: 0.005,
            seg_drift_cost: 0.5,
            seg_switch_cost: 8.0,
            seg_quiet: 0.3,
            seg_anchor_min: 0.3,
            seg_sigma_ms: 4.0,
            seg_clip: 3.0,
            seg_min_anchors: 4,
            seg_search_span: 0.01,
            seg_join_s: 8.0,
            seg_penalty: 24.0,
            seg_min_tempo_change: 0.001,
            seg_min_jump: 0.0,
            seg_min_bars: 16.0,
            seg_min_apart: 64,
            seg_margin: 0.25,
            seg_min_evidence: 2.5,
        }
    }
}

impl Params {
    /// Named octave conventions, as [`crate::Profile`] parses them: "none" (or "default")
    /// keeps these params as they are, "edm" applies [`edm_octaves`](Self::edm_octaves).
    pub fn with_profile(self, profile: &str) -> Option<Self> {
        match profile.parse().ok()? {
            crate::Profile::Default => Some(self),
            crate::Profile::Edm => Some(self.edm_octaves()),
        }
    }

    /// The "edm" octave convention: drum & bass (165-185 BPM) in half time, everything else
    /// folded into 100-200 BPM.
    pub fn edm_octaves(mut self) -> Self {
        self.octave_range = Some((100.0, 200.0));
        self.octave_half_range = Some((165.0, 185.0));
        self
    }

    /// A tempo mode and the segmentation knobs it uses: [`TempoMode::Variable`] looks harder
    /// than [`TempoMode::Auto`] (the defaults), and also counts same-tempo phase jumps.
    pub fn with_tempo(mut self, mode: TempoMode) -> Self {
        self.tempo = mode;
        if mode == TempoMode::Variable {
            self.seg_penalty = 12.0;
            self.seg_min_tempo_change = 0.0005;
            self.seg_min_jump = 0.1;
            self.seg_min_bars = 4.0;
            self.seg_min_apart = 16;
            self.seg_margin = 0.05;
            self.seg_min_evidence = 1.5;
        }
        self
    }

    /// Set a numeric knob by name (research and tuning: `beatgrid analyze --param key=value`).
    #[doc(hidden)]
    pub fn set(&mut self, key: &str, value: &str) -> anyhow::Result<()> {
        let f = || value.parse::<f64>().map_err(|e| anyhow::anyhow!("{key}={value}: {e}"));
        match key {
            "seg_window_s" => self.seg_window_s = f()?,
            "seg_hop_s" => self.seg_hop_s = f()?,
            "seg_step" => self.seg_step = f()?,
            "seg_drift_cost" => self.seg_drift_cost = f()?,
            "seg_switch_cost" => self.seg_switch_cost = f()?,
            "seg_quiet" => self.seg_quiet = f()?,
            "seg_anchor_min" => self.seg_anchor_min = f()?,
            "seg_sigma_ms" => self.seg_sigma_ms = f()?,
            "seg_clip" => self.seg_clip = f()?,
            "seg_min_anchors" => self.seg_min_anchors = f()? as usize,
            "seg_search_span" => self.seg_search_span = f()?,
            "seg_join_s" => self.seg_join_s = f()?,
            "seg_penalty" => self.seg_penalty = f()?,
            "seg_min_tempo_change" => self.seg_min_tempo_change = f()?,
            "seg_min_jump" => self.seg_min_jump = f()?,
            "seg_min_bars" => self.seg_min_bars = f()?,
            "seg_min_apart" => self.seg_min_apart = f()? as usize,
            "seg_margin" => self.seg_margin = f()?,
            "seg_min_evidence" => self.seg_min_evidence = f()?,
            _ => anyhow::bail!("unknown or non-numeric parameter {key:?}"),
        }
        Ok(())
    }
}

/// How the fit went: the intermediate estimates and the evidence behind them. For
/// debugging and research; the fields may change between versions.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Diagnostics {
    /// Tempo from the spacing of the network's beat peaks.
    pub bpm_coarse: f64,
    /// Tempo where the network's beat activation is most coherent with a pulse train.
    pub bpm_activation: f64,
    /// That coherence (0-1).
    pub coherence: f64,
    /// Peak of the onset flux folded at the tempo, over its median.
    pub flux_prominence: Option<f64>,
    /// Tempo refined on the audio, when the flux was prominent enough to use.
    pub bpm_flux: Option<f64>,
    /// Final tempo before the octave convention.
    pub bpm: f64,
    /// Whether the tempo was snapped to an exact whole or half BPM.
    pub snapped: bool,
    /// Where the tempo came from: "flux" (the audio) or "activation" (the network).
    pub tempo_source: &'static str,
    /// Coherence of the network's beats with the final tempo.
    pub coherence_final: f64,
    /// Largest deviation (ms) of a stretch of the track from the single tempo.
    pub bend_ms: f64,
    /// How far (ms) the grid moved from the network's phase onto the attack.
    pub attack_shift_ms: f64,
    /// Peak of the folded attack over its median.
    pub attack_prominence: f64,
    /// Height of the runner-up peak of the folded attack, as a fraction of the main one (0:
    /// a single attack). A second attack a few ms off the first is one a DJ might prefer.
    pub attack_second: f64,
    /// How far (ms) the attack walks under the grid from one end of the track to the other,
    /// measured in 8 stretches; `None` when fewer than 3 stretches have a clear attack.
    pub stretch_drift_ms: Option<f64>,
    /// How many of those 8 stretches have a clear attack under the grid.
    pub clear_stretches: usize,
    /// How far (ms) the exact whole or half BPM walks from the tempo measured on the audio,
    /// end to end (0 when the tempo was not snapped on the audio).
    pub snap_drift_ms: f64,
    /// Pooled log score of bar 1 over the runner-up.
    pub bar_margin: f64,
    /// Bar-1 votes by beat position, [network, arrangement changes], each summing to 1.
    pub bar_votes: [[f64; 4]; 2],
    /// The search for tempo changes; `None` with [`TempoMode::Constant`].
    pub segment_search: Option<SegmentSearch>,
}

pub(crate) fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// numpy.arange(start, stop, step)
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Python's round(): halves to even.
pub(crate) fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

pub fn coarse_bpm(act: &Activations, p: &Params) -> Option<f64> {
    let b = act.beat;
    let n = b.len();
    if n == 0 {
        return None;
    }
    // np.roll wraps around: a circular neighbourhood of ±3 frames.
    let frames: Vec<usize> = (0..n)
        .filter(|&i| {
            b[i] > 0.5 && (-3i64..=3).all(|k| b[i] >= b[((i as i64 + k).rem_euclid(n as i64)) as usize])
        })
        .collect();
    if frames.len() < 8 {
        return None;
    }
    let ioi: Vec<f64> = frames.windows(2).map(|w| (w[1] - w[0]) as f64 / act.fps).collect();
    let typical = median(&mut ioi.clone());
    let regular: Vec<f64> = ioi.iter().copied().filter(|x| (x / typical - 1.0).abs() < 0.15).collect();
    let mean = if regular.is_empty() { typical } else { regular.iter().sum::<f64>() / regular.len() as f64 };
    let bpm = 60.0 / mean;
    (p.min_bpm / 2.0 <= bpm && bpm <= p.max_bpm * 2.0).then_some(bpm)
}

/// sum a(t) e^{-2 pi i t / P} / sum a(t), over the frames in `range` (all if None).
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn abs(&self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn angle(&self) -> f64 {
        self.im.atan2(self.re)
    }
}

pub fn coherence_at(act: &Activations, bpm: f64, range: Option<(f64, f64)>) -> Complex {
    let period = 60.0 / bpm;
    let (mut re, mut im, mut total) = (0.0f64, 0.0f64, 0.0f64);
    for (i, &a) in act.beat.iter().enumerate() {
        let a = a as f64;
        if a <= 1e-3 {
            continue;
        }
        let t = act.time(i);
        if let Some((lo, hi)) = range {
            if t < lo || t >= hi {
                continue;
            }
        }
        let phase = -2.0 * PI * t / period;
        re += a * phase.cos();
        im += a * phase.sin();
        total += a;
    }
    if total <= 0.0 {
        return Complex { re: 0.0, im: 0.0 };
    }
    Complex { re: re / total, im: im / total }
}

pub fn fit_tempo(act: &Activations, bpm0: f64, p: &Params) -> (f64, Complex) {
    let frames: Vec<(f64, f64)> = act
        .beat
        .iter()
        .enumerate()
        .filter(|(_, &a)| a > 1e-3)
        .map(|(i, &a)| (act.time(i), a as f64))
        .collect();
    let total: f64 = frames.iter().map(|f| f.1).sum();
    let magnitude = |bpm: f64| -> f64 {
        let w = -2.0 * PI * bpm / 60.0;
        let (re, im) = frames.iter().fold((0.0, 0.0), |(re, im), &(t, a)| {
            let (s, c) = (w * t).sin_cos();
            (re + a * c, im + a * s)
        });
        re.hypot(im) / total.max(1e-300)
    };
    let best_of = |bpms: Vec<f64>| -> f64 {
        let scores: Vec<f64> = bpms.par_iter().map(|&b| magnitude(b)).collect();
        let i = scores.iter().enumerate().fold(0, |best, (j, v)| if *v > scores[best] { j } else { best });
        bpms[i]
    };
    let mut best = best_of(arange(bpm0 * (1.0 - p.search_span), bpm0 * (1.0 + p.search_span), 0.01));
    for step in [0.002, 0.0005] {
        best = best_of(arange(best - 10.0 * step, best + 10.5 * step, step));
    }
    (best, coherence_at(act, best, None))
}

pub fn phase_of(c: Complex, period: f64) -> f64 {
    (-c.angle() / (2.0 * PI) * period).rem_euclid(period)
}

pub fn wrap(x: f64, period: f64) -> f64 {
    (x + period / 2.0).rem_euclid(period) - period / 2.0
}

/// Largest deviation (ms) of 32-beat stretches from the single-tempo phase.
pub fn bend(act: &Activations, bpm: f64, t0: f64, window_beats: usize) -> f64 {
    let period = 60.0 / bpm;
    let span = window_beats as f64 * period;
    let last = (act.last_time() - span / 2.0).max(0.0);
    let mut offsets = Vec::new();
    for start in arange(0.0, last, span / 2.0) {
        let c = coherence_at(act, bpm, Some((start, start + span)));
        if c.abs() < 0.3 {
            continue;
        }
        offsets.push(wrap(phase_of(c, period) - t0, period));
    }
    if offsets.len() < 3 {
        return 0.0;
    }
    let m = median(&mut offsets.clone());
    offsets.iter().map(|o| (o - m).abs()).fold(0.0, f64::max) * 1000.0
}

/// Gaussian smoothing, sigma 1.5 samples, zero-padded (numpy convolve mode="same").
fn smooth(x: &[f64]) -> Vec<f64> {
    const SIGMA: f64 = 1.5;
    let radius = 4 * SIGMA.ceil() as i64;
    let kernel: Vec<f64> = (-radius..=radius).map(|i| (-0.5 * (i as f64 / SIGMA).powi(2)).exp()).collect();
    let sum: f64 = kernel.iter().sum();
    (0..x.len() as i64)
        .map(|i| {
            (-radius..=radius)
                .filter_map(|m| x.get((i + m) as usize).filter(|_| i + m >= 0).map(|v| v * kernel[(m + radius) as usize]))
                .sum::<f64>()
                / sum
        })
        .collect()
}

/// (fractional index, height) of the smoothed maximum, parabolically refined.
fn peak(folded: &[f64]) -> (f64, f64) {
    let s = smooth(folded);
    let i = s.iter().enumerate().fold(0, |best, (j, v)| if *v > s[best] { j } else { best });
    if i > 0 && i + 1 < s.len() {
        let (a, b, c) = (s[i - 1], s[i], s[i + 1]);
        let denom = a - 2.0 * b + c;
        if denom < 0.0 {
            return (i as f64 + 0.5 * (a - c) / denom, b - 0.125 * (a - c).powi(2) / denom);
        }
    }
    (i as f64, s[i])
}

/// Mean flux within ±half_s of every beat of a grid at bpm through t0.
pub fn fold(flux: &Flux, bpm: f64, t0: f64, half_s: f64) -> Vec<f64> {
    let period = 60.0 / bpm;
    let duration = flux.t_first + flux.values.len() as f64 * flux.dt;
    let half = (half_s / flux.dt).round() as i64;
    let width = (2 * half + 1) as usize;
    let mut sum = vec![0.0f64; width];
    let mut rows = 0usize;
    let k0 = (-t0 / period).ceil() as i64;
    let k1 = ((duration - t0) / period).floor() as i64;
    for k in k0..=k1 {
        let centre = ((t0 + k as f64 * period - flux.t_first) / flux.dt).round() as i64;
        if centre - half < 0 || centre + half >= flux.values.len() as i64 {
            continue;
        }
        for (j, s) in sum.iter_mut().enumerate() {
            *s += flux.values[(centre - half + j as i64) as usize] as f64;
        }
        rows += 1;
    }
    if rows > 0 {
        sum.iter_mut().for_each(|s| *s /= rows as f64);
    }
    sum
}

/// Shift (s) that puts the grid on the folded flux peak, and the peak's prominence.
pub fn attack_offset(flux: &Flux, bpm: f64, t0: f64, p: &Params) -> (f64, f64) {
    let folded = fold(flux, bpm, t0, p.attack_search_ms / 1000.0);
    let (i, height) = peak(&folded);
    let prominence = height / (median(&mut folded.clone()) + 1e-12);
    ((i - (folded.len() - 1) as f64 / 2.0) * flux.dt, prominence)
}

/// Height of the runner-up peak of the folded attack (the fold `attack_offset` measures) as a
/// fraction of the main one, both above the fold's median; 0 when there is none. A peak counts
/// when it is more than 5 ms from the main one and the fold dips at least 20% below it in
/// between.
pub fn second_attack(flux: &Flux, bpm: f64, t0: f64, p: &Params) -> f64 {
    let folded = fold(flux, bpm, t0, p.attack_search_ms / 1000.0);
    let s = smooth(&folded);
    let floor = median(&mut folded.clone());
    let i = s.iter().enumerate().fold(0, |best, (j, v)| if *v > s[best] { j } else { best });
    let main = s[i] - floor;
    if main.is_nan() || main <= 0.0 {
        return 0.0;
    }
    let gap = round_half_even(0.005 / flux.dt) as usize;
    let mut best = 0.0f64;
    for j in 1..s.len().saturating_sub(1) {
        if i.abs_diff(j) <= gap || s[j] < s[j - 1] || s[j] < s[j + 1] {
            continue;
        }
        let height = s[j] - floor;
        let dip = s[i.min(j)..=i.max(j)].iter().copied().fold(f64::INFINITY, f64::min) - floor;
        if height > best && dip < 0.8 * height {
            best = height;
        }
    }
    best / main
}

/// Stretches of the track the self-audit folds separately.
const STRETCHES: usize = 8;
/// How far (s) either side of each beat the self-audit looks for the attack.
const STRETCH_SEARCH_S: f64 = 0.06;
/// A stretch's folded attack needs a peak this many times its median to count.
const STRETCH_MIN_PROMINENCE: f64 = 1.5;

/// Does the attack stay put under the grid from one end of the track to the other? Folds the
/// flux around the grid's beats in each of 8 equal stretches of the track and finds the attack.
/// Returns how far it walks end to end (ms, on a Theil-Sen line through the stretches with a
/// clear attack; `None` when fewer than 3 have one) and how many stretches have one. A grid at
/// a slightly wrong tempo walks off the attacks; a right one stays put. The check `lab audit`
/// runs on reference grids, over the whole decoded track.
pub fn self_audit(flux: &Flux, bpm: f64, t0: f64) -> (Option<f64>, usize) {
    let period = 60.0 / bpm;
    let duration = flux.t_first + flux.values.len() as f64 * flux.dt;
    let half = round_half_even(STRETCH_SEARCH_S / flux.dt) as i64;
    let width = (2 * half + 1) as usize;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for s in 0..STRETCHES {
        let lo = duration * s as f64 / STRETCHES as f64;
        let hi = duration * (s + 1) as f64 / STRETCHES as f64;
        let mut sum = vec![0.0f64; width];
        let mut rows = 0usize;
        for k in ((lo - t0) / period).ceil() as i64..((hi - t0) / period).ceil() as i64 {
            let centre = ((t0 + k as f64 * period - flux.t_first) / flux.dt).round() as i64;
            if centre - half < 0 || centre + half >= flux.values.len() as i64 {
                continue;
            }
            for (j, v) in sum.iter_mut().enumerate() {
                *v += flux.values[(centre - half + j as i64) as usize] as f64;
            }
            rows += 1;
        }
        if rows < 8 {
            continue;
        }
        sum.iter_mut().for_each(|v| *v /= rows as f64);
        let (i, height) = peak(&sum);
        if height / (median(&mut sum.clone()) + 1e-12) < STRETCH_MIN_PROMINENCE {
            continue;
        }
        xs.push(s as f64);
        ys.push((i - half as f64) * flux.dt * 1000.0);
    }
    if xs.len() < 3 {
        return (None, xs.len());
    }
    let mut slopes = Vec::new();
    for a in 0..xs.len() {
        for b in a + 1..xs.len() {
            slopes.push((ys[b] - ys[a]) / (xs[b] - xs[a]));
        }
    }
    (Some(median(&mut slopes).abs() * (STRETCHES - 1) as f64), xs.len())
}

/// Mean flux by phase within the beat, one bin per flux sample.
pub fn fold_period(flux: &Flux, bpm: f64) -> Vec<f64> {
    let period = 60.0 / bpm;
    let bins = (period / flux.dt).ceil() as usize;
    let mut total = vec![0.0f64; bins];
    let mut count = vec![0u32; bins];
    // Phase advances by dt per sample; wrapped incrementally (same result as fmod to well
    // below a bin) rather than with a division per sample.
    let mut phase = flux.t_first.rem_euclid(period);
    for (i, &v) in flux.values.iter().enumerate() {
        if i % 4096 == 0 {
            phase = (flux.t_first + i as f64 * flux.dt).rem_euclid(period); // resync rounding
        }
        let idx = ((phase / flux.dt) as usize).min(bins - 1);
        total[idx] += v as f64;
        count[idx] += 1;
        phase += flux.dt;
        if phase >= period {
            phase -= period;
        }
    }
    total.iter().zip(&count).map(|(t, &c)| t / c.max(1) as f64).collect()
}

pub(crate) fn circular_peak_height(folded: &[f64]) -> f64 {
    let pad = 8.min(folded.len());
    let mut wrapped = folded[folded.len() - pad..].to_vec();
    wrapped.extend_from_slice(folded);
    wrapped.extend_from_slice(&folded[..pad]);
    let s = smooth(&wrapped);
    s[pad..s.len() - pad].iter().copied().fold(f64::MIN, f64::max)
}

pub(crate) fn height_at(flux: &Flux, bpm: f64) -> f64 {
    circular_peak_height(&fold_period(flux, bpm))
}

/// Tempo whose whole-period fold of the flux has the sharpest peak, near `bpm`.
pub fn tempo_by_flux(flux: &Flux, bpm: f64, p: &Params) -> (f64, f64) {
    let candidates = arange(bpm - p.flux_span, bpm + p.flux_span + p.flux_step / 2.0, p.flux_step);
    let heights: Vec<f64> = candidates.par_iter().map(|&b| height_at(flux, b)).collect();
    let i = heights.iter().enumerate().fold(0, |best, (j, v)| if *v > heights[best] { j } else { best });
    let mut best = candidates[i];
    if i > 0 && i + 1 < heights.len() {
        let (a, b, c) = (heights[i - 1], heights[i], heights[i + 1]);
        let denom = a - 2.0 * b + c;
        if denom < 0.0 {
            best += 0.5 * (a - c) / denom * p.flux_step;
        }
    }
    (best, height_at(flux, best))
}

pub fn apply_octave(mut bpm: f64, p: &Params) -> f64 {
    if let Some((lo, hi)) = p.octave_half_range {
        for candidate in [bpm / 2.0, bpm, bpm * 2.0] {
            if lo <= candidate && candidate < hi {
                return candidate / 2.0;
            }
        }
    }
    if let Some((lo, hi)) = p.octave_range {
        while bpm < lo {
            bpm *= 2.0;
        }
        while bpm >= hi {
            bpm /= 2.0;
        }
    }
    bpm
}

// Review reasons: why a DJ might want to listen before trusting a grid. What each catches
// and costs on the reference sets, and the signals that were tried and not kept, are in
// docs/results.md (research/experiments/review_flags.py).
pub const NO_ATTACK: &str = "no clear attack to place the grid on";
pub const WEAK_PULSE: &str = "weak or changing pulse";
pub const TEMPO_CHANGE: &str = "tempo may change";
pub const BAR_DISAGREE: &str = "bar 1 unclear: the downbeat cues disagree";
pub const OCTAVE_HALVED: &str = "octave chosen by the profile: the audio's pulse is twice as fast";
pub const OCTAVE_DOUBLED: &str = "octave chosen by the profile: the audio's pulse is half as fast";
pub const TEMPO_SEGMENTS: &str = "tempo changes: check the grid where it changes";

/// The review reasons for a fit (empty: nothing looked doubtful), from its diagnostics and the
/// grid's tempo after the octave convention. A grid in segments gets one reason, to check the
/// joins: the others describe a single-tempo grid.
pub fn review(d: &Diagnostics, grid_bpm: f64) -> Vec<&'static str> {
    if d.segment_search.as_ref().is_some_and(|s| s.accepted) {
        return vec![TEMPO_SEGMENTS];
    }
    let argmax = |v: &[f64; 4]| (0..4).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let mut reasons = Vec::new();
    // No usable attack: the audio could not refine the tempo, or under the final grid fewer
    // than 3 of the track's 8 stretches show one.
    if d.tempo_source != "flux" || d.stretch_drift_ms.is_none() {
        reasons.push(NO_ATTACK);
    }
    if d.coherence_final < 0.5 {
        reasons.push(WEAK_PULSE);
    }
    if d.bend_ms > 40.0 {
        reasons.push(TEMPO_CHANGE);
    }
    if argmax(&d.bar_votes[0]) != argmax(&d.bar_votes[1]) {
        reasons.push(BAR_DISAGREE);
    }
    // The profile, not the audio, chose the octave: a convention DJs differ on.
    if grid_bpm != d.bpm {
        reasons.push(if grid_bpm < d.bpm { OCTAVE_HALVED } else { OCTAVE_DOUBLED });
    }
    reasons
}

/// The result of [`fit`].
pub struct Fitted {
    /// The grid at one tempo, after the octave convention.
    pub grid: Grid,
    /// The grid in constant-tempo segments (one, the same as `grid`, unless the tempo changes
    /// and `Params::tempo` allows segments), after the octave convention.
    pub segments: SegmentedGrid,
    /// How it was found.
    pub diagnostics: Diagnostics,
    /// Reasons to check the grid by ear; empty when nothing looked doubtful.
    pub review: Vec<&'static str>,
}

/// Grid a track of `duration` seconds.
pub fn fit(act: &Activations, flux: &Flux, bands: &Bands, p: &Params, duration: f64) -> Option<Fitted> {
    let bpm0 = coarse_bpm(act, p)?;
    let (bpm_activation, c0) = fit_tempo(act, bpm0, p);
    let mut bpm = bpm_activation;
    let (mut snapped, mut source) = (false, "activation");

    // Tempo on the audio, judged at the best tempo the search finds.
    let (flux_bpm, height) = tempo_by_flux(flux, bpm, p);
    let folded = fold_period(flux, flux_bpm);
    let prominence = circular_peak_height(&folded) / (median(&mut folded.clone()) + 1e-12);
    let mut bpm_flux = None;
    let mut snap_drift_ms = 0.0;
    if prominence >= p.flux_min_prominence {
        bpm = flux_bpm;
        source = "flux";
        bpm_flux = Some(bpm);
        if p.snap {
            let half_track = flux.values.len() as f64 * flux.dt / 2.0;
            for &step in &p.snap_steps {
                let target = round_half_even(bpm / step) * step;
                if (target - bpm).abs() > p.snap_max_bpm {
                    continue;
                }
                let drift_ms = half_track * (target - bpm).abs() / bpm * 1000.0;
                if drift_ms <= p.snap_free_drift_ms || height_at(flux, target) >= height * (1.0 - p.snap_flux_tolerance) {
                    snap_drift_ms = 2.0 * drift_ms;
                    bpm = target;
                    snapped = true;
                    break;
                }
            }
        }
    }
    if source == "activation" && p.snap {
        for &step in &p.snap_steps {
            let target = round_half_even(bpm / step) * step;
            if (target - bpm).abs() > p.snap_max_bpm {
                continue;
            }
            if coherence_at(act, target, None).abs() >= c0.abs() * (1.0 - p.snap_tolerance) {
                bpm = target;
                snapped = true;
                break;
            }
        }
    }
    let c = coherence_at(act, bpm, None);
    let period = 60.0 / bpm;
    let mut t0 = phase_of(c, period);
    let bend_ms = bend(act, bpm, t0, p.bend_window);

    let (shift, attack_prominence) = attack_offset(flux, bpm, t0, p);
    let attack_second = second_attack(flux, bpm, t0, p);
    t0 += shift + p.offset_ms / 1000.0;
    let (stretch_drift_ms, clear_stretches) = self_audit(flux, bpm, t0);

    let bar = bars::decide(act, bands, bpm, t0, p.bar_w_network, p.bar_w_changes, p.bar_w_fills, None);
    let downbeat = t0 + bar.rotation as f64 * period;
    let grid = Grid { bpm: apply_octave(bpm, p), downbeat };
    let (found, segment_search) = match p.tempo {
        TempoMode::Constant => (None, None),
        _ => {
            let (found, search) = segments::fit(act, flux, bands, p, bpm, downbeat, duration);
            (found, Some(search))
        }
    };
    let segments = match found {
        Some(s) => segments::with_octave(&s, p),
        None => SegmentedGrid::from(grid),
    };

    let diagnostics = Diagnostics {
        bpm_coarse: bpm0,
        bpm_activation,
        coherence: c0.abs(),
        flux_prominence: Some(prominence),
        bpm_flux,
        bpm,
        snapped,
        tempo_source: source,
        coherence_final: c.abs(),
        bend_ms,
        attack_shift_ms: shift * 1000.0,
        attack_prominence,
        attack_second,
        stretch_drift_ms,
        clear_stretches,
        snap_drift_ms,
        bar_margin: bar.margin,
        bar_votes: bar.votes,
        segment_search,
    };
    let review = review(&diagnostics, grid.bpm);
    Some(Fitted { grid, segments, diagnostics, review })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 44100.0;

    /// Onset flux as the flux of a drum loop looks: a floor with a sharp attack on every beat at
    /// `bpm` from `first`, and optionally a second attack `ms` later at `height` of the first.
    fn flux_with_attacks(bpm: f64, first: f64, seconds: f64, second: Option<(f64, f64)>) -> Flux {
        let dt = 32.0 / RATE;
        let t_first = (256.0 + 16.0) / RATE;
        let period = 60.0 / bpm;
        let bump = |t: f64, at: f64| (-0.5 * ((t - at) / 0.0008).powi(2)).exp();
        let values = (0..(seconds / dt) as usize)
            .map(|i| {
                let t = t_first + i as f64 * dt;
                let beat = ((t - first) / period).round();
                let at = first + beat * period;
                let mut v = 1.0 + 10.0 * bump(t, at);
                if let Some((ms, height)) = second {
                    let k = ((t - first - ms / 1000.0) / period).round();
                    v += 10.0 * height * bump(t, first + k * period + ms / 1000.0);
                }
                v as f32
            })
            .collect();
        Flux { values, dt, t_first }
    }

    #[test]
    fn self_audit_holds_on_the_right_tempo_and_walks_on_a_wrong_one() {
        let flux = flux_with_attacks(128.0, 0.25, 240.0, None);
        let (drift, clear) = self_audit(&flux, 128.0, 0.25);
        assert_eq!(clear, STRETCHES);
        assert!(drift.unwrap() < 1.0, "drift {drift:?}");
        // 0.02 BPM off walks ~33 ms across the track's 8 stretches.
        let (drift, _) = self_audit(&flux, 128.02, 0.25);
        assert!(drift.unwrap() > 25.0, "drift {drift:?}");
        // No attack anywhere: no verdict.
        let flat = Flux { values: vec![1.0; flux.values.len()], ..flux };
        assert_eq!(self_audit(&flat, 128.0, 0.25), (None, 0));
    }

    #[test]
    fn a_second_attack_is_measured_against_the_first() {
        let p = Params::default();
        let single = flux_with_attacks(126.0, 0.3, 60.0, None);
        assert!(second_attack(&single, 126.0, 0.3, &p) < 0.1);
        let double = flux_with_attacks(126.0, 0.3, 60.0, Some((15.0, 0.6)));
        let ratio = second_attack(&double, 126.0, 0.3, &p);
        assert!((ratio - 0.6).abs() < 0.1, "ratio {ratio}");
        // Closer than 5 ms it is the same attack, not a second one.
        let close = flux_with_attacks(126.0, 0.3, 60.0, Some((3.0, 0.6)));
        assert!(second_attack(&close, 126.0, 0.3, &p) < 0.3);
    }

    fn clean() -> Diagnostics {
        Diagnostics {
            bpm_coarse: 128.0,
            bpm_activation: 128.0,
            coherence: 0.9,
            flux_prominence: Some(3.0),
            bpm_flux: Some(128.0),
            bpm: 128.0,
            snapped: true,
            tempo_source: "flux",
            coherence_final: 0.9,
            bend_ms: 5.0,
            attack_shift_ms: -8.0,
            attack_prominence: 3.0,
            attack_second: 0.1,
            stretch_drift_ms: Some(1.0),
            clear_stretches: 8,
            snap_drift_ms: 0.5,
            bar_margin: 6.0,
            bar_votes: [[0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]],
            segment_search: None,
        }
    }

    #[test]
    fn each_review_reason_has_its_cause() {
        assert!(review(&clean(), 128.0).is_empty());
        let only = |d: Diagnostics, grid_bpm: f64| review(&d, grid_bpm);
        assert_eq!(only(Diagnostics { tempo_source: "activation", ..clean() }, 128.0), [NO_ATTACK]);
        assert_eq!(only(Diagnostics { stretch_drift_ms: None, clear_stretches: 2, ..clean() }, 128.0), [NO_ATTACK]);
        assert_eq!(only(Diagnostics { coherence_final: 0.4, ..clean() }, 128.0), [WEAK_PULSE]);
        assert_eq!(only(Diagnostics { bend_ms: 60.0, ..clean() }, 128.0), [TEMPO_CHANGE]);
        let disagree = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]];
        assert_eq!(only(Diagnostics { bar_votes: disagree, ..clean() }, 128.0), [BAR_DISAGREE]);
        assert_eq!(only(Diagnostics { bpm: 174.0, ..clean() }, 87.0), [OCTAVE_HALVED]);
        assert_eq!(only(Diagnostics { bpm: 80.0, ..clean() }, 160.0), [OCTAVE_DOUBLED]);
        // A grid in segments: check the joins (the rest describe a single-tempo grid).
        let search = |accepted| Some(SegmentSearch { accepted, ..SegmentSearch::default() });
        let doubtful = Diagnostics { bend_ms: 60.0, coherence_final: 0.4, ..clean() };
        assert_eq!(only(Diagnostics { segment_search: search(true), ..doubtful.clone() }, 128.0), [TEMPO_SEGMENTS]);
        assert_eq!(only(Diagnostics { segment_search: search(false), ..doubtful }, 128.0), [WEAK_PULSE, TEMPO_CHANGE]);
        // Measured, but not reasons on their own (docs/results.md).
        let evidence = Diagnostics {
            attack_second: 0.9,
            stretch_drift_ms: Some(30.0),
            snap_drift_ms: 30.0,
            bar_margin: 0.1,
            ..clean()
        };
        assert!(review(&evidence, 128.0).is_empty());
    }
}
