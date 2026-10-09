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
use crate::Grid;

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
    /// Pooled log score of bar 1 over the runner-up.
    pub bar_margin: f64,
    /// Bar-1 votes by beat position, [network, arrangement changes], each summing to 1.
    pub bar_votes: [[f64; 4]; 2],
}

fn median(v: &mut [f64]) -> f64 {
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
fn round_half_even(x: f64) -> f64 {
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

fn circular_peak_height(folded: &[f64]) -> f64 {
    let pad = 8.min(folded.len());
    let mut wrapped = folded[folded.len() - pad..].to_vec();
    wrapped.extend_from_slice(folded);
    wrapped.extend_from_slice(&folded[..pad]);
    let s = smooth(&wrapped);
    s[pad..s.len() - pad].iter().copied().fold(f64::MIN, f64::max)
}

fn height_at(flux: &Flux, bpm: f64) -> f64 {
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

/// The result of [`fit`].
pub struct Fitted {
    /// The grid, after the octave convention.
    pub grid: Grid,
    /// How it was found.
    pub diagnostics: Diagnostics,
    /// Reasons to check the grid by ear; empty when nothing looked doubtful.
    pub review: Vec<&'static str>,
}

pub fn fit(act: &Activations, flux: &Flux, bands: &Bands, p: &Params) -> Option<Fitted> {
    let bpm0 = coarse_bpm(act, p)?;
    let (bpm_activation, c0) = fit_tempo(act, bpm0, p);
    let mut bpm = bpm_activation;
    let (mut snapped, mut source) = (false, "activation");

    // Tempo on the audio, judged at the best tempo the search finds.
    let (flux_bpm, height) = tempo_by_flux(flux, bpm, p);
    let folded = fold_period(flux, flux_bpm);
    let prominence = circular_peak_height(&folded) / (median(&mut folded.clone()) + 1e-12);
    let mut bpm_flux = None;
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
    t0 += shift + p.offset_ms / 1000.0;

    let bar = bars::decide(act, bands, bpm, t0, p.bar_w_network, p.bar_w_changes, p.bar_w_fills);
    let downbeat = t0 + bar.rotation as f64 * period;
    let grid = Grid { bpm: apply_octave(bpm, p), downbeat };

    // Measured on references the audit calls steady: catches 65% of wrong grids, flags 29%
    // of right ones (research/experiments/review_selfaudit.py).
    let argmax = |v: &[f64; 4]| (0..4).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let mut review = Vec::new();
    if source != "flux" {
        review.push("no clear attack to place the grid on");
    }
    if c.abs() < 0.5 {
        review.push("weak or changing pulse");
    }
    if bend_ms > 40.0 {
        review.push("tempo may change");
    }
    if argmax(&bar.votes[0]) != argmax(&bar.votes[1]) {
        review.push("bar 1 unclear: the downbeat cues disagree");
    }
    Some(Fitted {
        grid,
        diagnostics: Diagnostics {
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
            bar_margin: bar.margin,
            bar_votes: bar.votes,
        },
        review,
    })
}
