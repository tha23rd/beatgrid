//! Tempo changes: a port of research/beatgrid_lab/sections.py, which explains the method.
//!
//! 1. the local tempo in 16 s windows every 4 s, decoded jointly (Viterbi over tempo states)
//! 2. an anchor beat per window from the coherence phase at that tempo, numbered on
//! 3. the anchors split into the fewest straight lines (beat time against beat number)
//! 4. each segment fitted like a whole track on its own stretch; neighbours on one grid merged
//! 5. each change put on the new segment's downbeat where the attacks agree best
//! 6. the segments kept only where they differ from the constant grid and sit clearly better
//!    on the attacks there

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::bars;
use crate::dsp::{Bands, Flux};
use crate::fit::{self, Activations, Params};
use crate::grid::{Segment, SegmentedGrid};
use crate::TempoMode;

/// How the search for tempo changes went. For debugging and research; the fields may change
/// between versions.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SegmentSearch {
    /// Straight lines the anchors split into (1: one tempo holds).
    pub runs: usize,
    /// Segments after fitting each line's stretch and merging neighbours on one grid.
    pub fitted: usize,
    /// Beats where the segmented grid is more than 10 ms from the constant one.
    pub apart_beats: usize,
    /// Mean attack on those beats under the segmented grid and under the constant one, over the
    /// track's median onset flux.
    pub evidence: [f64; 2],
    /// Whether the segments replaced the constant grid.
    pub accepted: bool,
}

/// The activations with every frame outside [lo, hi) zeroed.
fn masked(act: &Activations, lo: f64, hi: f64) -> (Vec<f32>, Vec<f32>) {
    let keep = |i: usize| {
        let t = act.time(i);
        t >= lo && t < hi
    };
    let beat = act.beat.iter().enumerate().map(|(i, &v)| if keep(i) { v } else { 0.0 }).collect();
    let downbeat = act.downbeat.iter().enumerate().map(|(i, &v)| if keep(i) { v } else { 0.0 }).collect();
    (beat, downbeat)
}

/// The flux values whose time lies in [lo, hi).
fn flux_slice(flux: &Flux, lo: f64, hi: f64) -> Flux {
    let n = flux.values.len() as i64;
    let i0 = (((lo - flux.t_first) / flux.dt).ceil() as i64).clamp(0, n);
    let i1 = (((hi - flux.t_first) / flux.dt).ceil() as i64).clamp(i0, n);
    Flux {
        values: flux.values[i0 as usize..i1 as usize].to_vec(),
        dt: flux.dt,
        t_first: flux.t_first + i0 as f64 * flux.dt,
    }
}

// --- 1. tempo path ------------------------------------------------------------------------------

/// Tempos within an octave band around `centre`, log-spaced by `seg_step`.
fn tempo_states(centre: f64, p: &Params) -> Vec<f64> {
    let step = p.seg_step.ln_1p();
    let k = (2f64.sqrt().ln() / step).floor() as i64;
    (-k..=k).map(|i| centre * (i as f64 * step).exp()).collect()
}

fn window_starts(start: f64, end: f64, p: &Params) -> Vec<f64> {
    let n = (((end - start - p.seg_window_s) / p.seg_hop_s).floor() as i64 + 1).max(1);
    (0..n).map(|i| start + i as f64 * p.seg_hop_s).collect()
}

/// The beat activation's coherence with a pulse train at each tempo, over [lo, hi).
fn salience(act: &Activations, lo: f64, hi: f64, bpms: &[f64]) -> Vec<fit::Complex> {
    let frames: Vec<(f64, f64)> = (0..act.beat.len())
        .filter_map(|i| {
            let (t, a) = (act.time(i), act.beat[i] as f64);
            (t >= lo && t < hi && a > 1e-3).then_some((t, a))
        })
        .collect();
    let total: f64 = frames.iter().map(|f| f.1).sum();
    bpms.iter()
        .map(|&b| {
            if total <= 0.0 {
                return fit::Complex { re: 0.0, im: 0.0 };
            }
            let w = -2.0 * PI * b / 60.0;
            let (re, im) = frames.iter().fold((0.0, 0.0), |(re, im), &(t, a)| {
                let (s, c) = (w * t).sin_cos();
                (re + a * c, im + a * s)
            });
            fit::Complex { re: re / total, im: im / total }
        })
        .collect()
}

/// The most likely tempo state per window (Viterbi), and the coherence per window and state.
fn tempo_path(act: &Activations, starts: &[f64], bpms: &[f64], p: &Params) -> (Vec<usize>, Vec<Vec<fit::Complex>>) {
    let coh: Vec<Vec<fit::Complex>> = starts.par_iter().map(|&lo| salience(act, lo, lo + p.seg_window_s, bpms)).collect();
    let k = bpms.len();
    let strength: Vec<f64> = coh.iter().map(|row| row.iter().map(|c| c.abs()).fold(f64::MIN, f64::max)).collect();
    let loge: Vec<Vec<f64>> = coh.iter().map(|row| row.iter().map(|c| (c.abs() + 1e-3).ln()).collect()).collect();
    let trans = |from: usize, to: usize| match from.abs_diff(to) {
        0 => 0.0,
        1 => -p.seg_drift_cost,
        _ => -p.seg_switch_cost,
    };
    let mut score: Vec<f64> = if strength[0] >= p.seg_quiet { loge[0].clone() } else { vec![0.0; k] };
    let mut back = vec![vec![0usize; k]; starts.len()];
    for i in 1..starts.len() {
        let mut next = vec![0.0; k];
        for to in 0..k {
            // np.argmax: the first best.
            let (best, best_v) = score
                .iter()
                .enumerate()
                .map(|(from, &s)| (from, s + trans(from, to)))
                .fold((0, f64::NEG_INFINITY), |(b, bv), (f, v)| if v > bv { (f, v) } else { (b, bv) });
            back[i][to] = best;
            // Quiet windows (breakdowns) carry no evidence: they follow their neighbours.
            next[to] = best_v + if strength[i] >= p.seg_quiet { loge[i][to] } else { 0.0 };
        }
        score = next;
    }
    let mut path = vec![0usize; starts.len()];
    path[starts.len() - 1] = (0..k).fold(0, |b, j| if score[j] > score[b] { j } else { b });
    for i in (1..starts.len()).rev() {
        path[i - 1] = back[i][path[i]];
    }
    (path, coh)
}

// --- 2. anchors ---------------------------------------------------------------------------------

struct Anchors {
    number: Vec<f64>,
    time: Vec<f64>,
    weight: Vec<f64>,
}

fn anchors(starts: &[f64], bpms: &[f64], path: &[usize], coh: &[Vec<fit::Complex>], p: &Params) -> Anchors {
    let mut an = Anchors { number: Vec::new(), time: Vec::new(), weight: Vec::new() };
    let mut last_window = 0usize;
    for (w, &lo) in starts.iter().enumerate() {
        let c = coh[w][path[w]];
        if c.abs() < p.seg_anchor_min {
            continue;
        }
        let period = 60.0 / bpms[path[w]];
        let centre = lo + p.seg_window_s / 2.0;
        let phase = fit::phase_of(c, period);
        let t = phase + period * fit::round_half_even((centre - phase) / period);
        let number = match an.time.last() {
            Some(&prev) => {
                let gap = 0.5 * (period + 60.0 / bpms[path[last_window]]);
                an.number.last().unwrap() + fit::round_half_even((t - prev) / gap)
            }
            None => 0.0,
        };
        an.number.push(number);
        an.time.push(t);
        an.weight.push(c.abs());
        last_window = w;
    }
    an
}

// --- 3. segments --------------------------------------------------------------------------------

/// Robust straight line t = a + b n through weighted anchors: least squares, again without the
/// anchors more than `seg_clip` sigmas off, and the clipped cost. Returns (cost, a, b).
fn line_cost(n: &[f64], t: &[f64], w: &[f64], p: &Params) -> (f64, f64, f64) {
    let sigma = p.seg_sigma_ms / 1000.0;
    let fit = |keep: &dyn Fn(usize) -> bool| {
        let (mut sw, mut sn, mut st, mut snn, mut snt) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in (0..n.len()).filter(|&i| keep(i)) {
            sw += w[i];
            sn += w[i] * n[i];
            st += w[i] * t[i];
            snn += w[i] * n[i] * n[i];
            snt += w[i] * n[i] * t[i];
        }
        let den = sw * snn - sn * sn;
        if den <= 0.0 {
            return (st / sw, 0.0);
        }
        let b = (sw * snt - sn * st) / den;
        ((st - b * sn) / sw, b)
    };
    let (mut a, mut b) = fit(&|_| true);
    let r: Vec<f64> = (0..n.len()).map(|i| (t[i] - a - b * n[i]) / sigma).collect();
    let inliers: Vec<bool> = r.iter().map(|x| x.abs() <= p.seg_clip).collect();
    let count = inliers.iter().filter(|&&x| x).count();
    let mut r = r;
    if count >= 2 && count < n.len() {
        (a, b) = fit(&|i| inliers[i]);
        r = (0..n.len()).map(|i| (t[i] - a - b * n[i]) / sigma).collect();
    }
    let cost = (0..n.len()).map(|i| w[i] * (r[i] * r[i]).min(p.seg_clip * p.seg_clip)).sum();
    (cost, a, b)
}

/// Runs [i, j) of the anchors that each hold one line, minimising the summed cost plus
/// `penalty` per extra run; each holds at least `seg_min_anchors` anchors.
fn partition(an: &Anchors, penalty: f64, p: &Params) -> Vec<(usize, usize)> {
    let m = an.time.len();
    let least = p.seg_min_anchors;
    if m < 2 * least {
        return vec![(0, m)];
    }
    let cost = |i: usize, j: usize| line_cost(&an.number[i..j], &an.time[i..j], &an.weight[i..j], p).0;
    let costs: Vec<Vec<f64>> =
        (0..m).into_par_iter().map(|i| (0..=m).map(|j| if j >= i + least { cost(i, j) } else { f64::INFINITY }).collect()).collect();
    let mut best = vec![f64::INFINITY; m + 1];
    let mut prev = vec![0usize; m + 1];
    best[0] = -penalty;
    for j in least..=m {
        for i in 0..=j - least {
            if !best[i].is_finite() {
                continue;
            }
            let v = best[i] + costs[i][j] + penalty;
            if v < best[j] {
                best[j] = v;
                prev[j] = i;
            }
        }
    }
    let mut runs = Vec::new();
    let mut j = m;
    while j > 0 {
        runs.push((prev[j], j));
        j = prev[j];
    }
    runs.reverse();
    runs
}

// --- 4. fit each segment ------------------------------------------------------------------------

/// One stretch of the track fitted at one tempo: bpm (before the octave convention) and a
/// downbeat.
#[derive(Clone, Copy, Debug)]
struct Stretch {
    lo: f64,
    hi: f64,
    bpm: f64,
    downbeat: f64,
}

/// The constant-tempo fit of fit::fit, on [lo, hi) of the track only.
fn fit_stretch(act: &Activations, flux: &Flux, bands: &Bands, lo: f64, hi: f64, bpm0: f64, p: &Params) -> Stretch {
    let (beat, downbeat) = masked(act, lo, hi);
    let a = Activations { beat: &beat, downbeat: &downbeat, fps: act.fps };
    let f = flux_slice(flux, lo, hi);
    let (mut bpm, c0) = fit::fit_tempo(&a, bpm0, &Params { search_span: p.seg_search_span, ..p.clone() });
    let mut from_flux = false;
    let (flux_bpm, height) = fit::tempo_by_flux(&f, bpm, p);
    let folded = fit::fold_period(&f, flux_bpm);
    let prominence = fit::circular_peak_height(&folded) / (fit::median(&mut folded.clone()) + 1e-12);
    if prominence >= p.flux_min_prominence {
        bpm = flux_bpm;
        from_flux = true;
        if p.snap {
            let half = (hi - lo) / 2.0;
            for &step in &p.snap_steps {
                let target = fit::round_half_even(bpm / step) * step;
                if (target - bpm).abs() > p.snap_max_bpm {
                    continue;
                }
                let drift_ms = half * (target - bpm).abs() / bpm * 1000.0;
                if drift_ms <= p.snap_free_drift_ms || fit::height_at(&f, target) >= height * (1.0 - p.snap_flux_tolerance) {
                    bpm = target;
                    break;
                }
            }
        }
    }
    if !from_flux && p.snap {
        for &step in &p.snap_steps {
            let target = fit::round_half_even(bpm / step) * step;
            if (target - bpm).abs() > p.snap_max_bpm {
                continue;
            }
            if fit::coherence_at(&a, target, None).abs() >= c0.abs() * (1.0 - p.snap_tolerance) {
                bpm = target;
                break;
            }
        }
    }
    let period = 60.0 / bpm;
    let mut t0 = fit::phase_of(fit::coherence_at(&a, bpm, None), period);
    let (shift, _) = fit::attack_offset(&f, bpm, t0, p);
    t0 += shift + p.offset_ms / 1000.0;
    let bar = bars::decide(&a, bands, bpm, t0, p.bar_w_network, p.bar_w_changes, p.bar_w_fills, Some((lo, hi)));
    Stretch { lo, hi, bpm, downbeat: t0 + bar.rotation as f64 * period }
}

/// Whether two fits are one grid: the same tempo, and beats within 3 ms where they meet.
fn same_grid(x: &Stretch, y: &Stretch, at: f64) -> bool {
    if (x.bpm - y.bpm).abs() > 1e-6 * x.bpm {
        return false;
    }
    let period = 60.0 / x.bpm;
    let near = |s: &Stretch| s.downbeat + period * fit::round_half_even((at - s.downbeat) / period);
    fit::wrap(near(x) - near(y), period).abs() <= 0.003
}

// --- 5. joins -----------------------------------------------------------------------------------

/// The flux's peak in the 10 ms before each grid line (grid lines go 2 ms after the attack).
fn attack_evidence(flux: &Flux, times: &[f64]) -> Vec<f64> {
    let n = flux.values.len() as i64;
    times
        .iter()
        .map(|&t| {
            let lo = (fit::round_half_even((t - 0.008 - flux.t_first) / flux.dt) as i64).max(0);
            let hi = (fit::round_half_even((t + 0.002 - flux.t_first) / flux.dt) as i64 + 1).min(n);
            if hi > lo {
                flux.values[lo as usize..hi as usize].iter().fold(f32::MIN, |m, &v| m.max(v)) as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// Beats of a stretch's grid in [lo, hi).
fn grid_times(s: &Stretch, lo: f64, hi: f64) -> Vec<f64> {
    let period = 60.0 / s.bpm;
    let k0 = ((lo - s.downbeat) / period).ceil() as i64;
    let k1 = ((hi - s.downbeat) / period).ceil() as i64;
    (k0..k1).map(|k| s.downbeat + k as f64 * period).collect()
}

/// Where the grid changes from `a` to `b`: the downbeat of b, within `seg_join_s` of the rough
/// change, that puts a's beats before it and b's from it best on the attacks.
fn join(a: &Stretch, b: &Stretch, rough: f64, flux: &Flux, p: &Params) -> f64 {
    let (pa, pb) = (60.0 / a.bpm, 60.0 / b.bpm);
    let (lo, hi) = (rough - p.seg_join_s, rough + p.seg_join_s);
    let ta = grid_times(a, lo - 4.0 * pa, hi + 4.0 * pa);
    let tb = grid_times(b, lo - 4.0 * pb, hi + 4.0 * pb);
    let (ea, eb) = (attack_evidence(flux, &ta), attack_evidence(flux, &tb));
    let k0 = ((lo - b.downbeat) / (4.0 * pb)).ceil() as i64;
    let k1 = ((hi - b.downbeat) / (4.0 * pb)).floor() as i64;
    let (mut best, mut best_score) = (rough, f64::NEG_INFINITY);
    for k in k0..=k1 {
        let d = b.downbeat + 4.0 * pb * k as f64;
        let before: f64 = ta.iter().zip(&ea).filter(|(t, _)| **t <= d - pa / 2.0).map(|(_, e)| e).sum();
        let after: f64 = tb.iter().zip(&eb).filter(|(t, _)| **t >= d - 1e-9).map(|(_, e)| e).sum();
        if before + after > best_score {
            best = d;
            best_score = before + after;
        }
    }
    best
}

// --- 6. gate ------------------------------------------------------------------------------------

/// The nearest of `sorted` (at least two values) to each of `times`.
fn nearest_index(sorted: &[f64], t: f64) -> usize {
    let i = sorted.partition_point(|&x| x < t).clamp(1, sorted.len() - 1);
    if (sorted[i - 1] - t).abs() <= (sorted[i] - t).abs() {
        i - 1
    } else {
        i
    }
}

/// Whether the segmented grid should replace the constant one (see sections.gate).
fn gate(
    constant: &SegmentedGrid, segmented: &SegmentedGrid, flux: &Flux, start: f64, end: f64, p: &Params,
    search: &mut SegmentSearch,
) -> bool {
    // Python's Segmented.beats(start, end) includes `end`.
    let ts: Vec<f64> = segmented.beats(start, end + 1e-9).iter().map(|b| b.time).collect();
    let tc: Vec<f64> = constant.beats(start - 2.0, end + 2.0 + 1e-9).iter().map(|b| b.time).collect();
    if ts.len() < 2 || tc.len() < 2 {
        return false;
    }
    let apart: Vec<bool> = ts.iter().map(|&t| (tc[nearest_index(&tc, t)] - t).abs() > 0.010).collect();
    search.apart_beats = apart.iter().filter(|&&a| a).count();
    if search.apart_beats == 0 {
        return false;
    }
    let tc_in: Vec<f64> = constant.beats(start, end + 1e-9).iter().map(|b| b.time).collect();
    let c_apart: Vec<f64> = tc_in
        .iter()
        .copied()
        .filter(|&t| {
            let j = nearest_index(&ts, t);
            apart[j] && (ts[j] - t).abs() > 0.010
        })
        .collect();
    let floor = fit::median(&mut flux.values.iter().map(|&v| v as f64).collect::<Vec<_>>()) + 1e-12;
    let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let seg_apart: Vec<f64> = ts.iter().zip(&apart).filter(|(_, &a)| a).map(|(t, _)| *t).collect();
    let es = mean(attack_evidence(flux, &seg_apart)) / floor;
    let ec = if c_apart.is_empty() { 0.0 } else { mean(attack_evidence(flux, &c_apart)) / floor };
    search.evidence = [es, ec];
    search.apart_beats >= p.seg_min_apart && es >= p.seg_min_evidence && es >= ec * (1.0 + p.seg_margin)
}

/// Where a beat at `at` falls on a stretch's grid, as a fraction of a beat (-0.5 to 0.5).
fn phase_jump(a: &Stretch, at: f64) -> f64 {
    let x = (at - a.downbeat) / (60.0 / a.bpm);
    x - fit::round_half_even(x)
}

/// The first join this mode does not count (see sections.offending).
fn offending(fits: &[Stretch], changes: &[f64], start: f64, end: f64, p: &Params) -> Option<usize> {
    for (k, w) in fits.windows(2).enumerate() {
        if (w[1].bpm / w[0].bpm - 1.0).abs() < p.seg_min_tempo_change {
            let jump = phase_jump(&w[0], changes[k]).abs();
            if p.seg_min_jump <= 0.0 || jump < p.seg_min_jump {
                return Some(k);
            }
        }
    }
    let mut edges = vec![start];
    edges.extend(changes);
    edges.push(end);
    for (k, f) in fits.iter().enumerate() {
        if (edges[k + 1] - edges[k]) * f.bpm / 240.0 < p.seg_min_bars {
            if k == 0 {
                return Some(0);
            }
            if k == fits.len() - 1 {
                return Some(k - 1);
            }
            // Merge into the neighbour closer in tempo.
            let left = (f.bpm / fits[k - 1].bpm - 1.0).abs();
            let right = (fits[k + 1].bpm / f.bpm - 1.0).abs();
            return Some(if left <= right { k - 1 } else { k });
        }
    }
    None
}

// --- all together -------------------------------------------------------------------------------

/// A segmented grid (before the octave convention) for a track whose constant fit is `bpm`
/// through `downbeat`, or `None` when one tempo should stay.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    act: &Activations, flux: &Flux, bands: &Bands, p: &Params, bpm: f64, downbeat: f64, duration: f64,
) -> (Option<SegmentedGrid>, SegmentSearch) {
    let mut search = SegmentSearch::default();
    if p.tempo == TempoMode::Constant || duration < 2.0 * p.seg_window_s {
        return (None, search);
    }
    let bpms = tempo_states(bpm, p);
    let starts = window_starts(0.0, duration, p);
    let (path, coh) = tempo_path(act, &starts, &bpms, p);
    let an = anchors(&starts, &bpms, &path, &coh, p);
    let runs = partition(&an, p.seg_penalty, p);
    search.runs = runs.len();
    if runs.len() < 2 {
        return (None, search);
    }
    // Where the beat is clear: from the first anchor's window to the last one's.
    let start = (an.time[0] - p.seg_window_s / 2.0).max(0.0);
    let end = (an.time[an.time.len() - 1] + p.seg_window_s / 2.0).min(duration);
    // Rough changes: halfway between the last anchor of one run and the first of the next.
    let rough: Vec<f64> = runs[..runs.len() - 1].iter().map(|&(_, j)| 0.5 * (an.time[j - 1] + an.time[j])).collect();
    let mut edges = vec![0.0];
    edges.extend(&rough);
    edges.push(duration);
    let last = runs.len() - 1;
    let fits: Vec<Stretch> = runs
        .par_iter()
        .enumerate()
        .map(|(r, &(i, j))| {
            let lo = edges[r] + if r > 0 { p.seg_hop_s } else { 0.0 };
            let hi = edges[r + 1] - if r < last { p.seg_hop_s } else { 0.0 };
            let (_, _, slope) = line_cost(&an.number[i..j], &an.time[i..j], &an.weight[i..j], p);
            fit_stretch(act, flux, bands, lo, hi, 60.0 / slope, p)
        })
        .collect();
    // Neighbours on one grid are one segment.
    let mut merged = vec![fits[0]];
    let mut rough_kept = Vec::new();
    for (f, &at) in fits[1..].iter().zip(&rough) {
        let prev = *merged.last().unwrap();
        if same_grid(&prev, f, at) {
            merged.pop();
            merged.push(fit_stretch(act, flux, bands, prev.lo, f.hi, prev.bpm, p));
        } else {
            merged.push(*f);
            rough_kept.push(at);
        }
    }
    let mut rough = rough_kept;
    // Joins this mode does not count are merged too, one at a time.
    let mut changes = Vec::new();
    while merged.len() > 1 {
        changes = merged.windows(2).zip(&rough).map(|(w, &r)| join(&w[0], &w[1], r, flux, p)).collect();
        let Some(k) = offending(&merged, &changes, start, end, p) else {
            break;
        };
        let (a, b) = (merged[k], merged[k + 1]);
        let seed = if a.hi - a.lo >= b.hi - b.lo { a.bpm } else { b.bpm };
        merged.splice(k..k + 2, [fit_stretch(act, flux, bands, a.lo, b.hi, seed, p)]);
        rough.remove(k);
    }
    search.fitted = merged.len();
    if merged.len() < 2 {
        return (None, search);
    }
    let first = merged[0];
    let period = 60.0 / first.bpm;
    let mut segments = vec![Segment { start: first.downbeat.rem_euclid(4.0 * period), bpm: first.bpm, beat_in_bar: 1 }];
    for (f, &at) in merged[1..].iter().zip(&changes) {
        // A segment holds at least a bar.
        let prev = segments.last().unwrap();
        if at - prev.start < 4.0 * prev.period() - 1e-9 {
            continue;
        }
        segments.push(Segment { start: at, bpm: f.bpm, beat_in_bar: 1 });
    }
    if segments.len() < 2 {
        return (None, search);
    }
    let Ok(segmented) = SegmentedGrid::new(segments) else {
        return (None, search);
    };
    let constant = SegmentedGrid::from(crate::Grid { bpm, downbeat });
    search.accepted = gate(&constant, &segmented, flux, start, end, p, &mut search);
    (search.accepted.then_some(segmented), search)
}

/// The octave convention, per segment (each keeps its first beat).
pub fn with_octave(segmented: &SegmentedGrid, p: &Params) -> SegmentedGrid {
    let segments = segmented.segments().iter().map(|s| Segment { bpm: fit::apply_octave(s.bpm, p), ..*s }).collect();
    SegmentedGrid::new(segments).expect("halving or doubling a tempo keeps the segments apart")
}
