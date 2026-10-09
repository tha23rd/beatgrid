//! Bar 1: the network's downbeat fold pooled with arrangement changes (the biggest
//! beat-to-beat steps in loudness and brightness land on bar 1) and high-frequency bursts. A port of
//! research/beatgrid_lab/bars.py with the votes measured to help; weights 2:1 lifted bar
//! accuracy on Raveform from 86-89% to 95%.

use crate::dsp::Bands;
use crate::fit::Activations;

pub struct Decision {
    /// Which beat index modulo 4, counted from t0, is bar 1.
    pub rotation: usize,
    /// Pooled log score of the winner minus the runner-up.
    pub margin: f64,
    /// [network, changes] votes, each normalised to sum 1.
    pub votes: [[f64; 4]; 2],
}

fn beat_indices(bpm: f64, t0: f64, duration: f64) -> Vec<i64> {
    let period = 60.0 / bpm;
    let k0 = (-t0 / period).ceil() as i64;
    let k1 = ((duration - t0) / period).floor() as i64;
    (k0..=k1).collect()
}

fn fold4(values: &[f64], k: &[i64]) -> [f64; 4] {
    let mut sum = [0.0f64; 4];
    let mut n = [0usize; 4];
    for (v, &ki) in values.iter().zip(k) {
        let r = ki.rem_euclid(4) as usize;
        sum[r] += v;
        n[r] += 1;
    }
    std::array::from_fn(|r| if n[r] > 0 { sum[r] / n[r] as f64 } else { 0.0 })
}

pub fn vote_network(act: &Activations, bpm: f64, t0: f64) -> [f64; 4] {
    let period = 60.0 / bpm;
    let k = beat_indices(bpm, t0, act.last_time());
    let last = act.downbeat.len() as i64 - 1;
    let strength: Vec<f64> = k
        .iter()
        .map(|&ki| {
            let frame = ((t0 + ki as f64 * period) * act.fps).round() as i64;
            (-1..=1).map(|d| act.downbeat[(frame + d).clamp(0, last) as usize] as f64).fold(f64::MIN, f64::max)
        })
        .collect();
    fold4(&strength, &k)
}

fn robust_z(x: &[f64]) -> Vec<f64> {
    let med = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let n = s.len();
        if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
    };
    let m = med(x);
    let mad = med(&x.iter().map(|v| (v - m).abs()).collect::<Vec<_>>());
    x.iter().map(|v| (v - m) / (mad + 1e-9)).collect()
}

pub fn vote_changes(bands: &Bands, bpm: f64, t0: f64) -> [f64; 4] {
    const TOP: usize = 12;
    let period = 60.0 / bpm;
    let duration = bands.energy.len() as f64 * bands.dt;
    let k = beat_indices(bpm, t0, duration);
    if k.len() < 32 {
        return [1.0; 4];
    }
    let starts: Vec<f64> = k.iter().map(|&ki| t0 + ki as f64 * period).collect();
    // Mean log band energy within each beat.
    let mut sums = vec![[0.0f64; 12]; k.len()];
    let mut counts = vec![0usize; k.len()];
    for (i, row) in bands.energy.iter().enumerate() {
        let centre = (i + 1) as f64 * bands.dt;
        let which = starts.partition_point(|&s| s < centre) as i64 - 1;
        if which >= 0 && (which as usize) < k.len() {
            for (s, e) in sums[which as usize].iter_mut().zip(row) {
                *s += *e as f64;
            }
            counts[which as usize] += 1;
        }
    }
    let mut level = Vec::with_capacity(k.len());
    let mut centroid = Vec::with_capacity(k.len());
    for (s, &c) in sums.iter().zip(&counts) {
        let c = c.max(1) as f64;
        let w: Vec<f64> = s.iter().map(|v| (v / c).exp()).collect();
        let total: f64 = w.iter().sum();
        level.push((total + 1e-9).ln());
        centroid.push(w.iter().enumerate().map(|(b, v)| b as f64 * v).sum::<f64>() / (total + 1e-9));
    }
    let (zl, zc) = (robust_z(&level), robust_z(&centroid));
    let step: Vec<f64> = (0..k.len())
        .map(|i| {
            let p = i.saturating_sub(1);
            (zl[i] - zl[p]).abs() + (zc[i] - zc[p]).abs()
        })
        .collect();
    let mut order: Vec<usize> = (0..step.len()).collect();
    order.sort_by(|&a, &b| step[b].total_cmp(&step[a]).then(b.cmp(&a)));
    let mut chosen: Vec<usize> = Vec::new();
    for i in order {
        if chosen.iter().all(|&j| i.abs_diff(j) > 2) {
            chosen.push(i);
        }
        if chosen.len() == TOP {
            break;
        }
    }
    let mut votes = [0.0f64; 4];
    for i in chosen {
        votes[k[i].rem_euclid(4) as usize] += step[i];
    }
    let total: f64 = votes.iter().sum();
    votes.map(|v| v + 1e-3 * total)
}

/// High-frequency onset bursts (crashes, new hat patterns) by the beat they land on: the 16
/// beats with the most 2-10 kHz onsets above their local average vote for their position.
pub fn vote_fills(bands: &Bands, bpm: f64, t0: f64) -> [f64; 4] {
    const TOP: usize = 16;
    const CLICK: [usize; 3] = [8, 9, 10];
    let period = 60.0 / bpm;
    let duration = bands.energy.len() as f64 * bands.dt;
    let k = beat_indices(bpm, t0, duration);
    if k.len() < 32 {
        return [1.0; 4];
    }
    let starts: Vec<f64> = k.iter().map(|&ki| t0 + ki as f64 * period).collect();
    // Positive change of log energy per frame, summed over the click bands, averaged per beat.
    let mut sums = vec![0.0f64; k.len()];
    let mut counts = vec![0usize; k.len()];
    for (i, row) in bands.energy.iter().enumerate() {
        let prev = if i == 0 { row } else { &bands.energy[i - 1] };
        let onset: f64 = CLICK.iter().map(|&b| ((row[b] - prev[b]) as f64).max(0.0)).sum();
        let centre = (i + 1) as f64 * bands.dt;
        let which = starts.partition_point(|&s| s < centre) as i64 - 1;
        if which >= 0 && (which as usize) < k.len() {
            sums[which as usize] += onset;
            counts[which as usize] += 1;
        }
    }
    let density: Vec<f64> = sums.iter().zip(&counts).map(|(s, &c)| s / c.max(1) as f64).collect();
    // Minus a 16-beat moving average (numpy convolve mode="same": window centred, zero-padded).
    let n = density.len() as i64;
    let local: Vec<f64> = (0..n)
        .map(|i| {
            let avg: f64 = (i - 8..i + 8).filter(|&j| j >= 0 && j < n).map(|j| density[j as usize]).sum::<f64>() / 16.0;
            density[i as usize] - avg
        })
        .collect();
    let mut order: Vec<usize> = (0..local.len()).collect();
    order.sort_by(|&a, &b| local[b].total_cmp(&local[a]).then(b.cmp(&a)));
    let mut votes = [0.0f64; 4];
    for &i in order.iter().take(TOP) {
        votes[k[i].rem_euclid(4) as usize] += local[i].max(0.0);
    }
    let total: f64 = votes.iter().sum();
    votes.map(|v| v + 1e-3 * (total + 1e-9))
}

pub fn decide(act: &Activations, bands: &Bands, bpm: f64, t0: f64, w_network: f64, w_changes: f64, w_fills: f64) -> Decision {
    let network = vote_network(act, bpm, t0);
    let changes = vote_changes(bands, bpm, t0);
    let fills = vote_fills(bands, bpm, t0);
    let normalise = |v: [f64; 4]| {
        let v = v.map(|x| x.max(1e-9));
        let s: f64 = v.iter().sum();
        v.map(|x| x / s)
    };
    let (pn, pc, pf) = (normalise(network), normalise(changes), normalise(fills));
    let pooled: [f64; 4] = std::array::from_fn(|r| w_network * pn[r].ln() + w_changes * pc[r].ln() + w_fills * pf[r].ln());
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &b| pooled[b].total_cmp(&pooled[a]));
    Decision { rotation: order[0], margin: pooled[order[0]] - pooled[order[1]], votes: [pn, pc] }
}
