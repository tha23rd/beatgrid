//! Spectral features computed from the decoded audio. Each is a direct port of the research
//! pipeline (research/beatgrid_lab/pipeline.py and track.py) and is checked against it.

use realfft::RealFftPlanner;

/// Symmetric Hann window (numpy's `hanning`).
pub fn hanning(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()) as f32)
        .collect()
}

/// Magnitude (or power) spectra of every frame, handed to `each` one frame at a time.
fn frames(x: &[f32], n: usize, hop: usize, mut each: impl FnMut(usize, &[f32])) -> usize {
    if x.len() < n {
        return 0;
    }
    let count = (x.len() - n) / hop + 1;
    let window = hanning(n);
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    let mut magnitude = vec![0.0f32; n / 2 + 1];
    for f in 0..count {
        let start = f * hop;
        for (i, v) in input.iter_mut().enumerate() {
            *v = x[start + i] * window[i];
        }
        fft.process_with_scratch(&mut input, &mut output, &mut scratch).expect("fft sizes match");
        for (m, c) in magnitude.iter_mut().zip(&output) {
            *m = c.norm();
        }
        each(f, &magnitude);
    }
    count
}

/// Onset flux: rise of log-magnitude spectra (log1p(100 |X|)), window 512, hop 32 samples.
pub struct Flux {
    pub values: Vec<f32>,
    /// Seconds per value.
    pub dt: f64,
    /// Time of the first value: between the centres of the two frames it compares.
    pub t_first: f64,
}

pub const FLUX_WINDOW: usize = 512;
pub const FLUX_HOP: usize = 32;

pub fn onset_flux(x: &[f32], rate: u32) -> Flux {
    use rayon::prelude::*;

    let (n, hop) = (FLUX_WINDOW, FLUX_HOP);
    let bins = n / 2 + 1;
    let dt = hop as f64 / rate as f64;
    let t_first = (n as f64 / 2.0 + hop as f64 / 2.0) / rate as f64;
    if x.len() < n + hop {
        return Flux { values: vec![0.0], dt, t_first };
    }
    let count = (x.len() - n) / hop + 1;
    let window = hanning(n);
    // Blocks of frames in parallel; each block also computes the frame before it so the
    // first difference in the block is right.
    const BLOCK: usize = 4096;
    let blocks: Vec<Vec<f32>> = (1..count)
        .step_by(BLOCK)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&first| {
            let last = (first + BLOCK).min(count);
            let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
            let (mut input, mut output, mut scratch) = (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec());
            let mut spectrum = |f: usize, out: &mut [f32]| {
                let start = f * hop;
                for (i, v) in input.iter_mut().enumerate() {
                    *v = x[start + i] * window[i];
                }
                fft.process_with_scratch(&mut input, &mut output, &mut scratch).expect("fft sizes match");
                for (o, c) in out.iter_mut().zip(&output) {
                    *o = (100.0 * c.norm()).ln_1p();
                }
            };
            let mut previous = vec![0.0f32; bins];
            let mut current = vec![0.0f32; bins];
            spectrum(first - 1, &mut previous);
            let mut values = Vec::with_capacity(last - first);
            for f in first..last {
                spectrum(f, &mut current);
                values.push(current.iter().zip(&previous).map(|(c, p)| (c - p).max(0.0)).sum());
                std::mem::swap(&mut previous, &mut current);
            }
            values
        })
        .collect();
    Flux { values: blocks.concat(), dt, t_first }
}

/// Log energy in 12 log-spaced bands (40 Hz - 16 kHz), window 2048, hop 1024.
/// Frame i is centred at (i + 1) * dt.
pub struct Bands {
    pub energy: Vec<[f32; 12]>,
    pub dt: f64,
}

pub fn band_energy(x: &[f32], rate: u32) -> Bands {
    let (n, hop) = (2048usize, 1024usize);
    let bins = n / 2 + 1;
    // numpy: band = clip(searchsorted(geomspace(40, 16000, 13), f) - 1, -1, 11)
    let edges: Vec<f64> = (0..13).map(|i| 40.0 * (16000.0f64 / 40.0).powf(i as f64 / 12.0)).collect();
    let band_of: Vec<i32> = (0..bins)
        .map(|b| {
            let f = b as f64 * rate as f64 / n as f64;
            let left = edges.partition_point(|&e| e < f) as i32; // numpy side='left'
            (left - 1).clamp(-1, 11)
        })
        .collect();
    let mut energy = Vec::new();
    frames(x, n, hop, |_, magnitude| {
        let mut row = [0.0f64; 12];
        for (m, &band) in magnitude.iter().zip(&band_of) {
            if band >= 0 {
                row[band as usize] += (*m as f64) * (*m as f64);
            }
        }
        let mut out = [0.0f32; 12];
        for (o, r) in out.iter_mut().zip(row) {
            *o = (r as f32 + 1e-8).ln();
        }
        energy.push(out);
    });
    Bands { energy, dt: hop as f64 / rate as f64 }
}
