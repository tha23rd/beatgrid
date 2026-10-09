//! The activation network: the distilled student (research/beatgrid_lab/student), run in
//! plain Rust. Input is Beat This!'s log-mel spectrogram (22.05 kHz, 1024-point periodic Hann
//! STFT, hop 441 = 50 fps, 128 Slaney mel bands 30 Hz - 11 kHz, log1p(1000 x)); output is
//! a beat and a downbeat probability per frame.

use anyhow::{anyhow, bail, Result};
use realfft::RealFftPlanner;
use safetensors::SafeTensors;

/// Sample rate (Hz) the network listens at.
pub const RATE: u32 = 22050;
/// Activation frames per second.
pub const FPS: f64 = 50.0;
const N_FFT: usize = 1024;
const HOP: usize = 441;
const MELS: usize = 128;

static WEIGHTS: &[u8] = include_bytes!("../models/student.safetensors");

/// Resample to 22.05 kHz, trimming the resampler's delay so frame 0 is still time 0.
pub fn resample(x: &[f32], rate: u32) -> Result<Vec<f32>> {
    use rubato::audioadapter_buffers::direct::SequentialSliceOfSlices;
    use rubato::{Fft, FixedSync, Resampler};

    if rate == RATE {
        return Ok(x.to_vec());
    }
    let mut resampler = Fft::<f32>::new(rate as usize, RATE as usize, 4096, 1, FixedSync::Input)?;
    let channels = [x];
    let input = SequentialSliceOfSlices::new(&channels[..], 1, x.len())?;
    Ok(resampler.process_all(&input, x.len(), None)?.take_data())
}

/// Slaney mel scale (as torchaudio and librosa define it).
fn hz_to_mel(f: f64) -> f64 {
    let sp = 200.0 / 3.0;
    let log_step = 6.4f64.ln() / 27.0;
    if f >= 1000.0 {
        1000.0 / sp + (f / 1000.0).ln() / log_step
    } else {
        f / sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let sp = 200.0 / 3.0;
    let log_step = 6.4f64.ln() / 27.0;
    let min_log_mel = 1000.0 / sp;
    if m >= min_log_mel {
        1000.0 * (log_step * (m - min_log_mel)).exp()
    } else {
        m * sp
    }
}

/// torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, norm=None, "slaney"):
/// (frequency bins x mel bands), triangular, unnormalised.
pub fn mel_filterbank() -> Vec<[f32; MELS]> {
    let bins = N_FFT / 2 + 1;
    let freqs: Vec<f64> = (0..bins).map(|i| (RATE / 2) as f64 * i as f64 / (bins - 1) as f64).collect();
    let (m0, m1) = (hz_to_mel(30.0), hz_to_mel(11000.0));
    let points: Vec<f64> = (0..MELS + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (MELS + 1) as f64)).collect();
    freqs
        .iter()
        .map(|&f| {
            std::array::from_fn(|m| {
                let down = (f - points[m]) / (points[m + 1] - points[m]);
                let up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
                down.min(up).max(0.0) as f32
            })
        })
        .collect()
}

/// Log-mel frames (frames x 128) of 22.05 kHz audio.
pub fn log_mel(y: &[f32]) -> Vec<[f32; MELS]> {
    let pad = N_FFT / 2;
    if y.len() <= pad {
        return Vec::new();
    }
    // Reflect padding (no edge repeat), as torch.stft(center=True) does.
    let n = y.len();
    let at = |i: isize| -> f32 {
        let j = if i < 0 { -i } else if i >= n as isize { 2 * (n as isize - 1) - i } else { i };
        y[j as usize]
    };
    let frames = 1 + n / HOP;
    let window: Vec<f32> = (0..N_FFT).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32).collect();
    // Each frequency bin feeds at most two (triangular) bands: keep only the non-zero
    // weights. Same additions in the same order, so the same result to the bit.
    let fb: Vec<Vec<(usize, f32)>> = mel_filterbank()
        .iter()
        .map(|row| row.iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(m, w)| (m, *w)).collect())
        .collect();
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let (mut input, mut output, mut scratch) = (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec());
    let norm = 1.0 / (N_FFT as f32).sqrt();
    let mut out = Vec::with_capacity(frames);
    for t in 0..frames {
        let start = (t * HOP) as isize - pad as isize;
        for (i, v) in input.iter_mut().enumerate() {
            *v = at(start + i as isize) * window[i];
        }
        fft.process_with_scratch(&mut input, &mut output, &mut scratch).expect("fft sizes match");
        let mut mel = [0.0f32; MELS];
        for (c, row) in output.iter().zip(&fb) {
            let m = c.norm() * norm;
            if m != 0.0 {
                for &(band, w) in row {
                    mel[band] += w * m;
                }
            }
        }
        out.push(mel.map(|v| (1000.0 * v).ln_1p()));
    }
    out
}

struct Conv {
    weight: Vec<f32>, // (out, in * kernel...) row-major, as PyTorch stores it
    bias: Vec<f32>,
    out: usize,
    cols: usize, // in * kernel elements
}

impl Conv {
    fn load(t: &SafeTensors, name: &str) -> Result<Self> {
        let get = |n: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let v = t.tensor(n).map_err(|e| anyhow!("{n}: {e}"))?;
            let data = v.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            Ok((v.shape().to_vec(), data))
        };
        let (shape, weight) = get(&format!("{name}.weight"))?;
        let (_, bias) = get(&format!("{name}.bias"))?;
        let out = shape[0];
        Ok(Conv { cols: weight.len() / out, weight, bias, out })
    }

    /// out (self.out x n) = weight (self.out x cols) . col (cols x n) + bias
    fn matmul(&self, col: &[f32], n: usize, out: &mut [f32]) {
        for (o, row) in out.chunks_exact_mut(n).enumerate() {
            row.fill(self.bias[o]);
        }
        unsafe {
            matrixmultiply::sgemm(
                self.out, self.cols, n, 1.0,
                self.weight.as_ptr(), self.cols as isize, 1,
                col.as_ptr(), n as isize, 1,
                1.0, out.as_mut_ptr(), n as isize, 1,
            );
        }
    }
}

fn elu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = v.exp_m1();
        }
    }
}

/// 2-D convolution over (channels, time, freq) with kernel (3, kf), stride (1, sf),
/// padding (1 in time when kt == 3, pf in frequency). Processed in time blocks.
#[allow(clippy::too_many_arguments)] // the shape of the convolution, spelled out
fn conv2d(x: &[f32], cin: usize, t: usize, f: usize, conv: &Conv, kt: usize, kf: usize, sf: usize, pf: usize) -> (Vec<f32>, usize) {
    let pt = kt / 2;
    let fo = (f + 2 * pf - kf) / sf + 1;
    let mut out = vec![0.0f32; conv.out * t * fo];
    let block = 1024;
    let mut col = Vec::new();
    let mut tmp = Vec::new();
    for t0 in (0..t).step_by(block) {
        let tb = block.min(t - t0);
        let n = tb * fo;
        col.clear();
        col.resize(conv.cols * n, 0.0);
        for ci in 0..cin {
            for a in 0..kt {
                for b in 0..kf {
                    let row = (ci * kt + a) * kf + b;
                    let dst = &mut col[row * n..(row + 1) * n];
                    for ti in 0..tb {
                        let ts = (t0 + ti) as isize + a as isize - pt as isize;
                        if ts < 0 || ts >= t as isize {
                            continue;
                        }
                        for o in 0..fo {
                            let fs = (o * sf + b) as isize - pf as isize;
                            if fs >= 0 && (fs as usize) < f {
                                dst[ti * fo + o] = x[(ci * t + ts as usize) * f + fs as usize];
                            }
                        }
                    }
                }
            }
        }
        tmp.clear();
        tmp.resize(conv.out * n, 0.0);
        conv.matmul(&col, n, &mut tmp);
        for co in 0..conv.out {
            out[(co * t + t0) * fo..(co * t + t0 + tb) * fo].copy_from_slice(&tmp[co * n..(co + 1) * n]);
        }
    }
    (out, fo)
}

/// 1-D convolution over (channels, time), kernel k, dilation d, 'same' padding.
fn conv1d(x: &[f32], cin: usize, t: usize, conv: &Conv, k: usize, d: usize) -> Vec<f32> {
    let pad = d * (k - 1) / 2;
    let mut col = vec![0.0f32; cin * k * t];
    for ci in 0..cin {
        for j in 0..k {
            let dst = &mut col[(ci * k + j) * t..(ci * k + j + 1) * t];
            let shift = (j * d) as isize - pad as isize;
            let src = &x[ci * t..(ci + 1) * t];
            let (lo, hi) = ((-shift).max(0) as usize, (t as isize - shift).min(t as isize).max(0) as usize);
            for ti in lo..hi {
                dst[ti] = src[(ti as isize + shift) as usize];
            }
        }
    }
    let mut out = vec![0.0f32; conv.out * t];
    conv.matmul(&col, t, &mut out);
    out
}

/// The distilled beat and downbeat network. Load it once and share it: it holds only its
/// weights, so running it from many threads at once is fine.
pub struct Student {
    front: [Conv; 3],
    blocks: Vec<(Conv, Conv, Conv)>,
    head: Conv,
    kernel: usize,
}

impl Student {
    /// The network built into the library.
    pub fn embedded() -> Result<Self> {
        Self::from_bytes(WEIGHTS)
    }

    /// A network of the same architecture from safetensors weights.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let t = SafeTensors::deserialize(bytes)?;
        let mut blocks = Vec::new();
        while t.tensor(&format!("tcn.{}.conv1.weight", blocks.len())).is_ok() {
            let i = blocks.len();
            blocks.push((
                Conv::load(&t, &format!("tcn.{i}.conv1"))?,
                Conv::load(&t, &format!("tcn.{i}.conv2"))?,
                Conv::load(&t, &format!("tcn.{i}.mix"))?,
            ));
        }
        if blocks.is_empty() {
            bail!("no TCN blocks in the weights");
        }
        let kernel = blocks[0].0.cols / blocks[0].0.out;
        Ok(Student {
            front: [Conv::load(&t, "front.0")?, Conv::load(&t, "front.2")?, Conv::load(&t, "front.4")?],
            blocks,
            head: Conv::load(&t, "head")?,
            kernel,
        })
    }

    /// Beat and downbeat probabilities per frame.
    pub fn run(&self, mel: &[[f32; MELS]]) -> (Vec<f32>, Vec<f32>) {
        let t = mel.len();
        if t == 0 {
            return (Vec::new(), Vec::new());
        }
        let x: Vec<f32> = mel.iter().flat_map(|row| row.iter().copied()).collect(); // (1, t, 128)
        let (mut x, f1) = conv2d(&x, 1, t, MELS, &self.front[0], 3, 3, 3, 1);
        elu(&mut x);
        let (mut x, f2) = conv2d(&x, self.front[0].out, t, f1, &self.front[1], 3, 3, 3, 1);
        elu(&mut x);
        let (mut x, _) = conv2d(&x, self.front[1].out, t, f2, &self.front[2], 1, f2, 1, 0);
        elu(&mut x);
        let channels = self.front[2].out;
        for (i, (c1, c2, mix)) in self.blocks.iter().enumerate() {
            let d = 1 << i;
            let mut y = conv1d(&x, channels, t, c1, self.kernel, d);
            y.extend(conv1d(&x, channels, t, c2, self.kernel, 2 * d));
            elu(&mut y);
            let m = conv1d(&y, 2 * channels, t, mix, 1, 1);
            for (a, b) in x.iter_mut().zip(m) {
                *a += b;
            }
        }
        let logits = conv1d(&x, channels, t, &self.head, 1, 1);
        let sigmoid = |v: &f32| 1.0 / (1.0 + (-v).exp());
        (logits[..t].iter().map(sigmoid).collect(), logits[t..2 * t].iter().map(sigmoid).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_mel_scale_inverts() {
        for f in [30.0, 440.0, 999.0, 1000.0, 5000.0, 11000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
    }

    #[test]
    fn embedded_network_runs_on_silence() {
        let net = Student::embedded().unwrap();
        let mel = log_mel(&vec![0.0f32; RATE as usize * 3]);
        assert_eq!(mel.len(), 1 + RATE as usize * 3 / 441);
        let (beat, down) = net.run(&mel);
        assert_eq!(beat.len(), mel.len());
        assert!(beat.iter().chain(&down).all(|p| (0.0..=1.0).contains(p)));
    }
}
