//! Decoding to mono f32 at the file's own sample rate.
//!
//! The timeline matters more than anything else here: a grid is a list of times, and DJ
//! software stores them on the decoder-trimmed timeline (encoder delay and padding removed,
//! as ffmpeg does). A decoder that keeps the delay shifts every beat by 25-50 ms.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::TrackType;
use symphonia::core::io::MediaSourceStream;

/// Decoded mono audio.
pub struct Audio {
    /// Mono samples, downmixed as [`downmix`] does.
    pub samples: Vec<f32>,
    /// Sample rate (Hz), the file's own.
    pub rate: u32,
}

impl Audio {
    /// Length in seconds.
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.rate as f64
    }
}

/// Interleaved frames of `channels` channels to mono the way ffmpeg's `-ac 1` does, which
/// the research pipeline and the network were built on: stereo is (L + R) / sqrt 2, not the
/// average; more channels are averaged.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let scale = match channels {
        0 => return Vec::new(),
        1 => return interleaved.to_vec(),
        2 => std::f32::consts::FRAC_1_SQRT_2,
        n => 1.0 / n as f32,
    };
    interleaved.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() * scale).collect()
}

/// Decode an audio file (anything symphonia reads: MP3, FLAC, WAV, Ogg Vorbis, AAC/ALAC in
/// MP4, ...) to mono at its own sample rate, on ffmpeg's timeline.
pub fn decode(path: &Path) -> Result<Audio> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, stream, Default::default(), Default::default())
        .with_context(|| format!("unrecognised audio in {}", path.display()))?;
    let (track, params) = {
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| anyhow!("no audio track in {}", path.display()))?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| anyhow!("no codec parameters in {}", path.display()))?
            .clone();
        (track.id, params)
    };
    let rate = params.sample_rate.ok_or_else(|| anyhow!("no sample rate in {}", path.display()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .with_context(|| format!("no decoder for {}", path.display()))?;

    let mut samples = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        if packet.track_id != track {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(SymphoniaError::DecodeError(_)) => continue, // skip a corrupt packet, as ffmpeg does
            Err(e) => return Err(e).with_context(|| format!("decoding {}", path.display())),
        };
        let channels = buf.spec().channels().count();
        if channels == 0 {
            continue;
        }
        interleaved.resize(buf.samples_interleaved(), 0.0);
        buf.copy_to_slice_interleaved(&mut interleaved);
        samples.extend(downmix(&interleaved, channels));
    }
    Ok(Audio { samples, rate })
}
