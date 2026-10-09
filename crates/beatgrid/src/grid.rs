//! The grid itself: a constant tempo and where the bars fall, and the beats it implies.

use serde::{Deserialize, Serialize};

/// A constant-tempo grid in 4/4: `bpm`, and a downbeat (the first beat of a bar) at
/// `downbeat` seconds.
///
/// Times are seconds on the decoded timeline, with encoder delay and padding trimmed the
/// way ffmpeg trims them (the timeline DJ software stores grids on). `downbeat` can be any
/// downbeat of the grid, even one before the track starts; the methods below number beats
/// and bars from the first downbeat at or after time 0, so two grids that differ only in
/// which downbeat they store give the same beats.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    /// Tempo in beats per minute.
    pub bpm: f64,
    /// Time (s) of a downbeat.
    pub downbeat: f64,
}

/// One beat of a [`Grid`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Beat {
    /// Time in seconds.
    pub time: f64,
    /// Beat number: 0 is the first downbeat at or after time 0 ([`Grid::first_downbeat`]),
    /// negative before it.
    pub index: i64,
    /// Bar number: bar 1 starts at the first downbeat at or after time 0; beats before it
    /// (a pickup) are in bar 0.
    pub bar: i64,
    /// Position in the bar, 1 to 4; 1 is the downbeat.
    pub beat_in_bar: u8,
}

impl Grid {
    /// Seconds per beat.
    pub fn period(&self) -> f64 {
        60.0 / self.bpm
    }

    /// Time of the first beat at or after time 0.
    pub fn first_beat(&self) -> f64 {
        self.downbeat.rem_euclid(self.period())
    }

    /// Time of the first downbeat at or after time 0: beat 0, the start of bar 1.
    pub fn first_downbeat(&self) -> f64 {
        self.downbeat.rem_euclid(4.0 * self.period())
    }

    /// The beat number at `time`, fractional between beats: 0.0 at the first downbeat, 1.5
    /// halfway between its second and third beats. `beat_number_at(beat.time) == beat.index`.
    pub fn beat_number_at(&self, time: f64) -> f64 {
        (time - self.first_downbeat()) / self.period()
    }

    /// The time of beat number `beat` (fractional allowed); the inverse of
    /// [`beat_number_at`](Self::beat_number_at).
    pub fn time_of_beat(&self, beat: f64) -> f64 {
        self.first_downbeat() + beat * self.period()
    }

    /// Every beat with `start <= time < end`, in order. Use `beats(0.0, duration)` for a
    /// whole track.
    pub fn beats(&self, start: f64, end: f64) -> Vec<Beat> {
        let period = self.period();
        if !(period.is_finite() && period > 0.0 && start < end) {
            return Vec::new();
        }
        let first = self.first_downbeat();
        // One early, so rounding in the division never skips a beat exactly at `start`.
        let mut index = ((start - first) / period).ceil() as i64 - 1;
        let mut beats = Vec::new();
        loop {
            let time = first + index as f64 * period;
            if time >= end {
                return beats;
            }
            if time >= start {
                beats.push(Beat {
                    time,
                    index,
                    bar: index.div_euclid(4) + 1,
                    beat_in_bar: index.rem_euclid(4) as u8 + 1,
                });
            }
            index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beats_count_bars_from_the_first_downbeat() {
        // 120 BPM (0.5 s a beat) with a stored downbeat two bars before the track starts:
        // downbeats at 0.75, 2.75, ...
        let grid = Grid { bpm: 120.0, downbeat: 0.75 - 4.0 };
        assert!((grid.first_beat() - 0.25).abs() < 1e-12);
        assert!((grid.first_downbeat() - 0.75).abs() < 1e-12);
        let beats = grid.beats(0.0, 3.0);
        let summary: Vec<(i64, i64, u8)> = beats.iter().map(|b| (b.index, b.bar, b.beat_in_bar)).collect();
        assert_eq!(summary, vec![(-1, 0, 4), (0, 1, 1), (1, 1, 2), (2, 1, 3), (3, 1, 4), (4, 2, 1)]);
        assert!((beats[0].time - 0.25).abs() < 1e-12);
        assert!((beats[5].time - 2.75).abs() < 1e-12);
        for b in &beats {
            assert!((grid.beat_number_at(b.time) - b.index as f64).abs() < 1e-9);
            assert!((grid.time_of_beat(b.index as f64) - b.time).abs() < 1e-9);
        }
        // The same grid stored by another downbeat numbers its beats the same way.
        assert_eq!(Grid { bpm: 120.0, downbeat: 0.75 + 40.0 }.beats(0.0, 3.0), beats);
    }

    #[test]
    fn beats_are_half_open_and_empty_for_empty_ranges() {
        let grid = Grid { bpm: 120.0, downbeat: 0.0 };
        let beats = grid.beats(1.0, 2.0);
        assert_eq!(beats.iter().map(|b| b.time).collect::<Vec<_>>(), vec![1.0, 1.5]);
        assert!(grid.beats(2.0, 2.0).is_empty());
        assert!(grid.beats(2.0, 1.0).is_empty());
        assert!(grid.beats(f64::NAN, 1.0).is_empty());
    }
}
