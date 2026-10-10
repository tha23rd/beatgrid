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

/// One constant-tempo stretch of a [`SegmentedGrid`]: from a beat at `start` on, one beat
/// every `60 / bpm` seconds until the next segment starts. The same three numbers DJ software
/// stores per tempo marker.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// Time (s) of the segment's first beat.
    pub start: f64,
    /// Tempo in beats per minute.
    pub bpm: f64,
    /// Where the first beat falls in its bar, 1 to 4 (1: a downbeat). beatgrid starts every
    /// segment it finds on a downbeat.
    pub beat_in_bar: u8,
}

impl Segment {
    /// Seconds per beat.
    pub fn period(&self) -> f64 {
        60.0 / self.bpm
    }
}

/// Why [`SegmentedGrid::new`] refused its segments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentError(String);

impl std::fmt::Display for SegmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SegmentError {}

/// A grid whose tempo changes: constant-tempo [`Segment`]s one after another, the way DJ
/// software stores a grid with several tempo markers. A transition edit from 124 to 150 BPM is
/// two segments.
///
/// Segment *i* holds the beats `start + j * period` for `j >= 0`, up to the last one at least
/// half a beat before segment *i + 1* starts. The first segment also runs back before its
/// start, and the last one on forever, so a grid with one segment is a constant [`Grid`]
/// (`SegmentedGrid::from(grid)` gives the same beats).
///
/// Beat numbers run on across segments without gaps, 0 being the first downbeat at or after
/// time 0, as for [`Grid`]. Bars count on too: a new bar starts at every downbeat, and at any
/// beat whose position in the bar does not follow on from the beat before (a segment that
/// starts on its downbeat after a bar was cut short).
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentedGrid {
    segments: Vec<Segment>,
    /// Beats in each segment but the last (which has no end).
    counts: Vec<i64>,
    /// Number of each segment's first beat.
    first: Vec<i64>,
    /// Bar of each segment's first beat.
    bars: Vec<i64>,
}

impl SegmentedGrid {
    /// A grid from its segments, in order. Every tempo must be positive, every position in the
    /// bar 1 to 4, and every segment must start at least half a beat (of the segment before)
    /// after the one before.
    pub fn new(segments: Vec<Segment>) -> Result<Self, SegmentError> {
        if segments.is_empty() {
            return Err(SegmentError("a segmented grid needs at least one segment".into()));
        }
        for s in &segments {
            if !(s.start.is_finite() && s.bpm.is_finite() && s.bpm > 0.0 && (1..=4).contains(&s.beat_in_bar)) {
                return Err(SegmentError(format!("bad segment {s:?}")));
            }
        }
        for w in segments.windows(2) {
            if w[1].start - w[0].start < w[0].period() / 2.0 {
                return Err(SegmentError(format!(
                    "the segment at {} s starts less than half a beat after the one at {} s",
                    w[1].start, w[0].start
                )));
            }
        }
        let counts: Vec<i64> = segments
            .windows(2)
            .map(|w| ((w[1].start - w[0].period() / 2.0 - w[0].start) / w[0].period()).floor() as i64 + 1)
            .collect();
        let mut first = vec![0i64; segments.len()];
        for i in 1..segments.len() {
            first[i] = first[i - 1] + counts[i - 1];
        }
        let mut grid = SegmentedGrid { segments, counts, first, bars: Vec::new() };
        // Number from the first downbeat at or after time 0.
        let zero = grid.first_downbeat_number();
        grid.first.iter_mut().for_each(|f| *f -= zero);
        // Bars, from each segment's first beat on.
        let n = grid.segments.len();
        let mut bars = vec![0i64; n];
        for i in 1..n {
            let last = grid.segments[i - 1].beat_in_bar as i64 - 1 + grid.counts[i - 1] - 1;
            let (last_bar, last_pos) = (bars[i - 1] + last.div_euclid(4), last.rem_euclid(4) + 1);
            bars[i] = last_bar + i64::from(grid.segments[i].beat_in_bar as i64 <= last_pos);
        }
        grid.bars = bars;
        let (i, j) = grid.locate(0);
        let bar0 = grid.bars[i] + (grid.segments[i].beat_in_bar as i64 - 1 + j).div_euclid(4);
        grid.bars.iter_mut().for_each(|b| *b += 1 - bar0);
        Ok(grid)
    }

    /// The segments, in order.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Whether the grid has one tempo throughout (one segment).
    pub fn is_constant(&self) -> bool {
        self.segments.len() == 1
    }

    /// The grid as a constant [`Grid`], when it has one segment.
    pub fn constant(&self) -> Option<Grid> {
        self.is_constant().then(|| Grid { bpm: self.segments[0].bpm, downbeat: self.first_downbeat() })
    }

    /// Beat number of segment `i`'s first beat.
    pub fn start_beat(&self, i: usize) -> i64 {
        self.first[i]
    }

    /// Bar number of segment `i`'s first beat (bar 1 starts at beat 0).
    pub fn start_bar(&self, i: usize) -> i64 {
        self.bars[i]
    }

    /// The segment that holds `time`: the last one starting at or before it (the first one for
    /// times before it starts).
    pub fn segment_at(&self, time: f64) -> usize {
        self.segments.iter().rposition(|s| s.start <= time).unwrap_or(0)
    }

    /// The tempo at `time`.
    pub fn bpm_at(&self, time: f64) -> f64 {
        self.segments[self.segment_at(time)].bpm
    }

    /// Time of the first beat at or after time 0.
    pub fn first_beat(&self) -> f64 {
        let mut n = self.beat_number_at(0.0).ceil() - 1.0;
        while self.time_of_beat(n) < 0.0 {
            n += 1.0;
        }
        self.time_of_beat(n)
    }

    /// Time of the first downbeat at or after time 0: beat 0, the start of bar 1.
    pub fn first_downbeat(&self) -> f64 {
        self.time_of_beat(0.0)
    }

    /// Time of segment `i`'s last beat (segments but the last).
    fn last_time(&self, i: usize) -> f64 {
        let s = &self.segments[i];
        s.start + (self.counts[i] - 1) as f64 * s.period()
    }

    /// The beat number at `time`, fractional between beats; between the last beat of a segment
    /// and the first of the next it runs evenly. `beat_number_at(beat.time) == beat.index`.
    pub fn beat_number_at(&self, time: f64) -> f64 {
        let i = self.segment_at(time);
        let s = &self.segments[i];
        if i + 1 == self.segments.len() || time <= self.last_time(i) {
            return self.first[i] as f64 + (time - s.start) / s.period();
        }
        let last = self.last_time(i);
        (self.first[i] + self.counts[i] - 1) as f64 + (time - last) / (self.segments[i + 1].start - last)
    }

    /// The time of beat number `beat` (fractional allowed); the inverse of
    /// [`beat_number_at`](Self::beat_number_at).
    pub fn time_of_beat(&self, beat: f64) -> f64 {
        let i = self.first.iter().rposition(|&f| f as f64 <= beat).unwrap_or(0);
        let s = &self.segments[i];
        if i + 1 == self.segments.len() || beat <= (self.first[i] + self.counts[i] - 1) as f64 {
            return s.start + (beat - self.first[i] as f64) * s.period();
        }
        let last = self.last_time(i);
        last + (beat - (self.first[i] + self.counts[i] - 1) as f64) * (self.segments[i + 1].start - last)
    }

    /// (segment, beats into it) of beat number `n`.
    fn locate(&self, n: i64) -> (usize, i64) {
        let i = self.first.iter().rposition(|&f| f <= n).unwrap_or(0);
        (i, n - self.first[i])
    }

    fn first_downbeat_number(&self) -> i64 {
        let mut n = self.beat_number_at(0.0).ceil() as i64 - 1;
        loop {
            let (i, j) = self.locate(n);
            if self.time_of_beat(n as f64) >= 0.0 && (self.segments[i].beat_in_bar as i64 - 1 + j).rem_euclid(4) == 0 {
                return n;
            }
            n += 1;
        }
    }

    /// Every beat with `start <= time < end`, in order.
    pub fn beats(&self, start: f64, end: f64) -> Vec<Beat> {
        if start.partial_cmp(&end) != Some(std::cmp::Ordering::Less) {
            return Vec::new();
        }
        // One early, so rounding never skips a beat exactly at `start`.
        let mut index = self.beat_number_at(start).ceil() as i64 - 1;
        let mut beats = Vec::new();
        loop {
            let (i, j) = self.locate(index);
            let s = &self.segments[i];
            let time = s.start + j as f64 * s.period();
            if time >= end {
                return beats;
            }
            if time >= start {
                let pos = s.beat_in_bar as i64 - 1 + j;
                beats.push(Beat {
                    time,
                    index,
                    bar: self.bars[i] + pos.div_euclid(4),
                    beat_in_bar: pos.rem_euclid(4) as u8 + 1,
                });
            }
            index += 1;
        }
    }
}

impl From<Grid> for SegmentedGrid {
    /// A constant grid as one segment, starting at its first downbeat.
    fn from(grid: Grid) -> Self {
        SegmentedGrid::new(vec![Segment { start: grid.first_downbeat(), bpm: grid.bpm, beat_in_bar: 1 }])
            .expect("a constant grid is one valid segment")
    }
}

/// How a segment is written out: the segment, and the number and bar of its first beat.
#[derive(Serialize, Deserialize)]
struct NumberedSegment {
    start: f64,
    bpm: f64,
    beat_in_bar: u8,
    #[serde(default, skip_deserializing)]
    beat: i64,
    #[serde(default, skip_deserializing)]
    bar: i64,
}

impl Serialize for SegmentedGrid {
    /// A list of {"start", "bpm", "beat_in_bar", "beat", "bar"}: each segment, with the number
    /// and bar of its first beat.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let rows: Vec<NumberedSegment> = self
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| NumberedSegment {
                start: s.start,
                bpm: s.bpm,
                beat_in_bar: s.beat_in_bar,
                beat: self.first[i],
                bar: self.bars[i],
            })
            .collect();
        rows.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SegmentedGrid {
    /// The list [`Serialize`] writes; "beat" and "bar" are recomputed.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let rows = Vec::<NumberedSegment>::deserialize(deserializer)?;
        let segments = rows.into_iter().map(|r| Segment { start: r.start, bpm: r.bpm, beat_in_bar: r.beat_in_bar });
        SegmentedGrid::new(segments.collect()).map_err(serde::de::Error::custom)
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

    fn summary(beats: &[Beat]) -> Vec<(i64, i64, u8)> {
        beats.iter().map(|b| (b.index, b.bar, b.beat_in_bar)).collect()
    }

    #[test]
    fn a_constant_grid_is_one_segment() {
        for grid in [Grid { bpm: 120.0, downbeat: 0.75 - 4.0 }, Grid { bpm: 127.0, downbeat: 1.889441 }, Grid { bpm: 87.0, downbeat: -0.01 }] {
            let segmented = SegmentedGrid::from(grid);
            assert!(segmented.is_constant());
            assert_eq!(segmented.beats(0.0, 30.0), grid.beats(0.0, 30.0));
            assert_eq!(segmented.beats(7.3, 9.1), grid.beats(7.3, 9.1));
            assert!((segmented.first_beat() - grid.first_beat()).abs() < 1e-12);
            assert!((segmented.first_downbeat() - grid.first_downbeat()).abs() < 1e-12);
            for t in [0.0, 0.3, 5.0, 29.9] {
                assert!((segmented.beat_number_at(t) - grid.beat_number_at(t)).abs() < 1e-9);
                assert!((segmented.time_of_beat(t) - grid.time_of_beat(t)).abs() < 1e-9);
            }
            assert_eq!(segmented.constant().unwrap().beats(0.0, 30.0), grid.beats(0.0, 30.0));
        }
    }

    #[test]
    fn splitting_a_grid_on_a_beat_changes_nothing() {
        let grid = Grid { bpm: 120.0, downbeat: 0.75 };
        // A second segment from beat 10 (the third beat of bar 3), at the same tempo.
        let split = SegmentedGrid::new(vec![
            Segment { start: 0.75, bpm: 120.0, beat_in_bar: 1 },
            Segment { start: 0.75 + 10.0 * 0.5, bpm: 120.0, beat_in_bar: 3 },
        ])
        .unwrap();
        assert_eq!((split.start_beat(1), split.start_bar(1)), (10, 3));
        let (a, b) = (split.beats(0.0, 20.0), grid.beats(0.0, 20.0));
        assert_eq!(summary(&a), summary(&b));
        assert!(a.iter().zip(&b).all(|(x, y)| (x.time - y.time).abs() < 1e-12));
    }

    #[test]
    fn a_tempo_change_numbers_on_and_starts_a_bar() {
        // 124 BPM for 7.5 bars (30 beats from 0.3 s), then 150 BPM from a downbeat 0.1 s
        // later than a 31st beat would have been: a cut bar of two beats.
        let p = 60.0 / 124.0;
        let change = 0.3 + 30.0 * p + 0.1;
        let grid = SegmentedGrid::new(vec![
            Segment { start: 0.3, bpm: 124.0, beat_in_bar: 1 },
            Segment { start: change, bpm: 150.0, beat_in_bar: 1 },
        ])
        .unwrap();
        assert_eq!(grid.segment_at(change - 0.01), 0);
        assert_eq!(grid.bpm_at(change), 150.0);
        let beats = grid.beats(13.5, change + 1.0);
        assert_eq!(summary(&beats), vec![(28, 8, 1), (29, 8, 2), (30, 9, 1), (31, 9, 2), (32, 9, 3)]);
        assert!((beats[1].time - (0.3 + 29.0 * p)).abs() < 1e-12);
        assert!((beats[2].time - change).abs() < 1e-12);
        assert!((beats[3].time - (change + 0.4)).abs() < 1e-12);
        // Beat numbers run evenly across the join, and back.
        for t in [0.0, 5.0, 14.4, 14.6, 14.85, change, 20.0, 100.0] {
            let n = grid.beat_number_at(t);
            assert!((grid.time_of_beat(n) - t).abs() < 1e-9, "{t}");
        }
        assert!((grid.beat_number_at(change - 0.5 * (change - beats[1].time)) - 29.5).abs() < 1e-9);
        for b in &beats {
            assert!((grid.beat_number_at(b.time) - b.index as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn segments_must_be_valid() {
        let seg = |start: f64, bpm: f64, beat_in_bar: u8| Segment { start, bpm, beat_in_bar };
        assert!(SegmentedGrid::new(vec![]).is_err());
        assert!(SegmentedGrid::new(vec![seg(0.0, 0.0, 1)]).is_err());
        assert!(SegmentedGrid::new(vec![seg(0.0, 120.0, 5)]).is_err());
        assert!(SegmentedGrid::new(vec![seg(f64::NAN, 120.0, 1)]).is_err());
        // Closer than half a beat (0.25 s at 120) to the segment before.
        assert!(SegmentedGrid::new(vec![seg(1.0, 120.0, 1), seg(1.2, 140.0, 1)]).is_err());
        assert!(SegmentedGrid::new(vec![seg(1.0, 120.0, 1), seg(1.25, 140.0, 1)]).is_ok());
    }

    #[test]
    fn segmented_grids_serialize_with_their_numbers() {
        let grid = SegmentedGrid::new(vec![
            Segment { start: 0.5, bpm: 120.0, beat_in_bar: 1 },
            Segment { start: 8.6, bpm: 150.0, beat_in_bar: 1 },
        ])
        .unwrap();
        let json = serde_json::to_value(&grid).unwrap();
        assert_eq!(json[1]["beat"], 16);
        assert_eq!(json[1]["bar"], 5);
        assert_eq!(json[1]["bpm"], 150.0);
        let back: SegmentedGrid = serde_json::from_value(json).unwrap();
        assert_eq!(back, grid);
        assert!(serde_json::from_str::<SegmentedGrid>("[]").is_err());
    }
}
