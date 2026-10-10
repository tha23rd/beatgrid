//! beatgrid: DJ-grade beat grids from the command line.

use std::collections::BTreeMap;
use std::io::{self, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{anyhow, bail, Context, Result};
use beatgrid::{Analysis, Analyzer, Options, Profile, SegmentedGrid, TempoMode};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use serde::Serialize;

/// Show a progress line (on a terminal's stderr) from this many files up.
const PROGRESS_MIN_FILES: usize = 8;

#[derive(Parser)]
#[command(
    name = "beatgrid",
    version,
    about = "DJ-grade beat grids: the exact tempo, every beat on the attack, bar 1 on the one",
    after_help = "Examples:\n  \
        beatgrid analyze track.mp3 other.flac\n  \
        beatgrid analyze --profile edm --format csv music/*.mp3 > grids.csv\n  \
        beatgrid analyze --tempo auto edit.mp3\n  \
        beatgrid beats track.mp3 > beats.csv"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Grid tracks: tempo, bar 1, and anything worth checking by ear.
    ///
    /// Prints a table on a terminal and one JSON object per line when piped (or what --format
    /// asks for). A file that fails is reported and the rest carry on; the exit code is 1 if
    /// any failed.
    #[command(after_help = "Examples:\n  \
        beatgrid analyze track.mp3 other.flac\n  \
        beatgrid analyze --profile edm music/*.mp3 > grids.jsonl\n  \
        beatgrid analyze --format csv --jobs 4 *.flac > grids.csv\n  \
        beatgrid analyze --tempo auto edit.mp3")]
    Analyze {
        /// Audio files: MP3, FLAC, WAV, Ogg Vorbis, AAC or ALAC (MP4)
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Output format [default: text on a terminal, json otherwise]
        #[arg(long, value_enum)]
        format: Option<Format>,
        /// Tempo octave convention
        #[arg(long, value_enum, default_value_t = ProfileArg::Default)]
        profile: ProfileArg,
        /// Whether the grid may change tempo
        #[arg(long, value_enum, default_value_t = TempoArg::Constant)]
        tempo: TempoArg,
        /// Worker threads [default: one per CPU]
        #[arg(long, short)]
        jobs: Option<usize>,
        /// Read DIR/<file stem>.act (f32le beat then downbeat probabilities at 50 fps)
        /// instead of running the network, to check the fit against the research pipeline.
        #[arg(long, value_name = "DIR", hide = true)]
        activations_dir: Option<PathBuf>,
        /// Set a fit parameter (research): --param seg_penalty=20
        #[arg(long = "param", value_name = "KEY=VALUE", hide = true)]
        params: Vec<String>,
    },
    /// Every beat of a track: time (s), bar number, and position in the bar (1 = downbeat).
    ///
    /// Bar 1 starts at the first downbeat at or after time 0; beats before it are bar 0.
    #[command(after_help = "Examples:\n  \
        beatgrid beats track.mp3 > beats.csv\n  \
        beatgrid beats --format json --profile edm track.flac\n  \
        beatgrid beats --tempo auto edit.mp3")]
    Beats {
        /// An audio file
        file: PathBuf,
        /// Output format
        #[arg(long, value_enum, default_value_t = BeatsFormat::Csv)]
        format: BeatsFormat,
        /// Tempo octave convention
        #[arg(long, value_enum, default_value_t = ProfileArg::Default)]
        profile: ProfileArg,
        /// Whether the grid may change tempo
        #[arg(long, value_enum, default_value_t = TempoArg::Constant)]
        tempo: TempoArg,
    },
    /// Decode to mono f32le on stdout (for checking the timeline against ffmpeg).
    #[command(hide = true)]
    Decode { file: PathBuf },
    /// The network's beat then downbeat probabilities (50 fps) as f32le on stdout.
    #[command(hide = true)]
    Activations { file: PathBuf },
    /// Log-mel (f32le, frames x 128) of raw 22.05 kHz f32le samples.
    #[command(hide = true)]
    DebugMel { raw: PathBuf },
    /// Network probabilities for a raw log-mel file (frames x 128 f32le).
    #[command(hide = true)]
    DebugNet { mel: PathBuf },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// An aligned table
    Text,
    /// One JSON object per file and line, with the full analysis
    Json,
    /// A header and one row per file
    Csv,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BeatsFormat {
    /// time,bar,beat_in_bar with a header
    Csv,
    /// An array of {"time", "bar", "beat_in_bar"}
    Json,
}

#[derive(Clone, Copy, ValueEnum)]
enum ProfileArg {
    /// Keep the tempo octave the audio suggests
    #[value(alias = "none")]
    Default,
    /// Drum & bass in half time, everything else 100-200 BPM
    Edm,
}

#[derive(Clone, Copy, ValueEnum)]
enum TempoArg {
    /// One tempo for the whole track
    Constant,
    /// Constant-tempo segments where the track clearly changes tempo (edits, mashups)
    Auto,
    /// Segments wherever the evidence suggests a change: for tracks known to change tempo
    Variable,
}

fn options(profile: ProfileArg, tempo: TempoArg) -> Options {
    Options {
        profile: match profile {
            ProfileArg::Default => Profile::Default,
            ProfileArg::Edm => Profile::Edm,
        },
        tempo: match tempo {
            TempoArg::Constant => TempoMode::Constant,
            TempoArg::Auto => TempoMode::Auto,
            TempoArg::Variable => TempoMode::Variable,
        },
    }
}

/// Floats stored as little-endian f32.
fn read_f32le(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn write_f32le<'a>(values: impl IntoIterator<Item = &'a f32>) -> Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(out.flush()?)
}

/// One line per file in the chosen format.
struct Table {
    format: Format,
    file_width: usize,
}

#[derive(Serialize)]
struct JsonRow<'a> {
    file: String,
    bpm: Option<f64>,
    first_beat: Option<f64>,
    first_downbeat: Option<f64>,
    segments: Option<&'a SegmentedGrid>,
    duration: f64,
    review: &'a [String],
    analysis: &'a Analysis,
}

#[derive(Serialize)]
struct JsonError<'a> {
    file: String,
    error: &'a str,
}

/// Segments for the CSV: `start@bpm` each (`start@bpm/beat` when one does not start on a
/// downbeat), separated by spaces.
fn segments_field(segments: &SegmentedGrid) -> String {
    let one = |s: &beatgrid::Segment| match s.beat_in_bar {
        1 => format!("{:.6}@{}", s.start, s.bpm),
        b => format!("{:.6}@{}/{b}", s.start, s.bpm),
    };
    segments.segments().iter().map(one).collect::<Vec<_>>().join(" ")
}

/// A CSV field, quoted when it needs to be.
fn csv(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

impl Table {
    fn new(format: Format, files: &[PathBuf]) -> Self {
        let file_width = files.iter().map(|f| f.display().to_string().chars().count()).max().unwrap_or(0).max(4);
        Table { format, file_width }
    }

    fn header(&self) -> Option<String> {
        match self.format {
            Format::Text => Some(format!("{:<w$}  {:>8}  {:>8}  REVIEW", "FILE", "BPM", "DOWNBEAT", w = self.file_width)),
            Format::Csv => Some("file,bpm,first_beat,first_downbeat,duration,review,error,segments".to_string()),
            Format::Json => None,
        }
    }

    fn row(&self, file: &Path, result: &Result<Analysis>) -> Result<String> {
        let name = file.display().to_string();
        let grid = result.as_ref().ok().and_then(|a| a.grid);
        Ok(match (self.format, result) {
            (Format::Json, Ok(a)) => serde_json::to_string(&JsonRow {
                file: name,
                bpm: grid.map(|g| g.bpm),
                first_beat: grid.map(|g| g.first_beat()),
                first_downbeat: grid.map(|g| g.first_downbeat()),
                segments: a.segments.as_ref(),
                duration: a.duration,
                review: &a.review,
                analysis: a,
            })?,
            (Format::Json, Err(e)) => serde_json::to_string(&JsonError { file: name, error: &format!("{e:#}") })?,
            (Format::Csv, Ok(a)) => {
                let num = |v: Option<f64>| v.map(|v| format!("{v:.6}")).unwrap_or_default();
                format!(
                    "{},{},{},{},{:.3},{},,{}",
                    csv(&name),
                    grid.map(|g| g.bpm.to_string()).unwrap_or_default(),
                    num(grid.map(|g| g.first_beat())),
                    num(grid.map(|g| g.first_downbeat())),
                    a.duration,
                    csv(&a.review.join("; ")),
                    a.segments.as_ref().map(segments_field).unwrap_or_default()
                )
            }
            (Format::Csv, Err(e)) => format!("{},,,,,,{},", csv(&name), csv(&format!("{e:#}"))),
            (Format::Text, _) => {
                let segmented = result.as_ref().ok().and_then(|a| a.segments.as_ref()).filter(|s| !s.is_constant());
                let (bpm, down) = match (segmented, grid) {
                    // A grid in segments: its first segment here, the others on lines of their own.
                    (Some(s), _) => (format!("{:.3}", s.segments()[0].bpm), format!("{:.3}", s.first_downbeat())),
                    (None, Some(g)) => (format!("{:.3}", g.bpm), format!("{:.3}", g.first_downbeat())),
                    (None, None) => ("-".to_string(), "-".to_string()),
                };
                let review = match result {
                    Err(e) => format!("error: {e:#}"),
                    Ok(a) if a.review.is_empty() => "ok".to_string(),
                    Ok(a) => a.review.join("; "),
                };
                let mut line = format!("{name:<w$}  {bpm:>8}  {down:>8}  {review}", w = self.file_width);
                for (i, s) in segmented.map(|s| s.segments()).unwrap_or_default().iter().enumerate().skip(1) {
                    let at = segmented.unwrap();
                    let from = match s.beat_in_bar {
                        1 => format!("from bar {}", at.start_bar(i)),
                        b => format!("from bar {} beat {b}", at.start_bar(i)),
                    };
                    line += &format!("\n{:<w$}  {:>8.3}  {:>8.3}  {from}", "", s.bpm, s.start, w = self.file_width);
                }
                line
            }
        })
    }
}

fn analyze(
    files: Vec<PathBuf>, format: Option<Format>, options: Options, jobs: Option<usize>, activations_dir: Option<PathBuf>,
    overrides: Vec<String>,
) -> Result<ExitCode> {
    let stdout_tty = io::stdout().is_terminal();
    let format = format.unwrap_or(if stdout_tty { Format::Text } else { Format::Json });
    let mut params = options.params();
    for item in &overrides {
        let (key, value) = item.split_once('=').ok_or_else(|| anyhow!("--param {item}: expected KEY=VALUE"))?;
        params.set(key, value)?;
    }
    let analyzer = Analyzer::with_params(params)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs.unwrap_or(0)).build()?;
    let run = |file: &Path| -> Result<Analysis> {
        let Some(dir) = &activations_dir else {
            return analyzer.analyze_file(file);
        };
        let audio = beatgrid::audio::decode(file)?;
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let values = read_f32le(&dir.join(format!("{stem}.act")))?;
        let (beat, downbeat) = values.split_at(values.len() / 2);
        analyzer.analyze_with_activations(&audio.samples, audio.rate, beat, &downbeat[..beat.len()], beatgrid::network::FPS)
    };

    let table = Table::new(format, &files);
    let mut out = io::stdout().lock();
    if let Some(h) = table.header() {
        writeln!(out, "{h}")?;
    }
    // Errors go to stderr too, unless they are already in a table on the same terminal.
    let errors_to_stderr = !(format == Format::Text && stdout_tty);
    let progress = io::stderr().is_terminal() && files.len() >= PROGRESS_MIN_FILES;
    let clear_progress = || {
        if progress {
            eprint!("\r\x1b[2K");
        }
    };
    let stop = AtomicBool::new(false);
    let mut failed = 0;
    // Grid in parallel; print in order as results come in.
    let printed = std::thread::scope(|scope| -> Result<()> {
        let (tx, rx) = mpsc::channel();
        scope.spawn(|| {
            pool.install(|| {
                files.par_iter().enumerate().for_each_with(tx, |tx, (i, file)| {
                    if !stop.load(Ordering::Relaxed) {
                        let _ = tx.send((i, run(file)));
                    }
                })
            })
        });
        let result = (|| -> Result<()> {
            let (mut pending, mut next, mut done) = (BTreeMap::new(), 0, 0);
            for (i, result) in &rx {
                pending.insert(i, result);
                done += 1;
                clear_progress();
                while let Some(result) = pending.remove(&next) {
                    if let Err(e) = &result {
                        failed += 1;
                        if errors_to_stderr {
                            eprintln!("beatgrid: {}: {e:#}", files[next].display());
                        }
                    }
                    writeln!(out, "{}", table.row(&files[next], &result)?)?;
                    next += 1;
                }
                out.flush()?;
                if progress && done < files.len() {
                    eprint!("gridded {done}/{} files", files.len());
                }
            }
            Ok(())
        })();
        if result.is_err() {
            stop.store(true, Ordering::Relaxed); // stdout went away: stop starting files
        }
        result
    });
    clear_progress();
    printed?;
    if failed > 0 {
        eprintln!("beatgrid: {failed} of {} files failed", files.len());
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

fn beats(file: PathBuf, format: BeatsFormat, options: Options) -> Result<()> {
    let analysis = Analyzer::with_options(options)?.analyze_file(&file)?;
    let grid = analysis.segments.as_ref().ok_or_else(|| anyhow!("{}: no steady beat to grid", file.display()))?;
    if analysis.needs_review() {
        eprintln!("beatgrid: {}: worth checking by ear: {}", file.display(), analysis.review.join("; "));
    }
    let beats = grid.beats(0.0, analysis.duration);
    let mut out = BufWriter::new(io::stdout().lock());
    match format {
        BeatsFormat::Csv => {
            writeln!(out, "time,bar,beat_in_bar")?;
            for b in &beats {
                writeln!(out, "{:.6},{},{}", b.time, b.bar, b.beat_in_bar)?;
            }
        }
        BeatsFormat::Json => {
            #[derive(Serialize)]
            struct Row {
                time: f64,
                bar: i64,
                beat_in_bar: u8,
            }
            let rows: Vec<Row> = beats.iter().map(|b| Row { time: b.time, bar: b.bar, beat_in_bar: b.beat_in_bar }).collect();
            serde_json::to_writer(&mut out, &rows)?;
            writeln!(out)?;
        }
    }
    Ok(out.flush()?)
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Analyze { files, format, profile, tempo, jobs, activations_dir, params } => {
            return analyze(files, format, options(profile, tempo), jobs, activations_dir, params)
        }
        Command::Beats { file, format, profile, tempo } => beats(file, format, options(profile, tempo))?,
        Command::Decode { file } => write_f32le(&beatgrid::audio::decode(&file)?.samples)?,
        Command::Activations { file } => {
            let audio = beatgrid::audio::decode(&file)?;
            let (beat, down) = Analyzer::new()?.activations(&audio.samples, audio.rate)?;
            write_f32le(beat.iter().chain(&down))?
        }
        Command::DebugMel { raw } => {
            let mel = beatgrid::network::log_mel(&read_f32le(&raw)?);
            write_f32le(mel.iter().flatten())?
        }
        Command::DebugNet { mel } => {
            let v = read_f32le(&mel)?;
            if v.len() % 128 != 0 {
                bail!("{} is not frames x 128 floats", mel.display());
            }
            let rows: Vec<[f32; 128]> = v.chunks_exact(128).map(|c| c.try_into().unwrap()).collect();
            let (beat, down) = beatgrid::network::Student::embedded()?.run(&rows);
            write_f32le(beat.iter().chain(&down))?
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        // The reader went away (`beatgrid analyze ... | head`): nothing left to say.
        Err(e) if e.chain().any(|c| c.downcast_ref::<io::Error>().is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)) => {
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("beatgrid: {e:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis(review: &[&str]) -> Analysis {
        let grid = beatgrid::Grid { bpm: 128.0, downbeat: -0.5 };
        Analysis {
            grid: Some(grid),
            segments: Some(grid.into()),
            duration: 180.0,
            review: review.iter().map(|r| r.to_string()).collect(),
            diagnostics: None,
        }
    }

    #[test]
    fn csv_fields_are_quoted_when_needed() {
        assert_eq!(csv("plain.mp3"), "plain.mp3");
        assert_eq!(csv("a, b.mp3"), "\"a, b.mp3\"");
        assert_eq!(csv("say \"hi\".mp3"), "\"say \"\"hi\"\".mp3\"");
    }

    #[test]
    fn rows_in_every_format() {
        let files = [PathBuf::from("a.mp3"), PathBuf::from("longer name.mp3")];
        let ok = Ok(analysis(&[]));
        let flagged = Ok(analysis(&["tempo may change", "bar 1 unclear"]));
        let failed = Err(anyhow!("no such file"));

        let text = Table::new(Format::Text, &files);
        assert_eq!(text.header().unwrap(), "FILE                  BPM  DOWNBEAT  REVIEW");
        assert_eq!(text.row(&files[0], &ok).unwrap(), "a.mp3             128.000     1.375  ok");
        assert!(text.row(&files[0], &flagged).unwrap().ends_with("  tempo may change; bar 1 unclear"));
        assert!(text.row(&files[1], &failed).unwrap().ends_with("-         -  error: no such file"));

        let csv = Table::new(Format::Csv, &files);
        assert_eq!(
            csv.row(&files[1], &flagged).unwrap(),
            "longer name.mp3,128,0.437500,1.375000,180.000,tempo may change; bar 1 unclear,,1.375000@128"
        );
        assert_eq!(csv.row(&files[0], &failed).unwrap(), "a.mp3,,,,,,no such file,");

        let json = Table::new(Format::Json, &files);
        let v: serde_json::Value = serde_json::from_str(&json.row(&files[0], &ok).unwrap()).unwrap();
        assert_eq!(v["file"], "a.mp3");
        assert_eq!(v["first_downbeat"], 1.375);
        assert_eq!(v["analysis"]["grid"]["bpm"], 128.0);
        let v: serde_json::Value = serde_json::from_str(&json.row(&files[0], &failed).unwrap()).unwrap();
        assert_eq!(v["error"], "no such file");
    }

    #[test]
    fn segments_print_in_every_format() {
        let files = [PathBuf::from("edit.mp3")];
        let segmented = beatgrid::SegmentedGrid::new(vec![
            beatgrid::Segment { start: 0.5, bpm: 124.0, beat_in_bar: 1 },
            beatgrid::Segment { start: 60.5, bpm: 150.0, beat_in_bar: 1 },
        ])
        .unwrap();
        let mut a = analysis(&[beatgrid::fit::TEMPO_SEGMENTS]);
        a.segments = Some(segmented);
        let ok = Ok(a);
        let text = Table::new(Format::Text, &files).row(&files[0], &ok).unwrap();
        assert_eq!(
            text,
            format!("edit.mp3   124.000     0.500  {}\n           150.000    60.500  from bar 32", beatgrid::fit::TEMPO_SEGMENTS)
        );
        let csv = Table::new(Format::Csv, &files).row(&files[0], &ok).unwrap();
        assert!(csv.ends_with(",,0.500000@124 60.500000@150"), "{csv}");
        let v: serde_json::Value = serde_json::from_str(&Table::new(Format::Json, &files).row(&files[0], &ok).unwrap()).unwrap();
        assert_eq!(v["segments"][1]["start"], 60.5);
        assert_eq!(v["segments"][1]["bpm"], 150.0);
        assert_eq!(v["segments"][1]["bar"], 32);
        assert_eq!(v["analysis"]["segments"][0]["beat_in_bar"], 1);
    }

    #[test]
    fn review_reasons_print_as_they_are() {
        let files = [PathBuf::from("a.mp3")];
        let octave = beatgrid::fit::OCTAVE_HALVED;
        let flagged = Ok(analysis(&[beatgrid::fit::NO_ATTACK, octave]));
        let both = format!("{}; {octave}", beatgrid::fit::NO_ATTACK);
        assert!(Table::new(Format::Text, &files).row(&files[0], &flagged).unwrap().ends_with(&format!("  {both}")));
        assert!(Table::new(Format::Csv, &files).row(&files[0], &flagged).unwrap().ends_with(&format!(",{both},,1.375000@128")));
        let v: serde_json::Value = serde_json::from_str(&Table::new(Format::Json, &files).row(&files[0], &flagged).unwrap()).unwrap();
        assert_eq!(v["review"][1], octave);
    }
}
