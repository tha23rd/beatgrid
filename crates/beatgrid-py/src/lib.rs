//! Python bindings: `beatgrid.analyze(path)` and `beatgrid.analyze_many(paths)`.

use std::path::PathBuf;
use std::sync::OnceLock;

use beatgrid::{Analysis, Analyzer, Options, Profile, TempoMode};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// One analyzer per process: the network is loaded once and shared by every profile.
static ANALYZER: OnceLock<Analyzer> = OnceLock::new();

fn analyzer(profile: &str, tempo: &str) -> PyResult<Analyzer> {
    let profile: Profile = profile.parse().map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let tempo: TempoMode = tempo.parse().map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let shared = match ANALYZER.get() {
        Some(a) => a,
        None => {
            let a = Analyzer::new().map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
            ANALYZER.get_or_init(|| a)
        }
    };
    let mut a = shared.clone(); // shares the network
    a.set_options(Options { profile, tempo });
    Ok(a)
}

fn to_dict<'py>(py: Python<'py>, a: &Analysis) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("duration", a.duration)?;
    match (&a.grid, &a.segments) {
        (Some(g), Some(segmented)) => {
            let grid = PyDict::new(py);
            grid.set_item("bpm", g.bpm)?;
            grid.set_item("first_beat", g.first_beat())?;
            grid.set_item("first_downbeat", g.first_downbeat())?;
            // Every beat follows the segments: one, the constant grid, unless the tempo changes.
            let beats = segmented.beats(0.0, a.duration);
            grid.set_item("beats", beats.iter().map(|b| b.time).collect::<Vec<_>>())?;
            grid.set_item("is_downbeat", beats.iter().map(|b| b.beat_in_bar == 1).collect::<Vec<_>>())?;
            let segments = segmented
                .segments()
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let row = PyDict::new(py);
                    row.set_item("start", s.start)?;
                    row.set_item("bpm", s.bpm)?;
                    row.set_item("beat_in_bar", s.beat_in_bar)?;
                    row.set_item("beat", segmented.start_beat(i))?;
                    row.set_item("bar", segmented.start_bar(i))?;
                    Ok(row)
                })
                .collect::<PyResult<Vec<_>>>()?;
            grid.set_item("segments", segments)?;
            d.set_item("grid", grid)?;
        }
        _ => d.set_item("grid", py.None())?,
    }
    d.set_item("review", a.review.clone())?;
    if let Some(diag) = &a.diagnostics {
        d.set_item("snapped", diag.snapped)?;
        d.set_item("tempo_source", diag.tempo_source)?;
    }
    Ok(d)
}

/// Grid one audio file. Returns {"grid": {"bpm", "first_beat", "first_downbeat", "beats",
/// "is_downbeat", "segments"} | None, "duration", "review": [reasons to check it by ear; empty
/// when nothing looked doubtful], ...}.
///
/// profile: "none" (or "default") keeps the tempo octave the audio suggests; "edm" counts
/// drum & bass in half time and folds everything else into 100-200 BPM.
///
/// tempo: "constant" (one tempo), "auto" (constant-tempo segments where the track clearly
/// changes tempo) or "variable" (segments wherever the evidence suggests a change). "bpm",
/// "first_beat" and "first_downbeat" describe the best single-tempo grid; "beats" and
/// "is_downbeat" follow the segments, each {"start", "bpm", "beat_in_bar", "beat", "bar"}.
#[pyfunction]
#[pyo3(signature = (path, profile = "none", tempo = "constant"))]
fn analyze<'py>(py: Python<'py>, path: PathBuf, profile: &str, tempo: &str) -> PyResult<Bound<'py, PyDict>> {
    let analyzer = analyzer(profile, tempo)?;
    let a = py.detach(|| analyzer.analyze_file(&path)).map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    to_dict(py, &a)
}

/// Grid many files in parallel. Returns a list in the same order; a file that fails gives
/// {"error": message}.
#[pyfunction]
#[pyo3(signature = (paths, profile = "none", tempo = "constant"))]
fn analyze_many<'py>(py: Python<'py>, paths: Vec<PathBuf>, profile: &str, tempo: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let analyzer = analyzer(profile, tempo)?;
    let results: Vec<Result<Analysis, String>> =
        py.detach(|| paths.par_iter().map(|f| analyzer.analyze_file(f).map_err(|e| format!("{e:#}"))).collect());
    results
        .iter()
        .map(|r| match r {
            Ok(a) => to_dict(py, a),
            Err(e) => {
                let d = PyDict::new(py);
                d.set_item("error", e)?;
                Ok(d)
            }
        })
        .collect()
}

#[pymodule]
fn _beatgrid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_many, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
