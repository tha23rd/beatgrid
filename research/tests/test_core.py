import struct
import zlib

import numpy as np
import pytest

from beatgrid_lab.activations import Activations
from beatgrid_lab.grid import Grid
from beatgrid_lab.metrics import at_reference_octave, score
from beatgrid_lab import pipeline
from beatgrid_lab.pipeline import Params, attack_offset, onset_flux, review_reasons, run, second_attack, self_audit
from beatgrid_lab.refs.engine import decode_beat_data, to_grid


class FakeTrack:
    """Activations shaped like the network's: a narrow bump on each beat, sampled at 50 fps,
    with beats deliberately falling between frames."""

    def __init__(self, bpm, first_downbeat, seconds=240.0, jitter_ms=0.0, seed=0):
        rng = np.random.default_rng(seed)
        fps = 50.0
        t = np.arange(int(seconds * fps)) / fps
        period = 60.0 / bpm
        beats = first_downbeat + period * np.arange(int((seconds - first_downbeat) / period))
        beats = beats + rng.normal(0, jitter_ms / 1000, len(beats))
        bump = lambda centres: np.clip(sum(np.exp(-0.5 * ((t - c) * fps / 0.7) ** 2) for c in centres), 0, 1)
        self.activations = Activations(bump(beats), bump(beats[::4]), fps)
        self.truth = Grid.constant(bpm, first_downbeat)
        self.seconds = seconds

    def envelope(self, cutoff_hz):
        raise AssertionError("activation placement only")


def test_grid_roundtrip_and_beats():
    g = Grid.constant(128.0, 0.5)
    n, t = g.beats(0.0, 10.0)
    assert np.allclose(np.diff(t), 60 / 128)
    assert np.allclose(g.number_at(g.time_of(n)), n)
    assert t[n % 4 == 0][0] == pytest.approx(0.5)
    assert g.time_of(4.0)[0] == pytest.approx(0.5 + 4 * 60 / 128)
    assert g.number_at(0.5)[0] == pytest.approx(0.0)


def test_engine_blob_decodes():
    markers = [(-84333.85, -4, 765, 108), (16044515.21, 761, 0, 0)]
    body = struct.pack(">ddB", 44100.0, 16_100_000.0, 1)
    for _ in range(2):
        body += struct.pack(">q", len(markers)) + b"".join(struct.pack("<dqii", *m) for m in markers)
    blob = struct.pack(">I", len(body)) + zlib.compress(body)
    rate, count, default, adjusted = decode_beat_data(blob)
    grid = to_grid(adjusted, rate)
    assert rate == 44100.0 and len(default) == 2
    assert grid.bpm == pytest.approx(60 * 44100 * 765 / (16044515.21 + 84333.85))


def test_metrics_tolerances_and_octaves():
    ref = Grid.constant(126.0, 0.25)
    assert score(ref, ref, 0, 200).usable_10
    late = score(ref.shifted(0.015), ref, 0, 200)
    assert late.usable_20 and not late.usable_10 and late.bias_ms == pytest.approx(15, abs=0.01)
    assert not score(ref.renumbered(1), ref, 0, 200).bar_ok
    double = Grid.constant(252.0, 0.25)
    assert score(double, ref, 0, 200).octave == "2x"
    assert score(at_reference_octave(double, ref, 0, 200), ref, 0, 200).usable_10


@pytest.mark.parametrize("bpm,downbeat", [(128.0, 0.313), (126.5, 1.007), (174.0, 0.05), (122.0, 2.219)])
def test_pipeline_recovers_exact_tempo_phase_and_bar_from_20ms_frames(bpm, downbeat):
    track = FakeTrack(bpm, downbeat, jitter_ms=3.0, seed=int(bpm))
    result = run(track, Params(placement="activation", tempo_by_flux=False, offset_ms=0.0))
    s = score(result.grid, track.truth, 1.0, track.seconds - 1.0)
    assert result.diagnostics["snapped"]
    assert result.grid.bpm == pytest.approx(bpm, abs=1e-9)
    assert s.bar_ok
    assert s.median_abs_ms < 2.0, s


def test_pipeline_keeps_an_off_grid_tempo():
    track = FakeTrack(127.37, 0.4)
    result = run(track, Params(placement="activation", tempo_by_flux=False, offset_ms=0.0))
    assert not result.diagnostics["snapped"]
    assert result.grid.bpm == pytest.approx(127.37, abs=0.002)


def test_flux_placement_finds_the_click_wherever_the_grid_starts():
    """Decaying noise bursts (a kick's click) at known times. Wherever the grid starts
    within the search window, flux placement lands it on the same spot: just before the
    onset, because the log flux peaks as the burst enters the leading edge of the 512-sample
    window. That constant is part of the calibrated offset, not an error."""
    rate, bpm, first = 44100, 128.0, 0.5017
    period = 60 / bpm
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.002, rate * 60).astype(np.float32)
    burst = (rng.normal(0, 1, 2000) * np.exp(-np.arange(2000) / 300)).astype(np.float32)
    for t in first + period * np.arange(100):
        i = int(round(t * rate))
        if i + len(burst) < len(x):
            x[i : i + len(burst)] += burst
    flux, dt, t_first = onset_flux(x, rate)
    placed = []
    for error_ms in (-30.0, -7.0, 0.0, 12.0, 25.0):
        start = first + error_ms / 1000
        shift, prominence = attack_offset(flux, dt, t_first, bpm, start, Params())
        assert prominence > 3
        placed.append(start + shift)
    assert np.ptp(placed) < 0.0005
    assert -0.006 < np.mean(placed) - first < 0.0


def flux_with_attacks(bpm, first, seconds, second=None):
    """Onset flux as a drum loop's looks: a floor with a sharp attack on every beat, and
    optionally a second attack `ms` later at `height` of the first (the same as the Rust test)."""
    rate = 44100.0
    dt, t_first, period = 32 / rate, (256 + 16) / rate, 60 / bpm
    t = t_first + np.arange(int(seconds / dt)) * dt
    bump = lambda at: np.exp(-0.5 * ((t - at) / 0.0008) ** 2)
    v = 1 + 10 * bump(first + np.round((t - first) / period) * period)
    if second:
        ms, height = second
        v += 10 * height * bump(first + np.round((t - first - ms / 1000) / period) * period + ms / 1000)
    return v.astype(np.float32), dt, t_first


def test_self_audit_holds_on_the_right_tempo_and_walks_on_a_wrong_one():
    flux, dt, t_first = flux_with_attacks(128.0, 0.25, 240.0)
    drift, clear = self_audit(flux, dt, t_first, 128.0, 0.25)
    assert clear == pipeline.STRETCHES and drift < 1.0
    # 0.02 BPM off walks ~33 ms across the track's 8 stretches.
    drift, _ = self_audit(flux, dt, t_first, 128.02, 0.25)
    assert drift > 25.0
    assert self_audit(np.ones_like(flux), dt, t_first, 128.0, 0.25) == (None, 0)


def test_a_second_attack_is_measured_against_the_first():
    single = flux_with_attacks(126.0, 0.3, 60.0)
    assert second_attack(*single, 126.0, 0.3, Params()) < 0.1
    double = flux_with_attacks(126.0, 0.3, 60.0, (15.0, 0.6))
    assert second_attack(*double, 126.0, 0.3, Params()) == pytest.approx(0.6, abs=0.1)
    close = flux_with_attacks(126.0, 0.3, 60.0, (3.0, 0.6))  # the same attack, not a second one
    assert second_attack(*close, 126.0, 0.3, Params()) < 0.3


def test_each_review_reason_has_its_cause():
    clean = dict(bpm=128.0, bpm_grid=128.0, tempo_source="flux", coherence_final=0.9, bend_ms=5.0,
                 attack_second=0.1, stretch_drift_ms=1.0, clear_stretches=8, snap_drift_ms=0.5, bar_margin=6.0,
                 bar_votes={"network": [0.7, 0.1, 0.1, 0.1], "changes": [0.6, 0.2, 0.1, 0.1]})
    assert review_reasons(clean) == []
    cases = [
        ({"tempo_source": "activation"}, pipeline.NO_ATTACK),
        ({"stretch_drift_ms": None, "clear_stretches": 2}, pipeline.NO_ATTACK),
        ({"coherence_final": 0.4}, pipeline.WEAK_PULSE),
        ({"bend_ms": 60.0}, pipeline.TEMPO_CHANGE),
        ({"bar_votes": {"network": [0.7, 0.1, 0.1, 0.1], "changes": [0.1, 0.1, 0.7, 0.1]}}, pipeline.BAR_DISAGREE),
        ({"bpm": 174.0, "bpm_grid": 87.0}, pipeline.OCTAVE_HALVED),
        ({"bpm": 80.0, "bpm_grid": 160.0}, pipeline.OCTAVE_DOUBLED),
    ]
    for change, reason in cases:
        assert review_reasons({**clean, **change}) == [reason], change
    # Measured, but not reasons on their own (docs/results.md).
    assert review_reasons({**clean, "attack_second": 0.9, "stretch_drift_ms": 30.0, "snap_drift_ms": 30.0,
                           "bar_margin": 0.1}) == []


def test_segmented_grid_matches_a_constant_one_and_numbers_on_across_a_change():
    from beatgrid_lab.segmented import Segment, Segmented

    # One segment is a constant grid.
    g = Grid.constant(120.0, 0.75 - 4.0)
    s = Segmented.constant(120.0, 0.75 - 4.0)
    n_g, t_g = g.beats(0.0, 20.0)
    n_s, t_s = s.beats(0.0, 20.0)
    assert np.allclose(t_g, t_s) and np.array_equal(n_g % 4, n_s % 4)  # numbered from another origin
    # Splitting it on a beat changes nothing.
    split = Segmented([Segment(0.75, 120.0, 1), Segment(0.75 + 10 * 0.5, 120.0, 3)])
    assert np.allclose(split.beats(0.0, 20.0)[1], t_s) and np.array_equal(split.beats(0.0, 20.0)[0], n_s)
    # 124 BPM for 30 beats, then 150 from a downbeat 0.1 s late: a bar cut to two beats.
    p = 60 / 124
    change = 0.3 + 30 * p + 0.1
    edit = Segmented([Segment(0.3, 124.0, 1), Segment(change, 150.0, 1)])
    idx, t, bar, pos = edit.beat_list(13.5, change + 1.0)
    assert idx.tolist() == [28, 29, 30, 31, 32]
    assert bar.tolist() == [8, 8, 9, 9, 9] and pos.tolist() == [1, 2, 1, 2, 3]
    assert t[2] == pytest.approx(change) and t[3] == pytest.approx(change + 0.4)
    for x in (0.0, 5.0, 14.4, 14.6, change, 20.0, 100.0):
        assert edit.time_of(edit.number_at(x))[0] == pytest.approx(x)
    assert edit.bpm_at(change - 0.01) == 124.0 and edit.bpm_at(change) == 150.0
    with pytest.raises(ValueError):
        Segmented([Segment(1.0, 120.0, 1), Segment(1.2, 140.0, 1)])
