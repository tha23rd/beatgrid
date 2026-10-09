import struct
import zlib

import numpy as np
import pytest

from beatgrid_lab.activations import Activations
from beatgrid_lab.grid import Grid
from beatgrid_lab.metrics import at_reference_octave, score
from beatgrid_lab.pipeline import Params, attack_offset, onset_flux, run
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
