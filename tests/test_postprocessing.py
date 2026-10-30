"""Tests for the event post-processing (algo/mvafa_szcore/inference.py).

After the v1.3.1 input fix, some recordings still have a background
probability above THRESHOLD (Siena PN00: median p~0.72), which a fixed
threshold turns into an event every few minutes. The recording-adaptive gate
raises the threshold on long recordings so that at most (1 - GATE_QUANTILE) of
the windows can be positive.

Run from the repository root:  python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algo"))
from mvafa_szcore.inference import (  # noqa: E402
    GATE_MIN_RECORDING_SEC, GATE_QUANTILE, SMOOTH_K, STEP_SEC, THRESHOLD, WINDOW_SEC,
    _smooth, gate_threshold, probs_to_events)

N_LONG = int(2 * GATE_MIN_RECORDING_SEC / STEP_SEC)


def _recording(n, background, seed=0, burst=None):
    """Window probabilities around `background`, with an optional burst of
    p=0.99 over the window index range `burst`."""
    rng = np.random.default_rng(seed)
    probs = np.clip(background + 0.02 * rng.standard_normal(n), 0.0, 1.0)
    if burst is not None:
        probs[burst[0]:burst[1]] = 0.99
    return probs, np.arange(n) * STEP_SEC


def _inside(event, events):
    return any(a <= event[0] and event[1] <= b for a, b in events)


def test_gate_bounds_false_alarms_on_high_background():
    """Background above THRESHOLD everywhere: the fixed threshold flags the
    whole recording, the gate keeps only the burst."""
    burst = (N_LONG // 2, N_LONG // 2 + int(np.ceil(2 * (1 - GATE_QUANTILE) * N_LONG)) + 10)
    probs, starts = _recording(N_LONG, THRESHOLD + 0.08, burst=burst)
    rec = N_LONG * STEP_SEC

    fixed = probs_to_events(probs, starts, WINDOW_SEC)
    gated = probs_to_events(probs, starts, WINDOW_SEC, recording_sec=rec)

    assert sum(b - a for a, b in fixed) > 0.9 * rec
    assert len(gated) == 1
    onset, offset = gated[0]
    margin = SMOOTH_K * STEP_SEC
    assert starts[burst[0]] - margin <= onset < offset <= starts[burst[1] - 1] + WINDOW_SEC + margin


def test_gate_is_off_for_short_or_unknown_recordings():
    n = int(GATE_MIN_RECORDING_SEC / STEP_SEC) - 1
    probs, starts = _recording(n, THRESHOLD + 0.08, seed=1)
    smoothed = _smooth(probs, SMOOTH_K)
    short = GATE_MIN_RECORDING_SEC - 1.0

    assert gate_threshold(smoothed, THRESHOLD, short) == THRESHOLD
    assert gate_threshold(smoothed, THRESHOLD, None) == THRESHOLD
    assert (probs_to_events(probs, starts, WINDOW_SEC, recording_sec=short)
            == probs_to_events(probs, starts, WINDOW_SEC))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("background", [0.2, 0.6, 0.8])
def test_gate_only_removes_events(seed, background):
    """The gate never lowers the threshold, so every gated event lies inside
    an event of the fixed threshold."""
    probs, starts = _recording(N_LONG, background, seed=seed, burst=(100, 140))
    smoothed = _smooth(probs, SMOOTH_K)
    assert gate_threshold(smoothed, THRESHOLD, N_LONG * STEP_SEC) >= THRESHOLD

    fixed = probs_to_events(probs, starts, WINDOW_SEC)
    gated = probs_to_events(probs, starts, WINDOW_SEC, recording_sec=N_LONG * STEP_SEC)
    assert all(_inside(e, fixed) for e in gated)
