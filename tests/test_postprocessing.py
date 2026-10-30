"""Tests for the event post-processing (algo/mvafa_szcore/inference.py).

After the v1.3.1 input fix, some recordings still have a background
probability above THRESHOLD (Siena PN00: median p~0.72), which a fixed
threshold turns into an event every few minutes. On long recordings v1.3.2
uses a lower threshold and a recording-adaptive gate that raises it so that at
most (1 - GATE_QUANTILE) of the windows can be positive. Short recordings keep
the v1.3.1 setting.

Run from the repository root:  python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algo"))
from mvafa_szcore.inference import (  # noqa: E402
    GATE_MIN_RECORDING_SEC, GATE_QUANTILE, LONG_SMOOTH_K, LONG_THRESHOLD, STEP_SEC,
    THRESHOLD, WINDOW_SEC, _smooth, gate_threshold, probs_to_events)

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
    margin = LONG_SMOOTH_K * STEP_SEC
    assert starts[burst[0]] - margin <= onset < offset <= starts[burst[1] - 1] + WINDOW_SEC + margin


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("background", [0.2, 0.65, 0.8])
def test_short_recordings_keep_v131_setting(seed, background):
    """Under 60 min (every TUH-Sz file) or of unknown length: threshold 0.70,
    smoothing 7, no gate, as in v1.3.1."""
    n = int(GATE_MIN_RECORDING_SEC / STEP_SEC) - 1
    probs, starts = _recording(n, background, seed=seed, burst=(100, 140))
    short = GATE_MIN_RECORDING_SEC - 1.0
    v131 = probs_to_events(probs, starts, WINDOW_SEC, threshold=0.70, smooth_k=7,
                           gate_quantile=None)

    assert gate_threshold(_smooth(probs, 7), 0.70, short) == 0.70
    assert gate_threshold(_smooth(probs, 7), 0.70, None) == 0.70
    assert probs_to_events(probs, starts, WINDOW_SEC, recording_sec=short) == v131
    assert probs_to_events(probs, starts, WINDOW_SEC) == v131


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("background", [0.2, 0.6, 0.8])
def test_gate_only_removes_events(seed, background):
    """On long recordings the gate never lowers LONG_THRESHOLD, so every gated
    event lies inside an event of the ungated long-recording setting."""
    probs, starts = _recording(N_LONG, background, seed=seed, burst=(100, 140))
    rec = N_LONG * STEP_SEC
    assert gate_threshold(_smooth(probs, LONG_SMOOTH_K), LONG_THRESHOLD, rec) >= LONG_THRESHOLD

    ungated = probs_to_events(probs, starts, WINDOW_SEC, threshold=LONG_THRESHOLD,
                              smooth_k=LONG_SMOOTH_K)
    gated = probs_to_events(probs, starts, WINDOW_SEC, recording_sec=rec)
    assert all(_inside(e, ungated) for e in gated)
