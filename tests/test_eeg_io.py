"""Regression tests for the SzCORE input stage (algo/mvafa_szcore/eeg_io.py).

v1.3.0 did not recognise SzCORE's common-average labels ("Fp1-Avg"), so on
Siena / TUH / Dianalund / SeizeIT all 18 bipolar pairs were zero-filled and the
model returned p=0.949 for every window (~288 FP/day on the leaderboard). The
files written here follow epilepsy2bids' Eeg.saveEdf (the converter SzCORE
uses): 256 Hz, physical dimension "uV", 19 referential "<elec>-Avg" channels
or 18 double-banana bipolar channels.

Run from the repository root:  python -m pytest tests
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pyedflib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algo"))
from mvafa_szcore.eeg_io import BIPOLAR_PAIRS, _canonical, load_edf_as_bipolar  # noqa: E402

FS = 256
# epilepsy2bids Eeg.ELECTRODES_10_20 and Eeg.BIPOLAR_DBANANA (old T3-T6 names).
SZCORE_ELECTRODES = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz", "Cz",
                     "Pz", "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6"]
SZCORE_DBANANA = ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp1-F7", "F7-T3", "T3-T5",
                  "T5-O1", "Fz-Cz", "Cz-Pz", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
                  "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
OLD = {v: k for k, v in MODERN.items()}


def _modern(electrode):
    return MODERN.get(electrode, electrode).upper()


def _write_edf(path, labels, data_uv):
    headers = [{"label": lab, "dimension": "uV", "sample_frequency": FS,
                "physical_max": float(np.ceil(x.max())) + 1,
                "physical_min": float(np.floor(x.min())) - 1,
                "digital_max": 2 ** 14, "digital_min": -(2 ** 14)}
               for lab, x in zip(labels, data_uv)]
    pyedflib.highlevel.write_edf(str(path), data_uv, headers)
    return str(path)


def _electrode_signals(n_sec=20, seed=0):
    """A distinct signal per electrode (uV), keyed by modern upper-case name."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_sec * FS) / FS
    return {_modern(e): 40 * np.sin(2 * np.pi * (2 + i) * t) + 10 * rng.standard_normal(t.size)
            for i, e in enumerate(SZCORE_ELECTRODES)}


def _szcore_car(sig, gain=1.0):
    """19 x '<elec>-Avg' common-average channels, as epilepsy2bids writes them."""
    mean = np.mean(list(sig.values()), axis=0)
    data = np.stack([(sig[_modern(e)] - mean) * gain for e in SZCORE_ELECTRODES])
    return [f"{e}-Avg" for e in SZCORE_ELECTRODES], data


def _expected_pairs_volts(sig):
    return np.stack([sig[a] - sig[c] for a, c in BIPOLAR_PAIRS]) * 1e-6


def _load_without_zero_fill(edf):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = load_edf_as_bipolar(edf)
    zero_fill = [w for w in caught if "filled with zeros" in str(w.message)]
    assert not zero_fill, str(zero_fill[0].message)
    return out


@pytest.mark.parametrize("label, expected", [
    ("Fp1-Avg", "FP1"),          # SzCORE common average (the v1.3.0 failure)
    ("T3-Avg", "T7"),
    ("T6-Avg", "P8"),
    ("EEG FP1-REF", "FP1"),      # TUSZ
    ("EEG T5-LE", "P7"),
    ("EEG Fp1", "FP1"),          # raw Siena
    ("F7-T3", "F7-T7"),          # SzCORE CHB-MIT chb12_27-29 (unipolar-derived)
    ("T4-T6", "T8-P8"),
    ("FP1-F7", "FP1-F7"),        # raw / SzCORE CHB-MIT
    ("T8-P8-0", "T8-P8"),        # CHB-MIT duplicate index
])
def test_canonical_labels(label, expected):
    assert _canonical(label) == expected


def test_szcore_common_average_unipolar(tmp_path):
    """Siena / TUH / Dianalund / SeizeIT format."""
    sig = _electrode_signals()
    labels, data = _szcore_car(sig)
    edf = _write_edf(tmp_path / "szcore_car.edf", labels, data)

    out, sf = _load_without_zero_fill(edf)

    assert sf == FS and out.shape == (18, data.shape[1])
    # The common average cancels in every difference.
    np.testing.assert_allclose(out, _expected_pairs_volts(sig), atol=0.05e-6)


@pytest.mark.parametrize("labels", [
    SZCORE_DBANANA,
    [f"{OLD.get(a, a)}-{OLD.get(c, c)}" for a, c in BIPOLAR_PAIRS],
    [f"{a}-{c}" for a, c in BIPOLAR_PAIRS],
], ids=["szcore_dbanana", "old_names", "modern_names"])
def test_szcore_bipolar(tmp_path, labels):
    """CHB-MIT format: 18 bipolar channels in any order and 10-20 naming."""
    sig = _electrode_signals(seed=1)
    want = _expected_pairs_volts(sig)
    canon = [_canonical(lab) for lab in labels]
    data = np.empty_like(want)
    for k, (a, c) in enumerate(BIPOLAR_PAIRS):
        data[canon.index(f"{a}-{c}")] = want[k] * 1e6
    edf = _write_edf(tmp_path / "bipolar.edf", labels, data)

    out, _ = _load_without_zero_fill(edf)

    np.testing.assert_allclose(out, want, atol=0.05e-6)


def test_unrecognised_montage_raises(tmp_path):
    """Never silently predict on an all-zero input again."""
    data = np.random.default_rng(2).standard_normal((4, 10 * FS)) * 20
    edf = _write_edf(tmp_path / "unknown.edf", ["X1", "X2", "X3", "X4"], data)
    with pytest.raises(ValueError, match="No bipolar channel"):
        load_edf_as_bipolar(edf)


def test_inference_on_szcore_unipolar_is_not_constant(tmp_path):
    """End-to-end: the v1.3.0 symptom was one probability for every window."""
    from mvafa_szcore.inference import run_inference

    n_sec = 120
    gain = np.repeat(np.random.default_rng(3).uniform(0.3, 3.0, n_sec // 4), 4 * FS)
    labels, data = _szcore_car(_electrode_signals(n_sec=n_sec, seed=3), gain)
    edf = _write_edf(tmp_path / "szcore_e2e.edf", labels, data)

    probs, _, _, starts = run_inference(edf)

    assert len(probs) == len(starts) == n_sec // 4
    assert np.ptp(probs) > 1e-3, f"constant model output {probs[0]:.4f}"


@pytest.mark.parametrize("n_windows", [1, 3, 6, 7, 30])
def test_smoothing_keeps_one_value_per_window(n_windows):
    """np.convolve(mode="same") returned max(n, k) values for n < k windows."""
    from mvafa_szcore.inference import SMOOTH_K, _smooth, probs_to_events

    probs = np.full(n_windows, 0.99)
    starts = np.arange(n_windows) * 4.0
    smoothed = _smooth(probs, SMOOTH_K)
    assert len(smoothed) == n_windows
    if n_windows >= SMOOTH_K:
        np.testing.assert_allclose(
            smoothed, np.convolve(probs, np.ones(SMOOTH_K) / SMOOTH_K, mode="same"))
    for onset, offset in probs_to_events(probs, starts, 2.0, threshold=0.5, min_seizure_sec=0.0):
        assert 0.0 <= onset < offset <= starts[-1] + 2.0
