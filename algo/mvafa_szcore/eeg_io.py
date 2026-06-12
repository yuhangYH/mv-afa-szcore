"""EDF loading and channel alignment for SzCORE inference.

SzCORE standardises to 19 standard 10-20 channels. CHB-MIT was recorded as a
longitudinal bipolar montage (18 pairs). We derive the bipolar pairs
algebraically from the 19 referential channels so the model sees exactly the
same montage it was trained on.

Channel name aliases are normalised (T7=T3, P7=T5, T8=T4, P8=T6) to handle
older and newer 10-20 labelling conventions.
"""
from __future__ import annotations

import re
import warnings
from typing import Tuple

import numpy as np
import mne

# The 18 bipolar pairs as stored in the CHB-MIT corpus (presets.py).
# Each tuple is (anode, cathode) in canonical upper-case form.
BIPOLAR_PAIRS = [
    ("FP1", "F7"),  ("F7",  "T7"),  ("T7",  "P7"),  ("P7",  "O1"),
    ("FP1", "F3"),  ("F3",  "C3"),  ("C3",  "P3"),  ("P3",  "O1"),
    ("FP2", "F4"),  ("F4",  "C4"),  ("C4",  "P4"),  ("P4",  "O2"),
    ("FP2", "F8"),  ("F8",  "T8"),  ("T8",  "P8"),  ("P8",  "O2"),
    ("FZ",  "CZ"),  ("CZ",  "PZ"),
]

# Aliases: old 10-20 labels -> canonical labels used in BIPOLAR_PAIRS above.
_ALIASES = {
    "T3": "T7", "T4": "T8",
    "T5": "P7", "T6": "P8",
}

_NOTCH_HZ  = 60.0
_L_FREQ    = 0.5
_H_FREQ    = 40.0
_TARGET_SF = 256.0


def _canonical(name: str) -> str:
    n = name.upper().strip()
    n = re.sub(r"^EEG\s*", "", n)
    n = n.replace("-REF", "").replace("-LE", "").replace(" ", "")
    return _ALIASES.get(n, n)


def load_edf_as_bipolar(edf_path: str) -> Tuple[np.ndarray, float]:
    """Load an EDF and return a (18, T) float32 bipolar array at 256 Hz.

    Returns
    -------
    data  : np.ndarray shape (18, T) in µV
    sfreq : float — always 256.0
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    nyq = raw.info["sfreq"] / 2.0
    if _NOTCH_HZ < nyq:
        raw.notch_filter(_NOTCH_HZ, verbose="ERROR")
    raw.filter(_L_FREQ, _H_FREQ, verbose="ERROR")

    if abs(raw.info["sfreq"] - _TARGET_SF) > 1e-3:
        raw.resample(_TARGET_SF, verbose="ERROR")

    ch_map = {_canonical(ch): i for i, ch in enumerate(raw.ch_names)}
    eeg = raw.get_data() * 1e6          # V -> µV

    out = np.zeros((18, eeg.shape[1]), dtype=np.float32)
    for k, (anode, cathode) in enumerate(BIPOLAR_PAIRS):
        a_idx = ch_map.get(anode)
        c_idx = ch_map.get(cathode)
        if a_idx is None or c_idx is None:
            warnings.warn(
                f"Bipolar pair {anode}-{cathode}: missing electrode(s). "
                f"Available: {sorted(ch_map)}. Filled with zeros.",
                stacklevel=2,
            )
        else:
            out[k] = eeg[a_idx] - eeg[c_idx]

    return out, _TARGET_SF
