"""EDF loading and channel alignment for SzCORE inference.

SzCORE standardises to 19 standard 10-20 referential channels. MV-AFA was
trained on the CHB-MIT longitudinal bipolar montage (18 pairs), so we derive
those 18 bipolar signals algebraically from the referential channels.

CRITICAL — consistency with training:
The training pipeline reads signals with ``raw.get_data(picks)`` directly, i.e.
MNE's default Volt scaling and **no filtering / no rescaling** (only a per-window
z-score is applied later inside feature extraction). To keep inference features
on the same scale the model was trained on, we therefore:
  * do NOT band-pass / notch filter,
  * do NOT multiply by 1e6 (keep MNE Volts),
  * only resample to 256 Hz when the input rate differs (amplitude-preserving).
"""
from __future__ import annotations

import re
import warnings
from typing import Tuple

import numpy as np
import mne

# The 18 bipolar pairs, in the exact order used during training
# (CHB_MIT_TARGET_CHANNELS).
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

_TARGET_SF = 256.0


def _canonical(name: str) -> str:
    n = name.upper().strip()
    n = re.sub(r"^EEG\s*", "", n)
    n = re.sub(r"^POL\s*", "", n)
    n = n.replace("-REF", "").replace("-LE", "").replace(" ", "")
    # Strip a trailing duplicate index, e.g. CHB-MIT "T8-P8-0" -> "T8-P8".
    n = re.sub(r"-\d+$", "", n)
    return _ALIASES.get(n, n)


def load_edf_as_bipolar(edf_path: str) -> Tuple[np.ndarray, float]:
    """Load an EDF and return an (18, T) float32 bipolar array at 256 Hz.

    Signals are in MNE Volts (matching training); no filtering is applied.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # Resample only if needed (amplitude-preserving; training assumed 256 Hz).
    if abs(raw.info["sfreq"] - _TARGET_SF) > 1e-3:
        raw.resample(_TARGET_SF, verbose="ERROR")

    # Keep the FIRST occurrence of each canonical name (matches training).
    ch_map = {}
    for i, ch in enumerate(raw.ch_names):
        key = _canonical(ch)
        if key not in ch_map:
            ch_map[key] = i
    sig = raw.get_data()                 # Volts, no rescaling (matches training)

    out = np.zeros((18, sig.shape[1]), dtype=np.float32)
    missing = []
    for k, (anode, cathode) in enumerate(BIPOLAR_PAIRS):
        a_idx = ch_map.get(anode)
        c_idx = ch_map.get(cathode)
        if a_idx is not None and c_idx is not None:
            # SzCORE referential input: derive the bipolar signal.
            out[k] = sig[a_idx] - sig[c_idx]
            continue
        # Fallback: the recording may already be in bipolar montage with a
        # channel literally named e.g. "FP1-F7" (as in CHB-MIT).
        direct = ch_map.get(f"{anode}-{cathode}")
        if direct is not None:
            out[k] = sig[direct]
        else:
            missing.append(f"{anode}-{cathode}")

    if missing:
        warnings.warn(
            f"{len(missing)} bipolar channel(s) unavailable and filled with "
            f"zeros: {missing}. Available: {sorted(ch_map)}.",
            stacklevel=2,
        )

    return out, _TARGET_SF
