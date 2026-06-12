"""Sliding-window inference: EDF → seizure probability per window.

Pipeline (matches eeg_chbmit_multiview_gated.py training protocol)
-------------------------------------------------------------------
1. Load EDF → 18-ch bipolar array at 256 Hz
2. Slide 2-s windows with 4-s step (window_sec=2, step_sec=4)
3. Per window:
   a. Per-channel z-score normalisation → x_time (18, 512)
   b. extract_statistical_features → x_stat (40,)
   c. extract_tda_features (n_folds=10) → x_tda (12,)
   d. extract_psd_map → x_freq (1, 18, F)
4. Batched forward pass through MultiViewSeizureNet
5. Convert window probabilities to event (onset, offset) pairs
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .eeg_io import load_edf_as_bipolar
from .features import extract_statistical_features, extract_tda_features, extract_psd_map
from .model import MultiViewSeizureNet

# --- Must match training ---
WINDOW_SEC   = 2.0
STEP_SEC     = 4.0
SFREQ        = 256.0
N_CHANNELS   = 18
STAT_DIM     = 40
TDA_DIM      = 12
TDA_FOLDS    = 10
THRESHOLD    = 0.54

# --- Post-processing ---
MIN_SEIZURE_SEC = 5.0
MERGE_GAP_SEC   = 30.0

BATCH_SIZE = 16
MODEL_PATH = Path(__file__).parent / "weights" / "best_model.pt"


def _load_model(device: torch.device) -> MultiViewSeizureNet:
    model = MultiViewSeizureNet(
        time_channels=N_CHANNELS,
        stat_dim=STAT_DIM,
        tda_dim=TDA_DIM,
        common_dim=128,
        num_classes=2,
        dropout=0.2,
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(edf_path: str) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Run MV-AFA on an EDF file.

    Returns
    -------
    probs             : (N,) seizure probabilities, one per window
    sfreq             : 256.0
    window_sec        : 2.0
    window_starts_sec : (N,) window start times in seconds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)

    data, sfreq = load_edf_as_bipolar(edf_path)   # (18, T) at 256 Hz
    n_samples = data.shape[1]
    win  = int(round(WINDOW_SEC * sfreq))
    step = int(round(STEP_SEC   * sfreq))

    starts = list(range(0, n_samples - win + 1, step))
    if not starts:
        return np.array([]), sfreq, WINDOW_SEC, np.array([])

    time_list, freq_list, stat_list, tda_list = [], [], [], []
    for s in starts:
        seg = data[:, s : s + win].astype(np.float32)

        # per-channel z-score
        mu = seg.mean(axis=1, keepdims=True)
        sd = seg.std(axis=1, keepdims=True) + 1e-8
        seg_norm = (seg - mu) / sd

        time_list.append(seg_norm)
        freq_list.append(extract_psd_map(seg, sfreq=sfreq))
        stat_list.append(extract_statistical_features(seg))
        tda_list.append(extract_tda_features(seg, n_folds=TDA_FOLDS))

    time_arr = np.stack(time_list).astype(np.float32)   # (N, 18, 512)
    freq_arr = np.stack(freq_list).astype(np.float32)   # (N, 1, 18, F)
    stat_arr = np.stack(stat_list).astype(np.float32)   # (N, 40)
    tda_arr  = np.stack(tda_list).astype(np.float32)    # (N, 12)

    probs = []
    n = len(starts)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            logits, _ = model(
                torch.from_numpy(time_arr[i:i+BATCH_SIZE]).to(device),
                torch.from_numpy(freq_arr[i:i+BATCH_SIZE]).to(device),
                torch.from_numpy(stat_arr[i:i+BATCH_SIZE]).to(device),
                torch.from_numpy(tda_arr [i:i+BATCH_SIZE]).to(device),
            )
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)

    probs_arr = np.concatenate(probs)
    window_starts = np.array(starts, dtype=np.float32) / sfreq
    return probs_arr, sfreq, WINDOW_SEC, window_starts


def probs_to_events(
    probs: np.ndarray,
    window_starts_sec: np.ndarray,
    window_sec: float,
    threshold: float = THRESHOLD,
    min_seizure_sec: float = MIN_SEIZURE_SEC,
    merge_gap_sec: float = MERGE_GAP_SEC,
) -> List[Tuple[float, float]]:
    """Convert per-window probabilities to (onset, offset) seizure events."""
    positive = probs >= threshold
    if not positive.any():
        return []

    intervals = [(window_starts_sec[i], window_starts_sec[i] + window_sec)
                 for i, pos in enumerate(positive) if pos]

    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged if (e - s) >= min_seizure_sec]
