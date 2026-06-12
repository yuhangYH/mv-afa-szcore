"""Feature extraction — exact match to eeg_chbmit_multiview_gated.py training code."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

try:
    from ripser import ripser
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False


# ---------------------------------------------------------------------------
# Statistical features  (stat_dim = 40)
# ---------------------------------------------------------------------------
def _zero_crossing_rate(s: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.sign(s))) > 0))


def _slope_sign_changes(s: np.ndarray) -> float:
    d = np.diff(s)
    return float(np.sum(np.abs(np.diff(np.sign(d))) > 0))


def _hjorth(s: np.ndarray):
    var_x = np.var(s)
    dx = np.diff(s)
    var_dx = np.var(dx)
    if var_x < 1e-10:
        return 0.0, 0.0, 0.0
    mob = float(np.sqrt(var_dx / (var_x + 1e-12)))
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    comp = float(np.sqrt(var_ddx / (var_dx + 1e-12)) / (mob + 1e-12))
    return float(var_x), mob, comp


def _shannon_entropy(s: np.ndarray, bins: int = 64) -> float:
    counts, _ = np.histogram(s, bins=bins)
    counts = counts[counts > 0].astype(float)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _per_channel_stat(s: np.ndarray) -> list:
    std_ = float(np.std(s))
    mean_ = float(np.mean(s))
    var_  = float(np.var(s))
    sk_   = float(skew(s)) if std_ > 1e-8 else 0.0
    ku_   = float(kurtosis(s)) if std_ > 1e-8 else 0.0
    return [
        mean_, std_, var_, sk_, ku_,
        float(np.min(s)), float(np.max(s)), float(np.max(s) - np.min(s)),
        float(np.median(s)),
        float(np.percentile(s, 75) - np.percentile(s, 25)),
        float(np.sqrt(np.mean(s ** 2))),   # rms
        float(np.mean(s ** 2)),            # energy
        float(np.mean(np.abs(s))),         # mav
        float(np.mean(np.abs(np.diff(s)))),  # line_length
        _zero_crossing_rate(s),
        _slope_sign_changes(s),
        *_hjorth(s),
        _shannon_entropy(s),
    ]


def extract_statistical_features(seg: np.ndarray) -> np.ndarray:
    """Return 40-D aggregated stat vector (mean + std of 20 per-channel feats).

    Parameters
    ----------
    seg : (C, T) float32
    """
    feats = np.array([_per_channel_stat(seg[c]) for c in range(seg.shape[0])],
                     dtype=np.float32)            # (C, 20)
    agg = np.concatenate([feats.mean(axis=0), feats.std(axis=0)], axis=0)
    return np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# TDA features  (tda_dim = 12)
# ---------------------------------------------------------------------------
def _fold_point_cloud(seg: np.ndarray, n_folds: int = 10) -> np.ndarray:
    """Build (C*n_folds, L) point cloud from (C, T) window."""
    c, t = seg.shape
    fold_len = t // n_folds
    if fold_len < 2:
        n_folds = 1
        fold_len = t
    usable = n_folds * fold_len
    chunks = np.split(seg[:, :usable], n_folds, axis=1)
    points = []
    for chunk in chunks:
        for ch_sig in chunk:
            p = ch_sig.astype(np.float32)
            p = (p - p.mean()) / (p.std() + 1e-8)
            points.append(p)
    return np.stack(points, axis=0)    # (C*n_folds, fold_len)


def _dgm_summary(dgm) -> list:
    if dgm is None or len(dgm) == 0:
        return [0.0] * 6
    dgm = np.asarray(dgm, dtype=np.float32)
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return [0.0] * 6
    life = finite[:, 1] - finite[:, 0]
    life = life[life > 0]
    if len(life) == 0:
        return [0.0] * 6
    total = life.sum()
    p = life / (total + 1e-12)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return [float(len(life)), float(life.mean()), float(life.std()),
            float(life.max()), float(total), entropy]


def extract_tda_features(seg: np.ndarray, n_folds: int = 10) -> np.ndarray:
    """Return 12-D topological feature vector.

    Parameters
    ----------
    seg : (C, T) float32
    """
    if not _HAS_RIPSER:
        raise ImportError("ripser is required. Install with: pip install ripser")
    pc = _fold_point_cloud(seg, n_folds=n_folds)
    dgms = ripser(pc, maxdim=1)["dgms"]
    h0 = _dgm_summary(dgms[0] if len(dgms) > 0 else None)
    h1 = _dgm_summary(dgms[1] if len(dgms) > 1 else None)
    out = np.array(h0 + h1, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# PSD map for frequency branch  — shape (1, C, F)
# ---------------------------------------------------------------------------
def extract_psd_map(seg: np.ndarray, sfreq: float,
                    nperseg: int = 128, noverlap: int = 64,
                    fmin: float = 0.5, fmax: float = 40.0) -> np.ndarray:
    """Return normalized Welch PSD map, shape (1, C, F)."""
    psd_list = []
    n = seg.shape[1]
    np_ = min(nperseg, n)
    no_ = min(noverlap, max(0, np_ - 1))
    for ch in range(seg.shape[0]):
        freqs, pxx = welch(seg[ch], fs=sfreq, nperseg=np_, noverlap=no_)
        mask = (freqs >= fmin) & (freqs <= fmax)
        pxx = np.log1p(pxx[mask]).astype(np.float32)
        psd_list.append(pxx)
    psd = np.stack(psd_list, axis=0)       # (C, F)
    psd = (psd - psd.mean()) / (psd.std() + 1e-8)
    return psd[None].astype(np.float32)    # (1, C, F)
