"""
feat_utils.py — Utilitaires partagés pour le chargement et l'extraction de features.
Importé par 02_features.py, 05_monitor.py, api/serve.py.
"""
import numpy as np


def load_trace(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def extract_features(trace: np.ndarray, window: int = 100, n_fft: int = 10) -> np.ndarray:
    """
    Extraire les features par fenêtres non-chevauchantes.
    Par fenêtre de `window` pts : mean, std, énergie, FFT(n_fft premiers coef) = 3+n_fft features.
    Total : (len(trace) // window) × (3 + n_fft) features.
    """
    n_windows = len(trace) // window
    feats = np.empty(n_windows * (3 + n_fft), dtype=np.float32)
    for i in range(n_windows):
        w = trace[i * window: (i + 1) * window]
        base = i * (3 + n_fft)
        feats[base]             = w.mean()
        feats[base + 1]         = w.std()
        feats[base + 2]         = (w ** 2).sum()
        feats[base + 3: base + 3 + n_fft] = np.abs(np.fft.rfft(w))[:n_fft]
    return feats
