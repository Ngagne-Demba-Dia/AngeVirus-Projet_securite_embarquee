"""
feat_utils.py — Utilitaires partagés pour le chargement et l'extraction de features.
Importé par 02_features.py, 05_monitor.py, api/serve.py.
"""
import numpy as np


def load_trace(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def extract_features(trace: np.ndarray, window: int = 100, n_fft: int = 15) -> np.ndarray:
    """
    Extraire les features par fenêtres non-chevauchantes.
    Par fenêtre de `window` pts : mean, std, énergie, kurtosis, skewness, FFT(n_fft coef) = 5+n_fft.
    Total : (len(trace) // window) × (5 + n_fft) features.
    Exemple : 2500/100=25 fenêtres × (5+15) = 500 features.
    """
    n_windows = len(trace) // window
    n_feat    = 5 + n_fft
    feats = np.empty(n_windows * n_feat, dtype=np.float32)
    for i in range(n_windows):
        w    = trace[i * window: (i + 1) * window]
        base = i * n_feat
        mu   = w.mean()
        sig  = w.std() + 1e-8   # éviter division par zéro
        diff = w - mu
        feats[base]     = mu
        feats[base + 1] = sig
        feats[base + 2] = float((w ** 2).sum())
        feats[base + 3] = float(((diff / sig) ** 4).mean())  # kurtosis
        feats[base + 4] = float(((diff / sig) ** 3).mean())  # skewness
        feats[base + 5: base + n_feat] = np.abs(np.fft.rfft(w))[:n_fft]
    return feats
