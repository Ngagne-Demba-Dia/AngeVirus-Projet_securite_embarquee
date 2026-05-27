"""
02_features.py — Extraction de features par fenêtres glissantes.
Par fenêtre de 100 pts : mean, std, energie, FFT(10 premiers coef) = 13 features
25 fenêtres × 13 = 325 features par trace
Output: ../results/features_AES-TXXX.npz (X, y)
"""
import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path
from joblib import Parallel, delayed


from feat_utils import load_trace, extract_features


def process_row(row, window, n_fft):
    trace = load_trace(row["path"])
    feat  = extract_features(trace, window, n_fft)
    return feat, int(row["label"])


if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    feat_cfg = cfg["features"]
    WINDOW   = feat_cfg["window_size"]
    N_FFT    = feat_cfg["n_fft"]
    N_PER    = cfg["dataset"]["n_samples_per_class"]

    index = pd.read_parquet("../results/index.parquet")
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    n_feat = (2500 // WINDOW) * (3 + N_FFT)
    print(f"Features par trace : {n_feat}  (window={WINDOW}, n_fft={N_FFT})")
    print(f"Sampling : {N_PER} traces par (condition × methode)\n")

    for benchmark in sorted(index["benchmark"].unique()):
        bm_df = index[index["benchmark"] == benchmark].copy()

        sampled = (
            bm_df.groupby(["condition", "method"])
            .apply(lambda g: g.sample(min(N_PER, len(g)), random_state=42))
            .reset_index(drop=True)
        )

        print(f"{benchmark}: {len(sampled)} traces...")
        t0 = time.time()

        rows = [row for _, row in sampled.iterrows()]
        out_raw = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_row)(r, WINDOW, N_FFT) for r in rows
        )

        X = np.stack([r[0] for r in out_raw])
        y = np.array([r[1] for r in out_raw], dtype=np.int64)

        out_path = results_dir / f"features_{benchmark}.npz"
        np.savez_compressed(out_path, X=X, y=y)

        elapsed = time.time() - t0
        counts  = np.bincount(y, minlength=3)
        print(f"  X={X.shape}  classes={counts}  ({elapsed:.1f}s)  -> {out_path.name}")

    print("\nExtraction terminee.")
