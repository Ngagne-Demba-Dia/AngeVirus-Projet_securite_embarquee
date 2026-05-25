"""
CPA (Correlation Power Analysis) sur traces DPA Contest v4.2 (AES-128 RSM)
Dataset : DPA_contestv4_2_k01_part2
"""

import os
import bz2
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────────
_CANDIDATES = [
    Path("G:/datasets/DPA_contest_data"),       # Windows natif
    Path("/mnt/g/datasets/DPA_contest_data"),   # WSL
]
DATASET_ROOT = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
INDEX_FILE   = DATASET_ROOT / "dpav4_2_index"
TRACES_DIR   = DATASET_ROOT / "DPA_contestv4_2_k01_part2" / "DPA_contestv4_2" / "k01"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── AES S-Box ─────────────────────────────────────────────────────────────────
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.float32)


# ── Parseur index ─────────────────────────────────────────────────────────────
def parse_index(index_path, traces_dir, max_traces=None):
    """
    Chaque ligne du fichier index :
      KEY(32hex) PT(32hex) CT(32hex) f1 f2 f3 k00 DPACV42_XXXXXX.trc.bz2
    Retourne les listes (keys, plaintexts, filenames) pour les traces présentes.
    """
    keys, plaintexts, filenames = [], [], []
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            trc_file = traces_dir / parts[-1]
            if not trc_file.exists():
                continue
            key_bytes = bytes.fromhex(parts[0])
            pt_bytes  = bytes.fromhex(parts[1])
            keys.append(key_bytes)
            plaintexts.append(pt_bytes)
            filenames.append(trc_file)
            if max_traces and len(filenames) >= max_traces:
                break
    return keys, plaintexts, filenames


# ── Parseur trace LeCroy ──────────────────────────────────────────────────────
HEADER_PREFIX  = 11    # 0x23 + 10 zéros
WAVEDESC_SIZE  = 346   # WAVE_DESCRIPTOR = 0x015a
DATA_OFFSET    = HEADER_PREFIX + WAVEDESC_SIZE  # 357

def load_trace(path: Path, window=None):
    """
    Lit une trace LeCroy bz2.
    window=(start, end) : décompresse seulement header + end octets (34× plus rapide).
    """
    with bz2.open(path, "rb") as f:
        f.read(DATA_OFFSET)          # skip prefix + WAVEDESC (décompressé incrémentalement)
        if window is None:
            raw = f.read()
            data = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        else:
            if window[0] > 0:
                f.read(window[0])    # skip jusqu'au début de la fenêtre
            raw = f.read(window[1] - window[0])
            data = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    return data


# ── Chargement des traces ─────────────────────────────────────────────────────
def load_traces(filenames, window=None, verbose=True):
    traces = []
    for i, path in enumerate(filenames):
        if verbose and i % 100 == 0:
            print(f"  Chargement trace {i+1}/{len(filenames)}...", end="\r")
        traces.append(load_trace(path, window=window))
    if verbose:
        print(f"  {len(traces)} traces chargées.          ")
    return np.array(traces, dtype=np.float32)


# ── CPA ───────────────────────────────────────────────────────────────────────
def hw_model(plaintexts_byte, key_guess):
    """HW(SBox[pt ^ k]) pour un octet cible."""
    return HW[SBOX[(plaintexts_byte ^ key_guess) & 0xFF]]


def run_cpa(X, pt_bytes, target_byte=0):
    """
    CPA vectorisé : tous les 256 candidats en un seul produit matriciel.

    X        : (N, T) float32 — traces
    pt_bytes : (N, 16) uint8  — plaintext
    Retourne :
      corr_matrix : (256, T) corrélations de Pearson
      max_corr    : (256,)   max|corr| par candidat
      best_guess  : int      candidat avec la plus grande corrélation
    """
    N, T = X.shape
    pt_col = pt_bytes[:, target_byte].astype(np.uint8)  # (N,)

    # Hypothèses HW pour tous les candidats : (N, 256)
    H = HW[SBOX[(pt_col[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]) & 0xFF]]
    H = H.astype(np.float64)
    X64 = X.astype(np.float64)

    # Centrage
    Hc = H - H.mean(axis=0)   # (N, 256)
    Xc = X64 - X64.mean(axis=0)  # (N, T)

    # Covariance : (256, T) = (256, N) @ (N, T)
    num = Hc.T @ Xc            # (256, T)

    # Dénominateurs
    std_H = np.sqrt((Hc**2).sum(axis=0))  # (256,)
    std_X = np.sqrt((Xc**2).sum(axis=0))  # (T,)
    denom = std_H[:, None] * std_X[None, :]
    denom = np.where(denom < 1e-12, 1e-12, denom)

    corr_matrix = (num / denom).astype(np.float32)
    max_corr    = np.abs(corr_matrix).max(axis=1)
    best_guess  = int(np.argmax(max_corr))
    return corr_matrix, max_corr, best_guess


# ── Rank curve ────────────────────────────────────────────────────────────────
def _pearson_from_sums(sum_XH, sum_X, sum_X2, sum_H, sum_H2, n):
    """Pearson vectorisé à partir des accumulateurs (N traces vues)."""
    mu_x = sum_X / n
    mu_h = sum_H / n
    cov   = sum_XH / n - mu_h[:, None] * mu_x[None, :]
    var_x = np.maximum(sum_X2 / n - mu_x**2, 0.0)
    var_h = np.maximum(sum_H2 / n - mu_h**2, 0.0)
    denom = np.sqrt(var_h[:, None]) * np.sqrt(var_x[None, :])
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return cov / denom   # (256, T)


def compute_rank_curve(X, pt_bytes, real_key_byte, target_byte=0, steps=None):
    """
    Rang de la vraie clé en fonction du nombre de traces.
    Accumulation par batchs via matmul (évite la boucle Python par trace).
    """
    N, T = X.shape
    if steps is None:
        steps = list(range(100, N + 1, 100))
        if N not in steps:
            steps.append(N)
    steps = sorted(steps)

    pt_col = pt_bytes[:, target_byte].astype(np.uint8)
    H = HW[SBOX[(pt_col[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]) & 0xFF]].astype(np.float32)
    # (N, 256) float32

    sum_X  = np.zeros(T,        dtype=np.float64)
    sum_X2 = np.zeros(T,        dtype=np.float64)
    sum_H  = np.zeros(256,      dtype=np.float64)
    sum_H2 = np.zeros(256,      dtype=np.float64)
    sum_XH = np.zeros((256, T), dtype=np.float64)

    ranks = []
    step_set = set(steps)
    BATCH = 100

    for batch_start in range(0, N, BATCH):
        batch_end = min(batch_start + BATCH, N)
        h_b = H[batch_start:batch_end].astype(np.float64)   # (b, 256)
        x_b = X[batch_start:batch_end].astype(np.float64)   # (b, T)

        sum_XH += h_b.T @ x_b   # (256, b) @ (b, T) = (256, T)
        sum_H  += h_b.sum(axis=0)
        sum_H2 += (h_b**2).sum(axis=0)
        sum_X  += x_b.sum(axis=0)
        sum_X2 += (x_b**2).sum(axis=0)

        # Vérifier les steps qui tombent dans ce batch
        for n in range(batch_start + 1, batch_end + 1):
            if n in step_set:
                corr  = _pearson_from_sums(sum_XH, sum_X, sum_X2, sum_H, sum_H2, n)
                max_c = np.abs(corr).max(axis=1)
                sorted_idx = np.argsort(max_c)[::-1]
                rank = int(np.where(sorted_idx == real_key_byte)[0][0])
                ranks.append(rank)
                print(f"  N={n:5d}  rang={rank:3d}  "
                      f"corr_vraie={max_c[real_key_byte]:.4f}  "
                      f"corr_best={max_c[sorted_idx[0]]:.4f}")

    return steps[:len(ranks)], ranks


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_rank_curve(steps, ranks, real_key_byte, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, ranks, color="#2196F3", linewidth=2, marker="o", markersize=3)
    ax.axhline(0, color="#4CAF50", linestyle="--", linewidth=1.5, label="Rang 0 (clé trouvée)")
    ax.set_xlabel("Nombre de traces", fontsize=12)
    ax.set_ylabel("Rang de la vraie clé", fontsize=12)
    ax.set_title(f"CPA — Rang de la clé 0x{real_key_byte:02X} (byte 0)\nDPA Contest v4.2", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 256)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Courbe de rang sauvegardée : {out_path}")


def plot_correlation_bar(max_corr, real_key_byte, best_guess, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#F44336" if k == real_key_byte else "#90CAF9" for k in range(256)]
    ax.bar(range(256), max_corr, color=colors, width=1.0)
    ax.axvline(real_key_byte, color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"Vraie clé 0x{real_key_byte:02X}")
    ax.set_xlabel("Candidat clé k", fontsize=12)
    ax.set_ylabel("max|corr| sur le temps", fontsize=12)
    ax.set_title("CPA — Corrélation max par candidat (DPA Contest v4.2)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Bar chart corrélation sauvegardé : {out_path}")


def plot_best_trace_corr(corr_matrix, real_key_byte, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    T = corr_matrix.shape[1]
    t = np.arange(T)
    ax.plot(t, corr_matrix[real_key_byte], color="#F44336", linewidth=1.2,
            label=f"Vraie clé 0x{real_key_byte:02X}", alpha=0.9)
    best = np.argmax(np.abs(corr_matrix).max(axis=1))
    if best != real_key_byte:
        ax.plot(t, corr_matrix[best], color="#FF9800", linewidth=0.8,
                label=f"Meilleur candidat 0x{best:02X}", alpha=0.7)
    ax.set_xlabel("Échantillon temporel", fontsize=12)
    ax.set_ylabel("Corrélation de Pearson", fontsize=12)
    ax.set_title("CPA — Corrélation temporelle (DPA Contest v4.2)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Corrélation temporelle sauvegardée : {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    TARGET_BYTE = 0      # octet 0 de la clé
    MAX_TRACES  = 1000   # 1000 traces (2500 disponibles)

    # Fenêtre temporelle : 50 000 premiers échantillons
    # Les traces font 1.7M pts ; le pic de corrélation est à l'échantillon ~19585
    WINDOW = (0, 50_000)

    print("=" * 56)
    print("  CPA -- DPA Contest v4.2 (AES-128 RSM)")
    print("=" * 56)
    print()

    # 1. Chargement index
    print(f"[1/4] Lecture de l'index...")
    keys, plaintexts, filenames = parse_index(INDEX_FILE, TRACES_DIR, max_traces=MAX_TRACES)
    print(f"  {len(filenames)} traces trouvées.")

    if len(filenames) == 0:
        print("ERREUR : aucune trace trouvée. Vérifiez les chemins.")
        return

    real_key_byte = keys[0][TARGET_BYTE]
    print(f"  Vraie clé (byte {TARGET_BYTE}) : 0x{real_key_byte:02X} ({real_key_byte})")

    pt_bytes = np.array([[b for b in pt] for pt in plaintexts], dtype=np.uint8)

    # 2. Chargement des traces
    print(f"\n[2/4] Chargement des traces (fenêtre {WINDOW[0]}–{WINDOW[1]})...")
    X = load_traces(filenames, window=WINDOW)
    print(f"  Matrice traces : {X.shape}  dtype={X.dtype}")

    # 3. CPA finale (toutes les traces)
    print(f"\n[3/4] CPA sur {len(filenames)} traces...")
    corr_matrix, max_corr, best_guess = run_cpa(X, pt_bytes, target_byte=TARGET_BYTE)

    sorted_idx = np.argsort(max_corr)[::-1]
    rank_final = int(np.where(sorted_idx == real_key_byte)[0][0])

    print(f"\n  {'='*37}")
    print(f"  Vraie cle        : 0x{real_key_byte:02X}")
    print(f"  Meilleur candidat: 0x{best_guess:02X}")
    print(f"  Rang final       : {rank_final + 1} / 256")
    print(f"  Corr vraie cle   : {max_corr[real_key_byte]:.4f}")
    print(f"  Corr meilleur    : {max_corr[best_guess]:.4f}")
    print(f"  {'='*37}")

    # 4. Courbe de rang
    print(f"\n[4/4] Calcul de la courbe de rang...")
    steps_n = list(range(100, len(filenames) + 1, 100))
    if len(filenames) not in steps_n:
        steps_n.append(len(filenames))
    steps_done, ranks = compute_rank_curve(X, pt_bytes, real_key_byte,
                                           target_byte=TARGET_BYTE, steps=steps_n)

    # Figures
    plot_rank_curve(steps_done, [r + 1 for r in ranks], real_key_byte,
                    RESULTS_DIR / "01_rank_curve.png")
    plot_correlation_bar(max_corr, real_key_byte, best_guess,
                         RESULTS_DIR / "02_correlation_bar.png")
    plot_best_trace_corr(corr_matrix, real_key_byte,
                         RESULTS_DIR / "03_temporal_correlation.png")

    # Résumé
    traces_for_rank1 = None
    for s, r in zip(steps_done, ranks):
        if r == 0:
            traces_for_rank1 = s
            break

    print("\n" + "=" * 46)
    print("  RESULTATS FINAUX")
    print("=" * 46)
    print(f"  Traces utilisees   : {len(filenames)}")
    print(f"  Rang final         : {rank_final + 1} / 256")
    if traces_for_rank1:
        print(f"  Traces pour rang 1 : {traces_for_rank1}")
    else:
        print(f"  Rang 1 non atteint : CPA echoue (RSM)")
    print(f"  Vraie cle byte 0   : 0x{real_key_byte:02X}")
    print("=" * 46)


if __name__ == "__main__":
    main()
