"""
Template LDA — ASCAD octet TARGET
Cible : SBox[pt XOR k] XOR mask[0]  (valeur masquee, 256 classes)
Conforme au PDF section ASCAD etape 4.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path

_CANDIDATES = [
    Path('/mnt/g/DATA/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5'),
    Path('G:/DATA/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5'),
]
ASCAD_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

TARGET  = 2
N_TRAIN = 45000
TOP_K   = 100   # features SNR retenues

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

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement ASCAD...")
with h5py.File(ASCAD_PATH, 'r') as f:
    X_prof  = f['Profiling_traces/traces'][:].astype(np.float32)
    md_prof = f['Profiling_traces/metadata'][:]
    X_att   = f['Attack_traces/traces'][:].astype(np.float32)
    md_att  = f['Attack_traces/metadata'][:]

pt_prof   = md_prof['plaintext']
key_prof  = md_prof['key']
mask_prof = md_prof['masks']
pt_att    = md_att['plaintext']
key_att   = md_att['key']
mask_att  = md_att['masks']
TRUE_KEY  = int(key_att[0, TARGET])
print(f"Vraie cle octet {TARGET} : 0x{TRUE_KEY:02X}")

# ── Labels 256 classes — valeur masquee SBox[pt XOR k] XOR mask[0] ───────────
# mask[0] est le masque actif sur l'octet 2 dans ASCAD (pas mask[TARGET=2])
labels_prof = np.array([
    int(SBOX[int(pt_prof[i, TARGET]) ^ int(key_prof[i, TARGET])]) ^ int(mask_prof[i, 0])
    for i in range(len(X_prof))
], dtype=np.int64)

labels_att = np.array([
    int(SBOX[int(pt_att[i, TARGET]) ^ int(key_att[i, TARGET])]) ^ int(mask_att[i, 0])
    for i in range(len(X_att))
], dtype=np.int64)

# ── Selection SNR (top-K features) ───────────────────────────────────────────
print("Selection SNR des features...")
X_tr = X_prof[:N_TRAIN].astype(np.float64)
y_tr = labels_prof[:N_TRAIN]

class_means  = np.zeros((256, X_tr.shape[1]))
class_counts = np.zeros(256)
for c in range(256):
    idx = (y_tr == c)
    if idx.sum() > 0:
        class_means[c]  = X_tr[idx].mean(axis=0)
        class_counts[c] = idx.sum()
grand_mean  = X_tr.mean(axis=0)
between_var = (class_counts[:, None] * (class_means - grand_mean)**2).sum(axis=0) / N_TRAIN
within_var  = X_tr.var(axis=0) - between_var + 1e-10
snr_vals    = between_var / within_var
top_k       = np.argsort(-snr_vals)[:TOP_K]
print(f"Sample SNR max : {top_k[0]}  SNR={snr_vals[top_k[0]]:.4f}")

X_tr_sel  = X_tr[:, top_k]
X_att_sel = X_att[:, top_k].astype(np.float64)

# ── LDA 256 classes (valeur masquee complete) ─────────────────────────────────
print(f"Entrainement LDA (256 classes) sur {N_TRAIN} traces, {TOP_K} features SNR...")
lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=False)
lda.fit(X_tr_sel, labels_prof[:N_TRAIN])
print("Entrainement termine.")

acc_train = (lda.predict(X_tr_sel[:2000]) == labels_prof[:2000]).mean()
acc_att   = (lda.predict(X_att_sel) == labels_att).mean()
print(f"Accuracy (256 classes) — profiling : {acc_train*100:.1f}%  attaque : {acc_att*100:.1f}%"
      f"  (aleatoire = {100/256:.2f}%)")

# ── Attaque : scoring log-vraisemblance sur 256 candidats cle ─────────────────
# score(k) = sum_i log P(SBox[pt_i XOR k] XOR mask_i[0] | trace_i)
print("Calcul des scores d'attaque...")
log_p = lda.predict_log_proba(X_att_sel)  # (10000, 256)

guesses  = np.arange(256, dtype=np.int64)
pt_col   = pt_att[:, TARGET].astype(np.int64)
mask_col = mask_att[:, 0].astype(np.int64)   # mask[0] — meme index que le profiling
sbox_inter = (SBOX[(pt_col[:, None] ^ guesses[None, :]) & 0xFF].astype(np.int64)
              ^ mask_col[:, None]) & 0xFF     # (10000, 256)

classes  = lda.classes_.astype(np.int64)
sbox_idx = np.searchsorted(classes, sbox_inter)
log_scores = np.take_along_axis(log_p, sbox_idx, axis=1)  # (10000, 256)

# ── Courbe de rang ────────────────────────────────────────────────────────────
step = 50
ranks_n, ranks_v = [], []
cumul = np.cumsum(log_scores, axis=0)
for n in range(step, len(X_att) + 1, step):
    order = np.argsort(-cumul[n - 1])
    ranks_n.append(n)
    ranks_v.append(int(np.where(order == TRUE_KEY)[0][0]) + 1)

idx_rank1 = next((i for i, r in enumerate(ranks_v) if r == 1), -1)
print(f"Rang final ({len(X_att)} traces) : {ranks_v[-1]}/256")
if idx_rank1 >= 0:
    print(f"Rang 1 atteint a : {ranks_n[idx_rank1]} traces")
else:
    print("Rang 1 non atteint sur 10 000 traces")

plt.figure(figsize=(12, 5))
plt.plot(ranks_n, ranks_v, color='darkgreen', linewidth=2, label='LDA Template (256 classes, SNR top-100)')
plt.axhline(1, color='red', linestyle='--', linewidth=1.5, label='Rang 1 = cle retrouvee')
plt.xlabel("Traces d'attaque utilisees")
plt.ylabel("Rang de la vraie cle")
plt.title(f"Template LDA — ASCAD octet {TARGET} — Vraie cle 0x{TRUE_KEY:02X}")
plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/04_lda_rank.png', dpi=150)
print("Graphe sauvegarde : 04_lda_rank.png")
