"""
DL-SCA — Attaque par canal auxiliaire basee sur un CNN personnalise
Architecture CNN_SCA : 3 blocs Conv1D double + AvgPool + Dense
Dataset : ASCAD.h5 — AES-128 masque premier ordre sur ATMega8515
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

# ── Config ────────────────────────────────────────────────────────────────
ASCAD_PATH = '/mnt/g/DATA/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5'
TARGET     = 2
N_TRAIN    = 45000
N_VAL      = 5000       # dernieres traces de profiling pour validation
EPOCHS     = 50
BATCH      = 256
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

# ── SBOX ──────────────────────────────────────────────────────────────────
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

# ── Chargement des donnees ────────────────────────────────────────────────
print("Chargement ASCAD...")
with h5py.File(ASCAD_PATH, 'r') as f:
    X_prof  = f['Profiling_traces/traces'][:].astype(np.float32)
    md_prof = f['Profiling_traces/metadata'][:]
    X_att   = f['Attack_traces/traces'][:].astype(np.float32)
    md_att  = f['Attack_traces/metadata'][:]

pt_prof  = md_prof['plaintext']
key_prof = md_prof['key']
pt_att   = md_att['plaintext']
key_att  = md_att['key']
TRUE_KEY = int(key_att[0, TARGET])
print(f"Vraie cle octet {TARGET} : 0x{TRUE_KEY:02X}")

# Labels masques : SBox[pt XOR k] XOR mask  (premier ordre, signal clair)
mask_prof = md_prof['masks']
mask_att  = md_att['masks']
labels = np.array([
    int(SBOX[int(pt_prof[i, TARGET]) ^ int(key_prof[i, TARGET])]) ^ int(mask_prof[i, 0])
    for i in range(len(X_prof))
], dtype=np.int64)

# Normalisation par feature (standard SCA) : moyenne/std sur l'axe des traces
mu  = X_prof[:N_TRAIN].mean(axis=0)          # shape (700,)
std = X_prof[:N_TRAIN].std(axis=0) + 1e-8   # shape (700,)
X_prof = (X_prof - mu) / std
X_att  = (X_att  - mu) / std                # meme mu/std que profiling

# Split train / validation
X_tr, y_tr = X_prof[:N_TRAIN], labels[:N_TRAIN]
X_val, y_val = X_prof[N_TRAIN:N_TRAIN+N_VAL], labels[N_TRAIN:N_TRAIN+N_VAL]

# ── Architecture CNN_SCA ──────────────────────────────────────────────────
class CNN_SCA(nn.Module):
    """
    Architecture personnalisee : 3 blocs Conv1D (32/64/128) + AvgPool.
    Differente du papier de reference :
      - kernels 3 et 5 au lieu de 11
      - AvgPool au lieu de MaxPool
      - activation ReLU standard, pas de blocs VGG
      - tete Dense (512->256) avec Dropout 0.4
    """
    def __init__(self, n_input=700, n_classes=256):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1 : n_input -> n_input//2
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AvgPool1d(2),                           # 700 → 350
            # Bloc 2 : -> //2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AvgPool1d(2),                           # 350 → 175
            # Bloc 3 : -> //5
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AvgPool1d(5),                           # 175 → 35
        )
        flat = 128 * 35                                # = 4480
        self.head = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)       # (N, 1, 700)
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)      # logits (N, 256)

# ── Entrainement ──────────────────────────────────────────────────────────
model = CNN_SCA().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modele CNN_SCA : {n_params:,} parametres")

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

ds_tr  = TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr))
ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
ld_tr  = DataLoader(ds_tr,  batch_size=BATCH, shuffle=True,  num_workers=0)
ld_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=0)

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
print(f"\nEntrainement sur {N_TRAIN} traces, {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, total_ok, total_n = 0.0, 0, 0
    for xb, yb in ld_tr:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_ok   += (out.argmax(1) == yb).sum().item()
        total_n    += len(yb)
    scheduler.step()

    model.eval()
    v_loss, v_ok, v_n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in ld_val:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out   = model(xb)
            v_loss += criterion(out, yb).item() * len(yb)
            v_ok   += (out.argmax(1) == yb).sum().item()
            v_n    += len(yb)

    tr_loss = total_loss / total_n
    tr_acc  = total_ok  / total_n
    vl_loss = v_loss / v_n
    vl_acc  = v_ok   / v_n
    history['loss'].append(tr_loss)
    history['acc'].append(tr_acc)
    history['val_loss'].append(vl_loss)
    history['val_acc'].append(vl_acc)
    print(f"Epoch {epoch:3d}/{EPOCHS}  loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%"
          f"  val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

torch.save(model.state_dict(), '../results/cnn_sca.pt')
print("\nModele sauvegarde : DL-SCA/results/cnn_sca.pt")

# ── Courbe d'apprentissage ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['loss'],     label='Train loss')
ax1.plot(history['val_loss'], label='Val loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('CNN_SCA — Loss'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot([v*100 for v in history['acc']],     label='Train acc')
ax2.plot([v*100 for v in history['val_acc']], label='Val acc')
ax2.axhline(100/256, color='gray', linestyle='--', label=f'Aleatoire {100/256:.1f}%')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('CNN_SCA — Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/01_training_curves.png', dpi=150)
print("Courbes d'apprentissage : 01_training_curves.png")

# ── Attaque SCA ───────────────────────────────────────────────────────────
print("\nAttaque SCA sur 10000 traces...")
model.eval()
X_att_t = torch.from_numpy(X_att).to(DEVICE)

with torch.no_grad():
    log_p = F.log_softmax(model(X_att_t), dim=1).cpu().numpy()  # (10000, 256)

# Score : pour chaque cle k, somme log P(SBox[pt_i^k] XOR mask_i | trace_i)
guesses  = np.arange(256, dtype=np.int64)
pt_col   = pt_att[:, TARGET].astype(np.int64)
mask_col = mask_att[:, 0].astype(np.int64)
sbox_val = (SBOX[(pt_col[:,None] ^ guesses[None,:]) & 0xFF].astype(np.int64) ^ mask_col[:,None]) & 0xFF
log_scores = np.take_along_axis(log_p, sbox_val, axis=1)  # (10000, 256)

# Courbe de rang
step = 10
ranks_n, ranks_v = [], []
cumul = np.cumsum(log_scores, axis=0)
for n in range(step, len(X_att)+1, step):
    order = np.argsort(-cumul[n-1])
    ranks_n.append(n)
    ranks_v.append(int(np.where(order == TRUE_KEY)[0][0]) + 1)

idx1 = next((i for i, r in enumerate(ranks_v) if r == 1), -1)
print(f"Rang final ({len(X_att)} traces) : {ranks_v[-1]}/256")
if idx1 >= 0:
    print(f"Rang 1 atteint a : {ranks_n[idx1]} traces")
else:
    print("Rang 1 non atteint — augmenter le nombre d'epochs ou de traces")

plt.figure(figsize=(12, 5))
plt.plot(ranks_n, ranks_v, color='royalblue', linewidth=2, label='CNN_SCA (perso)')
plt.axhline(1, color='red', linestyle='--', linewidth=1.5, label='Rang 1 = cle trouvee')
plt.axhline(128, color='gray', linestyle=':', label='Aleatoire (rang 128)')
plt.xlabel("Traces d'attaque utilisees")
plt.ylabel("Rang de la vraie cle")
plt.title(f"DL-SCA CNN personnalise — ASCAD octet {TARGET} — Cle 0x{TRUE_KEY:02X}")
plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/02_rank_curve.png', dpi=150)
print("Courbe de rang : 02_rank_curve.png")
plt.show()
