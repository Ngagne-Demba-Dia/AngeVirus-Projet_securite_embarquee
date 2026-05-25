"""
DL-SCA — MLP reference (Benadjila et al. 2018)
Architecture : 5 couches entierement connectees, 200 neurones, BN + ReLU
Dataset : ASCAD.h5 — memes labels que CNN (SBox[pt XOR k] XOR mask[0])
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_CANDIDATES = [
    Path('/mnt/g/DATA/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5'),
    Path('G:/DATA/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5'),
]
ASCAD_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

TARGET  = 2
N_TRAIN = 45000
EPOCHS  = 20
BATCH   = 256
LR      = 1e-3
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")

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

# ── Chargement donnees ────────────────────────────────────────────────────────
print("Chargement ASCAD...")
with h5py.File(ASCAD_PATH, 'r') as f:
    X_prof  = f['Profiling_traces/traces'][:]
    md_prof = f['Profiling_traces/metadata'][:]
    X_att   = f['Attack_traces/traces'][:]
    md_att  = f['Attack_traces/metadata'][:]

pt_prof   = md_prof['plaintext']
key_prof  = md_prof['key']
mask_prof = md_prof['masks']
pt_att    = md_att['plaintext']
key_att   = md_att['key']
mask_att  = md_att['masks']
TRUE_KEY  = int(key_att[0, TARGET])
print(f"Vraie cle octet {TARGET} : 0x{TRUE_KEY:02X}")

# Labels : SBox[pt XOR k] XOR mask[0]
labels = np.array([
    int(SBOX[int(pt_prof[i, TARGET]) ^ int(key_prof[i, TARGET])]) ^ int(mask_prof[i, 0])
    for i in range(len(X_prof))
], dtype=np.int64)

# Normalisation min-max par trace
def normalize(X):
    X = X.astype(np.float32)
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)

X_train = normalize(X_prof[:N_TRAIN])
X_val   = normalize(X_prof[N_TRAIN:])
X_att_n = normalize(X_att)

y_train = labels[:N_TRAIN]
y_val   = labels[N_TRAIN:]
print(f"Train : {X_train.shape}  Val : {X_val.shape}  Attack : {X_att_n.shape}")

ds_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
ds_val   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
loader_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True,  num_workers=0)
loader_val   = DataLoader(ds_val,   batch_size=512,   shuffle=False, num_workers=0)

# ── Architecture MLP (Benadjila et al. 2018) ──────────────────────────────────
class MLP_SCA(nn.Module):
    """MLP 5 couches — architecture de reference ASCAD."""
    def __init__(self, n_samples=700, n_classes=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_samples, 200), nn.ReLU(), nn.BatchNorm1d(200),
            nn.Linear(200, 200),       nn.ReLU(), nn.BatchNorm1d(200),
            nn.Linear(200, 200),       nn.ReLU(), nn.BatchNorm1d(200),
            nn.Linear(200, 200),       nn.ReLU(), nn.BatchNorm1d(200),
            nn.Linear(200, n_classes),
        )

    def forward(self, x):
        return self.net(x)

mlp = MLP_SCA().to(DEVICE)
print(f"MLP parametres : {sum(p.numel() for p in mlp.parameters()):,}")

# ── Entrainement ──────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss()

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

print(f"\nEntrainement MLP — {EPOCHS} epochs")
print(f"{'Ep':>3} | {'Train loss':>10} | {'Val loss':>8} | {'Val acc':>7} | {'Time':>6}")
print("-" * 45)

for epoch in range(EPOCHS):
    t0 = time.time()
    mlp.train()
    total_loss = 0.0
    for xb, yb in loader_train:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(mlp(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    train_loss = total_loss / len(loader_train.dataset)

    mlp.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader_val:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = mlp(xb)
            val_loss += criterion(logits, yb).item() * len(xb)
            correct  += (logits.argmax(1) == yb).sum().item()
    val_loss /= len(loader_val.dataset)
    val_acc   = correct / len(loader_val.dataset)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"{epoch+1:>3} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>6.2%} | {time.time()-t0:>5.1f}s")

# ── Courbes d'apprentissage ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='Train', color='steelblue')
axes[0].plot(history['val_loss'],   label='Val',   color='darkorange')
axes[0].set_title('MLP SCA — Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CrossEntropy'); axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(history['val_acc'], color='darkorange')
axes[1].axhline(1/256, color='gray', linestyle='--', label='Aleatoire (0.39%)')
axes[1].set_title('MLP SCA — Val Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/03_mlp_training.png', dpi=150)
print("Courbes sauvegardees : 03_mlp_training.png")

torch.save(mlp.state_dict(), '../results/mlp_ascad.pt')
print("Modele sauvegarde : mlp_ascad.pt")

# ── Attaque : rang en fonction du nombre de traces ───────────────────────────
print("\nAttaque MLP sur traces ASCAD...")
mlp.eval()
X_t = torch.from_numpy(X_att_n).to(DEVICE)
with torch.no_grad():
    logits = mlp(X_t)
    log_p  = torch.log_softmax(logits, dim=1).cpu().numpy()  # (10000, 256)

guesses  = np.arange(256, dtype=np.uint8)
pt_col   = pt_att[:, TARGET].astype(np.uint16)
mask_col = mask_att[:, 0].astype(np.int64)
sbox_inter = (SBOX[(pt_col[:, None] ^ guesses[None, :]) & 0xFF].astype(np.int64)
              ^ mask_col[:, None]) & 0xFF   # (10000, 256)
log_key_scores = np.take_along_axis(log_p, sbox_inter.astype(np.int64), axis=1)

step = 50
ranks_n, ranks_v = [], []
cumul = np.cumsum(log_key_scores, axis=0)
for n in range(step, len(X_att) + 1, step):
    order = np.argsort(-cumul[n - 1])
    ranks_n.append(n)
    ranks_v.append(int(np.where(order == TRUE_KEY)[0][0]))

idx_r1 = next((i for i, r in enumerate(ranks_v) if r == 0), -1)
print(f"Rang final  ({len(X_att)} traces) : {ranks_v[-1] + 1}/256")
if idx_r1 >= 0:
    print(f"Rang 1 atteint a : {ranks_n[idx_r1]} traces")
else:
    print("Rang 1 non atteint sur 10 000 traces")

plt.figure(figsize=(10, 5))
plt.plot(ranks_n, [r + 1 for r in ranks_v], color='steelblue', linewidth=2, label='MLP (Benadjila 2018)')
plt.axhline(1, color='red', linestyle='--', linewidth=1.5, label='Rang 1 = cle retrouvee')
plt.xlabel("Traces d'attaque")
plt.ylabel("Rang de la vraie cle")
plt.title(f"MLP SCA — ASCAD octet {TARGET} — Vraie cle 0x{TRUE_KEY:02X}")
plt.yscale('log')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/04_mlp_rank.png', dpi=150)
print("Courbe de rang sauvegardee : 04_mlp_rank.png")
