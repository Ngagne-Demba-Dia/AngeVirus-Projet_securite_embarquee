"""
13_contrastive.py — Direction 2 : Self-supervised Contrastive Learning (SimCLR).
Méthode : Chen et al. 2020 "A Simple Framework for Contrastive Learning" (20 000 citations).
Objectif : atteindre 70%+ sur AES-T700 avec seulement 5% de données étiquetées.

Pipeline :
  Phase 1 — Pré-entraînement auto-supervisé (SANS labels) via NT-Xent loss
  Phase 2 — Fine-tuning supervisé avec 5%, 10%, 20% de labels

Fichiers générés (captures rapport) :
  results/13_CL_courbe_few_shot.png     — accuracy vs % données étiquetées
  results/13_CL_representations_tsne.png — t-SNE des représentations apprises
  results/13_CL_comparaison.png         — CL vs supervisé (CORAL)
  results/13_CL_bilan.png               — tableau visuel bilan
  results/13_CL_metrics.json            — métriques complètes
  results/cnn1d_CL_backbone_T700.pt     — backbone pré-entraîné (sans labels)
  results/cnn1d_CL_ft05_T700.pt         — fine-tuné 5% labels
  results/cnn1d_CL_ft10_T700.pt         — fine-tuné 10% labels
  results/cnn1d_CL_ft20_T700.pt         — fine-tuné 20% labels
"""
import copy
import json
import numpy as np
import yaml
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

from models import CNN1D

# ── Hyperparamètres ────────────────────────────────────────────────────────────
TARGET_BM      = "AES-T700"
PRETRAIN_EPOCHS = 200      # pré-entraînement sans labels (GPU)
FINETUNE_EPOCHS = 60       # fine-tuning supervisé
PRETRAIN_LR    = 3e-4
FINETUNE_LR    = 1e-4
BATCH_SIZE     = 256
TEMPERATURE    = 0.5       # température NT-Xent (SimCLR standard)
PROJ_DIM       = 128       # dimension projection head
LABEL_FRACS    = [0.05, 0.10, 0.20]   # fractions de labels testées
LABEL_NAMES    = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]

# Référence supervisée CORAL (Étape 4) pour comparaison
CORAL_RESULTS  = {0.05: 0.50, 0.10: 0.60, 0.20: 0.677}


# ── Augmentations pour traces de puissance ────────────────────────────────────
class PowerTraceAugment:
    """
    Augmentations pour features side-channel (500 dims).
    Simule les variations de mesure réelles : bruit, dérive, masquage.
    """
    def __init__(self, noise_std=0.15, mask_prob=0.15, scale_range=(0.85, 1.15)):
        self.noise_std   = noise_std
        self.mask_prob   = mask_prob
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Bruit gaussien (simule variabilité mesure)
        x = x + torch.randn_like(x) * self.noise_std

        # 2. Masquage aléatoire de features (simule capteurs défaillants)
        mask = torch.rand(x.shape[-1]) > self.mask_prob
        x = x * mask.to(x.device)

        # 3. Scaling (simule dérive d'amplitude)
        scale = torch.empty(1).uniform_(*self.scale_range).to(x.device)
        x = x * scale

        return x


# ── Backbone CNN1D avec extraction de features intermédiaires ─────────────────
class CNNBackbone(nn.Module):
    """CNN1D dont le forward retourne les features après avg pool (avant classifieur)."""
    def __init__(self, base_model: CNN1D):
        super().__init__()
        self.conv = base_model.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x.unsqueeze(1))   # (B, 128, 16)
        return out.flatten(1)              # (B, 2048)


# ── Projection Head (SimCLR) ──────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── NT-Xent Loss (SimCLR) ─────────────────────────────────────────────────────
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NT-Xent : les paires (z1[i], z2[i]) sont positives, tout le reste négatif.
    """
    N = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0)           # (2N, D)

    sim = torch.mm(z, z.t()) / temperature     # (2N, 2N)

    # Masquer la diagonale (self-similarity)
    mask = torch.eye(2 * N, dtype=bool, device=z.device)
    sim  = sim.masked_fill(mask, -1e9)

    # Labels : z1[i] ↔ z2[i]  →  labels[i] = i+N, labels[i+N] = i
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)

    return F.cross_entropy(sim, labels)


# ── Phase 1 : Pré-entraînement auto-supervisé ─────────────────────────────────
def pretrain(backbone, proj_head, X_unlabeled, device, augment,
             epochs, lr, batch_size, temperature):
    params = list(backbone.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t  = torch.FloatTensor(X_unlabeled)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    backbone.train(); proj_head.train()
    print(f"\n{'Epoch':>6} | {'NT-Xent Loss':>13}")
    print("-"*25)

    for ep in range(epochs):
        total_loss, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            x1 = augment(xb)
            x2 = augment(xb)

            z1 = proj_head(backbone(x1))
            z2 = proj_head(backbone(x2))
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1

        scheduler.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"{ep+1:>6} | {total_loss/n:>13.4f}")

    backbone.eval(); proj_head.eval()
    return backbone, proj_head


# ── Phase 2 : Fine-tuning supervisé (backbone gelé) ──────────────────────────
class CLClassifier(nn.Module):
    def __init__(self, backbone: CNNBackbone, n_classes: int = 3, freeze: bool = True):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.classifier(self.backbone(x))


def finetune_classifier(backbone, X_tr, y_tr, X_te, y_te, device,
                         epochs, lr, batch_size, freeze=True):
    model = CLClassifier(copy.deepcopy(backbone), freeze=freeze).to(device)
    params = model.classifier.parameters() if freeze else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te).to(device)).argmax(1).cpu().numpy()
    return model, float(accuracy_score(y_te, preds))


# ── Visualisation t-SNE des représentations ────────────────────────────────────
def plot_tsne(backbone, X, y, device, title, path):
    backbone.eval()
    with torch.no_grad():
        feats = backbone(torch.FloatTensor(X).to(device)).cpu().numpy()
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats[:2000])
    y_sub  = y[:2000]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "crimson"]
    for i, name in enumerate(LABEL_NAMES):
        mask = y_sub == i
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[i],
                   label=name, alpha=0.5, s=10)
    ax.set_title(title); ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    src = cfg["dataset"]["benchmarks"]["source"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  Target : {TARGET_BM}")

    # ── Charger et normaliser données T700 ────────────────────────────────────
    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_ms_mean_{TARGET_BM}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_ms_scale_{TARGET_BM}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    d = np.load(results_dir / f"features_{TARGET_BM}.npz")
    X_raw, y_all = d["X"].astype(np.float32), d["y"].astype(np.int64)
    X_all = scaler.transform(X_raw).astype(np.float32)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_all))
    n_te = int(len(X_all) * 0.2)
    X_te, y_te = X_all[idx[:n_te]], y_all[idx[:n_te]]
    X_tr, y_tr = X_all[idx[n_te:]], y_all[idx[n_te:]]

    print(f"Train (unlabeled pool) : {len(X_tr)} | Test : {len(X_te)}")

    # ── Initialiser backbone à partir du modèle multi-source T700 ─────────────
    base = CNN1D()
    ms_path = results_dir / f"cnn1d_ms_ft_{TARGET_BM}.pt"
    if ms_path.exists():
        sd = {k: v for k, v in torch.load(ms_path, map_location="cpu").items()
              if k in base.state_dict()}
        base.load_state_dict(sd, strict=False)
        print(f"Backbone initialisé depuis {ms_path.name}")

    backbone  = CNNBackbone(base).to(device)
    proj_head = ProjectionHead(2048, 512, PROJ_DIM).to(device)
    augment   = PowerTraceAugment()

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── Phase 1 : Pré-entraînement auto-supervisé ─────────────────────────────
    print(f"\n=== PHASE 1 : Pré-entraînement SimCLR ({PRETRAIN_EPOCHS} epochs, SANS labels) ===")
    with mlflow.start_run(run_name=f"SimCLR_Pretrain_{TARGET_BM}"):
        mlflow.log_params({
            "target": TARGET_BM, "pretrain_epochs": PRETRAIN_EPOCHS,
            "temperature": TEMPERATURE, "proj_dim": PROJ_DIM,
            "batch_size": BATCH_SIZE, "n_unlabeled": len(X_tr),
        })

        backbone, proj_head = pretrain(
            backbone, proj_head, X_tr, device, augment,
            PRETRAIN_EPOCHS, PRETRAIN_LR, BATCH_SIZE, TEMPERATURE
        )

    # Sauvegarder backbone pré-entraîné
    torch.save(backbone.state_dict(), results_dir / "cnn1d_CL_backbone_T700.pt")

    # t-SNE AVANT fine-tuning
    plot_tsne(backbone, X_te, y_te, device,
              f"Représentations SimCLR — {TARGET_BM} (après pré-entraînement sans labels)",
              results_dir / "13_CL_representations_tsne.png")
    print("t-SNE sauvegardé.")

    # ── Phase 2 : Fine-tuning avec différentes fractions de labels ────────────
    print(f"\n=== PHASE 2 : Fine-tuning supervisé (5%, 10%, 20% labels) ===")
    cl_results   = {}
    sup_results  = {}

    for frac in LABEL_FRACS:
        n_labeled = max(30, int(len(X_tr) * frac))
        idx_lab   = rng.choice(len(X_tr), n_labeled, replace=False)
        X_lab, y_lab = X_tr[idx_lab], y_tr[idx_lab]

        print(f"\n  [{frac:.0%} labels = {n_labeled} traces]")

        # -- Contrastif fine-tuning (backbone gelé)
        with mlflow.start_run(run_name=f"SimCLR_FT_{int(frac*100)}pct_{TARGET_BM}"):
            mlflow.log_params({"frac": frac, "n_labeled": n_labeled})

            model_ft, acc_cl = finetune_classifier(
                backbone, X_lab, y_lab, X_te, y_te, device,
                FINETUNE_EPOCHS, FINETUNE_LR, BATCH_SIZE, freeze=True
            )
            cl_results[frac] = acc_cl
            mlflow.log_metric("acc_contrastive", acc_cl)

            # Sauvegarder
            tag = f"{int(frac*100):02d}"
            torch.save(model_ft.state_dict(),
                       results_dir / f"cnn1d_CL_ft{tag}_T700.pt")

        # -- Supervisé pur (baseline) avec mêmes labels
        base_sup = copy.deepcopy(base).to(device)
        base_sup.eval()
        clf_sup = nn.Sequential(nn.Linear(2048, 3)).to(device)  # simple linear
        # Utiliser CNN1D complet comme baseline supervisé
        model_sup = CNN1D().to(device)
        opt_sup   = torch.optim.Adam(model_sup.parameters(), lr=FINETUNE_LR)
        crit_sup  = nn.CrossEntropyLoss()
        loader_sup = DataLoader(
            TensorDataset(torch.FloatTensor(X_lab), torch.LongTensor(y_lab)),
            batch_size=min(32, n_labeled), shuffle=True
        )
        model_sup.train()
        for _ in range(FINETUNE_EPOCHS):
            for xb, yb in loader_sup:
                xb, yb = xb.to(device), yb.to(device)
                opt_sup.zero_grad()
                crit_sup(model_sup(xb), yb).backward()
                opt_sup.step()
        model_sup.eval()
        with torch.no_grad():
            preds = model_sup(torch.FloatTensor(X_te).to(device)).argmax(1).cpu().numpy()
        acc_sup = float(accuracy_score(y_te, preds))
        sup_results[frac] = acc_sup

        print(f"    SimCLR : {acc_cl:.4f}  |  Supervisé pur : {acc_sup:.4f}  "
              f"|  CORAL ref : {CORAL_RESULTS[frac]:.4f}")

    # ── Figure 1 : Courbe few-shot ─────────────────────────────────────────────
    fracs_pct = [int(f*100) for f in LABEL_FRACS]
    cl_accs   = [cl_results[f]   for f in LABEL_FRACS]
    sup_accs  = [sup_results[f]  for f in LABEL_FRACS]
    coral_accs= [CORAL_RESULTS[f] for f in LABEL_FRACS]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fracs_pct, cl_accs,    "o-",  color="steelblue",  lw=2.5, ms=9,
            label="SimCLR (Direction 2)")
    ax.plot(fracs_pct, sup_accs,   "s--", color="crimson",    lw=2,   ms=8,
            label="Supervisé pur (baseline)")
    ax.plot(fracs_pct, coral_accs, "^:",  color="darkorange", lw=2,   ms=8,
            label="CORAL multi-source (Étape 4)")
    ax.axhline(0.70, color="green", linestyle=":", lw=1.5, label="Objectif 70%")
    for x, y in zip(fracs_pct, cl_accs):
        ax.annotate(f"{y:.2%}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="steelblue")
    ax.set_xlabel("% données étiquetées utilisées")
    ax.set_ylabel("Accuracy sur AES-T700")
    ax.set_title("Direction 2 — SimCLR : Accuracy vs Données étiquetées\n"
                 f"(pré-entraîné sur {len(X_tr)} traces SANS labels)")
    ax.set_xticks(fracs_pct)
    ax.set_xticklabels([f"{f}%" for f in fracs_pct])
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.3, 0.95)
    plt.tight_layout()
    out1 = results_dir / "13_CL_courbe_few_shot.png"
    plt.savefig(out1, dpi=150); plt.close()

    # ── Figure 2 : Comparaison barres ──────────────────────────────────────────
    x  = np.arange(len(LABEL_FRACS))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, cl_accs,    w, color="steelblue",  alpha=0.85, label="SimCLR (Direction 2)")
    b2 = ax.bar(x,     sup_accs,   w, color="crimson",    alpha=0.85, label="Supervisé pur")
    b3 = ax.bar(x + w, coral_accs, w, color="darkorange", alpha=0.85, label="CORAL (Étape 4)")
    for bars in [b1, b2, b3]:
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(0.70, color="green", linestyle=":", lw=1.5, label="Objectif 70%")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(f*100)}% labels" for f in LABEL_FRACS])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Comparaison : SimCLR vs Supervisé vs CORAL sur AES-T700")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = results_dir / "13_CL_comparaison.png"
    plt.savefig(out2, dpi=150); plt.close()

    # ── Figure 3 : Bilan tableau ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    rows = []
    for frac in LABEL_FRACS:
        gain = cl_results[frac] - CORAL_RESULTS[frac]
        rows.append([
            f"{int(frac*100)}% labels",
            f"{sup_results[frac]:.2%}",
            f"{CORAL_RESULTS[frac]:.2%}",
            f"{cl_results[frac]:.2%}",
            f"{gain:+.2%}",
            "✓ 70%+" if cl_results[frac] >= 0.70 else "proche" if cl_results[frac] >= 0.65 else "–"
        ])
    cols = ["Données étiquetées", "Supervisé pur", "CORAL (Étape 4)",
            "SimCLR (Dir.2)", "Gain vs CORAL", "Objectif"]
    cell_colors = []
    for row in rows:
        gain_val = float(row[4].replace("%","").replace("+",""))
        g_color = "#ccffcc" if gain_val > 0 else "#ffcccc"
        obj_color = "#ccffcc" if "✓" in row[5] else "#fff9cc"
        cell_colors.append(["#ecf0f1"]*4 + [g_color, obj_color])

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                   cellColours=cell_colors)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.2)
    ax.set_title("Direction 2 — Bilan Self-supervised Contrastive Learning (SimCLR)\n"
                 f"Cible : {TARGET_BM}  |  Pré-entraîné sur {len(X_tr)} traces sans labels",
                 fontsize=11, fontweight="bold", pad=15)
    out3 = results_dir / "13_CL_bilan.png"
    plt.tight_layout(); plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()

    # ── Métriques JSON ─────────────────────────────────────────────────────────
    metrics = {
        "target": TARGET_BM, "pretrain_epochs": PRETRAIN_EPOCHS,
        "n_unlabeled": len(X_tr), "temperature": TEMPERATURE,
        "results": {
            str(int(f*100))+"pct": {
                "simclr":     round(cl_results[f], 4),
                "supervised": round(sup_results[f], 4),
                "coral_ref":  round(CORAL_RESULTS[f], 4),
                "gain_vs_coral": round(cl_results[f] - CORAL_RESULTS[f], 4),
            } for f in LABEL_FRACS
        }
    }
    (results_dir / "13_CL_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Résumé terminal ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("BILAN DIRECTION 2 — SELF-SUPERVISED CONTRASTIVE LEARNING (SimCLR)")
    print("="*70)
    print(f"  Pré-entraîné sur {len(X_tr)} traces SANS labels ({TARGET_BM})")
    print(f"  {'Labels':>10} | {'SimCLR':>8} | {'Supervisé':>10} | {'CORAL ref':>10} | {'Gain':>8}")
    print("  " + "-"*55)
    for frac in LABEL_FRACS:
        gain = cl_results[frac] - CORAL_RESULTS[frac]
        flag = "✓ 70%+" if cl_results[frac] >= 0.70 else ""
        print(f"  {int(frac*100):>9}% | {cl_results[frac]:>8.2%} | "
              f"{sup_results[frac]:>10.2%} | {CORAL_RESULTS[frac]:>10.2%} | "
              f"{gain:>+8.2%} {flag}")
    print(f"\n  Fichiers captures rapport :")
    for out in [out1, out2, out3,
                results_dir/"13_CL_representations_tsne.png",
                results_dir/"13_CL_metrics.json"]:
        print(f"    {Path(out).name}")
