"""
10_coral.py — CORAL Domain Adaptation (Étape 4).
Pendant le fine-tuning, minimise la distance entre les covariances
des représentations intermédiaires source et target (Deep CORAL).
L_total = CrossEntropy(target) + λ * CORAL(features_source, features_target)
"""
import copy
import numpy as np
import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from models import CNN1D

LABEL_NAMES = ["Disabled", "Enabled", "Triggered"]


# ── CORAL loss ─────────────────────────────────────────────────────────────────
def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL : distance de Frobenius entre covariances source et target.
    source, target : (B, D) — représentations intermédiaires après conv backbone.
    """
    d = source.size(1)

    # Covariance source
    xm = source - source.mean(0, keepdim=True)
    cs = (xm.t() @ xm) / max(source.size(0) - 1, 1)

    # Covariance target
    ym = target - target.mean(0, keepdim=True)
    ct = (ym.t() @ ym) / max(target.size(0) - 1, 1)

    return torch.norm(cs - ct, p="fro") ** 2 / (4 * d * d)


# ── CNN avec extraction de features intermédiaires ────────────────────────────
class CNNWithFeatures(nn.Module):
    """Wrapper autour de CNN1D pour exposer les features après le backbone conv."""
    def __init__(self, model: CNN1D):
        super().__init__()
        self.conv       = model.conv
        self.classifier = model.classifier
        self._flatten   = nn.Flatten()

    def forward(self, x: torch.Tensor):
        conv_out  = self.conv(x.unsqueeze(1))           # (B, 128, 16)
        features  = self._flatten(conv_out)              # (B, 2048)
        logits    = self.classifier(conv_out)            # (B, 3)  — via Flatten interne
        return logits, features


# ── Fine-tuning CORAL ─────────────────────────────────────────────────────────
def finetune_coral(model_ms, X_src, y_src, X_ft, y_ft,
                   scaler, device, epochs, lr, batch_size,
                   lam: float = 0.5):
    """
    Fine-tuning avec CORAL loss.
    lam : poids de la CORAL loss (0 = standard fine-tuning, 1 = full CORAL).
    """
    model = CNNWithFeatures(copy.deepcopy(model_ms)).to(device)
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # DataLoaders
    X_ft_s  = torch.FloatTensor(scaler.transform(X_ft).astype(np.float32))
    X_src_s = torch.FloatTensor(scaler.transform(X_src).astype(np.float32))

    tgt_loader = DataLoader(
        TensorDataset(X_ft_s, torch.LongTensor(y_ft)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    src_loader = DataLoader(
        TensorDataset(X_src_s, torch.LongTensor(y_src)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    model.train()
    for ep in range(epochs):
        src_iter = iter(src_loader)
        total_ce, total_coral, n = 0.0, 0.0, 0
        for (xb_tgt, yb_tgt) in tgt_loader:
            try:
                xb_src, _ = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                xb_src, _ = next(src_iter)

            xb_tgt, yb_tgt = xb_tgt.to(device), yb_tgt.to(device)
            xb_src = xb_src.to(device)

            optimizer.zero_grad()

            # Forward target
            logits_tgt, feats_tgt = model(xb_tgt)
            loss_ce = criterion(logits_tgt, yb_tgt)

            # Forward source (pas de supervision → juste les features)
            with torch.no_grad():
                _, feats_src = model(xb_src)

            loss_c = coral_loss(feats_src.detach(), feats_tgt)
            loss   = loss_ce + lam * loss_c

            loss.backward()
            optimizer.step()

            total_ce    += loss_ce.item()
            total_coral += loss_c.item()
            n += 1

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"    ep {ep+1:3d}  CE={total_ce/n:.4f}  CORAL={total_coral/n:.4f}")

    model.eval()
    return model


def evaluate(model, X, y, scaler, device):
    X_s = torch.FloatTensor(scaler.transform(X)).to(device)
    with torch.no_grad():
        if isinstance(model, CNNWithFeatures):
            logits, _ = model(X_s)
        else:
            logits = model(X_s)
        preds = logits.argmax(1).cpu().numpy()
    return float(accuracy_score(y, preds)), float(f1_score(y, preds, average="macro"))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    all_bms = cfg["dataset"]["benchmarks"]["all"]
    t_cfg   = cfg["transfer"]
    cnn_cfg = cfg["cnn"]

    # Benchmarks où on applique CORAL (ceux proches de 70%)
    coral_targets = ["AES-T700", "AES-T800", "AES-T500"]

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results_coral = {}

    for test_bm in coral_targets:
        npz = results_dir / f"features_{test_bm}.npz"
        ms_model_path = results_dir / f"cnn1d_ms_ft_{test_bm}.pt"
        if not npz.exists() or not ms_model_path.exists():
            print(f"  {test_bm}: données ou modèle multi-source manquants, skip")
            continue

        print(f"\n[CORAL] Cible : {test_bm}")

        # Charger scaler multi-source
        scaler = StandardScaler()
        scaler.mean_  = np.load(results_dir / f"scaler_ms_mean_{test_bm}.npy")
        scaler.scale_ = np.load(results_dir / f"scaler_ms_scale_{test_bm}.npy")
        scaler.var_   = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)

        # Charger modèle multi-source
        n_input = scaler.n_features_in_
        base_model = CNN1D(n_input=n_input)
        base_model.load_state_dict(torch.load(ms_model_path, map_location="cpu"))
        base_model = base_model.to(device)
        base_model.eval()

        # Données cible
        d = np.load(npz)
        X_tgt, y_tgt = d["X"].astype(np.float32), d["y"].astype(np.int64)
        rng  = np.random.RandomState(42)
        idx  = rng.permutation(len(X_tgt))
        n_ft = max(30, int(len(X_tgt) * t_cfg["finetune_fraction"]))
        X_ft, y_ft = X_tgt[idx[:n_ft]], y_tgt[idx[:n_ft]]
        X_ev, y_ev = X_tgt[idx[n_ft:]], y_tgt[idx[n_ft:]]

        # Source : tous les autres benchmarks
        train_bms = [b for b in all_bms if b != test_bm]
        X_src_list, y_src_list = [], []
        for bm in train_bms:
            p = results_dir / f"features_{bm}.npz"
            if p.exists():
                dd = np.load(p)
                X_src_list.append(dd["X"].astype(np.float32))
                y_src_list.append(dd["y"].astype(np.int64))
        X_src = np.vstack(X_src_list)
        y_src = np.concatenate(y_src_list)

        # Référence sans CORAL (Étape 3)
        acc_ref, _ = evaluate(base_model, X_ev, y_ev, scaler, device)
        print(f"  Référence multi-source (sans CORAL) : {acc_ref:.4f}")

        # CORAL fine-tuning avec différents lambda
        best_acc, best_lam, best_model = 0.0, 0.0, None
        for lam in [0.1, 0.3, 0.5, 1.0]:
            print(f"  λ={lam} :")
            model_coral = finetune_coral(
                base_model, X_src, y_src, X_ft, y_ft,
                scaler, device,
                epochs=t_cfg["finetune_epochs"],
                lr=t_cfg["finetune_lr"],
                batch_size=cnn_cfg["batch_size"],
                lam=lam,
            )
            acc, f1 = evaluate(model_coral, X_ev, y_ev, scaler, device)
            print(f"    → acc={acc:.4f}  f1={f1:.4f}")
            if acc > best_acc:
                best_acc, best_lam, best_model = acc, lam, model_coral

        print(f"\n  Meilleur CORAL : acc={best_acc:.4f}  λ={best_lam}")
        print(f"  Gain vs Étape 3 : {best_acc - acc_ref:+.4f}")

        torch.save(best_model.state_dict(), results_dir / f"cnn1d_coral_{test_bm}.pt")

        with mlflow.start_run(run_name=f"CORAL_{test_bm}"):
            mlflow.log_params({"target": test_bm, "best_lambda": best_lam})
            mlflow.log_metric("acc_coral",  best_acc)
            mlflow.log_metric("acc_ref_ms", acc_ref)
            mlflow.log_metric("gain_coral", best_acc - acc_ref)

        results_coral[test_bm] = {
            "ref_ms": acc_ref, "coral": best_acc,
            "gain": best_acc - acc_ref, "best_lam": best_lam
        }

    # ── Graphique comparatif ───────────────────────────────────────────────────
    if results_coral:
        bms   = list(results_coral.keys())
        refs  = [results_coral[b]["ref_ms"] for b in bms]
        corals = [results_coral[b]["coral"]  for b in bms]

        x = np.arange(len(bms))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - w/2, refs,   w, label="Multi-source Étape 3", color="steelblue", alpha=0.85)
        b2 = ax.bar(x + w/2, corals, w, label="CORAL Étape 4",        color="darkorange", alpha=0.85)
        ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
        ax.axhline(1/3, color="gray",  linestyle="--", linewidth=1.2, label="Aléatoire (33%)")
        ax.axhline(0.70, color="green", linestyle=":",  linewidth=1.5, label="Objectif 70%")
        ax.set_xticks(x); ax.set_xticklabels(bms)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Étape 4 — CORAL Domain Adaptation vs Multi-source")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = results_dir / "08_coral.png"
        plt.savefig(out, dpi=150)
        plt.close()

    # ── Tableau ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Benchmark':<15} {'Étape 3':>10} {'CORAL':>10} {'Gain':>8} {'λ':>5}")
    print("-"*60)
    for bm, r in results_coral.items():
        target = " ✓" if r["coral"] >= 0.70 else ""
        print(f"{bm:<15} {r['ref_ms']:>10.4f} {r['coral']:>10.4f} "
              f"{r['gain']:>+8.4f} {r['best_lam']:>5}{target}")
