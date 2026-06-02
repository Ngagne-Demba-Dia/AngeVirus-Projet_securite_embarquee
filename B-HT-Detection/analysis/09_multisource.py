"""
09_multisource.py — Entraînement multi-source + fine-tuning (Étape 3).
Pour chaque benchmark cible, entraîne le CNN sur TOUS les autres benchmarks
(Leave-One-Benchmark-Out), puis fine-tune sur 20% du benchmark cible.
Objectif : réduire le domain shift en apprenant sur plusieurs distributions.
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


def load_features(results_dir: Path, bm: str):
    d = np.load(results_dir / f"features_{bm}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def train_cnn(X_tr, y_tr, cfg, device, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_tr).astype(np.float32)
    else:
        X_s = scaler.transform(X_tr).astype(np.float32)

    n_input = X_tr.shape[1]
    model = CNN1D(n_input=n_input).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_s), torch.LongTensor(y_tr)),
        batch_size=cfg["batch_size"], shuffle=True
    )

    model.train()
    for ep in range(cfg["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model, scaler


def finetune(model, X_ft, y_ft, scaler, cfg_t, cnn_cfg, device):
    model_ft = copy.deepcopy(model)
    for param in model_ft.parameters():
        param.requires_grad = True

    opt = torch.optim.Adam(model_ft.parameters(), lr=cfg_t["finetune_lr"])
    criterion = nn.CrossEntropyLoss()

    X_s = scaler.transform(X_ft).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_s), torch.LongTensor(y_ft)),
        batch_size=cnn_cfg["batch_size"], shuffle=True
    )

    model_ft.train()
    for _ in range(cfg_t["finetune_epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model_ft(xb), yb).backward()
            opt.step()

    model_ft.eval()
    return model_ft


def evaluate(model, X, y, scaler, device):
    X_s = torch.FloatTensor(scaler.transform(X)).to(device)
    with torch.no_grad():
        preds = model(X_s).argmax(1).cpu().numpy()
    return accuracy_score(y, preds), f1_score(y, preds, average="macro")


if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    all_bms = cfg["dataset"]["benchmarks"]["all"]
    bm_data = {}
    for bm in all_bms:
        p = results_dir / f"features_{bm}.npz"
        if p.exists():
            bm_data[bm] = load_features(results_dir, bm)
    available = sorted(bm_data.keys())
    print(f"Benchmarks disponibles : {available}")

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    t_cfg   = cfg["transfer"]
    cnn_cfg = cfg["cnn"]
    results_ms  = {}   # multi-source fine-tuned
    results_ref = {}   # référence étape 2 (single-source)

    # Charger résultats de référence (modèle T400 single-source) pour comparaison
    src = cfg["dataset"]["benchmarks"]["source"]
    model_src_path = results_dir / f"cnn1d_{src}.pt"
    scaler_src = StandardScaler()
    scaler_src.mean_  = np.load(results_dir / f"scaler_mean_{src}.npy")
    scaler_src.scale_ = np.load(results_dir / f"scaler_scale_{src}.npy")
    scaler_src.var_   = scaler_src.scale_ ** 2
    scaler_src.n_features_in_ = len(scaler_src.mean_)

    print("\n=== ENTRAÎNEMENT MULTI-SOURCE (Leave-One-Benchmark-Out) ===\n")

    for test_bm in available:
        train_bms = [b for b in available if b != test_bm]
        if len(train_bms) < 2:
            continue

        print(f"[{test_bm}] Train sur : {train_bms}")

        # Données multi-source
        X_train = np.vstack([bm_data[b][0] for b in train_bms])
        y_train = np.concatenate([bm_data[b][1] for b in train_bms])

        # Données cible
        X_tgt, y_tgt = bm_data[test_bm]
        rng   = np.random.RandomState(42)
        idx   = rng.permutation(len(X_tgt))
        n_ft  = max(30, int(len(X_tgt) * t_cfg["finetune_fraction"]))
        X_ft, y_ft = X_tgt[idx[:n_ft]], y_tgt[idx[:n_ft]]
        X_ev, y_ev = X_tgt[idx[n_ft:]], y_tgt[idx[n_ft:]]

        with mlflow.start_run(run_name=f"MultiSource_{test_bm}"):
            mlflow.log_params({
                "target": test_bm,
                "train_sources": "+".join(train_bms),
                "n_finetune": n_ft,
            })

            # 1. Entraîner modèle multi-source
            model_ms, scaler_ms = train_cnn(X_train, y_train, cnn_cfg, device)

            # 2. Zero-shot multi-source
            acc_zs, f1_zs = evaluate(model_ms, X_ev, y_ev, scaler_ms, device)
            mlflow.log_metric("acc_zeroshot_ms", acc_zs)
            print(f"  Zero-shot multi-source : {acc_zs:.4f}")

            # 3. Fine-tuning sur cible
            model_ft = finetune(model_ms, X_ft, y_ft, scaler_ms, t_cfg, cnn_cfg, device)
            acc_ft, f1_ft = evaluate(model_ft, X_ev, y_ev, scaler_ms, device)
            mlflow.log_metric("acc_finetuned_ms", acc_ft)
            mlflow.log_metric("f1_finetuned_ms", f1_ft)
            print(f"  Fine-tuned  multi-source : {acc_ft:.4f}  (gain vs zero-shot: {acc_ft-acc_zs:+.4f})")

            # Sauvegarder
            torch.save(model_ft.state_dict(), results_dir / f"cnn1d_ms_ft_{test_bm}.pt")
            np.save(results_dir / f"scaler_ms_mean_{test_bm}.npy", scaler_ms.mean_)
            np.save(results_dir / f"scaler_ms_scale_{test_bm}.npy", scaler_ms.scale_)

            results_ms[test_bm] = {
                "zeroshot": acc_zs, "finetuned": acc_ft,
                "f1_zs": f1_zs, "f1_ft": f1_ft, "n_ft": n_ft
            }

    # ── Graphique comparatif ───────────────────────────────────────────────────
    bms     = [b for b in available if b in results_ms]
    zs_ms   = [results_ms[b]["zeroshot"]  for b in bms]
    ft_ms   = [results_ms[b]["finetuned"] for b in bms]

    x = np.arange(len(bms))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, zs_ms, w, label="Zero-shot multi-source", color="steelblue", alpha=0.85)
    b2 = ax.bar(x + w/2, ft_ms, w, label="Fine-tuned multi-source (20%)", color="darkorange", alpha=0.85)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1.2, label="Aléatoire (33%)")
    ax.axhline(0.70, color="green", linestyle=":", linewidth=1.5, label="Objectif 70%")
    ax.set_xticks(x); ax.set_xticklabels(bms, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Étape 3 — Multi-source LOBO + fine-tuning")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = results_dir / "07_multisource.png"
    plt.savefig(out, dpi=150)
    plt.close()

    # ── Tableau final ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Benchmark':<15} {'ZS multi-src':>14} {'FT multi-src':>14} {'n_ft':>6}")
    print("-"*70)
    for bm in bms:
        r = results_ms[bm]
        target = " ✓ 70%+" if r["finetuned"] >= 0.70 else ""
        print(f"{bm:<15} {r['zeroshot']:>14.4f} {r['finetuned']:>14.4f} {r['n_ft']:>6}{target}")
    print(f"\nGraphique sauvegardé : {out}")
