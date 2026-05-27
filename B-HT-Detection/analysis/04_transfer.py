"""
04_transfer.py — Domain shift + Transfer Learning / Fine-tuning.
Pipeline :
  1. Charger le CNN entraîné sur AES-T400 (source)
  2. Tester zero-shot sur T500/T600/T700/T800/T1100 → mesure du domain shift
  3. Fine-tuning : geler le backbone conv, entraîner seulement le classifieur
     sur 10% des traces du domaine cible
  4. Comparer zero-shot vs fine-tuned → courbe de généralisation
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models import CNN1D

LABEL_NAMES = ["Disabled", "Enabled", "Triggered"]


def load_features(results_dir: Path, bm: str):
    d = np.load(results_dir / f"features_{bm}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def evaluate(model, X, y, scaler, device) -> tuple:
    model.eval()
    X_s = torch.FloatTensor(scaler.transform(X)).to(device)
    with torch.no_grad():
        y_pred = model(X_s).argmax(1).cpu().numpy()
    return accuracy_score(y, y_pred), f1_score(y, y_pred, average="macro"), y_pred


def finetune(model, X_ft, y_ft, scaler, device, lr, epochs, batch, freeze_backbone):
    model_ft = copy.deepcopy(model)

    if freeze_backbone:
        # Geler les couches conv — entraîner seulement le classifieur
        for param in model_ft.conv.parameters():
            param.requires_grad = False
        trainable = list(model_ft.classifier.parameters())
    else:
        trainable = list(model_ft.parameters())

    opt = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_s = scaler.transform(X_ft).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_s), torch.LongTensor(y_ft)),
        batch_size=batch, shuffle=True
    )

    model_ft.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model_ft(xb), yb).backward()
            opt.step()

    # Dégeler tout
    for param in model_ft.parameters():
        param.requires_grad = True

    return model_ft


if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    src = cfg["dataset"]["benchmarks"]["source"]
    targets = cfg["dataset"]["benchmarks"]["targets"]
    all_bms = [src] + targets

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Charger source + scaler
    X_src, y_src = load_features(results_dir, src)
    scaler = StandardScaler()
    scaler.fit(X_src)

    # Charger le modèle source entraîné
    model_src = CNN1D().to(device)
    weights_path = results_dir / f"cnn1d_{src}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Modele non trouve : {weights_path}. Lancer 03_train.py d'abord.")
    model_src.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Modele CNN charge : {weights_path}")

    t_cfg = cfg["transfer"]
    results = {}

    for bm in all_bms:
        npz = results_dir / f"features_{bm}.npz"
        if not npz.exists():
            print(f"  {bm}: features non trouvees, skip")
            continue

        X_tgt, y_tgt = load_features(results_dir, bm)

        # Partition : fraction fine-tuning / évaluation
        rng  = np.random.RandomState(42)
        idx  = rng.permutation(len(X_tgt))
        n_ft = max(30, int(len(X_tgt) * t_cfg["finetune_fraction"]))
        X_ft, y_ft = X_tgt[idx[:n_ft]], y_tgt[idx[:n_ft]]
        X_ev, y_ev = X_tgt[idx[n_ft:]], y_tgt[idx[n_ft:]]

        with mlflow.start_run(run_name=f"Transfer_{src}->{bm}"):
            mlflow.log_params({"source": src, "target": bm,
                               "n_finetune": n_ft,
                               "freeze_backbone": t_cfg["freeze_backbone"]})

            # 1. Zero-shot
            acc_zs, f1_zs, _ = evaluate(model_src, X_ev, y_ev, scaler, device)
            mlflow.log_metric("acc_zeroshot", acc_zs)
            mlflow.log_metric("f1_zeroshot", f1_zs)

            # 2. Fine-tuning
            model_ft = finetune(
                model_src, X_ft, y_ft, scaler, device,
                lr=t_cfg["finetune_lr"], epochs=t_cfg["finetune_epochs"],
                batch=cfg["cnn"]["batch_size"], freeze_backbone=t_cfg["freeze_backbone"]
            )
            acc_ft, f1_ft, y_pred_ft = evaluate(model_ft, X_ev, y_ev, scaler, device)
            mlflow.log_metric("acc_finetuned", acc_ft)
            mlflow.log_metric("f1_finetuned", f1_ft)

            # Sauvegarder le modèle fine-tuné
            if bm != src:
                torch.save(model_ft.state_dict(), results_dir / f"cnn1d_ft_{bm}.pt")

            print(f"{bm:15}  zero-shot={acc_zs:.4f}  fine-tuned={acc_ft:.4f}"
                  f"  gain={acc_ft-acc_zs:+.4f}  (n_ft={n_ft})")
            results[bm] = {"zero_shot": acc_zs, "finetuned": acc_ft,
                           "f1_zs": f1_zs, "f1_ft": f1_ft, "n_ft": n_ft}

    # ── Courbe domain shift ────────────────────────────────────────────────────
    bms    = [b for b in all_bms if b in results]
    zs_acc = [results[b]["zero_shot"] for b in bms]
    ft_acc = [results[b]["finetuned"] for b in bms]
    x = np.arange(len(bms))

    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.35
    bars_zs = ax.bar(x - w/2, zs_acc, w, label="Zero-shot", color="steelblue", alpha=0.85)
    bars_ft = ax.bar(x + w/2, ft_acc, w, label="Fine-tuned (10%)", color="darkorange", alpha=0.85)
    ax.bar_label(bars_zs, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_ft, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1.2, label="Aleatoire (33%)")
    ax.set_xticks(x); ax.set_xticklabels(bms, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Domain Shift : CNN entraîné sur {src} → autres benchmarks")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = results_dir / "03_domain_shift.png"
    plt.savefig(out, dpi=150)
    plt.close()

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Benchmark':<15} {'Zero-shot':>10} {'Fine-tuned':>11} {'Gain':>6} {'n_ft':>6}")
    print("-"*65)
    for bm in bms:
        r = results[bm]
        gain = r["finetuned"] - r["zero_shot"]
        print(f"{bm:<15} {r['zero_shot']:>10.4f} {r['finetuned']:>11.4f} {gain:>+6.4f} {r['n_ft']:>6}")
    print(f"\nCourbe domain shift sauvegardee : {out}")
