"""
12_adversarial_training.py — Direction 1 : Adversarial Training (PGD-AT).
Méthode : Madry et al. 2018 "Towards Deep Learning Models Resistant to Adversarial Attacks".
Objectif : rendre le détecteur HT robuste aux attaques FGSM/PGD avec ε=0.2 (seuil critique).

Fichiers générés (captures rapport) :
  results/12_AT_courbe_robustesse.png    — comparaison avant/après AT
  results/12_AT_trade_off.png            — clean accuracy vs adversarial accuracy
  results/12_AT_bilan_securite.png       — tableau visuel bilan
  results/cnn1d_AT_AES-T800.pt           — modèle adversarialement entraîné
  results/12_AT_metrics.json             — métriques complètes
"""
import copy
import json
import numpy as np
import yaml
import mlflow
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models import CNN1D

# ── Paramètres AT (basés sur notre analyse : ε=0.2 = seuil critique) ──────────
EPS          = 0.2    # budget adversarial = seuil critique identifié
ALPHA        = 0.05   # pas PGD = ε/4
PGD_STEPS    = 10     # itérations PGD pendant l'entraînement (GPU : on peut se permettre plus)
AT_EPOCHS    = 100    # epochs adversarial training (GPU)
AT_LR        = 3e-5   # lr faible pour préserver la connaissance initiale
BATCH_SIZE   = 256    # batch GPU
TARGET_BM    = "AES-T800"
LABEL_NAMES  = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
TRIGGER_CLASS = 2


# ── PGD Attack (targeted : TrojanTriggered → TrojanDisabled) ──────────────────
def pgd_attack_train(model, xb, yb, eps, alpha, n_steps, device):
    """
    PGD untargeted pour l'entraînement AT.
    On perturbe dans la direction qui maximise la loss (rend la classification difficile).
    """
    x_adv = xb.clone().detach() + torch.empty_like(xb).uniform_(-eps, eps)
    x_adv = x_adv.detach()
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        loss = criterion(model(x_adv), yb)
        model.zero_grad()
        loss.backward()
        x_adv = (x_adv + alpha * x_adv.grad.sign()).detach()
        x_adv = torch.clamp(x_adv, xb - eps, xb + eps)

    return x_adv.detach()


# ── FGSM + PGD pour évaluation ────────────────────────────────────────────────
def fgsm_eval(model, X_t, eps, device, target=0):
    X_adv = X_t.clone().detach().requires_grad_(True)
    crit  = nn.CrossEntropyLoss()
    model.zero_grad()
    tgt = torch.LongTensor([target]*len(X_adv)).to(device)
    crit(model(X_adv), tgt).backward()
    return (X_adv - eps * X_adv.grad.sign()).detach()


def pgd_eval(model, X_t, eps, alpha, n_steps, device, target=0):
    X_adv = X_t.clone().detach()
    crit  = nn.CrossEntropyLoss()
    tgt   = torch.LongTensor([target]*len(X_adv)).to(device)
    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        crit(model(X_adv), tgt).backward()
        X_adv = (X_adv - alpha * X_adv.grad.sign()).detach()
        X_adv = torch.clamp(X_adv, X_t - eps, X_t + eps)
    return X_adv.detach()


def masquage_rate(model, X_trig, eps, alpha, n_steps, device, mode="pgd"):
    """Taux de masquage (TrojanTriggered classifié comme TrojanDisabled)."""
    if mode == "fgsm":
        X_adv = fgsm_eval(model, X_trig, eps, device)
    else:
        X_adv = pgd_eval(model, X_trig, eps, alpha, n_steps, device)
    with torch.no_grad():
        preds = model(X_adv).argmax(1).cpu().numpy()
    return float(np.mean(preds == 0))


def clean_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()
    return float(accuracy_score(y, preds))


# ── Adversarial Training (PGD-AT Madry 2018) ──────────────────────────────────
def adversarial_train(model, X_train, y_train, scaler, device,
                      eps, alpha, pgd_steps, epochs, lr, batch_size):
    model_at = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model_at.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_s = torch.FloatTensor(scaler.transform(X_train).astype(np.float32))
    loader = DataLoader(
        TensorDataset(X_s, torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    print(f"\n{'Epoch':>6} | {'Loss clean':>11} | {'Loss adv':>9}")
    print("-"*35)

    for ep in range(epochs):
        model_at.train()
        total_clean, total_adv, n = 0.0, 0.0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # Générer exemples adversariaux avec PGD
            x_adv = pgd_attack_train(model_at, xb, yb, eps, alpha, pgd_steps, device)

            optimizer.zero_grad()

            # Loss sur clean + adversarial (50/50)
            loss_clean = criterion(model_at(xb),    yb)
            loss_adv   = criterion(model_at(x_adv), yb)
            loss       = 0.5 * loss_clean + 0.5 * loss_adv

            loss.backward()
            optimizer.step()

            total_clean += loss_clean.item()
            total_adv   += loss_adv.item()
            n += 1

        scheduler.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"{ep+1:>6} | {total_clean/n:>11.4f} | {total_adv/n:>9.4f}")

    model_at.eval()
    return model_at


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    src = cfg["dataset"]["benchmarks"]["source"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  ε={EPS}  α={ALPHA}  steps={PGD_STEPS}")

    # ── Charger meilleur modèle (CORAL T800) ──────────────────────────────────
    model_path  = results_dir / f"cnn1d_coral_{TARGET_BM}.pt"
    scaler_mean = results_dir / f"scaler_ms_mean_{TARGET_BM}.npy"
    scaler_scl  = results_dir / f"scaler_ms_scale_{TARGET_BM}.npy"

    if not model_path.exists():
        print(f"CORAL {TARGET_BM} non trouvé, fallback modèle source")
        model_path  = results_dir / f"cnn1d_{src}.pt"
        scaler_mean = results_dir / f"scaler_mean_{src}.npy"
        scaler_scl  = results_dir / f"scaler_scale_{src}.npy"

    scaler = StandardScaler()
    scaler.mean_  = np.load(scaler_mean)
    scaler.scale_ = np.load(scaler_scl)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    model_orig = CNN1D()
    sd = {k: v for k, v in torch.load(model_path, map_location="cpu").items()
          if k in CNN1D().state_dict()}
    model_orig.load_state_dict(sd, strict=False)
    model_orig = model_orig.to(device)
    model_orig.eval()
    print(f"Modèle chargé : {model_path.name}  →  {device}")

    # ── Charger données T800 ───────────────────────────────────────────────────
    d = np.load(results_dir / f"features_{TARGET_BM}.npz")
    X_all, y_all = d["X"].astype(np.float32), d["y"].astype(np.int64)

    rng  = np.random.RandomState(42)
    idx  = rng.permutation(len(X_all))
    n_tr = int(len(X_all) * 0.7)
    X_tr, y_tr = X_all[idx[:n_tr]], y_all[idx[:n_tr]]
    X_te, y_te = X_all[idx[n_tr:]], y_all[idx[n_tr:]]

    # Traces TrojanTriggered pour évaluation du masquage
    mask    = y_te == TRIGGER_CLASS
    X_trig  = torch.FloatTensor(scaler.transform(X_te[mask]).astype(np.float32)).to(device)
    y_trig  = y_te[mask]
    X_te_t  = torch.FloatTensor(scaler.transform(X_te).astype(np.float32)).to(device)

    # ── Évaluation AVANT AT ────────────────────────────────────────────────────
    print("\n=== Évaluation AVANT Adversarial Training ===")
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    before_clean   = clean_accuracy(model_orig, X_te_t, y_te)
    before_fgsm    = [masquage_rate(model_orig, X_trig, e, ALPHA, PGD_STEPS, device, "fgsm") for e in epsilons]
    before_pgd     = [masquage_rate(model_orig, X_trig, e, ALPHA, PGD_STEPS, device, "pgd")  for e in epsilons]
    print(f"  Clean accuracy : {before_clean:.4f}")
    print(f"  FGSM masquage  : {[f'{v:.2f}' for v in before_fgsm]}")
    print(f"  PGD  masquage  : {[f'{v:.2f}' for v in before_pgd]}")

    # ── Adversarial Training ───────────────────────────────────────────────────
    print(f"\n=== Adversarial Training PGD-AT (ε={EPS}, {AT_EPOCHS} epochs) ===")
    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"AdversarialTraining_PGD_eps{EPS}_{TARGET_BM}"):
        mlflow.log_params({
            "model":      f"CORAL_{TARGET_BM}",
            "eps":        EPS, "alpha": ALPHA,
            "pgd_steps":  PGD_STEPS, "at_epochs": AT_EPOCHS,
            "at_lr":      AT_LR, "batch_size": BATCH_SIZE,
        })

        model_at = adversarial_train(
            model_orig, X_tr, y_tr, scaler, device,
            EPS, ALPHA, PGD_STEPS, AT_EPOCHS, AT_LR, BATCH_SIZE
        )

        # ── Évaluation APRÈS AT ────────────────────────────────────────────────
        print("\n=== Évaluation APRÈS Adversarial Training ===")
        after_clean = clean_accuracy(model_at, X_te_t, y_te)
        after_fgsm  = [masquage_rate(model_at, X_trig, e, ALPHA, PGD_STEPS, device, "fgsm") for e in epsilons]
        after_pgd   = [masquage_rate(model_at, X_trig, e, ALPHA, PGD_STEPS, device, "pgd")  for e in epsilons]
        print(f"  Clean accuracy : {after_clean:.4f}  (Δ={after_clean-before_clean:+.4f})")
        print(f"  FGSM masquage  : {[f'{v:.2f}' for v in after_fgsm]}")
        print(f"  PGD  masquage  : {[f'{v:.2f}' for v in after_pgd]}")

        # Log MLflow
        mlflow.log_metric("clean_acc_before", before_clean)
        mlflow.log_metric("clean_acc_after",  after_clean)
        mlflow.log_metric("pgd_masquage_before_eps02", before_pgd[epsilons.index(0.2)])
        mlflow.log_metric("pgd_masquage_after_eps02",  after_pgd[epsilons.index(0.2)])
        mlflow.log_metric("fgsm_masquage_after_eps02", after_fgsm[epsilons.index(0.2)])

        # Sauvegarder modèle AT
        at_path = results_dir / f"cnn1d_AT_{TARGET_BM}.pt"
        torch.save(model_at.state_dict(), at_path)
        print(f"\nModèle AT sauvegardé : {at_path.name}")

    # ── Figure 1 : Courbe robustesse avant/après ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epsilons, before_pgd, "o--", color="crimson",    lw=2, label="PGD avant AT")
    ax.plot(epsilons, after_pgd,  "o-",  color="steelblue",  lw=2, label="PGD après AT")
    ax.plot(epsilons, before_fgsm,"s--", color="darkorange",  lw=1.5, alpha=0.7, label="FGSM avant AT")
    ax.plot(epsilons, after_fgsm, "s-",  color="green",       lw=1.5, alpha=0.7, label="FGSM après AT")
    ax.axvline(EPS, color="gray", linestyle=":", lw=1.5, label=f"ε critique={EPS}")
    ax.set_xlabel("Perturbation ε"); ax.set_ylabel("Taux de masquage (Triggered→Disabled)")
    ax.set_title("Robustesse adversariale\nAvant vs Après AT")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    categories = [f"ε={e}" for e in epsilons]
    x = np.arange(len(epsilons))
    w = 0.35
    b1 = ax2.bar(x - w/2, before_pgd, w, color="crimson",   alpha=0.8, label="PGD avant AT")
    b2 = ax2.bar(x + w/2, after_pgd,  w, color="steelblue", alpha=0.8, label="PGD après AT")
    ax2.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
    ax2.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(categories, rotation=15)
    ax2.set_ylabel("Taux de masquage")
    ax2.set_title(f"PGD masquage avant/après AT\n(modèle {TARGET_BM})")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Direction 1 — Adversarial Training PGD-AT (ε={EPS})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out1 = results_dir / "12_AT_courbe_robustesse.png"
    plt.savefig(out1, dpi=150); plt.close()

    # ── Figure 2 : Trade-off clean vs adversarial ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([before_clean], [before_pgd[epsilons.index(0.2)]],
               s=200, color="crimson", zorder=5, label=f"Avant AT\nacc={before_clean:.2%} masq={before_pgd[epsilons.index(0.2)]:.2%}")
    ax.scatter([after_clean],  [after_pgd[epsilons.index(0.2)]],
               s=200, color="steelblue", marker="*", zorder=5, label=f"Après AT\nacc={after_clean:.2%} masq={after_pgd[epsilons.index(0.2)]:.2%}")
    ax.annotate("", xy=(after_clean, after_pgd[epsilons.index(0.2)]),
                xytext=(before_clean, before_pgd[epsilons.index(0.2)]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.set_xlabel("Clean Accuracy"); ax.set_ylabel(f"PGD Masquage (ε={EPS})")
    ax.set_title("Trade-off : Clean Accuracy vs Robustesse Adversariale\n(bas = robuste, haut = vulnérable)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.invert_yaxis()
    out2 = results_dir / "12_AT_trade_off.png"
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()

    # ── Figure 3 : Bilan sécurité visuel ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table_data = [
        ["Métrique", "Avant AT", "Après AT", "Amélioration"],
        ["Clean Accuracy", f"{before_clean:.2%}", f"{after_clean:.2%}",
         f"{after_clean-before_clean:+.2%}"],
        [f"FGSM masquage ε={EPS}", f"{before_fgsm[epsilons.index(EPS)]:.2%}",
         f"{after_fgsm[epsilons.index(EPS)]:.2%}",
         f"{after_fgsm[epsilons.index(EPS)]-before_fgsm[epsilons.index(EPS)]:+.2%}"],
        [f"PGD masquage ε={EPS}", f"{before_pgd[epsilons.index(EPS)]:.2%}",
         f"{after_pgd[epsilons.index(EPS)]:.2%}",
         f"{after_pgd[epsilons.index(EPS)]-before_pgd[epsilons.index(EPS)]:+.2%}"],
        ["PGD masquage ε=0.5", f"{before_pgd[epsilons.index(0.5)]:.2%}",
         f"{after_pgd[epsilons.index(0.5)]:.2%}",
         f"{after_pgd[epsilons.index(0.5)]-before_pgd[epsilons.index(0.5)]:+.2%}"],
    ]
    colors = [["#2c3e50"]*4] + [["#ecf0f1", "#e74c3c", "#3498db", "#27ae60"]]*len(table_data[1:])
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   cellColours=[["#ecf0f1", "#ffcccc", "#cce5ff", "#ccffcc"]]*len(table_data[1:]))
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 2.0)
    ax.set_title(f"Bilan Sécurité — Direction 1 : Adversarial Training PGD-AT\n"
                 f"Modèle : CORAL {TARGET_BM}  |  ε critique = {EPS}",
                 fontsize=12, fontweight="bold", pad=20)
    out3 = results_dir / "12_AT_bilan_securite.png"
    plt.tight_layout(); plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()

    # ── Métriques JSON ─────────────────────────────────────────────────────────
    metrics = {
        "model": f"CORAL_{TARGET_BM}", "eps_critique": EPS,
        "at_epochs": AT_EPOCHS, "at_lr": AT_LR,
        "before": {"clean_acc": round(before_clean, 4),
                   "fgsm_masquage": {str(e): round(v, 4) for e, v in zip(epsilons, before_fgsm)},
                   "pgd_masquage":  {str(e): round(v, 4) for e, v in zip(epsilons, before_pgd)}},
        "after":  {"clean_acc": round(after_clean, 4),
                   "fgsm_masquage": {str(e): round(v, 4) for e, v in zip(epsilons, after_fgsm)},
                   "pgd_masquage":  {str(e): round(v, 4) for e, v in zip(epsilons, after_pgd)}},
        "gain_robustesse_pgd_eps02": round(before_pgd[epsilons.index(EPS)] - after_pgd[epsilons.index(EPS)], 4),
        "cout_clean_acc":            round(before_clean - after_clean, 4),
    }
    (results_dir / "12_AT_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Résumé terminal ────────────────────────────────────────────────────────
    gain = before_pgd[epsilons.index(EPS)] - after_pgd[epsilons.index(EPS)]
    print("\n" + "="*65)
    print("BILAN DIRECTION 1 — ADVERSARIAL TRAINING PGD-AT")
    print("="*65)
    print(f"  ε critique traité    : {EPS}")
    print(f"  Clean accuracy       : {before_clean:.2%} → {after_clean:.2%}  ({after_clean-before_clean:+.2%})")
    print(f"  PGD masquage ε={EPS} : {before_pgd[epsilons.index(EPS)]:.2%} → {after_pgd[epsilons.index(EPS)]:.2%}  (-{gain:.2%})")
    print(f"  FGSM masquage ε={EPS}: {before_fgsm[epsilons.index(EPS)]:.2%} → {after_fgsm[epsilons.index(EPS)]:.2%}")
    print(f"\n  Fichiers captures rapport :")
    print(f"    {out1.name}")
    print(f"    {out2.name}")
    print(f"    {out3.name}")
    print(f"    12_AT_metrics.json")
    print(f"    cnn1d_AT_{TARGET_BM}.pt")
