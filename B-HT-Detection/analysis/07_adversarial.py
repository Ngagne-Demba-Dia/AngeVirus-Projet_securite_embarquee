"""
07_adversarial.py — Attaque adversariale sur le détecteur HT (golden-free).

Scénario :
  Un attaquant connaît le modèle de détection (white-box).
  Il possède un chip avec Trojan ACTIF (label=2 TrojanTriggered).
  Il cherche à le faire classifier comme SAIN (label=0 TrojanDisabled)
  en perturbant légèrement les features de puissance.

Méthodes :
  - FGSM  (Fast Gradient Sign Method) — attaque en un pas
  - PGD   (Projected Gradient Descent) — attaque itérative plus forte
"""
import numpy as np
import yaml
import mlflow
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models import CNN1D

LABEL_NAMES = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
TARGET_CLASS  = 0   # TrojanDisabled — classe cible de l'attaque
TRIGGER_CLASS = 2   # TrojanTriggered — classe source à tromper


# ── FGSM ──────────────────────────────────────────────────────────────────────
def fgsm_attack(model, X_t, epsilon: float, target: int) -> torch.Tensor:
    """Targeted FGSM : minimise la loss pour la classe cible."""
    X_adv = X_t.clone().detach().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    target_t  = torch.LongTensor([target] * len(X_adv))

    model.zero_grad()
    loss = criterion(model(X_adv), target_t)
    loss.backward()

    # Gradient descent vers la classe cible
    return (X_adv - epsilon * X_adv.grad.sign()).detach()


# ── PGD ───────────────────────────────────────────────────────────────────────
def pgd_attack(model, X_t, epsilon: float, alpha: float,
               n_steps: int, target: int) -> torch.Tensor:
    """Targeted PGD : itérations FGSM contraintes dans la boule L∞."""
    X_adv = X_t.clone().detach()
    criterion = nn.CrossEntropyLoss()
    target_t  = torch.LongTensor([target] * len(X_adv))

    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        loss = criterion(model(X_adv), target_t)
        model.zero_grad()
        loss.backward()
        grad_sign = X_adv.grad.sign().detach()
        X_adv = (X_adv - alpha * grad_sign).detach()
        # Projection dans la boule L∞
        X_adv = torch.clamp(X_adv, X_t - epsilon, X_t + epsilon)

    return X_adv


# ── Évaluation ────────────────────────────────────────────────────────────────
def eval_attack(model, X_adv, y_true):
    with torch.no_grad():
        preds = model(X_adv).argmax(1).numpy()
    success = np.mean(preds == TARGET_CLASS)   # taux de masquage du Trojan
    acc     = accuracy_score(y_true, preds)
    return success, acc, preds


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    src = cfg["dataset"]["benchmarks"]["source"]

    # Charger le meilleur modèle disponible (fine-tuné T700 si possible)
    model = CNN1D()
    ft_path = results_dir / "cnn1d_ft_AES-T700.pt"
    if ft_path.exists():
        model.load_state_dict(torch.load(ft_path, map_location="cpu"))
        model_name = "CNN_FT_T700"
        data_bm    = "AES-T700"
        print("Modèle fine-tuné T700 chargé.")
    else:
        model.load_state_dict(torch.load(results_dir / f"cnn1d_{src}.pt", map_location="cpu"))
        model_name = f"CNN_{src}"
        data_bm    = src
        print(f"Modèle source {src} chargé.")
    model.eval()

    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_mean_{src}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_scale_{src}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # Charger les données TrojanTriggered uniquement
    d = np.load(results_dir / f"features_{data_bm}.npz")
    X_all, y_all = d["X"].astype(np.float32), d["y"].astype(np.int64)
    mask = y_all == TRIGGER_CLASS
    X_trig, y_trig = X_all[mask], y_all[mask]
    print(f"Traces TrojanTriggered : {len(X_trig)}")

    X_s   = torch.FloatTensor(scaler.transform(X_trig))
    y_np  = y_trig

    # Référence sans attaque
    acc0, _, _ = eval_attack(model, X_s, y_np)
    print(f"Sans attaque — accuracy={1-acc0:.4f}  masquage={acc0:.4f}")

    # ── FGSM sweep ────────────────────────────────────────────────────────────
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    fgsm_success, pgd_success = [], []

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"AdversarialAttack_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("attack_target", "TrojanDisabled")
        mlflow.log_param("attack_source", "TrojanTriggered")

        print(f"\n{'ε':>6} {'FGSM masquage':>15} {'PGD masquage':>14}")
        print("-"*40)

        for eps in epsilons:
            # FGSM
            X_fgsm = fgsm_attack(model, X_s, eps, TARGET_CLASS)
            succ_f, _, _ = eval_attack(model, X_fgsm, y_np)
            fgsm_success.append(succ_f)
            mlflow.log_metric(f"fgsm_success_eps{eps}", succ_f)

            # PGD (10 itérations, alpha = eps/4)
            X_pgd = pgd_attack(model, X_s, eps, alpha=eps/4,
                               n_steps=10, target=TARGET_CLASS)
            succ_p, _, _ = eval_attack(model, X_pgd, y_np)
            pgd_success.append(succ_p)
            mlflow.log_metric(f"pgd_success_eps{eps}", succ_p)

            print(f"{eps:>6.2f} {succ_f:>15.4f} {succ_p:>14.4f}")

        # ε minimal pour 50% de succès PGD
        for eps, s in zip(epsilons, pgd_success):
            if s >= 0.5:
                mlflow.log_metric("pgd_eps_50pct_success", eps)
                print(f"\nPGD atteint 50% de masquage à ε={eps}")
                break

    # ── Courbe succès d'attaque ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epsilons, fgsm_success, "o-", color="crimson", linewidth=2, label="FGSM (1 pas)")
    ax.plot(epsilons, pgd_success,  "s-", color="darkorange", linewidth=2, label="PGD (10 pas)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Seuil 50%")
    ax.set_xlabel("Perturbation ε (L∞ sur features normalisées)")
    ax.set_ylabel("Taux de masquage du Trojan")
    ax.set_title("Succès de l'attaque adversariale\n(Triggered → Disabled)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Comparaison features originales vs adversariales (1 exemple, PGD eps=1.0)
    ax2 = axes[1]
    if 1.0 in epsilons:
        eps_idx = epsilons.index(1.0)
        X_pgd_1 = pgd_attack(model, X_s[:1], 1.0, alpha=0.25, n_steps=10, target=TARGET_CLASS)
        orig = X_s[0].numpy()
        adv  = X_pgd_1[0].numpy()
        diff = np.abs(adv - orig)
        feat_idx = np.arange(len(orig))
        ax2.bar(feat_idx, diff, color="darkorange", alpha=0.7, width=1.0)
        ax2.set_xlabel("Index feature (325 features)")
        ax2.set_ylabel("|perturbation|")
        ax2.set_title("Perturbation PGD (ε=1.0) sur une trace Triggered")
        ax2.grid(alpha=0.3)

    plt.suptitle(f"Attaque adversariale white-box — modèle {model_name}", fontsize=12)
    plt.tight_layout()
    out = results_dir / "06_adversarial.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"\nCourbe adversariale sauvegardée : {out}")

    # ── Résumé sécurité ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("BILAN SÉCURITÉ — Attaque adversariale white-box")
    print("="*55)
    print(f"  Modèle cible  : {model_name}")
    print(f"  Attaque       : TrojanTriggered → TrojanDisabled")
    print(f"  FGSM ε=0.1    : {fgsm_success[epsilons.index(0.1)]:.1%} de masquage")
    print(f"  PGD  ε=0.1    : {pgd_success[epsilons.index(0.1)]:.1%} de masquage")
    print(f"  PGD  ε=1.0    : {pgd_success[epsilons.index(1.0)]:.1%} de masquage")
    print("  → Vulnérable aux attaques white-box avec perturbation modérée")
    print("  → Défense recommandée : adversarial training (AT)")
