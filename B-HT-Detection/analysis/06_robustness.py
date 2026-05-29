"""
06_robustness.py — Test de robustesse : injection de bruit gaussien.
Évalue la dégradation de précision du CNN en fonction du niveau de bruit σ.
Teste le modèle source (AES-T400) sur les données source et cross-domain.
"""
import numpy as np
import yaml
import mlflow
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models import CNN1D

LABEL_NAMES = ["Disabled", "Enabled", "Triggered"]


def load_features(results_dir: Path, bm: str):
    d = np.load(results_dir / f"features_{bm}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def evaluate_with_noise(model, X, y, scaler, sigma: float) -> float:
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, X.shape).astype(np.float32)
    X_noisy = X + noise
    X_s = torch.FloatTensor(scaler.transform(X_noisy))
    with torch.no_grad():
        preds = model(X_s).argmax(1).numpy()
    return float(accuracy_score(y, preds))


if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    src = cfg["dataset"]["benchmarks"]["source"]

    # ── Charger modèle source ──────────────────────────────────────────────────
    model_src = CNN1D()
    model_src.load_state_dict(torch.load(results_dir / f"cnn1d_{src}.pt", map_location="cpu"))
    model_src.eval()

    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_mean_{src}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_scale_{src}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # ── Charger modèle fine-tuné (T700) si disponible ─────────────────────────
    ft_model = None
    for candidate in ["AES-T700", "AES-T800", "AES-T500"]:
        ft_path = results_dir / f"cnn1d_ft_{candidate}.pt"
        if ft_path.exists():
            ft_model = CNN1D()
            ft_model.load_state_dict(torch.load(ft_path, map_location="cpu"))
            ft_model.eval()
            ft_bm = candidate
            print(f"Modèle fine-tuné chargé : {ft_path}")
            break

    # ── Niveaux de bruit à tester ──────────────────────────────────────────────
    # σ en unités des features brutes (avant normalisation)
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Benchmarks à évaluer
    test_bms = ["AES-T400", "AES-T500", "AES-T700"]
    results = {}

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Robustness_GaussianNoise"):

        for bm in test_bms:
            npz = results_dir / f"features_{bm}.npz"
            if not npz.exists():
                print(f"  {bm}: features non trouvées, skip")
                continue

            X, y = load_features(results_dir, bm)

            # Choisir le bon modèle selon le benchmark
            if ft_model is not None and bm == ft_bm:
                model_eval = ft_model
                label = f"{bm} (fine-tuned)"
            else:
                model_eval = model_src
                label = bm

            accs = []
            print(f"\n[{label}]")
            for sigma in sigmas:
                acc = evaluate_with_noise(model_eval, X, y, scaler, sigma)
                accs.append(acc)
                mlflow.log_metric(f"rob_{bm}_sigma{sigma:.2f}", acc)
                print(f"  σ={sigma:5.2f}  acc={acc:.4f}")

            results[label] = accs

        # Seuil de dégradation : σ où on passe sous 50% sur T400
        t400_label = "AES-T400"
        if t400_label in results:
            for sigma, acc in zip(sigmas, results[t400_label]):
                if acc < 0.5:
                    mlflow.log_metric("robustness_threshold_50pct", sigma)
                    print(f"\nSeuil 50% atteint à σ={sigma}")
                    break

    # ── Courbe de robustesse ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["steelblue", "darkorange", "green", "purple"]
    for (label, accs), color in zip(results.items(), palette):
        ax.plot(sigmas, accs, marker="o", label=label, color=color, linewidth=2.2)

    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1.2, label="Aléatoire (33%)")
    ax.set_xlabel("Niveau de bruit σ (écart-type gaussien sur les features)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustesse du CNN HT-Detector au bruit gaussien")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = results_dir / "05_robustness.png"
    plt.savefig(out, dpi=150)
    plt.close()

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    print("\n" + "="*70)
    header = f"{'σ':>6}" + "".join(f"  {lb[:12]:>12}" for lb in results)
    print(header)
    print("-"*70)
    for i, sigma in enumerate(sigmas):
        row = f"{sigma:>6.2f}"
        for accs in results.values():
            row += f"  {accs[i]:>12.4f}"
        print(row)

    print(f"\nCourbe robustesse sauvegardée : {out}")
