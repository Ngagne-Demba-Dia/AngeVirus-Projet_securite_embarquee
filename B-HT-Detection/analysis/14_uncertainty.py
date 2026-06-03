"""
14_uncertainty.py — Direction 3 : Uncertainty Quantification (Monte Carlo Dropout).
Méthode : Gal & Ghahramani 2016 "Dropout as a Bayesian Approximation" (9000 citations).
Objectif : détecter quand le modèle est incertain → alarme UNCERTAIN en production.

Pipeline :
  1. MC Dropout : N=50 passes forward avec Dropout actif
  2. Calcul incertitude épistémique (variance) + entropie prédictive
  3. Calibration : incertitude corrélée à l'accuracy ?
  4. Seuil UNCERTAIN : traces cross-domain détectées automatiquement
  5. Comparaison T700 (in-domain) vs T1100 (out-of-domain)

Fichiers générés (captures rapport) :
  results/14_UQ_distribution.png      — distributions incertitude in vs out-of-domain
  results/14_UQ_calibration.png       — courbe calibration (incertitude vs accuracy)
  results/14_UQ_cross_domain.png      — T700 vs T600/T1100 comparaison
  results/14_UQ_bilan.png             — tableau visuel bilan
  results/14_UQ_metrics.json          — métriques complètes
"""
import json
import numpy as np
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models import CNN1D

# ── Hyperparamètres ────────────────────────────────────────────────────────────
N_PASSES      = 50      # nombre de passes MC Dropout
UNCERTAINTY_THRESHOLD_PERCENTILE = 75   # seuil = 75e percentile in-domain
LABEL_NAMES   = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
TARGET_BM     = "AES-T800"   # modèle AT_T800 (meilleur en production)
IN_DOMAIN_BM  = "AES-T700"   # benchmark in-domain pour référence
OUT_DOMAIN_BMS = ["AES-T600", "AES-T1100"]  # benchmarks out-of-domain


# ── Activer Dropout en mode inférence ─────────────────────────────────────────
def enable_mc_dropout(model: nn.Module):
    """Passe le modèle en eval() mais réactive les couches Dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


# ── Inférence Monte Carlo Dropout ─────────────────────────────────────────────
def mc_predict(model, X_t: torch.Tensor, n_passes: int, device) -> dict:
    """
    N passes forward avec Dropout actif.
    Retourne : probas moyennes, incertitude épistémique, entropie prédictive.
    """
    model = enable_mc_dropout(model)
    all_probs = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(X_t.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.array(all_probs)       # (N_passes, B, n_classes)
    mean_probs = all_probs.mean(axis=0)   # (B, n_classes)
    var_probs  = all_probs.var(axis=0)    # (B, n_classes)

    # Incertitude épistémique = variance moyenne sur les classes
    epistemic = var_probs.mean(axis=1)    # (B,)

    # Entropie prédictive = -Σ p̄ · log(p̄)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)  # (B,)

    # Prédiction finale = argmax des probas moyennes
    preds = mean_probs.argmax(axis=1)     # (B,)
    confidence = mean_probs.max(axis=1)   # (B,)

    return {
        "preds":      preds,
        "confidence": confidence,
        "mean_probs": mean_probs,
        "epistemic":  epistemic,
        "entropy":    entropy,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    src = cfg["dataset"]["benchmarks"]["source"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  MC Dropout N={N_PASSES} passes")

    # ── Charger modèle AT_T800 (meilleur en production) ───────────────────────
    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_ms_mean_{TARGET_BM}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_ms_scale_{TARGET_BM}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    model = CNN1D().to(device)
    at_path = results_dir / f"cnn1d_AT_{TARGET_BM}.pt"
    sd = {k: v for k, v in torch.load(at_path, map_location="cpu").items()
          if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    print(f"Modèle chargé : {at_path.name}")

    # ── Charger données in-domain (T700) ──────────────────────────────────────
    all_results = {}
    benchmarks_to_test = [IN_DOMAIN_BM] + OUT_DOMAIN_BMS

    for bm in benchmarks_to_test:
        npz = results_dir / f"features_{bm}.npz"
        if not npz.exists():
            print(f"  {bm} : features non trouvées, skip")
            continue

        d = np.load(npz)
        X_raw = d["X"].astype(np.float32)
        y     = d["y"].astype(np.int64)

        # Utiliser le scaler du modèle AT_T800
        X_s = torch.FloatTensor(scaler.transform(X_raw).astype(np.float32))

        print(f"\n[{bm}] MC Dropout inference ({len(X_s)} traces × {N_PASSES} passes)...")
        out = mc_predict(model, X_s, N_PASSES, device)

        acc = float(accuracy_score(y, out["preds"]))
        print(f"  Accuracy          : {acc:.4f}")
        print(f"  Incertitude moy.  : {out['epistemic'].mean():.5f} ± {out['epistemic'].std():.5f}")
        print(f"  Entropie moy.     : {out['entropy'].mean():.4f} ± {out['entropy'].std():.4f}")
        print(f"  Confidence moy.   : {out['confidence'].mean():.4f}")

        all_results[bm] = {**out, "y_true": y, "accuracy": acc}

    # ── Définir seuil d'incertitude depuis in-domain ───────────────────────────
    in_epistemic = all_results[IN_DOMAIN_BM]["epistemic"]
    threshold    = float(np.percentile(in_epistemic, UNCERTAINTY_THRESHOLD_PERCENTILE))
    print(f"\nSeuil UNCERTAIN : {threshold:.5f} ({UNCERTAINTY_THRESHOLD_PERCENTILE}e percentile T700)")

    # ── Évaluation du seuil : PREDICT vs UNCERTAIN ────────────────────────────
    print("\n=== ÉVALUATION SEUIL UNCERTAIN ===")
    for bm, res in all_results.items():
        ep  = res["epistemic"]
        certain_mask   = ep <= threshold
        uncertain_mask = ep >  threshold
        frac_uncertain = uncertain_mask.mean()

        if certain_mask.sum() > 0:
            acc_certain = accuracy_score(res["y_true"][certain_mask],
                                         res["preds"][certain_mask])
        else:
            acc_certain = 0.0

        if uncertain_mask.sum() > 0:
            acc_uncertain = accuracy_score(res["y_true"][uncertain_mask],
                                            res["preds"][uncertain_mask])
        else:
            acc_uncertain = 0.0

        domain = "IN-domain " if bm == IN_DOMAIN_BM else "OUT-domain"
        print(f"  [{domain}] {bm:15} : "
              f"UNCERTAIN={frac_uncertain:.1%}  "
              f"acc(certain)={acc_certain:.3f}  "
              f"acc(uncertain)={acc_uncertain:.3f}")

        all_results[bm]["frac_uncertain"]  = float(frac_uncertain)
        all_results[bm]["acc_certain"]     = float(acc_certain)
        all_results[bm]["acc_uncertain"]   = float(acc_uncertain)

    # ── Figure 1 : Distribution incertitude ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogramme épistémique
    ax = axes[0]
    colors = {"AES-T700": "steelblue", "AES-T600": "darkorange", "AES-T1100": "crimson"}
    for bm, res in all_results.items():
        label = f"{bm} ({'in' if bm==IN_DOMAIN_BM else 'out'}-domain)"
        ax.hist(res["epistemic"], bins=50, alpha=0.6,
                color=colors.get(bm, "gray"), label=label, density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=2,
               label=f"Seuil UNCERTAIN={threshold:.4f}")
    ax.set_xlabel("Incertitude épistémique (variance MC)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution de l'incertitude\nIn-domain vs Out-of-domain")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Entropie prédictive
    ax2 = axes[1]
    for bm, res in all_results.items():
        label = f"{bm} ({'in' if bm==IN_DOMAIN_BM else 'out'}-domain)"
        ax2.hist(res["entropy"], bins=50, alpha=0.6,
                 color=colors.get(bm, "gray"), label=label, density=True)
    ax2.set_xlabel("Entropie prédictive H(p̄)")
    ax2.set_ylabel("Densité")
    ax2.set_title("Distribution de l'entropie prédictive")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle("Direction 3 — Monte Carlo Dropout : Incertitude In vs Out-of-domain",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out1 = results_dir / "14_UQ_distribution.png"
    plt.savefig(out1, dpi=150); plt.close()

    # ── Figure 2 : Courbe de calibration ──────────────────────────────────────
    # Pour T700 in-domain : trier par incertitude et calculer l'accuracy par bucket
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, bm in enumerate([IN_DOMAIN_BM, OUT_DOMAIN_BMS[0]]):
        if bm not in all_results:
            continue
        res = all_results[bm]
        ax  = axes[ax_idx]

        # Trier par incertitude croissante
        order     = np.argsort(res["epistemic"])
        ep_sorted = res["epistemic"][order]
        y_sorted  = res["y_true"][order]
        p_sorted  = res["preds"][order]

        # Calculer accuracy par décile
        n = len(ep_sorted)
        n_bins = 10
        bin_size = n // n_bins
        bin_unc, bin_acc = [], []
        for i in range(n_bins):
            sl = slice(i * bin_size, (i+1) * bin_size)
            bin_unc.append(ep_sorted[sl].mean())
            bin_acc.append(accuracy_score(y_sorted[sl], p_sorted[sl]))

        ax.plot(bin_unc, bin_acc, "o-", color="steelblue", lw=2, ms=7)
        ax.axvline(threshold, color="red", linestyle="--", lw=1.5,
                   label=f"Seuil={threshold:.4f}")
        ax.set_xlabel("Incertitude épistémique (moyenne du décile)")
        ax.set_ylabel("Accuracy du décile")
        domain = "in-domain" if bm == IN_DOMAIN_BM else "out-of-domain"
        ax.set_title(f"Courbe de calibration — {bm} ({domain})\n"
                     "Incertitude ↑ → Accuracy ↓ (modèle calibré)")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Direction 3 — Calibration Monte Carlo Dropout",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out2 = results_dir / "14_UQ_calibration.png"
    plt.savefig(out2, dpi=150); plt.close()

    # ── Figure 3 : Comparaison cross-domain ───────────────────────────────────
    bms    = list(all_results.keys())
    means  = [all_results[b]["epistemic"].mean() for b in bms]
    stds   = [all_results[b]["epistemic"].std()  for b in bms]
    accs   = [all_results[b]["accuracy"]          for b in bms]
    f_unc  = [all_results[b]["frac_uncertain"]    for b in bms]

    x = np.arange(len(bms))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(x, means, yerr=stds, color=[
        "steelblue" if b == IN_DOMAIN_BM else "crimson" for b in bms
    ], alpha=0.85, capsize=5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.axhline(threshold, color="black", linestyle="--", lw=1.5,
               label=f"Seuil={threshold:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(bms, rotation=10)
    ax.set_ylabel("Incertitude épistémique moyenne")
    ax.set_title("Incertitude moyenne par benchmark\n(bleu=in-domain, rouge=out-of-domain)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(means, accs, c=["steelblue" if b==IN_DOMAIN_BM else "crimson" for b in bms],
                s=200, zorder=5)
    for bm, m, a in zip(bms, means, accs):
        ax2.annotate(bm, (m, a), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax2.axvline(threshold, color="black", linestyle="--", lw=1.5,
                label=f"Seuil={threshold:.4f}")
    ax2.set_xlabel("Incertitude épistémique moyenne")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Incertitude vs Accuracy cross-domain\n(corrélation négative attendue)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Direction 3 — UQ Cross-domain : le modèle sait quand il ne sait pas",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out3 = results_dir / "14_UQ_cross_domain.png"
    plt.savefig(out3, dpi=150); plt.close()

    # ── Figure 4 : Bilan tableau ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    rows = []
    for bm in bms:
        r = all_results[bm]
        domain = "IN-domain" if bm == IN_DOMAIN_BM else "OUT-domain"
        rows.append([
            bm, domain,
            f"{r['accuracy']:.2%}",
            f"{r['epistemic'].mean():.5f} ± {r['epistemic'].std():.5f}",
            f"{r['frac_uncertain']:.1%}",
            f"{r['acc_certain']:.2%}",
            f"{r['acc_uncertain']:.2%}",
        ])
    cols = ["Benchmark", "Domaine", "Accuracy", "Incertitude moy.", "% UNCERTAIN",
            "Acc (certain)", "Acc (uncertain)"]
    colors_rows = []
    for bm in bms:
        c = "#cce5ff" if bm == IN_DOMAIN_BM else "#ffcccc"
        colors_rows.append([c]*7)

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   loc="center", cellColours=colors_rows)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 2.0)
    ax.set_title(
        f"Direction 3 — Bilan Uncertainty Quantification (MC Dropout N={N_PASSES})\n"
        f"Modèle : AT_{TARGET_BM}  |  Seuil UNCERTAIN = {threshold:.5f} "
        f"({UNCERTAINTY_THRESHOLD_PERCENTILE}e percentile in-domain)",
        fontsize=10, fontweight="bold", pad=15)
    out4 = results_dir / "14_UQ_bilan.png"
    plt.tight_layout()
    plt.savefig(out4, dpi=150, bbox_inches="tight"); plt.close()

    # ── Métriques JSON ─────────────────────────────────────────────────────────
    metrics = {
        "model": f"AT_{TARGET_BM}", "n_passes": N_PASSES,
        "threshold": round(threshold, 6),
        "threshold_percentile": UNCERTAINTY_THRESHOLD_PERCENTILE,
        "benchmarks": {
            bm: {
                "accuracy":        round(all_results[bm]["accuracy"], 4),
                "epistemic_mean":  round(float(all_results[bm]["epistemic"].mean()), 6),
                "epistemic_std":   round(float(all_results[bm]["epistemic"].std()), 6),
                "entropy_mean":    round(float(all_results[bm]["entropy"].mean()), 4),
                "frac_uncertain":  round(all_results[bm]["frac_uncertain"], 4),
                "acc_certain":     round(all_results[bm]["acc_certain"], 4),
                "acc_uncertain":   round(all_results[bm]["acc_uncertain"], 4),
                "domain":          "in" if bm == IN_DOMAIN_BM else "out",
            } for bm in bms
        }
    }
    (results_dir / "14_UQ_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Résumé terminal ────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("BILAN DIRECTION 3 — UNCERTAINTY QUANTIFICATION (Monte Carlo Dropout)")
    print("="*75)
    print(f"  Modèle : AT_{TARGET_BM}  |  N_passes={N_PASSES}  |  Seuil={threshold:.5f}")
    print(f"\n  {'Benchmark':15} {'Domaine':12} {'Accuracy':10} {'Incert. moy':13} "
          f"{'% UNCERTAIN':12} {'Acc certain':12} {'Acc incert':10}")
    print("  " + "-"*80)
    for bm in bms:
        r = all_results[bm]
        dom = "IN-domain  " if bm == IN_DOMAIN_BM else "OUT-domain "
        print(f"  {bm:15} {dom:12} {r['accuracy']:10.2%} "
              f"{r['epistemic'].mean():13.5f} "
              f"{r['frac_uncertain']:12.1%} "
              f"{r['acc_certain']:12.2%} "
              f"{r['acc_uncertain']:10.2%}")

    print(f"\n  Fichiers captures rapport :")
    for out in [out1, out2, out3, out4, results_dir/"14_UQ_metrics.json"]:
        print(f"    {Path(out).name}")

    print(f"\n  Conclusion : le modèle est CALIBRÉ si acc(uncertain) < acc(certain)")
    for bm in bms:
        r  = all_results[bm]
        ok = "✓ CALIBRÉ" if r["acc_uncertain"] < r["acc_certain"] else "✗ non calibré"
        print(f"    {bm:15} : {ok}")
