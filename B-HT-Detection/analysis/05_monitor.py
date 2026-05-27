"""
05_monitor.py — Simulation de monitoring temps réel (golden-free).
Scénario : flux de traces normales (TrojanDisabled) → injection de traces
           TrojanTriggered à l'index N → observer la latence de détection.
Décision toutes les window_traces traces via vote majoritaire + seuil de confiance.
"""
import numpy as np
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from models import CNN1D
from feat_utils import load_trace, extract_features

LABEL_NAMES  = ["Disabled", "Enabled", "Triggered"]
LABEL_COLORS = {0: "green", 1: "orange", 2: "red"}
STATE_NAMES  = {0: "OK", 1: "WARN", 2: "ALERT"}


def run_monitor(model, scaler, device,
                stream_paths,       # list of (path, true_label) tuples
                window_n,           # taille du buffer de décision
                alert_thresh,       # seuil confiance pour ALERT
                window_size=100, n_fft=10):
    """
    Simule un flux de traces :
    - Charge chaque trace, extrait ses features
    - Maintient un buffer glissant de window_n features
    - Toutes les window_n traces : vote majoritaire → état courant
    - Déclenche une alerte si état==Triggered et confiance >= alert_thresh
    """
    model.eval()
    predictions  = []   # classe prédite à chaque pas
    confidences  = []   # confiance associée
    true_labels  = []   # label réel
    states       = []   # "OK" / "WARN" / "ALERT"
    alert_times  = []   # indices des alertes

    buffer_feats = []

    for i, (path, true_lbl) in enumerate(stream_paths):
        trace = load_trace(path)
        feat  = extract_features(trace, window=window_size, n_fft=n_fft)
        buffer_feats.append(feat)
        true_labels.append(true_lbl)

        if len(buffer_feats) < window_n:
            predictions.append(-1)
            confidences.append(0.0)
            states.append("WAIT")
            continue

        # Classer le dernier buffer complet
        win = np.stack(buffer_feats[-window_n:])
        win_s = scaler.transform(win)
        X_t = torch.FloatTensor(win_s).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(X_t), dim=1).cpu().numpy()  # (W, 3)

        votes   = probs.argmax(axis=1)
        agg_cls = np.bincount(votes, minlength=3).argmax()
        agg_conf = probs[:, agg_cls].mean()

        predictions.append(int(agg_cls))
        confidences.append(float(agg_conf))

        if agg_cls == 2 and agg_conf >= alert_thresh:
            states.append("ALERT")
            alert_times.append(i)
        elif agg_cls == 1:
            states.append("WARN")
        else:
            states.append("OK")

        # Sliding : retirer la trace la plus ancienne
        if len(buffer_feats) > window_n * 2:
            buffer_feats = buffer_feats[-window_n:]

    return predictions, confidences, true_labels, states, alert_times


def plot_monitor(predictions, confidences, true_labels, states, alert_times,
                 inject_at, alert_thresh, out_path):
    n = len(predictions)
    xs = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 2]})

    # --- Ax0 : labels réels ---
    ax0 = axes[0]
    true_colors = [LABEL_COLORS.get(t, "gray") for t in true_labels]
    ax0.scatter(xs, [0.5] * n, c=true_colors, marker="|", s=300, linewidths=2)
    ax0.set_yticks([]); ax0.set_ylabel("Réel", fontsize=10)
    ax0.set_title("HT Real-Time Monitoring — Simulation flux traces")

    # --- Ax1 : prédictions ---
    ax1 = axes[1]
    pred_colors = [LABEL_COLORS.get(p, "lightgray") for p in predictions]
    ax1.scatter(xs, [0.5] * n, c=pred_colors, marker="|", s=300, linewidths=2)
    ax1.set_yticks([]); ax1.set_ylabel("Prédit", fontsize=10)

    # Point d'injection
    for ax in [ax0, ax1]:
        ax.axvline(inject_at, color="black", linestyle="--", linewidth=1.5,
                   label=f"Injection (t={inject_at})")
    for t in alert_times[:1]:
        ax1.axvline(t, color="red", linestyle=":", linewidth=2,
                    label=f"1er alert (t={t})")

    patches = [mpatches.Patch(color=c, label=LABEL_NAMES[i])
               for i, c in LABEL_COLORS.items()]
    ax1.legend(handles=patches, loc="upper left", fontsize=8)

    # --- Ax2 : confiance ---
    ax2 = axes[2]
    valid = [(i, c) for i, c in enumerate(confidences) if predictions[i] >= 0]
    if valid:
        vi, vc = zip(*valid)
        ax2.plot(vi, vc, color="steelblue", linewidth=1.5, label="Confiance agrégée")
    ax2.axhline(alert_thresh, color="red", linestyle="--",
                label=f"Seuil alerte ({alert_thresh:.0%})")
    ax2.axvline(inject_at, color="black", linestyle="--")
    if alert_times:
        ax2.axvline(alert_times[0], color="red", linestyle=":",
                    label=f"1er alert t={alert_times[0]}")
    ax2.set_ylim(0, 1); ax2.set_ylabel("Confiance")
    ax2.set_xlabel("Index trace"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Fond coloré par état
    for i, st in enumerate(states):
        if st == "ALERT":
            ax2.axvspan(i, i + 1, color="red", alpha=0.12)
        elif st == "WARN":
            ax2.axvspan(i, i + 1, color="orange", alpha=0.08)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Monitoring plot : {out_path}")


if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = cfg["dataset"]["benchmarks"]["source"]
    mon_cfg = cfg["monitor"]
    feat_cfg = cfg["features"]

    # Utiliser le meilleur modèle disponible (fine-tuné > base)
    # Priorité : modèles fine-tunés par accuracy décroissante
    best_bm, best_weights = src, results_dir / f"cnn1d_{src}.pt"
    for candidate in ["AES-T700", "AES-T800", "AES-T500"]:
        ft_path = results_dir / f"cnn1d_ft_{candidate}.pt"
        if ft_path.exists():
            best_bm, best_weights = candidate, ft_path
            break

    print(f"Modele utilise : {best_weights.name}  (benchmark de monitoring : {best_bm})")
    model = CNN1D().to(device)
    if not best_weights.exists():
        raise FileNotFoundError(f"{best_weights} introuvable — lancer 03_train.py d'abord.")
    model.load_state_dict(torch.load(best_weights, map_location=device))

    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_mean_{src}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_scale_{src}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # Construire le flux depuis l'index (benchmark avec le meilleur modèle)
    index = pd.read_parquet(results_dir / "index.parquet")
    bm_idx = index[index["benchmark"] == best_bm]

    N_NORMAL   = mon_cfg["inject_at"] + 20   # traces normales avant + après inject
    N_INJECT   = 60                            # traces Triggered à injecter

    paths_dis = bm_idx[bm_idx["condition"] == "TrojanDisabled"]["path"].values[:N_NORMAL]
    paths_trg = bm_idx[bm_idx["condition"] == "TrojanTriggered"]["path"].values[:N_INJECT]

    if len(paths_dis) == 0 or len(paths_trg) == 0:
        raise RuntimeError("Index vide pour le benchmark source. Relancer 01_indexer.py.")

    INJECT_AT = min(mon_cfg["inject_at"], len(paths_dis))
    stream = ([(p, 0) for p in paths_dis[:INJECT_AT]] +
              [(p, 2) for p in paths_trg] +
              [(p, 0) for p in paths_dis[INJECT_AT:]])

    print(f"Flux : {INJECT_AT} Disabled + {len(paths_trg)} Triggered + {len(paths_dis)-INJECT_AT} Disabled")
    print(f"Window={mon_cfg['window_traces']} traces  |  Seuil={mon_cfg['alert_threshold']}")

    preds, confs, true_lbls, states, alerts = run_monitor(
        model, scaler, device, stream,
        window_n=mon_cfg["window_traces"],
        alert_thresh=mon_cfg["alert_threshold"],
        window_size=feat_cfg["window_size"],
        n_fft=feat_cfg["n_fft"],
    )

    plot_monitor(preds, confs, true_lbls, states, alerts,
                 INJECT_AT, mon_cfg["alert_threshold"],
                 results_dir / "04_monitor.png")

    # Résultats texte
    if alerts:
        delay = alerts[0] - INJECT_AT
        print(f"\n[RESULTAT] 1er alerte a t={alerts[0]} — delai={delay} trace(s) apres injection")
        print(f"           Total alertes : {len(alerts)} sur {len([s for s in states if s=='ALERT'])}")
    else:
        print("\n[RESULTAT] Aucune alerte declenchee — seuil trop eleve ou modele insuffisant")

    # Distribution des états
    from collections import Counter
    cnt = Counter(states)
    print(f"Distribution etats : OK={cnt['OK']}  WARN={cnt['WARN']}  ALERT={cnt['ALERT']}  WAIT={cnt['WAIT']}")
