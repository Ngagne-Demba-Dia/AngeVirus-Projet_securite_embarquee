"""
03_train.py — Entraînement SVM / Random Forest / XGBoost / CNN1D + MLflow tracking.
Golden-free : pas de chip de référence — classification 3 classes directe.
Train sur AES-T400, test sur AES-T500 (domain shift baseline).
"""
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              classification_report)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost non installe — XGBoost sera ignore")

LABEL_NAMES = ["Disabled", "Enabled", "Triggered"]

from models import CNN1D


# ── Utilitaires ────────────────────────────────────────────────────────────────
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_classical(name, clf, X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="macro")
    cm   = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=LABEL_NAMES)
    return pipe, acc, f1, cm, report


def _predict_batched(model, X_np, device, batch_size=512):
    """Inférence par batch pour éviter l'OOM sur GPU."""
    preds = []
    X_t = torch.FloatTensor(X_np)
    for i in range(0, len(X_t), batch_size):
        with torch.no_grad():
            preds.append(model(X_t[i:i+batch_size].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def train_cnn(X_tr, y_tr, X_te, y_te, cfg, device):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    model = CNN1D(dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr_s), torch.LongTensor(y_tr)),
        batch_size=cfg["batch_size"], shuffle=True
    )

    print(f"  {'Ep':>3} | {'Train loss':>10} | {'Val acc':>7}")
    for ep in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        if (ep + 1) % 5 == 0 or ep == cfg["epochs"] - 1:
            model.eval()
            y_pred_t = _predict_batched(model, X_te_s, device)
            val_acc = accuracy_score(y_te, y_pred_t)
            print(f"  {ep+1:>3} | {total_loss/len(y_tr):>10.4f} | {val_acc:>6.2%}")

    model.eval()
    y_pred = _predict_batched(model, X_te_s, device)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred, average="macro")
    cm     = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=LABEL_NAMES)
    return model, scaler, acc, f1, cm, report


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Chargement features
    def load(bm):
        d = np.load(results_dir / f"features_{bm}.npz")
        return d["X"].astype(np.float32), d["y"].astype(np.int64)

    src = cfg["dataset"]["benchmarks"]["source"]

    # ── Charger tous les benchmarks disponibles pour la CV ────────────────────
    all_bms = cfg["dataset"]["benchmarks"]["all"]
    bm_data = {}
    for bm in all_bms:
        p = results_dir / f"features_{bm}.npz"
        if p.exists():
            bm_data[bm] = load(bm)
    available_bms = sorted(bm_data.keys())
    print(f"Benchmarks disponibles : {available_bms}")

    # ── Validation croisée 5-fold par benchmark (Leave-One-Benchmark-Out) ─────
    # Chaque fold : 1 benchmark en test, les autres en train
    # Pas de data leakage : les traces du même benchmark ne sont jamais
    # dans train ET test du même fold.
    print("\n=== VALIDATION CROISEE LEAVE-ONE-BENCHMARK-OUT ===")
    cv_results = []
    for test_bm in available_bms:
        train_bms = [b for b in available_bms if b != test_bm]
        X_cv_tr = np.vstack([bm_data[b][0] for b in train_bms])
        y_cv_tr = np.concatenate([bm_data[b][1] for b in train_bms])
        X_cv_te, y_cv_te = bm_data[test_bm]

        # Évaluer avec Random Forest (rapide, bon proxy)
        from sklearn.pipeline import Pipeline as _P
        pipe_cv = _P([("sc", StandardScaler()),
                      ("rf", RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42))])
        pipe_cv.fit(X_cv_tr, y_cv_tr)
        acc_cv = accuracy_score(y_cv_te, pipe_cv.predict(X_cv_te))
        f1_cv  = f1_score(y_cv_te, pipe_cv.predict(X_cv_te), average="macro")
        cv_results.append({"test_bm": test_bm, "train_bms": "+".join(train_bms),
                            "accuracy": round(acc_cv, 4), "f1_macro": round(f1_cv, 4)})
        print(f"  Fold test={test_bm:12}  acc={acc_cv:.4f}  f1={f1_cv:.4f}")

    df_cv = pd.DataFrame(cv_results)
    print(f"\nCV moyen : acc={df_cv['accuracy'].mean():.4f} ± {df_cv['accuracy'].std():.4f}"
          f"  f1={df_cv['f1_macro'].mean():.4f} ± {df_cv['f1_macro'].std():.4f}")
    df_cv.to_csv(results_dir / "cv_benchmark_results.csv", index=False)

    # Log CV dans MLflow
    with mlflow.start_run(run_name="CV_LeaveOneBenchmarkOut_RF"):
        mlflow.log_metric("cv_acc_mean",  df_cv["accuracy"].mean())
        mlflow.log_metric("cv_acc_std",   df_cv["accuracy"].std())
        mlflow.log_metric("cv_f1_mean",   df_cv["f1_macro"].mean())
        mlflow.log_artifact(str(results_dir / "cv_benchmark_results.csv"))

    # ── Train/Test final : src → premier target disponible ────────────────────
    X_tr, y_tr = bm_data[src]
    tgt_candidates = [b for b in cfg["dataset"]["benchmarks"]["targets"] if b in bm_data]
    tgt = tgt_candidates[0] if tgt_candidates else src
    if tgt != src:
        X_te, y_te = bm_data[tgt]
        print(f"\nTrain final : {src} ({len(X_tr)})  |  Test : {tgt} ({len(X_te)})")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_tr, y_tr, test_size=0.2,
                                                   stratify=y_tr, random_state=42)
        tgt = f"{src}_split"
        print(f"Pas d'autre benchmark — split 80/20 sur {src}")

    # Classifieurs classiques
    classical = [
        ("SVM_RBF", SVC(kernel=cfg["models"]["svm_kernel"],
                        C=cfg["models"]["svm_c"], probability=True, random_state=42)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=cfg["models"]["n_estimators_rf"], n_jobs=-1, random_state=42)),
    ]
    if HAS_XGB:
        classical.append(("XGBoost", xgb.XGBClassifier(
            n_estimators=cfg["models"]["xgb_n_estimators"],
            learning_rate=cfg["models"]["xgb_learning_rate"],
            max_depth=cfg["models"]["xgb_max_depth"],
            use_label_encoder=False, eval_metric="mlogloss", random_state=42)))

    summary = []

    for name, clf in classical:
        print(f"\n[{name}] Training...")
        with mlflow.start_run(run_name=f"{name}_{src}->{tgt}"):
            mlflow.log_params({"model": name, "train": src, "test": tgt})
            pipe, acc, f1, cm, report = train_classical(name, clf, X_tr, y_tr, X_te, y_te)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.sklearn.log_model(pipe, name)

            cm_path = results_dir / f"cm_{name}.png"
            plot_cm(cm, f"{name} {src}→{tgt} (acc={acc:.3f})", cm_path)
            mlflow.log_artifact(str(cm_path))

            print(f"  Accuracy={acc:.4f}  F1={f1:.4f}")
            print(report)
            summary.append({"Model": name, "Train": src, "Test": tgt,
                             "Accuracy": round(acc, 4), "F1_macro": round(f1, 4)})

            if name == "RandomForest":
                fi = pipe.named_steps["clf"].feature_importances_
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.bar(range(len(fi)), fi, color="steelblue", alpha=0.7)
                ax.set_title("Random Forest — Feature Importance (325 features)")
                ax.set_xlabel("Feature index")
                ax.set_ylabel("Importance")
                fi_path = results_dir / "01_rf_feature_importance.png"
                plt.tight_layout()
                plt.savefig(fi_path, dpi=150)
                plt.close()
                mlflow.log_artifact(str(fi_path))

    # CNN 1D
    print(f"\n[CNN1D] Training...")
    cnn_cfg = cfg["cnn"]
    with mlflow.start_run(run_name=f"CNN1D_{src}->{tgt}"):
        mlflow.log_params({"model": "CNN1D", "train": src, "test": tgt,
                           "epochs": cnn_cfg["epochs"], "lr": cnn_cfg["lr"]})
        model, scaler, acc, f1, cm, report = train_cnn(X_tr, y_tr, X_te, y_te, cnn_cfg, device)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        cm_path = results_dir / "cm_CNN1D.png"
        plot_cm(cm, f"CNN1D {src}→{tgt} (acc={acc:.3f})", cm_path)
        mlflow.log_artifact(str(cm_path))

        torch.save(model.state_dict(), results_dir / f"cnn1d_{src}.pt")
        np.save(results_dir / f"scaler_mean_{src}.npy", scaler.mean_)
        np.save(results_dir / f"scaler_scale_{src}.npy", scaler.scale_)

        print(f"  Accuracy={acc:.4f}  F1={f1:.4f}")
        print(report)
        summary.append({"Model": "CNN1D", "Train": src, "Test": tgt,
                         "Accuracy": round(acc, 4), "F1_macro": round(f1, 4)})

    # Bilan
    df_sum = pd.DataFrame(summary).sort_values("Accuracy", ascending=False)
    df_sum.to_csv(results_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print(f"BILAN  —  Train={src}  Test={tgt}")
    print(df_sum.to_string(index=False))

    # Bar chart comparatif
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "darkorange", "green", "purple"]
    bars = ax.bar(df_sum["Model"], df_sum["Accuracy"], color=colors[:len(df_sum)], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(1 / 3, color="gray", linestyle="--", label="Aleatoire (33%)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy"); ax.set_title(f"Comparaison modeles — {src}→{tgt}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "02_model_comparison.png", dpi=150)
    plt.close()
    print("Bilan sauvegarde.")

    # ── MLflow Model Registry ──────────────────────────────────────────────────
    # Trouver le meilleur run et l'enregistrer dans le Registry
    best_row = df_sum.iloc[0]
    best_acc = best_row["Accuracy"]
    best_model_name = best_row["Model"]

    # Écrire l'accuracy pour le Quality Gate Jenkins
    (results_dir / "best_accuracy.txt").write_text(f"{best_acc:.4f}")

    # Exporter les métriques au format JSON pour DVC (dvc metrics show / diff)
    import json
    metrics = {row["Model"]: {"accuracy": row["Accuracy"], "f1_macro": row["F1_macro"]}
               for _, row in df_sum.iterrows()}
    metrics["_best"] = {"model": best_model_name, "accuracy": best_acc,
                        "train": src, "test": tgt}
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        experiment = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model = '{best_model_name}'",
                order_by=["metrics.accuracy DESC"],
                max_results=1,
            )
            if runs:
                best_run_id = runs[0].info.run_id
                print(f"\nEnregistrement dans MLflow Registry : model={best_model_name}  run={best_run_id}")

                mv = mlflow.register_model(
                    model_uri=f"runs:/{best_run_id}/{best_model_name}",
                    name="HT-Detector",
                )

                stage = "Production" if best_acc >= 0.90 else "Staging"
                client.transition_model_version_stage(
                    name="HT-Detector",
                    version=mv.version,
                    stage=stage,
                )
                print(f"  Version {mv.version} promue en {stage}  (accuracy={best_acc:.4f})")
    except Exception as e:
        print(f"[WARN] Model Registry non disponible (MLflow server requis) : {e}")
        print("       Lancer mlflow server ou docker compose up mlflow pour activer le Registry.")
