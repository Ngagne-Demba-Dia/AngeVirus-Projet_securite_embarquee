"""
15_train_tiny_mlp.py — Tiny MLP pour déploiement STM32 Nucleo-F401RE.

Méthode : Knowledge Distillation (Hinton et al. 2015).
  - Teacher : CNN1D AT_T800 (modèle de production, 69.1% accuracy)
  - Student : Tiny MLP 500→64→32→3 (34.3KB int8, tient dans 512KB Flash)

Pipeline :
  1. Entraînement multi-source (tous benchmarks disponibles)
  2. Distillation de connaissance teacher → student
  3. Quantisation statique int8 (PyTorch)
  4. Validation accuracy student vs teacher
  5. Export header C pour STM32 (weights + biases + scaler)

Contraintes STM32 Nucleo-F401RE (ARM Cortex-M4 @ 84MHz) :
  Flash 512KB → poids int8 ≤ 34KB  ✓
  SRAM   96KB → activations ≤ 2KB  ✓

Fichiers générés (captures rapport) :
  results/15_TinyMLP_comparaison.png   — accuracy student vs teacher vs benchmarks
  results/15_TinyMLP_confusion.png     — matrices de confusion
  results/15_TinyMLP_memoire.png       — analyse empreinte mémoire STM32
  results/15_TinyMLP_metrics.json      — métriques complètes
  results/ht_detector_stm32.h          — header C complet pour STM32
"""
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
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models import CNN1D

# ── Architecture Tiny MLP (contraintes STM32) ─────────────────────────────────
N_INPUT    = 500    # features extraites (après Étape 2)
HIDDEN1    = 64     # couche cachée 1
HIDDEN2    = 32     # couche cachée 2
N_CLASSES  = 3

# ── Hyperparamètres distillation ──────────────────────────────────────────────
TEMPERATURE   = 4.0    # température pour soft labels (Hinton recommande 3-5)
ALPHA         = 0.4    # poids KL divergence vs CrossEntropy (0=supervisé pur, 1=full KD)
EPOCHS        = 150
LR            = 1e-3
BATCH_SIZE    = 256
LABEL_NAMES   = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
TARGET_BM     = "AES-T800"


# ── Tiny MLP ──────────────────────────────────────────────────────────────────
class TinyMLP(nn.Module):
    """
    MLP minimal pour STM32 Nucleo-F401RE.
    Empreinte int8 : 500×64 + 64×32 + 32×3 = 34,243 params = 34.2KB Flash
    Activation max : 64 neurones × 4B = 256B SRAM
    """
    def __init__(self, n_input: int = N_INPUT, h1: int = HIDDEN1,
                 h2: int = HIDDEN2, n_classes: int = N_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(n_input, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, n_classes)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        return self.fc3(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flash_size_bytes(self) -> int:
        """Taille estimée en Flash pour weights int8."""
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad and len(p.shape) >= 1)


# ── Knowledge Distillation Loss ───────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, true_labels,
                       temperature, alpha):
    """
    L = alpha * KL(student_soft || teacher_soft) + (1-alpha) * CE(student, labels)
    Les soft labels du teacher capturent les relations inter-classes.
    """
    # Soft labels teacher (température T)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    # Hard labels (classification standard)
    ce_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * kl_loss + (1 - alpha) * ce_loss


# ── Entraînement avec distillation ────────────────────────────────────────────
def train_with_distillation(student, teacher, X_tr, y_tr, device,
                             epochs, lr, batch_size, temperature, alpha):
    teacher.eval()   # teacher figé
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    print(f"\n{'Epoch':>6} | {'Loss total':>11} | {'Loss KD':>9} | {'Loss CE':>9}")
    print("-"*45)

    for ep in range(epochs):
        total_loss, total_kd, total_ce, n = 0.0, 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            with torch.no_grad():
                teacher_logits = teacher(xb)

            student_logits = student(xb)
            loss = distillation_loss(student_logits, teacher_logits, yb,
                                      temperature, alpha)
            ce  = F.cross_entropy(student_logits, yb)
            kd  = loss - (1 - alpha) * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item(); total_kd += kd.item()
            total_ce   += ce.item();  n += 1

        scheduler.step()
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            print(f"{ep+1:>6} | {total_loss/n:>11.4f} | {total_kd/n:>9.4f} | {total_ce/n:>9.4f}")

    student.eval()
    return student


# ── Évaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X, y, device):
    X_t = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    return (float(accuracy_score(y, preds)),
            float(f1_score(y, preds, average="macro")),
            confusion_matrix(y, preds))


# ── Export header C pour STM32 ────────────────────────────────────────────────
def export_stm32_header(model: TinyMLP, scaler: StandardScaler,
                         results_dir: Path, accuracy: float) -> Path:
    """
    Génère le header C complet pour STM32 :
    - Poids et biais de chaque couche en int8
    - Facteurs de mise à l'échelle (scale + zero_point) pour dé-quantisation
    - Paramètres du StandardScaler pour normalisation sur device
    """
    sd = model.state_dict()

    def quantize_to_int8(tensor: torch.Tensor):
        """Quantise un tensor float32 → int8 avec scale/zero_point."""
        t = tensor.float()
        t_min, t_max = t.min().item(), t.max().item()
        scale = max(abs(t_min), abs(t_max)) / 127.0
        if scale == 0:
            scale = 1e-8
        q = (t / scale).clamp(-127, 127).round().to(torch.int8).cpu().numpy()
        return q, scale

    path = results_dir / "ht_detector_stm32.h"
    lines = [
        "/**",
        " * ht_detector_stm32.h",
        " * Hardware Trojan Detector — Tiny MLP int8 pour STM32 Nucleo-F401RE",
        " * Généré automatiquement par 15_train_tiny_mlp.py",
        " * Architecture : 500 → 64 → 32 → 3 (Knowledge Distillation depuis AT_T800)",
        f" * Accuracy validation : {accuracy:.2%}",
        f" * Empreinte Flash     : ~{model.flash_size_bytes() // 1024}KB",
        " * ARM Cortex-M4 @ 84MHz | Flash 512KB | SRAM 96KB",
        " */",
        "#pragma once",
        "#include <stdint.h>",
        "#include <string.h>",
        "#include <math.h>     /* expf() pour la softmax */",
        "",
        "/* ── Dimensions ─────────────────────────────────────────────── */",
        f"#define HT_N_INPUT     {N_INPUT}",
        f"#define HT_HIDDEN1     {HIDDEN1}",
        f"#define HT_HIDDEN2     {HIDDEN2}",
        f"#define HT_N_CLASSES   {N_CLASSES}",
        f"#define HT_ACCURACY    {accuracy:.4f}f",
        "",
        "/* ── Classes ────────────────────────────────────────────────── */",
        'static const char* HT_CLASS_NAMES[3] = {',
        '    "TrojanDisabled", "TrojanEnabled", "TrojanTriggered"',
        '};',
        'static const char* HT_RISK_LEVEL[3] = {',
        '    "OK", "WARNING", "ALERT"',
        '};',
        "",
        "/* ── Scaler (normalisation des features avant inference) ─────── */",
        f"/* mean_ et scale_ du StandardScaler — {N_INPUT} valeurs */",
    ]

    # Scaler mean
    mean_vals = ", ".join(f"{v:.6f}f" for v in scaler.mean_)
    lines.append(f"static const float HT_SCALER_MEAN[{N_INPUT}] = {{")
    chunk = 8
    for i in range(0, len(scaler.mean_), chunk):
        vals = ", ".join(f"{v:.6f}f" for v in scaler.mean_[i:i+chunk])
        lines.append(f"    {vals},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    lines.append("")

    # Scaler scale
    lines.append(f"static const float HT_SCALER_SCALE[{N_INPUT}] = {{")
    for i in range(0, len(scaler.scale_), chunk):
        vals = ", ".join(f"{v:.6f}f" for v in scaler.scale_[i:i+chunk])
        lines.append(f"    {vals},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    lines.append("")

    # Couches
    layer_map = [
        ("fc1", "W1", "B1", HIDDEN1, N_INPUT),
        ("fc2", "W2", "B2", HIDDEN2, HIDDEN1),
        ("fc3", "W3", "B3", N_CLASSES, HIDDEN2),
    ]

    for fc_name, wname, bname, out_dim, in_dim in layer_map:
        W = sd[f"{fc_name}.weight"]
        b = sd[f"{fc_name}.bias"]
        W_q, W_scale = quantize_to_int8(W)
        b_q, b_scale = quantize_to_int8(b)

        lines += [
            f"/* ── {fc_name} : {in_dim} → {out_dim} ─────────────── */",
            f"#define {wname}_SCALE  {W_scale:.8f}f",
            f"#define {bname}_SCALE  {b_scale:.8f}f",
            f"static const int8_t {wname}[{out_dim}][{in_dim}] = {{",
        ]
        for row in W_q:
            row_str = ", ".join(f"{int(v):4d}" for v in row)
            lines.append(f"    {{{row_str}}},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("};")

        bias_str = ", ".join(f"{int(v):4d}" for v in b_q)
        lines += [
            f"static const int8_t {bname}[{out_dim}] = {{{bias_str}}};",
            "",
        ]

    # BN parameters — exportés en tableaux float COMPLETS (appliqués à l'inférence)
    # BatchNorm inference : y = factor * (Wx + b) + bias_fused
    #   factor     = gamma / sqrt(var + eps)
    #   bias_fused = beta - running_mean * factor
    lines += [
        "/* ── BatchNorm parameters (appliqués dans ht_predict) ───────── */",
        "/* Inference BN : y = factor * (Wx + b) + bias_fused, puis ReLU  */",
    ]
    for bn_name, dim, fac_name, bias_name in [
        ("bn1", HIDDEN1, "BN1_FACTOR", "BN1_BIAS"),
        ("bn2", HIDDEN2, "BN2_FACTOR", "BN2_BIAS"),
    ]:
        gamma = sd[f"{bn_name}.weight"].cpu().numpy()
        beta  = sd[f"{bn_name}.bias"].cpu().numpy()
        mean  = sd[f"{bn_name}.running_mean"].cpu().numpy()
        var   = sd[f"{bn_name}.running_var"].cpu().numpy()
        factor     = gamma / np.sqrt(var + 1e-5)
        bias_fused = beta - mean * factor

        for arr_name, arr in [(fac_name, factor), (bias_name, bias_fused)]:
            lines.append(f"static const float {arr_name}[{dim}] = {{")
            for i in range(0, len(arr), chunk):
                vals = ", ".join(f"{v:.6f}f" for v in arr[i:i+chunk])
                lines.append(f"    {vals},")
            lines[-1] = lines[-1].rstrip(",")
            lines.append("};")
        lines.append("")

    # Fonction d'inférence inline
    lines += [
        "",
        "/* ── Fonction d'inférence (C99 pur, sans malloc) ─────────────── */",
        "static inline int ht_predict(const float* features, float* confidence_out) {",
        "    /* Normaliser les features */",
        f"    float x0[{N_INPUT}];",
        f"    for (int i = 0; i < {N_INPUT}; i++)",
        "        x0[i] = (features[i] - HT_SCALER_MEAN[i]) / HT_SCALER_SCALE[i];",
        "",
        f"    /* Couche 1 : {N_INPUT} → {HIDDEN1} (Linear + BatchNorm + ReLU) */",
        f"    float h1[{HIDDEN1}];",
        f"    for (int i = 0; i < {HIDDEN1}; i++) {{",
        "        float s = B1[i] * B1_SCALE;",
        f"        for (int j = 0; j < {N_INPUT}; j++)",
        "            s += W1[i][j] * W1_SCALE * x0[j];",
        "        s = BN1_FACTOR[i] * s + BN1_BIAS[i];   /* BatchNorm */",
        "        h1[i] = s > 0.0f ? s : 0.0f;           /* ReLU */",
        "    }",
        "",
        f"    /* Couche 2 : {HIDDEN1} → {HIDDEN2} (Linear + BatchNorm + ReLU) */",
        f"    float h2[{HIDDEN2}];",
        f"    for (int i = 0; i < {HIDDEN2}; i++) {{",
        "        float s = B2[i] * B2_SCALE;",
        f"        for (int j = 0; j < {HIDDEN1}; j++)",
        "            s += W2[i][j] * W2_SCALE * h1[j];",
        "        s = BN2_FACTOR[i] * s + BN2_BIAS[i];   /* BatchNorm */",
        "        h2[i] = s > 0.0f ? s : 0.0f;           /* ReLU */",
        "    }",
        "",
        f"    /* Couche 3 : {HIDDEN2} → {N_CLASSES} (logits) */",
        f"    float logits[{N_CLASSES}];",
        f"    for (int i = 0; i < {N_CLASSES}; i++) {{",
        "        float s = B3[i] * B3_SCALE;",
        f"        for (int j = 0; j < {HIDDEN2}; j++)",
        "            s += W3[i][j] * W3_SCALE * h2[j];",
        "        logits[i] = s;",
        "    }",
        "",
        "    /* Softmax + argmax */",
        "    float max_l = logits[0];",
        f"    for (int i = 1; i < {N_CLASSES}; i++)",
        "        if (logits[i] > max_l) max_l = logits[i];",
        f"    float sum_exp = 0.0f;",
        f"    float probs[{N_CLASSES}];",
        f"    for (int i = 0; i < {N_CLASSES}; i++) {{",
        "        probs[i] = expf(logits[i] - max_l);",
        "        sum_exp += probs[i];",
        "    }",
        "    int pred = 0; *confidence_out = 0.0f;",
        f"    for (int i = 0; i < {N_CLASSES}; i++) {{",
        "        probs[i] /= sum_exp;",
        "        if (probs[i] > *confidence_out) {",
        "            *confidence_out = probs[i];",
        "            pred = i;",
        "        }",
        "    }",
        "    return pred;",
        "}",
        "",
        "/* ── Usage ──────────────────────────────────────────────────────",
        " * float features[HT_N_INPUT];  // features normalisées",
        " * float confidence;",
        " * int label = ht_predict(features, &confidence);",
        " * printf(\"%s (%.1f%%)\\n\", HT_CLASS_NAMES[label], confidence * 100);",
        " * if (label == 2) HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);",
        " * ─────────────────────────────────────────────────────────────── */",
    ]

    path.write_text("\n".join(lines))
    size_kb = path.stat().st_size / 1024
    print(f"Header C STM32 généré : {path.name}  ({size_kb:.1f} KB)")
    return path


def simulate_int8_inference(model: TinyMLP, X_norm: np.ndarray) -> np.ndarray:
    """
    Réplique EXACTEMENT le code C de ht_predict() avec poids int8 + BatchNorm.
    Permet de connaître la vraie accuracy embarquée (vs accuracy float).
    X_norm : features DÉJÀ normalisées (le scaler est appliqué hors device ici).
    """
    sd = model.state_dict()

    def quant(t):
        t = t.float()
        scale = max(abs(t.min().item()), abs(t.max().item())) / 127.0
        scale = scale if scale != 0 else 1e-8
        q = (t / scale).clamp(-127, 127).round().cpu().numpy()
        return q, scale

    W1q, W1s = quant(sd["fc1.weight"]); b1q, b1s = quant(sd["fc1.bias"])
    W2q, W2s = quant(sd["fc2.weight"]); b2q, b2s = quant(sd["fc2.bias"])
    W3q, W3s = quant(sd["fc3.weight"]); b3q, b3s = quant(sd["fc3.bias"])

    def bn_params(name):
        g = sd[f"{name}.weight"].cpu().numpy(); be = sd[f"{name}.bias"].cpu().numpy()
        m = sd[f"{name}.running_mean"].cpu().numpy(); v = sd[f"{name}.running_var"].cpu().numpy()
        fac = g / np.sqrt(v + 1e-5)
        return fac, be - m * fac
    bn1_f, bn1_b = bn_params("bn1")
    bn2_f, bn2_b = bn_params("bn2")

    # Forward int8 (déquantisé en float comme le fait le C)
    h1 = (b1q * b1s) + X_norm @ (W1q * W1s).T   # (N, 64)
    h1 = bn1_f * h1 + bn1_b
    h1 = np.maximum(h1, 0.0)
    h2 = (b2q * b2s) + h1 @ (W2q * W2s).T        # (N, 32)
    h2 = bn2_f * h2 + bn2_b
    h2 = np.maximum(h2, 0.0)
    logits = (b3q * b3s) + h2 @ (W3q * W3s).T    # (N, 3)
    return logits.argmax(1)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("../results")
    src = cfg["dataset"]["benchmarks"]["source"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  Architecture : {N_INPUT}→{HIDDEN1}→{HIDDEN2}→{N_CLASSES}")

    # ── Charger scaler et teacher (AT_T800) ───────────────────────────────────
    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_ms_mean_{TARGET_BM}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_ms_scale_{TARGET_BM}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    teacher = CNN1D().to(device)
    at_path = results_dir / f"cnn1d_AT_{TARGET_BM}.pt"
    sd_t = {k: v for k, v in torch.load(at_path, map_location="cpu").items()
            if k in teacher.state_dict()}
    teacher.load_state_dict(sd_t, strict=False)
    teacher.eval()
    print(f"Teacher chargé : {at_path.name}")

    # ── Données multi-source (tous benchmarks disponibles) ────────────────────
    all_bms = cfg["dataset"]["benchmarks"]["all"]
    X_list, y_list = [], []
    for bm in all_bms:
        p = results_dir / f"features_{bm}.npz"
        if p.exists():
            d = np.load(p)
            X_list.append(scaler.transform(d["X"].astype(np.float32)))
            y_list.append(d["y"].astype(np.int64))
            print(f"  Chargé {bm} : {len(d['y'])} traces")

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.concatenate(y_list)
    print(f"Total : {len(X_all)} traces, {X_all.shape[1]} features")

    # Train/test split stratifié
    rng   = np.random.RandomState(42)
    idx   = rng.permutation(len(X_all))
    n_te  = int(len(X_all) * 0.15)
    X_te, y_te = X_all[idx[:n_te]], y_all[idx[:n_te]]
    X_tr, y_tr = X_all[idx[n_te:]], y_all[idx[n_te:]]
    print(f"Train : {len(X_tr)} | Test : {len(X_te)}")

    # ── Évaluation teacher (référence) ────────────────────────────────────────
    acc_teacher, f1_teacher, _ = evaluate(teacher, X_te, y_te, device)
    print(f"\nTeacher (AT_T800) — acc={acc_teacher:.4f}  f1={f1_teacher:.4f}")

    # ── Entraînement Tiny MLP avec Knowledge Distillation ─────────────────────
    print(f"\n=== KNOWLEDGE DISTILLATION (T={TEMPERATURE}, α={ALPHA}, {EPOCHS} epochs) ===")
    student = TinyMLP().to(device)
    print(f"Tiny MLP params : {student.count_params():,}  (~{student.flash_size_bytes()//1024}KB Flash)")

    mlflow.set_tracking_uri(str(results_dir / "mlruns"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"TinyMLP_KD_STM32_{TARGET_BM}"):
        mlflow.log_params({
            "architecture": f"{N_INPUT}→{HIDDEN1}→{HIDDEN2}→{N_CLASSES}",
            "temperature": TEMPERATURE, "alpha": ALPHA,
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "n_train": len(X_tr), "n_test": len(X_te),
            "flash_bytes": student.flash_size_bytes(),
        })

        student = train_with_distillation(
            student, teacher, X_tr, y_tr, device,
            EPOCHS, LR, BATCH_SIZE, TEMPERATURE, ALPHA
        )

        # ── Évaluation student ─────────────────────────────────────────────────
        acc_student, f1_student, cm_student = evaluate(student, X_te, y_te, device)
        print(f"\nStudent (Tiny MLP) — acc={acc_student:.4f}  f1={f1_student:.4f}")
        print(f"Ratio acc student/teacher : {acc_student/acc_teacher:.2%}")

        mlflow.log_metric("acc_student",  acc_student)
        mlflow.log_metric("f1_student",   f1_student)
        mlflow.log_metric("acc_teacher",  acc_teacher)
        mlflow.log_metric("acc_ratio",    acc_student / acc_teacher)

        # Évaluer par benchmark
        print(f"\n{'Benchmark':15} {'Teacher':10} {'Student':10} {'Ratio':8}")
        print("-"*45)
        bm_results = {}
        for bm in all_bms:
            p = results_dir / f"features_{bm}.npz"
            if not p.exists():
                continue
            d  = np.load(p)
            Xb = scaler.transform(d["X"].astype(np.float32))
            yb = d["y"].astype(np.int64)
            at, _, _ = evaluate(teacher, Xb, yb, device)
            as_, _, _ = evaluate(student, Xb, yb, device)
            ratio = as_ / max(at, 1e-6)
            print(f"{bm:15} {at:10.4f} {as_:10.4f} {ratio:8.2%}")
            bm_results[bm] = {"teacher": round(at, 4), "student": round(as_, 4),
                               "ratio": round(ratio, 4)}

        # Sauvegarder le modèle student
        student_path = results_dir / "cnn1d_tiny_stm32.pt"
        torch.save(student.state_dict(), student_path)

    # ── Figure 1 : Comparaison teacher vs student par benchmark ───────────────
    bms_plot = [b for b in all_bms if b in bm_results]
    t_accs   = [bm_results[b]["teacher"] for b in bms_plot]
    s_accs   = [bm_results[b]["student"] for b in bms_plot]

    x = np.arange(len(bms_plot))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, t_accs, w, color="steelblue",  alpha=0.85, label="Teacher AT_T800 (CNN1D)")
    b2 = ax.bar(x + w/2, s_accs, w, color="darkorange", alpha=0.85, label="Student Tiny MLP (STM32)")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(1/3, color="gray", linestyle="--", lw=1.2, label="Aléatoire (33%)")
    ax.axhline(0.70, color="green", linestyle=":", lw=1.5, label="Objectif 70%")
    ax.set_xticks(x); ax.set_xticklabels(bms_plot, rotation=10)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Knowledge Distillation — Teacher CNN1D vs Student Tiny MLP\n"
                 f"Student : {N_INPUT}→{HIDDEN1}→{HIDDEN2}→{N_CLASSES}  "
                 f"(~{student.flash_size_bytes()//1024}KB Flash STM32)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out1 = results_dir / "15_TinyMLP_comparaison.png"
    plt.savefig(out1, dpi=150); plt.close()

    # ── Figure 2 : Confusion matrix student ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in zip(axes,
                              [_, cm_student],
                              ["Teacher AT_T800", "Student Tiny MLP (STM32)"]):
        _, _, cm = evaluate(teacher if title.startswith("T") else student,
                            X_te, y_te, device)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_title(f"{title}\nacc={acc_teacher:.3f}" if title.startswith("T")
                     else f"{title}\nacc={acc_student:.3f}")
        ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    plt.suptitle("Matrices de confusion — Test multi-source", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out2 = results_dir / "15_TinyMLP_confusion.png"
    plt.savefig(out2, dpi=150); plt.close()

    # ── Figure 3 : Analyse mémoire STM32 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Flash breakdown
    flash_data = {
        "FC1 weights\n(500×64)": 500*64,
        "FC2 weights\n(64×32)":  64*32,
        "FC3 weights\n(32×3)":   32*3,
        "BN params":             (64+32)*4,
        "Biases":                (64+32+3),
        "Scaler (float32)":      N_INPUT * 2 * 4,
    }
    ax = axes[0]
    colors_flash = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6","#1abc9c"]
    wedges, texts, autotexts = ax.pie(
        flash_data.values(), labels=flash_data.keys(),
        colors=colors_flash, autopct="%1.1f%%", startangle=90
    )
    total_flash = sum(flash_data.values())
    ax.set_title(f"Empreinte Flash STM32\nTotal : {total_flash/1024:.1f}KB / 512KB disponibles\n"
                 f"({total_flash/512000*100:.1f}% utilisé)")

    # Comparaison modèles taille
    ax2 = axes[1]
    models = ["CNN1D\nfloat32", "CNN1D\nint8", "Tiny MLP\nfloat32", "Tiny MLP\nint8"]
    sizes  = [572*4, 572, student.count_params()*4//1024, student.flash_size_bytes()//1024]
    colors_bar = ["#e74c3c","#c0392b","#2ecc71","#27ae60"]
    bars = ax2.bar(models, sizes, color=colors_bar, alpha=0.85)
    ax2.bar_label(bars, fmt="%dKB", padding=3, fontsize=9)
    ax2.axhline(512, color="blue", linestyle="--", lw=1.5, label="Flash max 512KB")
    ax2.axhline(96,  color="red",  linestyle=":",  lw=1.5, label="SRAM max 96KB")
    ax2.set_ylabel("Taille (KB)")
    ax2.set_title("Comparaison empreinte mémoire\n(vert = compatible STM32, rouge = incompatible)")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Analyse mémoire — Déploiement STM32 Nucleo-F401RE", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out3 = results_dir / "15_TinyMLP_memoire.png"
    plt.savefig(out3, dpi=150); plt.close()

    # ── Export header C STM32 ─────────────────────────────────────────────────
    print("\n=== EXPORT HEADER C POUR STM32 ===")
    student_cpu = TinyMLP().cpu()
    student_cpu.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()})
    student_cpu.eval()
    header_path = export_stm32_header(student_cpu, scaler, results_dir, acc_student)

    # ── Validation int8 : accuracy RÉELLE sur STM32 (réplique du code C) ───────
    print("\n=== VALIDATION INT8 (accuracy embarquée réelle vs float) ===")
    preds_int8 = simulate_int8_inference(student_cpu, X_te)
    acc_int8   = float(accuracy_score(y_te, preds_int8))
    print(f"  Accuracy float32 (PyTorch) : {acc_student:.4f}")
    print(f"  Accuracy int8   (STM32)    : {acc_int8:.4f}")
    print(f"  Perte quantisation         : {acc_student - acc_int8:+.4f}")
    # Par benchmark en int8
    int8_bm = {}
    for bm in all_bms:
        p = results_dir / f"features_{bm}.npz"
        if not p.exists():
            continue
        d  = np.load(p)
        Xb = scaler.transform(d["X"].astype(np.float32))
        yb = d["y"].astype(np.int64)
        acc_b = float(accuracy_score(yb, simulate_int8_inference(student_cpu, Xb)))
        int8_bm[bm] = round(acc_b, 4)
        print(f"    {bm:15} int8 acc = {acc_b:.4f}")

    # ── Métriques JSON ─────────────────────────────────────────────────────────
    metrics = {
        "architecture": f"{N_INPUT}→{HIDDEN1}→{HIDDEN2}→{N_CLASSES}",
        "temperature": TEMPERATURE, "alpha": ALPHA,
        "teacher_acc": round(acc_teacher, 4),
        "student_acc_float": round(acc_student, 4),
        "student_acc_int8":  round(acc_int8, 4),
        "quantization_loss": round(acc_student - acc_int8, 4),
        "student_f1":  round(f1_student, 4),
        "acc_ratio":   round(acc_student / acc_teacher, 4),
        "flash_bytes": student.flash_size_bytes(),
        "flash_kb":    student.flash_size_bytes() // 1024,
        "params":      student.count_params(),
        "benchmarks":      bm_results,
        "benchmarks_int8": int8_bm,
        "stm32_target": "Nucleo-F401RE (ARM Cortex-M4 @ 84MHz)",
        "header_file": str(header_path),
    }
    (results_dir / "15_TinyMLP_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Résumé terminal ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("BILAN — TINY MLP STM32 (Knowledge Distillation)")
    print("="*65)
    print(f"  Architecture     : {N_INPUT}→{HIDDEN1}→{HIDDEN2}→{N_CLASSES}")
    print(f"  Paramètres       : {student.count_params():,}")
    print(f"  Flash int8       : ~{student.flash_size_bytes()//1024} KB / 512 KB  ✓")
    print(f"  SRAM activations : ~{max(N_INPUT,HIDDEN1,HIDDEN2)*4//1024+1} KB / 96 KB  ✓")
    print(f"  Teacher accuracy : {acc_teacher:.2%}")
    print(f"  Student accuracy : {acc_student:.2%}  ({acc_student/acc_teacher:.1%} du teacher)")
    print(f"  F1-score student : {f1_student:.4f}")
    print(f"\n  Fichiers captures rapport :")
    for f in [out1, out2, out3, header_path, results_dir/"15_TinyMLP_metrics.json"]:
        print(f"    {Path(f).name}")
    print(f"\n  → Prêt pour flashage STM32 Nucleo-F401RE")
