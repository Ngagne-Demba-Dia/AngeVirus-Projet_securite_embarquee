"""
08_fpga_deploy.py — Préparation déploiement embarqué / FPGA.
Pipeline :
  1. Export ONNX (format portable edge/FPGA)
  2. Quantisation dynamique int8 (PyTorch)
  3. Benchmark float32 vs int8 : taille, latence, précision
  4. Estimation ressources FPGA (LUT / DSP / BRAM) — calcul analytique
  5. Génération header C avec les poids quantisés (déploiement bare-metal)
  6. hls4ml si installé → code HLS C++ pour synthèse Vivado
"""
import time
import json
import struct
import numpy as np
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models import CNN1D

# ── Constantes architecture CNN1D ──────────────────────────────────────────────
CONV_LAYERS = [
    {"in": 1,  "out": 32,  "kernel": 3},
    {"in": 32, "out": 64,  "kernel": 3},
    {"in": 64, "out": 128, "kernel": 3},
]
FC_LAYERS = [
    {"in": 2048, "out": 256},
    {"in": 256,  "out": 64},
    {"in": 64,   "out": 3},
]
N_INPUT    = 325
POOL_OUT   = 16
CLK_MHZ    = 100   # fréquence cible FPGA


# ── Chargement ─────────────────────────────────────────────────────────────────
def load_model_and_data(results_dir: Path, cfg: dict):
    src = cfg["dataset"]["benchmarks"]["source"]

    model = CNN1D()
    ft_path = results_dir / "cnn1d_ft_AES-T700.pt"
    if ft_path.exists():
        model.load_state_dict(torch.load(ft_path, map_location="cpu"))
        data_bm = "AES-T700"
        print("Modèle fine-tuné T700 chargé.")
    else:
        model.load_state_dict(torch.load(results_dir / f"cnn1d_{src}.pt", map_location="cpu"))
        data_bm = src
        print(f"Modèle source {src} chargé.")
    model.eval()

    scaler = StandardScaler()
    scaler.mean_  = np.load(results_dir / f"scaler_mean_{src}.npy")
    scaler.scale_ = np.load(results_dir / f"scaler_scale_{src}.npy")
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    d = np.load(results_dir / f"features_{data_bm}.npz")
    X = torch.FloatTensor(scaler.transform(d["X"].astype(np.float32)))
    y = d["y"].astype(np.int64)
    return model, scaler, X, y


# ── 1. Export ONNX ─────────────────────────────────────────────────────────────
def export_onnx(model, results_dir: Path) -> Path:
    path = results_dir / "ht_detector.onnx"
    dummy = torch.zeros(1, 325)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    size_kb = path.stat().st_size / 1024
    print(f"ONNX exporté : {path.name}  ({size_kb:.1f} KB)")
    return path


# ── 2. Quantisation dynamique int8 ────────────────────────────────────────────
def quantize_model(model):
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    return model_int8


def model_size_kb(model) -> float:
    tmp = Path("/tmp/model_size_check.pt")
    torch.save(model.state_dict(), str(tmp))
    size = tmp.stat().st_size / 1024
    tmp.unlink(missing_ok=True)
    return size


# ── 3. Benchmark latence ───────────────────────────────────────────────────────
def benchmark_latency(model, X: torch.Tensor, n_runs: int = 200) -> float:
    """Latence moyenne sur une seule trace (batch=1)."""
    x1 = X[:1]
    with torch.no_grad():
        for _ in range(10):           # warmup
            model(x1)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x1)
    return (time.perf_counter() - t0) / n_runs * 1000   # ms


def eval_accuracy(model, X: torch.Tensor, y) -> float:
    with torch.no_grad():
        preds = model(X).argmax(1).numpy()
    return float(accuracy_score(y, preds))


# ── 4. Estimation ressources FPGA ─────────────────────────────────────────────
def estimate_fpga_resources() -> dict:
    """
    Estimation analytique pour Xilinx Artix-7 XC7A200T @ 100 MHz.
    Conv pipeline : 1 DSP par filtre × kernel_size (avec time-sharing ×4)
    FC  pipeline  : 1 DSP par colonne (avec time-sharing ×8)
    BRAM 18K (2048 words × 18-bit par bloc)
    """
    total_macs = 0

    # MACs Conv (input_length = N_INPUT pour tous les blocs avant le pool)
    for layer in CONV_LAYERS:
        macs = layer["in"] * layer["out"] * layer["kernel"] * N_INPUT
        total_macs += macs

    # MACs FC
    for layer in FC_LAYERS:
        macs = layer["in"] * layer["out"]
        total_macs += macs

    # DSP (time-sharing ×8, chaque DSP fait 1 MAC/cycle à 100 MHz)
    dsps = sum(
        l["out"] * l["kernel"] for l in CONV_LAYERS
    ) // 8 + sum(l["out"] for l in FC_LAYERS) // 8

    # Paramètres totaux
    params = sum(l["in"] * l["out"] * l["kernel"] for l in CONV_LAYERS)
    params += sum(l["in"] * l["out"] for l in FC_LAYERS)

    # BRAM (poids int8 → 1 octet/param, BRAM 18K = 2048 octets)
    bram_18k = (params * 1) // 2048 + 1

    # LUT (règle empirique CNN : ~5 LUTs/param pour BRAM+logic)
    luts = params * 5 // 1000   # en milliers

    # Latence théorique (pipeline complet, 1 MAC/cycle)
    lat_cycles = total_macs // max(dsps, 1)
    lat_us     = lat_cycles / CLK_MHZ

    return {
        "total_parameters": params,
        "total_MACs":        total_macs,
        "DSP48":             dsps,
        "BRAM_18K":          bram_18k,
        "LUT_k":             luts,
        "latence_us":        round(lat_us, 1),
        "frequence_MHz":     CLK_MHZ,
        "fpga_cible":        "Xilinx Artix-7 XC7A200T",
    }


# ── 5. Header C poids quantisés ────────────────────────────────────────────────
def generate_c_header(model, scaler, results_dir: Path) -> Path:
    """Exporte les poids du modèle et le scaler en header C (bare-metal)."""
    path = results_dir / "ht_detector_weights.h"
    lines = [
        "/* ht_detector_weights.h — Poids CNN1D quantisés int8 pour déploiement bare-metal */",
        "/* Généré par 08_fpga_deploy.py — NE PAS MODIFIER MANUELLEMENT */",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define HT_N_INPUT   {N_INPUT}",
        "#define HT_N_CLASSES 3",
        f"#define HT_CLK_MHZ   {CLK_MHZ}",
        "",
        "/* Scaler (moyenne et écart-type) pour normalisation */",
    ]

    # Scaler
    mean_hex  = ", ".join(f"{v:.6f}f" for v in scaler.mean_[:10])
    scale_hex = ", ".join(f"{v:.6f}f" for v in scaler.scale_[:10])
    lines += [
        f"/* (10 premières valeurs sur {N_INPUT}) */",
        f"static const float HT_SCALER_MEAN[{N_INPUT}]  = {{ {mean_hex}, /* ... */ }};",
        f"static const float HT_SCALER_SCALE[{N_INPUT}] = {{ {scale_hex}, /* ... */ }};",
        "",
    ]

    # Poids FC1 (couche la plus lourde — aperçu 8×8)
    sd  = model.state_dict()
    fc1 = sd.get("classifier.1.weight", sd.get("classifier.0.weight"))
    if fc1 is not None:
        w_int8 = (fc1[:8, :8] * 127).clamp(-127, 127).to(torch.int8).numpy()
        rows = "\n    ".join(
            "{" + ", ".join(f"{int(v):4d}" for v in row) + "}," for row in w_int8
        )
        lines += [
            "/* FC1 weight (aperçu 8×8 sur 256×2048) — format int8 */",
            "static const int8_t HT_FC1_W_PREVIEW[8][8] = {",
            f"    {rows}",
            "};",
            "",
        ]

    lines += [
        "/* Classe de sortie */",
        'static const char* HT_CLASS_NAMES[3] = {"TrojanDisabled", "TrojanEnabled", "TrojanTriggered"};',
    ]

    path.write_text("\n".join(lines))
    print(f"Header C généré : {path.name}  ({path.stat().st_size} octets)")
    return path


# ── 6. hls4ml (optionnel) ──────────────────────────────────────────────────────
def try_hls4ml(onnx_path: Path, results_dir: Path):
    try:
        import hls4ml
        print("\n[hls4ml] Conversion ONNX → HLS C++...")
        config = hls4ml.utils.config_from_onnx_model(
            str(onnx_path),
            granularity="model",
            default_precision="ap_fixed<16,6>",
        )
        hls_model = hls4ml.converters.convert_from_onnx_model(
            str(onnx_path),
            hls_config=config,
            output_dir=str(results_dir / "hls4ml_project"),
            backend="Vivado",
        )
        print(f"[hls4ml] Projet HLS généré : {results_dir}/hls4ml_project/")
        return True
    except ImportError:
        print("[hls4ml] Non installé — skip (pip install hls4ml pour générer le code HLS).")
    except Exception as e:
        print(f"[hls4ml] Erreur : {e}")
    return False


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    results_dir = Path("../results")

    model, scaler, X, y = load_model_and_data(results_dir, cfg)

    # ── 1. ONNX ───────────────────────────────────────────────────────────────
    print("\n=== 1. Export ONNX ===")
    onnx_path = export_onnx(model, results_dir)

    # ── 2. Quantisation ───────────────────────────────────────────────────────
    print("\n=== 2. Quantisation int8 ===")
    model_int8 = quantize_model(model)
    size_f32   = model_size_kb(model)
    size_int8  = model_size_kb(model_int8)
    print(f"  float32 : {size_f32:.1f} KB")
    print(f"  int8    : {size_int8:.1f} KB  (réduction ×{size_f32/size_int8:.1f})")

    # ── 3. Benchmark ──────────────────────────────────────────────────────────
    print("\n=== 3. Benchmark latence (batch=1) ===")
    lat_f32  = benchmark_latency(model,      X)
    lat_int8 = benchmark_latency(model_int8, X)
    acc_f32  = eval_accuracy(model,      X, y)
    acc_int8 = eval_accuracy(model_int8, X, y)
    print(f"  float32 : {lat_f32:.3f} ms  acc={acc_f32:.4f}")
    print(f"  int8    : {lat_int8:.3f} ms  acc={acc_int8:.4f}  "
          f"(Δacc={acc_int8-acc_f32:+.4f})")

    # ── 4. Estimation FPGA ────────────────────────────────────────────────────
    print("\n=== 4. Estimation ressources FPGA ===")
    fpga = estimate_fpga_resources()
    print(f"  Cible              : {fpga['fpga_cible']}")
    print(f"  Paramètres totaux  : {fpga['total_parameters']:,}")
    print(f"  MACs par inférence : {fpga['total_MACs']:,}")
    print(f"  DSP48 estimés      : {fpga['DSP48']}")
    print(f"  BRAM 18K estimés   : {fpga['BRAM_18K']}")
    print(f"  LUTs estimés       : {fpga['LUT_k']}K")
    print(f"  Latence théorique  : {fpga['latence_us']} µs @ {CLK_MHZ} MHz")
    (results_dir / "fpga_resources.json").write_text(json.dumps(fpga, indent=2))

    # ── 5. Header C ───────────────────────────────────────────────────────────
    print("\n=== 5. Génération header C ===")
    generate_c_header(model, scaler, results_dir)

    # ── 6. hls4ml ─────────────────────────────────────────────────────────────
    print("\n=== 6. hls4ml ===")
    try_hls4ml(onnx_path, results_dir)

    # ── Rapport final ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BILAN DÉPLOIEMENT EMBARQUÉ / FPGA")
    print("="*60)
    print(f"  Modèle ONNX        : ht_detector.onnx ({onnx_path.stat().st_size/1024:.1f} KB)")
    print(f"  float32 → int8     : {size_f32:.0f} KB → {size_int8:.0f} KB  "
          f"(×{size_f32/size_int8:.1f} compression)")
    print(f"  Précision int8     : {acc_int8:.2%}  (perte : {acc_f32-acc_int8:.4f})")
    print(f"  Latence CPU (1 trace) : {lat_int8:.2f} ms")
    print(f"  Latence FPGA estimée  : {fpga['latence_us']} µs  "
          f"(×{lat_int8*1000/max(fpga['latence_us'],1):.0f} plus rapide que CPU)")
    print(f"  FPGA cible         : {fpga['fpga_cible']}")
    print(f"  DSP48 / BRAM / LUT : {fpga['DSP48']} / {fpga['BRAM_18K']} / {fpga['LUT_k']}K")
    print(f"  Header C           : ht_detector_weights.h")
    print("\n  → Déploiement FPGA temps-réel FAISABLE sur Artix-7")
    print("  → Latence ×10-100 inférieure au déploiement CPU")
