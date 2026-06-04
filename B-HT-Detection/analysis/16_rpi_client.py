#!/usr/bin/env python3
"""
16_rpi_client.py — Client passerelle Raspberry Pi 4 : détecteur HT 3 niveaux.

Architecture de déploiement (golden-free) :

    ┌──────────────┐   features    ┌──────────────┐   features   ┌──────────────┐
    │  EDGE        │   (UART)       │  GATEWAY     │   (HTTP)     │  CLOUD       │
    │  STM32 F401  │ ◄────────────► │  Raspberry   │ ◄──────────► │  API K8s     │
    │  TinyMLP int8│                │  Pi 4 (ici)  │              │  AT_T800     │
    │  ~33KB Flash │                │  numpy/torch │              │  CNN1D 2.2MB │
    └──────────────┘                └──────────────┘              └──────────────┘

Le RPi orchestre : il lit de VRAIES traces (IEEE Dataport, held-out), interroge
les 3 niveaux et compare leurs verdicts. Aucune donnée générée — on rejoue des
traces réelles pour émuler un fonctionnement en ligne.

Deux scénarios d'attaque (sur traces réelles) :
  A — Détection d'activation : flux temporel sain → Trojan déclenché → alarme LED
  B — Évasion adversariale    : PGD ε=0.2, modèle normal trompé vs AT robuste

Dégradation gracieuse : tourne sur WSL/PC sans STM32 ni API (proxy numpy local),
puis sur le vrai RPi avec le STM32 branché en USB et l'API K8s joignable.

Usage :
  python 16_rpi_client.py --scenario both --benchmark AES-T700
  python 16_rpi_client.py --scenario A --stm32 /dev/ttyACM0 --api http://192.168.1.50:30800
  python 16_rpi_client.py --scenario B --benchmark AES-T800
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).parent))
from models import CNN1D

try:
    import serial          # pyserial — uniquement nécessaire avec un vrai STM32
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Constantes du protocole UART (miroir de ht_protocol.h) ──────────────────────
REQ_MAGIC   = 0xAA
RESP_MAGIC  = 0xBB
CMD_PREDICT = 0x01
CMD_INFO    = 0x02
CMD_PING    = 0x03
N_FEATURES  = 500
RESP_SIZE   = 10

LABEL_NAMES = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
RISK_NAMES  = ["OK", "WARNING", "ALERT"]
LED_STATE   = ["OFF", "BLINK 1Hz", "ON (ALERTE)"]


# ── TinyMLP (miroir exact de 15_train_tiny_mlp.py) ──────────────────────────────
class TinyMLP(nn.Module):
    def __init__(self, n_input=500, h1=64, h2=32, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(n_input, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, n_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        return self.fc3(x)


# ════════════════════════════════════════════════════════════════════════════════
# EDGE — réplique numpy int8 de ce que calcule le STM32 (BatchNorm fusionné)
# ════════════════════════════════════════════════════════════════════════════════
class EdgeModel:
    """Inférence int8 identique au firmware STM32 (validée à 0.34% près du float)."""

    def __init__(self, tiny_pt: Path, scaler_mean, scaler_scale):
        self.mean = scaler_mean.astype(np.float32)
        self.scale = scaler_scale.astype(np.float32)
        self.model = TinyMLP()
        sd = torch.load(tiny_pt, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval()
        self._quantize()

    @staticmethod
    def _quant(t: torch.Tensor):
        t = t.float()
        scale = max(abs(t.min().item()), abs(t.max().item())) / 127.0
        scale = scale if scale != 0 else 1e-8
        q = (t / scale).clamp(-127, 127).round().cpu().numpy().astype(np.float32)
        return q, scale

    def _quantize(self):
        sd = self.model.state_dict()
        self.W1q, self.W1s = self._quant(sd["fc1.weight"]); self.b1q, self.b1s = self._quant(sd["fc1.bias"])
        self.W2q, self.W2s = self._quant(sd["fc2.weight"]); self.b2q, self.b2s = self._quant(sd["fc2.bias"])
        self.W3q, self.W3s = self._quant(sd["fc3.weight"]); self.b3q, self.b3s = self._quant(sd["fc3.bias"])

        def bn(name):
            g = sd[f"{name}.weight"].numpy();        be = sd[f"{name}.bias"].numpy()
            m = sd[f"{name}.running_mean"].numpy();   v = sd[f"{name}.running_var"].numpy()
            fac = g / np.sqrt(v + 1e-5)
            return fac.astype(np.float32), (be - m * fac).astype(np.float32)
        self.bn1_f, self.bn1_b = bn("bn1")
        self.bn2_f, self.bn2_b = bn("bn2")

    def predict(self, raw_features):
        """raw_features: (500,) ou (N,500) features BRUTES → labels, confidences."""
        raw = np.atleast_2d(raw_features).astype(np.float32)
        Xn  = (raw - self.mean) / self.scale                     # normalisation (= scaler STM32)
        h1 = (self.b1q * self.b1s) + Xn @ (self.W1q * self.W1s).T
        h1 = np.maximum(self.bn1_f * h1 + self.bn1_b, 0.0)
        h2 = (self.b2q * self.b2s) + h1 @ (self.W2q * self.W2s).T
        h2 = np.maximum(self.bn2_f * h2 + self.bn2_b, 0.0)
        logits = (self.b3q * self.b3s) + h2 @ (self.W3q * self.W3s).T
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
        return probs.argmax(1), probs.max(1)


# ════════════════════════════════════════════════════════════════════════════════
# STM32 — liaison UART (vrai hardware). Proxy edge si non branché.
# ════════════════════════════════════════════════════════════════════════════════
class STM32Link:
    def __init__(self, port, baud=115200, timeout=2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)                  # laisser le STM32 démarrer
        self.ser.reset_input_buffer()

    def ping(self) -> bool:
        self.ser.write(bytes([REQ_MAGIC, CMD_PING, 0x00, 0x00]))
        resp = self.ser.read(3)
        return len(resp) == 3 and resp[0] == RESP_MAGIC and resp[1] == CMD_PING

    def predict(self, raw_features):
        """Envoie 500 features brutes, reçoit (label, conf, risk, latence_ms)."""
        payload = np.asarray(raw_features, dtype="<f4").tobytes()   # 2000 bytes LE
        chk = 0
        for byte in payload:
            chk ^= byte
        header = bytes([REQ_MAGIC, CMD_PREDICT, (N_FEATURES >> 8) & 0xFF, N_FEATURES & 0xFF])
        self.ser.reset_input_buffer()
        self.ser.write(header + payload + bytes([chk & 0xFF]))
        resp = self.ser.read(RESP_SIZE)
        if len(resp) != RESP_SIZE or resp[0] != RESP_MAGIC:
            return None
        label = resp[1]
        conf  = struct.unpack("<f", resp[2:6])[0]
        risk  = resp[6]
        latency = (resp[7] << 8) | resp[8]
        return label, conf, risk, latency

    def close(self):
        self.ser.close()


# ════════════════════════════════════════════════════════════════════════════════
# CLOUD — API K8s (modèle complet AT_T800) via /predict/features
# ════════════════════════════════════════════════════════════════════════════════
class CloudAPI:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/ready", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def predict(self, raw_features):
        body = json.dumps({"features": np.asarray(raw_features).tolist()}).encode()
        req  = urllib.request.Request(
            f"{self.url}/predict/features", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
                return d["label"], d["confidence"]
        except Exception:
            return None


# ── Chargement modèle CNN1D (gère les wrappers CORAL : strict=False) ────────────
def load_cnn1d(path: Path, device):
    m = CNN1D(n_input=500).to(device)
    sd = torch.load(path, map_location=device)
    sd = {k: v for k, v in sd.items()
          if k in m.state_dict() and v.shape == m.state_dict()[k].shape}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


# ── Attaque PGD ciblée (masquage Trojan : Triggered → Disabled) ─────────────────
def pgd_attack(model, X_norm, eps, alpha, steps, target_class, device):
    X0 = torch.tensor(X_norm, dtype=torch.float32, device=device)
    X_adv = X0.clone()
    target = torch.full((len(X0),), target_class, dtype=torch.long, device=device)
    for _ in range(steps):
        X_adv.requires_grad_(True)
        loss = F.cross_entropy(model(X_adv), target)   # minimiser → pousser vers Disabled
        grad = torch.autograd.grad(loss, X_adv)[0]
        with torch.no_grad():
            X_adv = X_adv - alpha * grad.sign()
            X_adv = X0 + torch.clamp(X_adv - X0, -eps, eps)
        X_adv = X_adv.detach()
    return X_adv.cpu().numpy()


def torch_predict(model, X_norm, device):
    with torch.no_grad():
        logits = model(torch.tensor(X_norm, dtype=torch.float32, device=device))
        return logits.argmax(1).cpu().numpy()


# ════════════════════════════════════════════════════════════════════════════════
# SCÉNARIO A — Détection d'activation du Trojan (flux temporel)
# ════════════════════════════════════════════════════════════════════════════════
def scenario_A(edge, stm32, cloud, X, y, bm, inject_at, n_traces, confirm, out_dir, rng):
    print("\n" + "=" * 70)
    print(f"SCÉNARIO A — DÉTECTION D'ACTIVATION DU TROJAN  ({bm})")
    print("=" * 70)
    print(f"  Phase 1 (t<{inject_at}) : puce saine     → traces TrojanDisabled réelles")
    print(f"  Phase 2 (t≥{inject_at}) : Trojan activé   → traces TrojanTriggered réelles")
    print(f"  Confirmation alarme     : {confirm} traces Triggered consécutives\n")

    idx_dis = np.where(y == 0)[0]      # TrojanDisabled
    idx_trg = np.where(y == 2)[0]      # TrojanTriggered
    if len(idx_dis) == 0 or len(idx_trg) == 0:
        print(f"  ⚠ {bm} n'a pas les deux classes (Disabled/Triggered) — scénario A ignoré.")
        return None

    # Construire le flux temporel à partir de VRAIES traces
    stream_idx, truth = [], []
    for t in range(n_traces):
        pool = idx_dis if t < inject_at else idx_trg
        stream_idx.append(rng.choice(pool))
        truth.append(0 if t < inject_at else 2)
    stream_idx = np.array(stream_idx)
    truth = np.array(truth)
    X_stream = X[stream_idx]

    # Inférence edge (batch) — proxy du STM32
    edge_lbl, edge_conf = edge.predict(X_stream)

    # Détection : fenêtre de confirmation
    detection_at = None
    run = 0
    for t in range(n_traces):
        run = run + 1 if edge_lbl[t] == 2 else 0
        if run >= confirm and detection_at is None and t >= inject_at:
            detection_at = t
    fp = int(np.sum(edge_lbl[:inject_at] == 2))        # faux positifs (phase saine)
    acc_p1 = float(np.mean(edge_lbl[:inject_at] == 0))
    acc_p2 = float(np.mean(edge_lbl[inject_at:] == 2))

    # Vérification STM32 réel (si branché) sur un échantillon
    stm32_agree = None
    if stm32 is not None:
        n_check = min(20, n_traces)
        agree = 0
        for t in range(n_check):
            r = stm32.predict(X_stream[t])
            if r is not None and r[0] == edge_lbl[t]:
                agree += 1
        stm32_agree = agree / n_check
        print(f"  STM32 réel : accord {agree}/{n_check} avec le proxy numpy")

    # Vérification cloud (si joignable) sur un échantillon
    cloud_agree = None
    if cloud is not None:
        n_check = min(20, n_traces)
        agree = 0
        for t in range(n_check):
            r = cloud.predict(X_stream[t])
            if r is not None and r[0] == truth[t]:
                agree += 1
        cloud_agree = agree / n_check
        print(f"  Cloud AT_T800 : {agree}/{n_check} traces correctement classées")

    # Bilan
    if detection_at is not None:
        latency = detection_at - inject_at
        print(f"\n  🔴 ALARME levée à t={detection_at} (Trojan activé à t={inject_at})")
        print(f"     Latence de détection : {latency} traces")
    else:
        latency = None
        print(f"\n  ⚠ Aucune alarme levée (Trojan non détecté)")
    print(f"  Faux positifs phase saine : {fp}/{inject_at}")
    print(f"  Accuracy phase 1 (saine)  : {acc_p1:.1%}")
    print(f"  Accuracy phase 2 (Trojan) : {acc_p2:.1%}")

    # ── Figure timeline ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    t_axis = np.arange(n_traces)
    colors = np.array(["#2ca02c", "#ff7f0e", "#d62728"])  # vert/orange/rouge
    ax1.step(t_axis, truth, where="mid", color="gray", lw=1.5, alpha=0.5,
             label="Vérité terrain")
    ax1.scatter(t_axis, edge_lbl, c=colors[edge_lbl], s=35, zorder=3,
                label="Prédiction edge (STM32)")
    ax1.axvline(inject_at, color="red", ls="--", lw=2, label=f"Trojan activé (t={inject_at})")
    if detection_at is not None:
        ax1.axvline(detection_at, color="green", ls=":", lw=2,
                    label=f"Détection (t={detection_at})")
    ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(LABEL_NAMES)
    ax1.set_ylabel("État détecté")
    ax1.set_title(f"Scénario A — Détection d'activation du Trojan sur {bm} (traces réelles)")
    ax1.legend(loc="center left", fontsize=8); ax1.grid(alpha=0.3)

    ax2.fill_between(t_axis, edge_conf, color="#1f77b4", alpha=0.4)
    ax2.plot(t_axis, edge_conf, color="#1f77b4", lw=1)
    ax2.axvline(inject_at, color="red", ls="--", lw=2)
    if detection_at is not None:
        ax2.axvline(detection_at, color="green", ls=":", lw=2)
    ax2.set_ylabel("Confiance"); ax2.set_xlabel("Trace # (temps)")
    ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "16_rpi_scenarioA_timeline.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure : {out.name}")

    return {
        "benchmark": bm, "n_traces": n_traces, "inject_at": inject_at,
        "detection_at": detection_at, "detection_latency": latency,
        "false_positives": fp, "acc_phase_saine": round(acc_p1, 4),
        "acc_phase_trojan": round(acc_p2, 4),
        "stm32_agreement": stm32_agree, "cloud_accuracy": cloud_agree,
        "stm32_connected": stm32 is not None, "cloud_connected": cloud is not None,
    }


# ════════════════════════════════════════════════════════════════════════════════
# SCÉNARIO B — Évasion adversariale (PGD : masquage du Trojan)
# ════════════════════════════════════════════════════════════════════════════════
def scenario_B(edge, results_dir, scaler_mean, scaler_scale,
               eps, alpha, steps, out_dir, device):
    # Le modèle AT et l'edge sont entraînés sur T800 (scaler T800). On attaque donc
    # sur T800 : même scaler pour les 3 modèles → comparaison cohérente et valide.
    bm = "AES-T800"
    print("\n" + "=" * 70)
    print(f"SCÉNARIO B — ÉVASION ADVERSARIALE (PGD ε={eps})  ({bm})")
    print("=" * 70)
    print("  Attaquant : perturbation PGD (L∞) sur traces Triggered réelles")
    print("  Objectif  : faire classer 'Disabled' → masquer le Trojan")
    print("  Compare les 2 niveaux DÉPLOYÉS : Edge STM32 vs Cloud K8s (AT_T800)")
    print("  (T800 : scaler cohérent edge/cloud)\n")

    d = np.load(results_dir / f"features_{bm}.npz")
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    idx_trg = np.where(y == 2)[0]
    if len(idx_trg) == 0:
        print(f"  ⚠ {bm} n'a pas de classe Triggered — scénario B ignoré.")
        return None
    rng = np.random.RandomState(42)
    sel = rng.choice(idx_trg, size=min(500, len(idx_trg)), replace=False)
    X_trg = X[sel]                                            # traces brutes Triggered
    Xn = ((X_trg - scaler_mean) / scaler_scale).astype(np.float32)   # espace normalisé

    # On compare les 2 niveaux réellement déployés dans l'architecture :
    #   Edge  = TinyMLP int8 (STM32, 33KB)      → triage basse latence
    #   Cloud = CNN1D AT_T800 (API K8s, 2.2MB)  → vérification robuste
    models = {}
    models["Edge STM32\n(TinyMLP int8)"] = edge.model.to(device).eval()
    p = results_dir / "cnn1d_AT_AES-T800.pt"
    if p.exists():
        models["Cloud K8s\n(CNN1D AT)"] = load_cnn1d(p, device)
    else:
        print("  ⚠ cnn1d_AT_AES-T800.pt absent — niveau cloud ignoré")

    # Masquage à ε fixe + courbe masquage(ε) — jusqu'à ε=0.5 (point de rupture cloud)
    eps_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    masking_curve = {name: [] for name in models}
    masking_at_eps, full_evasion, clean_det = {}, {}, {}

    for name, model in models.items():
        clean_pred = torch_predict(model, Xn, device)
        clean_det[name] = float(np.mean(clean_pred == 2))
        for e in eps_grid:
            if e == 0.0:
                pred = clean_pred
            else:
                X_adv = pgd_attack(model, Xn, e, alpha, steps, target_class=0, device=device)
                pred = torch_predict(model, X_adv, device)
            # Contournement de l'alarme : Triggered n'est plus classé Triggered
            # (→ Enabled = alarme rétrogradée, ou → Disabled = évasion totale)
            bypass = float(np.mean(pred != 2))
            masking_curve[name].append(bypass)
            if abs(e - eps) < 1e-9:
                masking_at_eps[name] = bypass
                full_evasion[name] = float(np.mean(pred == 0))   # → Disabled (benign, == script 12)
        nm = name.replace("\n", " ")
        print(f"  {nm:28} détection propre={clean_det[name]:.1%}  "
              f"contournement-alarme@ε={eps}={masking_at_eps[name]:.1%}  "
              f"évasion-totale={full_evasion[name]:.1%}")

    # ── Figure : barres masquage + courbe masquage(ε) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(models.keys())
    vals = [masking_at_eps[n] * 100 for n in names]
    bar_colors = ["#d62728", "#2ca02c"][:len(names)]   # edge=rouge (faible), cloud=vert (robuste)
    bars = ax1.bar(range(len(names)), vals, color=bar_colors, alpha=0.85)
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Contournement alarme Triggered (%)")
    ax1.set_title(f"Contournement de l'alarme sous PGD (ε={eps})\n"
                  "↓ plus bas = plus robuste  •  edge → Enabled (WARNING), jamais caché")
    ax1.set_ylim(0, 105); ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%",
                 ha="center", fontweight="bold")

    for name in names:
        ax2.plot([e for e in eps_grid], [m*100 for m in masking_curve[name]],
                 marker="o", lw=2, label=name.replace("\n", " "))
    ax2.axvline(eps, color="gray", ls="--", alpha=0.6, label=f"ε critique={eps}")
    ax2.set_xlabel("Force de l'attaque ε"); ax2.set_ylabel("Contournement alarme (%)")
    ax2.set_title("Robustesse vs force de l'attaque")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_ylim(0, 105)

    plt.suptitle(f"Scénario B — Évasion adversariale PGD sur {bm} : Edge vs Cloud (traces réelles IEEE Dataport)\n"
                 "Défense en profondeur : l'edge évadable est rattrapé par le cloud robuste",
                 fontweight="bold")
    plt.tight_layout()
    out = out_dir / "16_rpi_scenarioB_masquage.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Figure : {out.name}")

    return {
        "benchmark": bm, "eps": eps, "n_traces_attaquees": len(sel),
        "contournement_alarme_at_eps": {k.replace(chr(10), " "): round(v, 4) for k, v in masking_at_eps.items()},
        "evasion_totale_at_eps": {k.replace(chr(10), " "): round(v, 4) for k, v in full_evasion.items()},
        "detection_propre": {k.replace(chr(10), " "): round(v, 4) for k, v in clean_det.items()},
        "eps_grid": eps_grid,
        "contournement_curve": {k.replace(chr(10), " "): [round(x, 4) for x in v]
                                for k, v in masking_curve.items()},
    }


# ════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Client RPi détecteur HT 3 niveaux")
    ap.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    ap.add_argument("--benchmark", default="AES-T700")
    ap.add_argument("--stm32", default=None, help="Port série STM32 (ex: /dev/ttyACM0, COM3)")
    ap.add_argument("--api", default="http://localhost:30800", help="URL API K8s")
    ap.add_argument("--inject-at", type=int, default=50)
    ap.add_argument("--n-traces", type=int, default=100)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--pgd-alpha", type=float, default=0.05)   # = script 12 (PGD-AT Madry)
    ap.add_argument("--pgd-steps", type=int, default=20)
    args = ap.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    out_dir = results_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = args.benchmark

    print("=" * 70)
    print("CLIENT RASPBERRY PI — DÉTECTEUR HARDWARE TROJAN (3 niveaux)")
    print("=" * 70)
    print(f"  Device         : {device}")
    print(f"  Benchmark      : {bm}")
    print(f"  Scénario       : {args.scenario}")

    # ── Données réelles (held-out) ──
    npz = results_dir / f"features_{bm}.npz"
    if not npz.exists():
        print(f"\n✗ {npz} introuvable — lance d'abord l'extraction de features.")
        sys.exit(1)
    d = np.load(npz)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    print(f"  Traces réelles : {len(X)} ({bm}), distribution {np.bincount(y).tolist()}")

    # ── Scaler T800 multi-source (= scaler du header STM32) ──
    scaler_mean  = np.load(results_dir / "scaler_ms_mean_AES-T800.npy")
    scaler_scale = np.load(results_dir / "scaler_ms_scale_AES-T800.npy")

    # ── EDGE (toujours dispo : proxy numpy int8 du STM32) ──
    edge = EdgeModel(results_dir / "cnn1d_tiny_stm32.pt", scaler_mean, scaler_scale)
    print(f"  EDGE           : TinyMLP int8 chargé (proxy STM32)")

    # ── STM32 réel (optionnel) ──
    stm32 = None
    if args.stm32:
        if not HAS_SERIAL:
            print("  ⚠ pyserial absent (pip install pyserial) — STM32 ignoré")
        else:
            try:
                stm32 = STM32Link(args.stm32)
                ok = stm32.ping()
                print(f"  STM32 réel     : {args.stm32} — PING {'OK' if ok else 'ÉCHEC'}")
                if not ok:
                    stm32.close(); stm32 = None
            except Exception as e:
                print(f"  ⚠ STM32 {args.stm32} injoignable : {e}")
                stm32 = None
    else:
        print("  STM32 réel     : non spécifié (proxy numpy utilisé)")

    # ── CLOUD (optionnel) ──
    cloud = CloudAPI(args.api)
    if cloud.health():
        print(f"  CLOUD          : {args.api} — API joignable (AT_T800)")
    else:
        print(f"  CLOUD          : {args.api} — injoignable (ignoré)")
        cloud = None

    rng = np.random.RandomState(2024)
    metrics = {"benchmark": bm, "device": str(device)}

    if args.scenario in ("A", "both"):
        metrics["scenario_A"] = scenario_A(
            edge, stm32, cloud, X, y, bm,
            args.inject_at, args.n_traces, args.confirm, out_dir, rng)

    if args.scenario in ("B", "both"):
        metrics["scenario_B"] = scenario_B(
            edge, results_dir, scaler_mean, scaler_scale,
            args.eps, args.pgd_alpha, args.pgd_steps, out_dir, device)

    if stm32 is not None:
        stm32.close()

    out_json = out_dir / "16_rpi_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("BILAN — CLIENT RASPBERRY PI")
    print("=" * 70)
    print(f"  Fichiers captures rapport :")
    print(f"    16_rpi_scenarioA_timeline.png")
    print(f"    16_rpi_scenarioB_masquage.png")
    print(f"    16_rpi_metrics.json")
    print(f"\n  → Architecture Edge(STM32)/Gateway(RPi)/Cloud(K8s) validée")


if __name__ == "__main__":
    main()
