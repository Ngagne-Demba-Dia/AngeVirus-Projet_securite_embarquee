"""
api/serve.py — FastAPI inference server (golden-free HT detection).
Endpoints :
  GET  /health         → liveness probe (K8s)
  GET  /ready          → readiness probe (modèle chargé ?)
  GET  /model/info     → infos du modèle en production
  POST /predict        → classifier une trace (2500 floats)
  POST /predict/batch  → classifier plusieurs traces en lot
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

# Ajouter le chemin des utilitaires partagés
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.models import CNN1D
from analysis.feat_utils import extract_features

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ht-api")

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "/models"))
SRC_BM     = os.getenv("SOURCE_BENCHMARK", "AES-T400")
CONF_FILE  = Path(os.getenv("CONFIG_PATH", "configs/config.yaml"))

LABEL_NAMES = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
LABEL_RISK  = {"TrojanDisabled": "OK", "TrojanEnabled": "WARNING", "TrojanTriggered": "ALERT"}

app = FastAPI(
    title="HT-Detection API",
    description="Golden-free Hardware Trojan detection via power side-channel ML",
    version="1.1.0",
)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# ── État global ────────────────────────────────────────────────────────────────
state = {
    "model":     None,
    "scaler":    None,
    "cfg":       None,
    "loaded_at": None,
    "source_bm": SRC_BM,
    "n_predict": 0,
}


# ── Chargement du modèle au démarrage ─────────────────────────────────────────
@app.on_event("startup")
def load_model():
    log.info("Chargement du modèle CNN1D...")

    try:
        with open(CONF_FILE) as f:
            state["cfg"] = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning(f"Config non trouvée ({CONF_FILE}) — utilisation des defaults")
        state["cfg"] = {"features": {"window_size": 100, "n_fft": 10}}

    # Priorité : AT_T800 > CORAL T800 > CORAL T700 > fine-tuned T700 > source T400
    candidates = [
        ("cnn1d_AT_AES-T800.pt",    "scaler_ms_mean_AES-T800.npy",  "scaler_ms_scale_AES-T800.npy",  "AT_T800"),
        ("cnn1d_coral_AES-T800.pt", "scaler_ms_mean_AES-T800.npy",  "scaler_ms_scale_AES-T800.npy",  "CORAL_T800"),
        ("cnn1d_coral_AES-T700.pt", "scaler_ms_mean_AES-T700.npy",  "scaler_ms_scale_AES-T700.npy",  "CORAL_T700"),
        ("cnn1d_ft_AES-T700.pt",    f"scaler_mean_{SRC_BM}.npy",    f"scaler_scale_{SRC_BM}.npy",    "FT_T700"),
        (f"cnn1d_{SRC_BM}.pt",      f"scaler_mean_{SRC_BM}.npy",    f"scaler_scale_{SRC_BM}.npy",    SRC_BM),
    ]

    weights, mean_f, scale_f, model_id = None, None, None, None
    for w, m, s, mid in candidates:
        if (MODEL_DIR / w).exists() and (MODEL_DIR / m).exists():
            weights, mean_f, scale_f, model_id = MODEL_DIR/w, MODEL_DIR/m, MODEL_DIR/s, mid
            break

    if weights is None:
        log.error("Aucun modèle trouvé dans MODEL_DIR")
        return

    model = CNN1D()
    # Filtrer les clés incompatibles (ex: _flatten du wrapper CORAL)
    sd = {k: v for k, v in torch.load(weights, map_location="cpu").items()
          if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    state["model"]     = model
    state["source_bm"] = model_id

    scaler = StandardScaler()
    scaler.mean_  = np.load(mean_f)
    scaler.scale_ = np.load(scale_f)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    state["scaler"] = scaler

    state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info(f"Modèle chargé : {weights.name}  (id: {model_id})")


# ── Schemas Pydantic ───────────────────────────────────────────────────────────
class TraceRequest(BaseModel):
    samples: list[float] = Field(..., min_length=100, max_length=10000,
                                  description="Trace de puissance (2500 échantillons)")
    benchmark: Optional[str] = Field(None, description="Benchmark d'origine (informatif)")

class BatchRequest(BaseModel):
    traces: list[list[float]] = Field(..., max_length=256,
                                       description="Lot de traces (max 256)")

class FeatureRequest(BaseModel):
    features: list[float] = Field(..., min_length=100, max_length=2000,
                                   description="Features déjà extraites (500), envoyées par l'edge")
    benchmark: Optional[str] = Field(None, description="Benchmark d'origine (informatif)")

class PredictResponse(BaseModel):
    label:      int
    state:      str          # TrojanDisabled / TrojanEnabled / TrojanTriggered
    risk:       str          # OK / WARNING / ALERT
    confidence: float
    latency_ms: float

class FeatureContrib(BaseModel):
    rank:       int
    name:       str
    importance: float
    value:      float

class ExplainResponse(BaseModel):
    label:        int
    state:        str
    risk:         str
    confidence:   float
    top_features: list[FeatureContrib]
    latency_ms:   float

class UncertainResponse(BaseModel):
    label:       int
    state:       str
    risk:        str
    confidence:  float
    epistemic:   float   # variance MC Dropout — incertitude épistémique
    entropy:     float   # entropie prédictive
    is_uncertain: bool   # True si incertitude > seuil
    decision:    str     # "PREDICT" ou "UNCERTAIN"
    n_passes:    int
    latency_ms:  float

# Seuil MC Dropout calibré sur T700 in-domain (75e percentile)
MC_UNCERTAINTY_THRESHOLD = 0.00046
MC_N_PASSES = 20   # 20 passes en production (équilibre latence/précision)


# ── Noms des 325 features (25 fenêtres × 13 : mean, std, energy, fft×10) ─────
def _feature_names(window_size: int = 100, n_fft: int = 10) -> list[str]:
    n_windows = 2500 // window_size
    names = []
    for w in range(n_windows):
        names += [f"w{w}_mean", f"w{w}_std", f"w{w}_energy"]
        names += [f"w{w}_fft{k}" for k in range(n_fft)]
    return names


# ── Fonction d'inference ───────────────────────────────────────────────────────
def _infer(samples: list[float]) -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    cfg = state["cfg"]["features"]
    trace = np.array(samples, dtype=np.float32)
    feat  = extract_features(trace, window=cfg["window_size"], n_fft=cfg["n_fft"])

    X = state["scaler"].transform(feat.reshape(1, -1))
    X_t = torch.FloatTensor(X)

    with torch.no_grad():
        logits = state["model"](X_t)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    label = int(probs.argmax())
    state["n_predict"] += 1

    return {
        "label":      label,
        "state":      LABEL_NAMES[label],
        "risk":       LABEL_RISK[LABEL_NAMES[label]],
        "confidence": float(probs[label]),
    }


def _infer_features(features: list[float]) -> dict:
    """Inference sur features DÉJÀ extraites (envoyées par l'edge STM32/RPi).
    Évite de transmettre la trace brute (2500 pts) — l'edge envoie 500 features."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    feat = np.array(features, dtype=np.float32)
    X = state["scaler"].transform(feat.reshape(1, -1))
    X_t = torch.FloatTensor(X)

    with torch.no_grad():
        logits = state["model"](X_t)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    label = int(probs.argmax())
    state["n_predict"] += 1

    return {
        "label":      label,
        "state":      LABEL_NAMES[label],
        "risk":       LABEL_RISK[LABEL_NAMES[label]],
        "confidence": float(probs[label]),
    }


# ── Attribution par gradient (saliency map sur le CNN) ────────────────────────
def _gradient_explain(samples: list[float], top_n: int = 10) -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    cfg = state["cfg"]["features"]
    trace = np.array(samples, dtype=np.float32)
    feat = extract_features(trace, window=cfg["window_size"], n_fft=cfg["n_fft"])

    X = state["scaler"].transform(feat.reshape(1, -1))
    X_t = torch.FloatTensor(X).requires_grad_(True)

    state["model"].zero_grad()
    with torch.enable_grad():
        logits = state["model"](X_t)
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        label = int(probs.argmax())
        logits[0, label].backward()

    importance = X_t.grad[0].abs().detach().numpy()
    feat_names = _feature_names(cfg["window_size"], cfg["n_fft"])

    top_idx = importance.argsort()[::-1][:top_n]
    top_features = [
        {
            "rank": int(r + 1),
            "name": feat_names[i],
            "importance": round(float(importance[i]), 6),
            "value": round(float(feat[i]), 4),
        }
        for r, i in enumerate(top_idx)
    ]

    return {
        "label":        label,
        "state":        LABEL_NAMES[label],
        "risk":         LABEL_RISK[LABEL_NAMES[label]],
        "confidence":   round(float(probs[label]), 4),
        "top_features": top_features,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", status_code=200, tags=["Probes"])
def health():
    """Liveness probe — répond toujours 200 si le service est up."""
    return {"status": "up"}


@app.get("/ready", tags=["Probes"])
def ready():
    """Readiness probe — 200 si le modèle est chargé, 503 sinon."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non prêt")
    return {"status": "ready"}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Informations sur le modèle en production."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "source_benchmark": state["source_bm"],
        "architecture":     "CNN1D (325 features → 3 classes)",
        "classes":          LABEL_NAMES,
        "loaded_at":        state["loaded_at"],
        "n_predictions":    state["n_predict"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: TraceRequest):
    """
    Classifier une trace de puissance AES.
    - Input  : 2500 échantillons float (1 chiffrement AES)
    - Output : état du Trojan + confiance
    """
    t0 = time.perf_counter()
    result = _infer(req.samples)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if result["risk"] == "ALERT":
        log.warning(f"ALERTE HT détecté! conf={result['confidence']:.3f}")

    return result


@app.post("/predict/features", response_model=PredictResponse, tags=["Inference"])
def predict_features(req: FeatureRequest):
    """
    Classifier à partir de features DÉJÀ extraites (flux edge → cloud).
    - Input  : 500 features (extraites sur le RPi/STM32)
    - Output : verdict du modèle cloud AT_T800
    Permet de comparer le verdict edge (Tiny MLP) au verdict cloud (modèle complet).
    """
    t0 = time.perf_counter()
    result = _infer_features(req.features)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if result["risk"] == "ALERT":
        log.warning(f"ALERTE HT détecté (features)! conf={result['confidence']:.3f}")

    return result


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(req: BatchRequest):
    """
    Classifier un lot de traces. Retourne la distribution des états
    et un score d'alerte global (fraction de traces Triggered).
    """
    t0 = time.perf_counter()
    results = [_infer(trace) for trace in req.traces]

    labels  = [r["label"] for r in results]
    counts  = {LABEL_NAMES[i]: labels.count(i) for i in range(3)}
    alert_frac = counts["TrojanTriggered"] / len(results)

    return {
        "n_traces":      len(results),
        "distribution":  counts,
        "alert_fraction": round(alert_frac, 4),
        "global_risk":   "ALERT" if alert_frac > 0.3 else ("WARNING" if alert_frac > 0.1 else "OK"),
        "predictions":   results,
        "latency_ms":    round((time.perf_counter() - t0) * 1000, 2),
    }


def _mc_infer(samples: list[float], n_passes: int) -> dict:
    """Monte Carlo Dropout : N passes forward avec Dropout actif."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    cfg   = state["cfg"]["features"]
    trace = np.array(samples, dtype=np.float32)
    feat  = extract_features(trace, window=cfg["window_size"], n_fft=cfg["n_fft"])
    X     = state["scaler"].transform(feat.reshape(1, -1))
    X_t   = torch.FloatTensor(X)

    # Activer Dropout en inférence
    model = state["model"]
    model.eval()
    for module in model.modules():
        if hasattr(module, 'p') and hasattr(module, 'training'):
            if 'Dropout' in type(module).__name__:
                module.train()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            probs = torch.softmax(model(X_t), dim=1).numpy()[0]
            all_probs.append(probs)

    model.eval()   # remettre en eval complet
    all_probs  = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    epistemic  = float(all_probs.var(axis=0).mean())
    entropy    = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
    label      = int(mean_probs.argmax())
    state["n_predict"] += 1

    return {
        "label":       label,
        "state":       LABEL_NAMES[label],
        "risk":        LABEL_RISK[LABEL_NAMES[label]],
        "confidence":  round(float(mean_probs[label]), 4),
        "epistemic":   round(epistemic, 6),
        "entropy":     round(entropy, 4),
        "is_uncertain": epistemic > MC_UNCERTAINTY_THRESHOLD,
        "decision":    "UNCERTAIN" if epistemic > MC_UNCERTAINTY_THRESHOLD else "PREDICT",
    }


@app.post("/predict/uncertain", response_model=UncertainResponse, tags=["Inference"])
def predict_uncertain(req: TraceRequest):
    """
    Prédiction avec quantification d'incertitude (Monte Carlo Dropout).
    Si is_uncertain=True, la prédiction est peu fiable — domaine probablement inconnu.
    """
    t0 = time.perf_counter()
    result = _mc_infer(req.samples, MC_N_PASSES)
    result["n_passes"]   = MC_N_PASSES
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if result["is_uncertain"]:
        log.warning(f"UNCERTAIN détecté! epistemic={result['epistemic']:.6f}")
    return result


@app.post("/explain", response_model=ExplainResponse, tags=["Inference"])
def explain(req: TraceRequest):
    """
    Classifier une trace et expliquer la décision par attribution de gradient.
    Retourne les 10 features ayant le plus influencé la prédiction du CNN.
    """
    t0 = time.perf_counter()
    result = _gradient_explain(req.samples)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


# ── Lancement local ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
