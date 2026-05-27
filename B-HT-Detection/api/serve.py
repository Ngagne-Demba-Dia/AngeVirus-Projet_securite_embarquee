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
    version="1.0.0",
)

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

    weights = MODEL_DIR / f"cnn1d_{SRC_BM}.pt"
    mean_f  = MODEL_DIR / f"scaler_mean_{SRC_BM}.npy"
    scale_f = MODEL_DIR / f"scaler_scale_{SRC_BM}.npy"

    if not weights.exists():
        log.error(f"Modèle non trouvé : {weights}")
        return

    model = CNN1D()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    state["model"] = model

    scaler = StandardScaler()
    scaler.mean_  = np.load(mean_f)
    scaler.scale_ = np.load(scale_f)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    state["scaler"] = scaler

    state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info(f"Modèle chargé : {weights}  (source: {SRC_BM})")


# ── Schemas Pydantic ───────────────────────────────────────────────────────────
class TraceRequest(BaseModel):
    samples: list[float] = Field(..., min_length=100, max_length=10000,
                                  description="Trace de puissance (2500 échantillons)")
    benchmark: Optional[str] = Field(None, description="Benchmark d'origine (informatif)")

class BatchRequest(BaseModel):
    traces: list[list[float]] = Field(..., max_length=256,
                                       description="Lot de traces (max 256)")

class PredictResponse(BaseModel):
    label:      int
    state:      str          # TrojanDisabled / TrojanEnabled / TrojanTriggered
    risk:       str          # OK / WARNING / ALERT
    confidence: float
    latency_ms: float


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


# ── Lancement local ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
