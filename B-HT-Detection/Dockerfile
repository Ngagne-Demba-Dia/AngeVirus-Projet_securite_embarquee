# ── Stage 1 : build (dépendances lourdes) ─────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2 : image finale (pipeline ML) ──────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Ngagne Demba Dia"
LABEL description="HT-Detection ML pipeline — training stage"

WORKDIR /app

# Copier les paquets installés
COPY --from=builder /install /usr/local

# Copier le code source
COPY analysis/ ./analysis/
COPY configs/  ./configs/

# Volumes :
#   /data    → dataset hardware_trojan (monté en lecture seule depuis l'hôte)
#   /app/results → outputs (features, modèles, plots)
VOLUME ["/data", "/app/results"]

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Point d'entrée : on passe le script en argument
# docker run ht-pipeline python analysis/01_indexer.py
CMD ["python", "--version"]
