# Dashboard — Détection de Hardware Trojan (golden-free) + MLOps

Application **Streamlit** présentant l'intégralité du projet B et permettant de
**tester le détecteur en direct** sur de vraies traces.

## Pages

| Page | Contenu |
|------|---------|
| 🏠 **Accueil** | Contexte, architecture 3-niveaux (Edge/Gateway/Cloud), chiffres clés |
| 🔬 **Démo Live** | Rejoue une trace held-out réelle → verdict **edge int8** (réplique STM32) + **cloud** (API K8s) + **attaque PGD** interactive |
| 📊 **Résultats ML** | Baselines, CORAL, Adversarial Training, SimCLR, MC-Dropout, TinyMLP, scénarios RPi |
| ⚙️ **MLOps & Infra** | MLflow, DVC, Jenkins, Kubernetes, API, monitoring, modèle de menace |

Toutes les figures et métriques sont lues depuis `../results/` — **vrais artefacts**,
aucune donnée générée.

## Lancer en local

```bash
cd B-HT-Detection
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Ouvre http://localhost:8501

### Démo Live — connexion au cloud (optionnel)
La page Démo Live interroge l'API cloud si elle est joignable. Pour l'activer :

```bash
# soit l'API locale
uvicorn api.serve:app --port 30800
# soit le service Kubernetes
kubectl port-forward svc/ht-api 30800:80
```

Sans API, l'**edge int8** fonctionne seul (aucune dépendance réseau).

## Dépendances de données (Démo Live)
- `results/cnn1d_tiny_stm32.pt` — modèle edge (TinyMLP)
- `results/scaler_ms_mean_AES-T800.npy`, `..._scale_...` — normalisation
- `results/features_<benchmark>.npz` — traces réelles held-out (X, y)

Les pages Résultats ML et MLOps ne nécessitent que les figures `results/*.png`
et les métriques `results/*.json`.
