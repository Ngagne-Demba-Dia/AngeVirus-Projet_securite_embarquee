#!/usr/bin/env python3
"""
app.py — Dashboard Streamlit : Détection de Hardware Trojan (golden-free) + MLOps.

Présente l'intégralité du projet B : du problème (dérive de domaine) jusqu'au
déploiement edge (STM32) / gateway (RPi) / cloud (K8s), avec une démo live qui
interroge le VRAI modèle (TinyMLP int8 local + API cloud) sur des traces réelles.

4 pages :
  🏠 Accueil        — contexte, architecture 3-niveaux, chiffres clés
  🔬 Démo Live      — tester le détecteur en direct (edge int8 + cloud + attaque PGD)
  📊 Résultats ML   — baselines, CORAL, AT, SimCLR, MC-Dropout, TinyMLP, scénarios RPi
  ⚙️  MLOps & Infra  — MLflow, DVC, Jenkins, Kubernetes, API, monitoring, sécurité

Lancer :  streamlit run dashboard/app.py
Données  : lit B-HT-Detection/results/ (figures, métriques, modèles — VRAIS artefacts)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import streamlit as st

# ── Chemins ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent      # B-HT-Detection/
RESULTS  = ROOT / "results"
ANALYSIS = ROOT / "analysis"

st.set_page_config(page_title="HT Detection — Golden-Free + MLOps",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

LABELS = ["TrojanDisabled", "TrojanEnabled", "TrojanTriggered"]
DESCR  = ["puce saine — aucun Trojan actif",
          "Trojan présent mais latent (overhead structurel)",
          "Trojan DÉCLENCHÉ — charge utile active"]
LED    = ["🟢", "🟡", "🔴"]
RISK   = ["OK", "WARNING", "ALERT"]
BENCHMARKS = ["AES-T400", "AES-T500", "AES-T600", "AES-T700", "AES-T800", "AES-T1100"]


# ── Helpers chargement ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def img(name, **kw):
    """Affiche une figure si elle existe, sinon une note discrète."""
    p = RESULTS / name
    if p.exists():
        st.image(str(p), use_container_width=True, **kw)
    else:
        st.caption(f"_(figure manquante : {name})_")


def img_grid(items, cols=2):
    """items : liste de (fichier, légende)."""
    columns = st.columns(cols)
    for i, (name, cap) in enumerate(items):
        with columns[i % cols]:
            img(name, caption=cap)


@st.cache_data(show_spinner=False)
def load_traces(bm):
    p = RESULTS / f"features_{bm}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


@st.cache_resource(show_spinner="Chargement des modèles (edge int8 + cloud)…")
def load_engine():
    """Importe 16_rpi_client.py (nom à chiffre → importlib) et instancie l'edge."""
    spec = importlib.util.spec_from_file_location("rpi_client", str(ANALYSIS / "16_rpi_client.py"))
    rpi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rpi)
    mean  = np.load(RESULTS / "scaler_ms_mean_AES-T800.npy")
    scale = np.load(RESULTS / "scaler_ms_scale_AES-T800.npy")
    edge  = rpi.EdgeModel(RESULTS / "cnn1d_tiny_stm32.pt", mean, scale)
    return rpi, edge


def verdict_card(label, conf, titre):
    st.markdown(f"**{titre}**")
    color = ["#1b7a3d", "#b58900", "#c0392b"][label]
    st.markdown(
        f"<div style='border:2px solid {color};border-radius:10px;padding:14px 16px'>"
        f"<span style='font-size:1.6em'>{LED[label]} <b>{LABELS[label]}</b></span><br>"
        f"<span style='color:gray'>{DESCR[label]}</span><br>"
        f"Confiance : <b>{conf:.1%}</b> &nbsp;•&nbsp; Risque : <b>{RISK[label]}</b></div>",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — navigation
# ════════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🛡️ HT Detection")
st.sidebar.caption("Golden-free · Power side-channel · MLOps")
page = st.sidebar.radio("Navigation", [
    "🏠 Accueil",
    "🔬 Démo Live",
    "📊 Résultats ML",
    "⚙️ MLOps & Infra",
])
st.sidebar.divider()
st.sidebar.markdown(
    "**Master Sécurité des Systèmes Embarqués**  \nUCAD — Projet B  \n"
    "Détection de chevaux de Troie matériels par apprentissage automatique")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("Détection de Hardware Trojans — sans puce de référence")
    st.markdown(
        "Détecter un **cheval de Troie matériel** (logique malveillante insérée dans un "
        "circuit) à partir de la **consommation de courant** mesurée, **sans disposer "
        "d'une puce saine de référence** (*golden-free*). Approche : extraction de "
        "features side-channel + apprentissage automatique, jusqu'au **déploiement "
        "embarqué** (STM32) et au **MLOps** complet (CI/CD, versioning, monitoring).")

    tm = load_json("15_TinyMLP_metrics.json") or {}
    cl = load_json("13_CL_metrics.json") or {}
    bint = (tm.get("benchmarks_int8") or {})

    c = st.columns(4)
    c[0].metric("Edge — Flash", f"{tm.get('flash_kb', 33)} KB", "TinyMLP int8 / STM32")
    c[1].metric("Edge AES-T800 (int8)", f"{bint.get('AES-T800', 0.9526):.1%}", "sur device")
    c[2].metric("SimCLR @ 5% labels", f"{cl.get('results', {}).get('5pct', {}).get('simclr', 0.6592):.1%}",
                f"+{cl.get('results', {}).get('5pct', {}).get('gain_vs_coral', 0.159):.0%} vs CORAL")
    c[3].metric("Évasion totale @ ε=0.2", "0 %", "défense en profondeur")

    c = st.columns(4)
    c[0].metric("Benchmarks réels", "6", "AES-T400 → T1100")
    c[1].metric("Features / trace", "500", "25 fenêtres × 20")
    c[2].metric("Modèles entraînés", "21", "CNN1D / CORAL / AT…")
    c[3].metric("Latence edge", "~1–5 ms", "Cortex-M4 @ 84 MHz")

    st.divider()
    st.subheader("Architecture de déploiement — 3 niveaux")
    st.markdown("""
```
┌─────────────────────┐    courant    ┌──────────────────────┐   UART 115200   ┌──────────────────────┐   HTTP   ┌──────────────────────┐
│   PUCE VICTIME      │  ──────────►  │   EDGE               │ ──────────────► │   GATEWAY            │ ───────► │   CLOUD              │
│   AES + Trojan      │               │   STM32 F401RE        │                 │   Raspberry Pi 4     │          │   API K8s            │
│   (FPGA/IEEE)       │               │   TinyMLP int8        │ ◄────────────── │   orchestration      │ ◄─────── │   CNN1D AT_T800      │
│   traces réelles    │               │   33 KB · ~1-5 ms     │   verdict edge  │   proxy numpy int8   │  verdict │   2.2 MB · robuste   │
└─────────────────────┘               └──────────────────────┘                 └──────────────────────┘          └──────────────────────┘
```
""")
    st.caption("L'edge assure un triage basse latence ; les cas douteux sont escaladés "
               "au cloud, bien plus difficile à tromper par PGD (cf. scénario B).")

    st.divider()
    st.subheader("Chaîne de traitement")
    s = st.columns(5)
    for col, (t, d) in zip(s, [
        ("1 · Données", "Traces de courant réelles (IEEE Dataport), 6 benchmarks AES-T*"),
        ("2 · Features", "500 descripteurs / trace : stats + énergie + FFT par fenêtre"),
        ("3 · Modèles", "CNN1D, adaptation de domaine CORAL, multi-source"),
        ("4 · Robustesse", "Adversarial Training, SimCLR semi-supervisé, MC-Dropout"),
        ("5 · Embarqué", "Distillation → TinyMLP int8 → STM32 / RPi / Cloud"),
    ]):
        col.markdown(f"**{t}**  \n{d}")

    st.info("➡️ Onglet **Démo Live** : tester le détecteur en direct sur une vraie trace. "
            "Onglet **Résultats ML** : toutes les expériences. Onglet **MLOps** : l'infra.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DÉMO LIVE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Démo Live":
    st.title("🔬 Démo Live — tester le détecteur")
    st.markdown(
        "On rejoue une **vraie trace held-out** (jamais vue à l'entraînement) et on "
        "interroge les niveaux déployés : **edge int8** (réplique exacte du STM32) et "
        "**cloud** (API K8s, si joignable). Puis on lance une **attaque adversariale "
        "PGD** pour voir qui résiste.")

    try:
        rpi, edge = load_engine()
    except Exception as e:
        st.error(f"Impossible de charger les modèles : {e}")
        st.stop()

    cc = st.columns([1, 1, 1.3])
    bm = cc[0].selectbox("Benchmark", BENCHMARKS, index=4)
    X, y = load_traces(bm)
    if X is None:
        st.warning(f"`features_{bm}.npz` introuvable dans results/. "
                   "Lance l'extraction de features pour ce benchmark.")
        st.stop()

    classe = cc[1].selectbox("Classe à tirer", ["(aléatoire)"] + LABELS)
    api_url = cc[2].text_input("URL API cloud", value="http://localhost:30800")

    if "trace_idx" not in st.session_state:
        st.session_state.trace_idx = int(np.random.randint(len(X)))
    if st.button("🎲 Tirer une nouvelle trace"):
        pool = np.arange(len(X)) if classe == "(aléatoire)" else np.where(y == LABELS.index(classe))[0]
        st.session_state.trace_idx = int(np.random.choice(pool))

    idx = st.session_state.trace_idx
    raw = X[idx]
    st.caption(f"Trace #{idx} — vérité terrain : **{LED[y[idx]]} {LABELS[y[idx]]}**")

    st.line_chart(raw, height=160, use_container_width=True)

    # ── Inférence propre ──
    st.subheader("Verdict des détecteurs (trace intacte)")
    lab, conf = edge.predict(raw)
    lab, conf = int(lab[0]), float(conf[0])

    cloud = rpi.CloudAPI(api_url)
    cloud_up = cloud.health()
    cres = cloud.predict(raw.tolist()) if cloud_up else None

    v = st.columns(2)
    with v[0]:
        verdict_card(lab, conf, "🔹 EDGE — STM32 (TinyMLP int8, 33 KB)")
        st.caption("Calcul identique au firmware embarqué (BatchNorm fusionné, int8).")
    with v[1]:
        if cres is not None:
            verdict_card(int(cres[0]), float(cres[1]), "☁️ CLOUD — API K8s (CNN1D AT)")
            st.caption("Modèle complet robuste servi par FastAPI sur Kubernetes.")
        else:
            st.markdown("**☁️ CLOUD — API K8s (CNN1D AT)**")
            st.warning("API injoignable. Démarre-la, ex. :\n\n"
                       "`kubectl port-forward svc/ht-api 30800:80`\n\n"
                       "ou `uvicorn api.serve:app --port 30800`")

    ok = (y[idx] == lab)
    st.success("✅ Edge correct — verdict = vérité terrain." if ok
               else "⚠️ Edge en désaccord avec la vérité (cas limite — d'où l'intérêt du cloud).")

    # ── Attaque adversariale PGD ──
    st.divider()
    st.subheader("🎯 Attaque adversariale (PGD) — masquer le Trojan")
    st.markdown("L'attaquant ajoute une perturbation imperceptible pour faire passer une "
                "trace **Triggered** pour **Disabled**. On teste l'edge sous attaque.")

    if y[idx] != 2:
        st.info("Tire une trace de classe **TrojanTriggered** pour lancer l'attaque "
                "(menu « Classe à tirer » → TrojanTriggered).")
    else:
        eps = st.slider("Force de l'attaque ε (budget L∞)", 0.0, 0.5, 0.2, 0.05)
        import torch
        dev = torch.device("cpu")
        Xn = ((raw - edge.mean) / edge.scale).astype(np.float32)[None]
        if eps > 0:
            Xadv = rpi.pgd_attack(edge.model, Xn, eps, 0.05, 20, 0, dev)
        else:
            Xadv = Xn
        raw_adv = (Xadv[0] * edge.scale + edge.mean).astype(np.float32)
        alab, aconf = edge.predict(raw_adv)
        alab, aconf = int(alab[0]), float(aconf[0])

        a = st.columns(2)
        with a[0]:
            verdict_card(2, conf, "Avant attaque (ε=0)")
        with a[1]:
            verdict_card(alab, aconf, f"Après PGD (ε={eps})")
        if alab == 2:
            st.success(f"🛡️ Edge **résiste** à ε={eps} : le Trojan reste détecté (Triggered).")
        elif alab == 1:
            st.warning(f"⚠️ Alarme **rétrogradée** ALERT→WARNING (→ Enabled). "
                       "Le Trojan n'est pas caché : un anomalie subsiste → le cloud tranche.")
        else:
            st.error(f"🔓 Évasion **totale** à ε={eps} : edge trompé (→ Disabled). "
                     "C'est précisément le rôle du cloud robuste de rattraper ce cas.")
        st.caption("Rappel (scénario B, 500 traces) : edge contournable ~49% à ε=0.2 "
                   "mais 0% d'évasion totale ; cloud AT ~0.4%. Défense en profondeur.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RÉSULTATS ML
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Résultats ML":
    st.title("📊 Résultats des expériences")
    t = st.tabs(["Baseline & dérive", "Multi-source & CORAL", "D1 · Adversarial Training",
                 "D2 · SimCLR", "D3 · MC-Dropout", "Edge · TinyMLP", "Déploiement · RPi"])

    with t[0]:
        st.subheader("Baselines et problème de dérive de domaine")
        m = load_json("metrics.json") or {}
        st.markdown("Entraîner sur un benchmark et tester sur un autre (**cross-domaine**) "
                    "fait s'effondrer la performance — c'est le cœur du problème golden-free.")
        cols = st.columns(4)
        for col, name in zip(cols, ["RandomForest", "SVM_RBF", "XGBoost", "CNN1D"]):
            if name in m:
                col.metric(name, f"{m[name]['accuracy']:.1%}", f"F1 {m[name]['f1_macro']:.2f}")
        best = m.get("_best", {})
        if best:
            st.caption(f"Meilleur cross-domaine : **{best.get('model')}** "
                       f"{best.get('accuracy', 0):.1%} (train {best.get('train')} → test {best.get('test')}) "
                       "— à peine mieux que le hasard (33%). D'où l'adaptation de domaine.")
        img_grid([("02_model_comparison.png", "Comparaison des modèles"),
                  ("01_rf_feature_importance.png", "Importance des features (RF)"),
                  ("03_domain_shift.png", "Dérive de domaine entre benchmarks"),
                  ("cm_CNN1D.png", "Matrice de confusion — CNN1D")])
        with st.expander("Matrices de confusion (tous modèles)"):
            img_grid([("cm_RandomForest.png", "RandomForest"), ("cm_SVM_RBF.png", "SVM RBF"),
                      ("cm_XGBoost.png", "XGBoost"), ("cm_CNN1D.png", "CNN1D")])

    with t[1]:
        st.subheader("Adaptation de domaine — multi-source & CORAL")
        st.markdown("**CORAL** aligne les statistiques (covariances) entre domaines source "
                    "et cible ; l'entraînement **multi-source** généralise mieux à une puce inconnue.")
        img_grid([("07_multisource.png", "Entraînement multi-source"),
                  ("08_coral.png", "Adaptation CORAL"),
                  ("05_robustness.png", "Robustesse"),
                  ("06_adversarial.png", "Sensibilité adversariale (étude initiale)")])
        with st.expander("Visualisations CORAL supplémentaires"):
            img_grid([("coral1.png", "CORAL — 1"), ("coral2.png", "CORAL — 2"),
                      ("coral3.png", "CORAL — 3")])

    with t[2]:
        st.subheader("Direction 1 — Adversarial Training (PGD-AT, Madry 2018)")
        d = load_json("12_AT_metrics.json") or {}
        b, a = d.get("before", {}), d.get("after", {})
        cols = st.columns(3)
        cols[0].metric("Accuracy propre (avant→après)",
                       f"{a.get('clean_acc', 0):.1%}", f"{a.get('clean_acc', 0)-b.get('clean_acc', 0):+.1%}")
        cols[1].metric("Masquage PGD ε=0.2", f"{b.get('pgd_masquage', {}).get('0.2', 0):.0%}",
                       "modèle déjà robuste à ε réaliste")
        cols[2].metric("Masquage PGD ε=0.5 (après AT)",
                       f"{a.get('pgd_masquage', {}).get('0.5', 0):.0%}", "point de rupture")
        st.markdown("À ε réaliste (≤0.2) le détecteur CNN1D est **déjà robuste** (0% de "
                    "masquage) ; l'AT déplace le point de rupture mais coûte de la précision propre.")
        img_grid([("12_AT_courbe_robustesse.png", "Robustesse vs force d'attaque"),
                  ("12_AT_trade_off.png", "Compromis précision / robustesse"),
                  ("12_AT_bilan_securite.png", "Bilan sécurité")])

    with t[3]:
        st.subheader("Direction 2 — SimCLR (apprentissage contrastif semi-supervisé)")
        d = load_json("13_CL_metrics.json") or {}
        r = d.get("results", {})
        cols = st.columns(3)
        for col, k in zip(cols, ["5pct", "10pct", "20pct"]):
            if k in r:
                col.metric(f"SimCLR @ {k.replace('pct','%')} labels", f"{r[k]['simclr']:.1%}",
                           f"+{r[k]['gain_vs_coral']:.0%} vs CORAL")
        st.markdown("Avec seulement **5% d'étiquettes**, SimCLR atteint **65.9%** — soit "
                    "**+15.9 points** vs CORAL — en exploitant les traces non étiquetées.")
        img_grid([("13_CL_courbe_few_shot.png", "Courbe few-shot (labels vs accuracy)"),
                  ("13_CL_comparaison.png", "SimCLR vs supervisé vs CORAL"),
                  ("13_CL_representations_tsne.png", "Représentations apprises (t-SNE)"),
                  ("13_CL_bilan.png", "Bilan")])

    with t[4]:
        st.subheader("Direction 3 — MC-Dropout (quantification d'incertitude)")
        d = load_json("14_UQ_metrics.json") or {}
        bdict = d.get("benchmarks", {})
        t700 = bdict.get("AES-T700", {})
        cols = st.columns(3)
        cols[0].metric("Accuracy si « certain »", f"{t700.get('acc_certain', 0):.1%}", "T700")
        cols[1].metric("Accuracy si « incertain »", f"{t700.get('acc_uncertain', 0):.1%}",
                       f"{t700.get('acc_uncertain',0)-t700.get('acc_certain',0):+.1%}")
        cols[2].metric("Passes stochastiques", f"{d.get('n_passes', 50)}", "Monte-Carlo")
        st.markdown("Le modèle **sait dire qu'il ne sait pas** : sur T700, les prédictions "
                    "« certaines » atteignent **71.5%** vs **50.8%** quand l'incertitude épistémique "
                    "est élevée → on peut **escalader** les cas incertains au cloud / à l'humain.")
        img_grid([("14_UQ_calibration.png", "Calibration"),
                  ("14_UQ_distribution.png", "Distribution de l'incertitude"),
                  ("14_UQ_cross_domain.png", "Incertitude in- vs out-of-domain"),
                  ("14_UQ_bilan.png", "Bilan")])

    with t[5]:
        st.subheader("Edge — distillation vers TinyMLP int8 (STM32)")
        d = load_json("15_TinyMLP_metrics.json") or {}
        cols = st.columns(4)
        cols[0].metric("Architecture", d.get("architecture", "500→64→32→3"))
        cols[1].metric("Flash", f"{d.get('flash_kb', 33)} KB", f"{d.get('params', 34435):,} params")
        cols[2].metric("Accuracy int8", f"{d.get('student_acc_int8', 0.5456):.1%}",
                       f"perte quant. {d.get('quantization_loss', 0.0034):.2%}")
        cols[3].metric("vs teacher", f"×{d.get('acc_ratio', 1.24):.2f}", "élève > maître")
        st.markdown("La **distillation de connaissances** transfère le maître (CNN1D AT) vers un "
                    "**TinyMLP** quantifié **int8** tenant en **33 KB** de Flash, avec une perte de "
                    "quantification de seulement **0.34%** — déployable sur Cortex-M4.")
        img_grid([("15_TinyMLP_comparaison.png", "Élève vs maître par benchmark"),
                  ("15_TinyMLP_confusion.png", "Matrice de confusion (int8)"),
                  ("15_TinyMLP_memoire.png", "Empreinte mémoire")])
        bint = d.get("benchmarks_int8", {})
        if bint:
            st.markdown("**Accuracy int8 par benchmark (sur device) :**")
            cs = st.columns(len(bint))
            for col, (k, val) in zip(cs, bint.items()):
                col.metric(k, f"{val:.1%}")

    with t[6]:
        st.subheader("Déploiement — client Raspberry Pi 3-niveaux")
        st.markdown("**Scénario A** — détection d'activation : un flux temporel sain bascule "
                    "en Trojan déclenché, l'alarme se lève avec ~1 trace de latence.")
        img("16_rpi_scenarioA_timeline.png", caption="Scénario A — détection d'activation")
        st.markdown("**Scénario B** — évasion adversariale : l'edge est contournable "
                    "(alarme rétrogradée) mais le cloud robuste préserve l'alerte.")
        img("16_rpi_scenarioB_masquage.png", caption="Scénario B — Edge vs Cloud sous PGD")
        d = load_json("16_rpi_metrics.json") or {}
        if d:
            with st.expander("Métriques brutes (16_rpi_metrics.json)"):
                st.json(d)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MLOPS & INFRA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ MLOps & Infra":
    st.title("⚙️ MLOps & Infrastructure")
    st.markdown("Le projet n'est pas qu'un notebook : suivi d'expériences, versioning des "
                "données, CI/CD, conteneurisation, orchestration et monitoring de production.")
    t = st.tabs(["MLflow", "DVC", "Jenkins CI/CD", "Kubernetes", "API serving",
                 "Monitoring", "Sécurité"])

    with t[0]:
        st.subheader("MLflow — suivi d'expériences & registre de modèles")
        img_grid([("MLFlow_run.png", "Runs MLflow"),
                  ("MLflow_Transfer Learning.png", "Expérience Transfer Learning"),
                  ("run_cnn_detais.png", "Détails d'un run CNN"),
                  ("model-details.png", "Fiche modèle enregistré")])
        img("model_train.png", caption="Suivi d'entraînement")

    with t[1]:
        st.subheader("DVC — versioning des données & pipelines")
        st.markdown("Les jeux de traces et artefacts sont versionnés avec **DVC** "
                    "(reproductibilité, séparation code/données).")
        img("versionnage_DVC.png", caption="Versionnage DVC")

    with t[2]:
        st.subheader("Jenkins — intégration & déploiement continus")
        img_grid([("Jenkins_dashboard.png", "Tableau de bord Jenkins"),
                  ("build_succes.png", "Build réussi"),
                  ("buil-fil.png", "Pipeline de build"),
                  ("jenkins_coral.png", "Pipeline CORAL")])

    with t[3]:
        st.subheader("Kubernetes — orchestration du service")
        st.markdown("Le modèle robuste (CNN1D AT_T800) est servi en production sur un cluster "
                    "**Kubernetes** (scalabilité, résilience).")
        img("kubernetes_pods.png", caption="Pods Kubernetes")

    with t[4]:
        st.subheader("API serving — FastAPI")
        st.markdown("Endpoints `/predict` (trace brute) et `/predict/features` (features "
                    "pré-extraites, flux edge→cloud) ; `/ready` pour la santé.")
        img_grid([("API_predict.png", "Réponse /predict"),
                  ("ht_predict1.png", "Prédiction HT via API"),
                  ("test_endpoint_D3.png", "Test d'endpoint")])

    with t[5]:
        st.subheader("Monitoring — dérive & santé en production")
        img_grid([("ht-monitoring.png", "Monitoring du détecteur"),
                  ("04_monitor.png", "Surveillance de dérive"),
                  ("qualité_gate_D3.png", "Quality gate (CI)")])

    with t[6]:
        st.subheader("Sécurité — modèle de menace")
        reg = load_json("threat_registry.json")
        st.markdown("Registre des menaces considérées (attaques d'évasion, dérive, "
                    "empoisonnement…) et contre-mesures.")
        if reg:
            st.json(reg)
        else:
            st.caption("threat_registry.json introuvable.")


# ── Pied de page ─────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Données : traces réelles IEEE Dataport · Aucune donnée générée.")
