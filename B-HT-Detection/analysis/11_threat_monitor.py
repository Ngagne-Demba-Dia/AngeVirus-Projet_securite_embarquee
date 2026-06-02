"""
11_threat_monitor.py — LLM-Agent de surveillance continue des menaces Hardware Trojan.

Surveille en continu :
  - arXiv : nouveaux articles HT detection ML
  - IEEE DataPort : nouveaux datasets Trust-Hub
  - Trust-Hub benchmark updates

Pipeline :
  1. Fetch arXiv API pour nouveaux papiers (quotidien/hebdomadaire)
  2. Analyser la pertinence via Claude API (résumé + extraction méthodes)
  3. Détecter nouveaux datasets ou benchmarks
  4. Notifier (log + email optionnel) + déclencher Jenkins si nouveau dataset
  5. Maintenir un registre des menaces identifiées

Usage :
  python 11_threat_monitor.py --mode once       # scan unique
  python 11_threat_monitor.py --mode watch      # surveillance continue (toutes les 6h)
  python 11_threat_monitor.py --mode report     # rapport complet
"""
import os
import json
import time
import argparse
import logging
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ht-threat-monitor")

# ── Configuration ──────────────────────────────────────────────────────────────
SEMANTIC_API  = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API     = "https://export.arxiv.org/api/query"   # fallback
ARXIV_QUERIES = [
    "hardware trojan detection machine learning",
    "hardware trojan deep learning side channel",
    "hardware trojan adversarial attack",
    "hardware trojan transfer learning domain",
    "hardware trojan neural network golden free",
]
REGISTRY_PATH = Path("../results/threat_registry.json")
REPORT_PATH   = Path("../results/threat_report.md")
JENKINS_URL   = os.getenv("JENKINS_URL", "http://localhost:8080")
JENKINS_JOB   = os.getenv("JENKINS_JOB", "Hardware_Trojan-Detection-Pipeline")
JENKINS_TOKEN = os.getenv("JENKINS_TOKEN", "")

# Mots-clés indiquant un nouveau dataset ou benchmark
DATASET_KEYWORDS = [
    "trust-hub", "trusthub", "ieee dataport", "aes-t", "fpga trojan",
    "side-channel dataset", "power trace dataset", "benchmark hardware trojan",
    "new dataset", "novel benchmark"
]

# Mots-clés indiquant une nouvelle menace (attaque adversariale)
THREAT_KEYWORDS = [
    "adversarial attack", "evasion attack", "hardware trojan obfuscation",
    "trojan hiding", "adversarial perturbation", "bypass detection",
    "evade hardware trojan", "fooling detector"
]


# ── Fetch Semantic Scholar ─────────────────────────────────────────────────────
def fetch_semantic_papers(query: str, max_results: int = 10,
                           days_back: int = 30) -> list[dict]:
    """Récupère les papiers via Semantic Scholar API (phrase search native)."""
    cutoff = (datetime.now() - timedelta(days=days_back)).year
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,authors,year,venue,externalIds,publicationDate",
    }
    url = f"{SEMANTIC_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "HT-Monitor/1.0"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            wait = (attempt + 1) * 15
            log.warning(f"Erreur Semantic Scholar ({query[:30]}...) tentative {attempt+1}/3 : {e}")
            if attempt < 2:
                time.sleep(wait)
            else:
                return []

    papers = []
    for item in data.get("data", []):
        year = item.get("year") or 0
        if year < cutoff:
            continue
        title    = item.get("title", "").strip()
        abstract = (item.get("abstract") or "")[:800]
        authors  = [a.get("name", "") for a in item.get("authors", [])[:3]]
        pub_date = item.get("publicationDate") or str(year)
        ext_ids  = item.get("externalIds", {})
        arxiv_id = ext_ids.get("ArXiv", "")
        url_paper = (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                     else f"https://api.semanticscholar.org/graph/v1/paper/{item.get('paperId','')}")

        if not title:
            continue

        papers.append({
            "id":        hashlib.md5((title + str(year)).encode()).hexdigest()[:12],
            "title":     title,
            "authors":   authors,
            "published": pub_date[:10],
            "url":       url_paper,
            "abstract":  abstract,
            "source":    "SemanticScholar",
        })

    return papers


# ── Fetch arXiv (fallback) ─────────────────────────────────────────────────────
def fetch_arxiv_papers(query: str, max_results: int = 10,
                        days_back: int = 30) -> list[dict]:
    """Fallback arXiv fetch avec requête simple."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        log.warning(f"arXiv fallback échoué : {e}")
        return []

    papers = []
    root = ET.fromstring(content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title    = entry.findtext("atom:title",   "", ns).strip()
        summary  = entry.findtext("atom:summary", "", ns).strip()
        paper_id = entry.findtext("atom:id",      "", ns).strip()
        published = entry.findtext("atom:published", "", ns)[:10]
        authors  = [a.findtext("atom:name", "", ns)
                    for a in entry.findall("atom:author", ns)]
        if not title:
            continue
        papers.append({
            "id":        hashlib.md5(paper_id.encode()).hexdigest()[:12],
            "title":     title, "authors": authors[:3],
            "published": published, "url": paper_id,
            "abstract":  summary[:800], "source": "arXiv",
        })
    return papers


# ── Analyse de pertinence ──────────────────────────────────────────────────────
def analyze_relevance(paper: dict) -> dict:
    """
    Analyse la pertinence d'un papier sans LLM externe (rule-based).
    Catégories : nouveau_dataset, nouvelle_attaque, nouvelle_methode, hors_sujet.
    """
    text = (paper["title"] + " " + paper["abstract"]).lower()

    is_dataset  = any(kw in text for kw in DATASET_KEYWORDS)
    is_attack   = any(kw in text for kw in THREAT_KEYWORDS)
    is_ml       = any(kw in text for kw in [
        "neural network", "deep learning", "machine learning", "cnn", "lstm",
        "random forest", "svm", "xgboost", "transfer learning", "domain adaptation"
    ])
    # Exiger la phrase exacte "hardware trojan" pour éviter les faux positifs
    is_ht = ("hardware trojan" in text or
             "hardware trojan" in text or
             "trojan detection" in text or
             "ht detection" in text or
             "hardware-trojan" in text)

    # Score de pertinence
    score = 0
    if is_ht:   score += 3
    if is_ml:   score += 2
    if is_dataset: score += 2
    if is_attack:  score += 2
    if "side-channel" in text or "power trace" in text: score += 2
    if "golden" in text and "free" in text: score += 1

    category = "hors_sujet"
    if score >= 5:
        if is_dataset:
            category = "nouveau_dataset"
        elif is_attack:
            category = "nouvelle_attaque"
        else:
            category = "nouvelle_methode"

    return {
        **paper,
        "relevance_score": score,
        "category":        category,
        "is_dataset":      is_dataset,
        "is_attack":       is_attack,
        "alert":           score >= 3 and is_ht,
    }


# ── Registre des menaces ───────────────────────────────────────────────────────
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"papers": {}, "last_scan": None, "alerts": []}


def save_registry(registry: dict):
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


def is_new(paper: dict, registry: dict) -> bool:
    return paper["id"] not in registry["papers"]


# ── Trigger Jenkins ────────────────────────────────────────────────────────────
def trigger_jenkins(reason: str):
    """Déclenche un build Jenkins si un nouveau dataset est détecté."""
    if not JENKINS_TOKEN:
        log.info(f"Jenkins trigger désactivé (pas de token) — raison : {reason}")
        return False

    url = f"{JENKINS_URL}/job/{JENKINS_JOB}/build?token={JENKINS_TOKEN}"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            log.info(f"Jenkins build déclenché : {reason}")
            return True
    except Exception as e:
        log.warning(f"Impossible de déclencher Jenkins : {e}")
        return False


# ── Rapport Markdown ───────────────────────────────────────────────────────────
def generate_report(registry: dict):
    alerts     = [p for p in registry["papers"].values() if p.get("alert")]
    datasets   = [p for p in alerts if p["category"] == "nouveau_dataset"]
    attacks    = [p for p in alerts if p["category"] == "nouvelle_attaque"]
    methods    = [p for p in alerts if p["category"] == "nouvelle_methode"]

    lines = [
        "# Rapport de surveillance — Hardware Trojan Threats",
        f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Dernier scan : {registry.get('last_scan', 'N/A')}*",
        "",
        f"## Résumé",
        f"- **Total papiers analysés** : {len(registry['papers'])}",
        f"- **Alertes actives** : {len(alerts)}",
        f"- **Nouveaux datasets** : {len(datasets)}",
        f"- **Nouvelles attaques** : {len(attacks)}",
        f"- **Nouvelles méthodes** : {len(methods)}",
        "",
    ]

    if datasets:
        lines += ["## Nouveaux Datasets/Benchmarks", ""]
        for p in sorted(datasets, key=lambda x: x["published"], reverse=True)[:5]:
            lines += [
                f"### [{p['title'][:80]}]({p['url']})",
                f"- **Date** : {p['published']}",
                f"- **Auteurs** : {', '.join(p['authors'])}",
                f"- **Score** : {p['relevance_score']}/10",
                f"- **Résumé** : {p['abstract'][:200]}...",
                "",
            ]

    if attacks:
        lines += ["## Nouvelles Attaques Détectées", ""]
        for p in sorted(attacks, key=lambda x: x["relevance_score"], reverse=True)[:5]:
            lines += [
                f"### [{p['title'][:80]}]({p['url']})",
                f"- **Date** : {p['published']}",
                f"- **Score danger** : {p['relevance_score']}/10",
                f"- **Résumé** : {p['abstract'][:200]}...",
                "",
            ]

    if methods:
        lines += ["## Nouvelles Méthodes ML/DL", ""]
        for p in sorted(methods, key=lambda x: x["relevance_score"], reverse=True)[:10]:
            lines += [
                f"### [{p['title'][:80]}]({p['url']})",
                f"- **Date** : {p['published']}",
                f"- **Score** : {p['relevance_score']}/10",
                "",
            ]

    lines += [
        "## Impact sur notre pipeline",
        "",
        "| Catégorie | Action recommandée |",
        "|-----------|-------------------|",
        "| Nouveau dataset | Intégrer dans DVC + relancer 02_features.py |",
        "| Nouvelle attaque | Mettre à jour 07_adversarial.py + seuil ε |",
        "| Nouvelle méthode | Évaluer pour Étape 5 d'optimisation |",
        "",
        "---",
        "*Ce rapport est généré automatiquement par 11_threat_monitor.py*",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    log.info(f"Rapport généré : {REPORT_PATH}")


# ── Scan principal ─────────────────────────────────────────────────────────────
def run_scan(days_back: int = 30) -> dict:
    log.info(f"Début du scan (fenêtre : {days_back} jours)...")
    registry = load_registry()
    new_papers, new_alerts = 0, 0

    for query in ARXIV_QUERIES:
        log.info(f"  Semantic Scholar : '{query[:50]}'")
        papers = fetch_semantic_papers(query, max_results=10, days_back=days_back)
        if not papers:
            log.info(f"  arXiv fallback : '{query[:50]}'")
            papers = fetch_arxiv_papers(query, max_results=10, days_back=days_back)
        time.sleep(5)  # Semantic Scholar : max 100 req/min

        for paper in papers:
            if not is_new(paper, registry):
                continue

            analyzed = analyze_relevance(paper)
            registry["papers"][analyzed["id"]] = analyzed
            new_papers += 1

            if analyzed["alert"]:
                new_alerts += 1
                log.warning(
                    f"  ALERTE [{analyzed['category']}] "
                    f"score={analyzed['relevance_score']} : {paper['title'][:70]}"
                )

                # Déclencher Jenkins si nouveau dataset
                if analyzed["category"] == "nouveau_dataset":
                    trigger_jenkins(f"Nouveau dataset détecté : {paper['title'][:50]}")

    registry["last_scan"] = datetime.now().isoformat()
    save_registry(registry)
    generate_report(registry)

    log.info(f"Scan terminé : {new_papers} nouveaux papiers, {new_alerts} alertes")
    return registry


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HT Threat Monitor")
    parser.add_argument("--mode",     default="once",
                        choices=["once", "watch", "report"],
                        help="Mode : once (scan unique), watch (continu), report (rapport)")
    parser.add_argument("--interval", default=6, type=int,
                        help="Intervalle en heures pour le mode watch (défaut: 6)")
    parser.add_argument("--days",     default=30, type=int,
                        help="Fenêtre temporelle en jours (défaut: 30)")
    args = parser.parse_args()

    if args.mode == "report":
        registry = load_registry()
        generate_report(registry)
        total   = len(registry["papers"])
        alerts  = sum(1 for p in registry["papers"].values() if p.get("alert"))
        print(f"\nRegistre : {total} papiers, {alerts} alertes")
        print(f"Rapport  : {REPORT_PATH}")

    elif args.mode == "once":
        registry = run_scan(days_back=args.days)
        alerts = [p for p in registry["papers"].values() if p.get("alert")]
        print(f"\n{'='*60}")
        print(f"RÉSULTATS DU SCAN")
        print(f"{'='*60}")
        print(f"Total papiers analysés : {len(registry['papers'])}")
        print(f"Alertes                : {len(alerts)}")
        for p in sorted(alerts, key=lambda x: x['relevance_score'], reverse=True)[:5]:
            print(f"  [{p['category']:20}] score={p['relevance_score']} | {p['title'][:60]}")
        print(f"\nRapport complet : {REPORT_PATH}")

    elif args.mode == "watch":
        log.info(f"Mode surveillance continue — scan toutes les {args.interval}h")
        while True:
            run_scan(days_back=args.days)
            next_scan = datetime.now() + timedelta(hours=args.interval)
            log.info(f"Prochain scan : {next_scan.strftime('%Y-%m-%d %H:%M')}")
            time.sleep(args.interval * 3600)
