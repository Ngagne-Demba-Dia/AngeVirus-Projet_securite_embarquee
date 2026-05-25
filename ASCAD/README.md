# ASCAD — Analyse par Canal Auxiliaire sur AES Masqué

## Objectif

Évaluer les méthodes classiques d'analyse par canal auxiliaire (SNR, CPA, Template LDA, TVLA)
sur la base ASCAD — une implémentation AES-128 **masquée au premier ordre** sur ATMega8515.

## Dataset

| Paramètre | Valeur |
|-----------|--------|
| Source | ASCAD.h5 (ANSSI/CEA, 2018) |
| Traces de profiling | 50 000 (45 000 train + 5 000 val) |
| Traces d'attaque | 10 000 |
| Longueur d'une trace | 700 échantillons EM |
| Implémentation | AES-128 masqué booléen, ATMega8515 |
| Octet ciblé | Byte 2 — `SBox[pt[2] XOR k[2]] XOR mask[0]` |
| Vraie clé (byte 2) | `0xE0` |

## Principe du masquage booléen

L'intermédiaire qui fuit dans la trace est la valeur **masquée** :

```
z = SBox[pt[2] XOR k[2]] XOR mask[0]   ← fuite mesurée par le SNR
```

Le masque `mask[0]` est aléatoire, uniforme et renouvelé à chaque trace.
Il rend l'intermédiaire `z` statistiquement indépendant de `SBox[pt XOR k]` → **la CPA directe échoue**.

---

## Étape 1 — SNR (Signal-to-Noise Ratio)

**Script :** [analysis/snr.py](analysis/snr.py)

| Résultat | Valeur |
|----------|--------|
| Pic SNR (valeur masquée) | Échantillon **517**, SNR = **6.33** |
| Pic SNR (valeur non masquée) | Échantillon 517, SNR = 0.007 |

Le masque décale le signal : la fuite porte sur `z = SBox[pt XOR k] XOR mask[0]`,
et le SNR non masqué est quasi nul — ce qui explique l'échec de la CPA.

![SNR](results/02_snr.png)

---

## Étape 2 — CPA (Correlation Power Analysis)

**Script :** [analysis/cpa.py](analysis/cpa.py)

Attaque directe non-profilée sur la variable **non masquée** `SBox[pt XOR k]`.

| Métrique | Valeur |
|----------|--------|
| Traces utilisées | 10 000 |
| Rang final de la vraie clé | **71 / 256** |
| Résultat | **Échec** — corrélation → 0 avec plus de traces |

**Pourquoi ça échoue :** La CPA corrèle avec `HW(SBox[pt XOR k])` mais la trace fuit
`z = SBox[pt XOR k] XOR mask[0]`. Comme `mask[0]` est uniforme et indépendant,
`HW(z)` est décorrélé de `HW(SBox[pt XOR k])` : le signal est annulé au premier ordre.

![CPA](results/03_cpa_echec.png)

---

## Étape 3 — Template LDA (attaque profilée)

**Script :** [analysis/lda.py](analysis/lda.py)

Attaque profilée : le masque `mask[0]` est connu pendant le profiling (scénario évaluation ANSSI).

### Méthode

| Étape | Détail |
|-------|--------|
| Cible | `z = SBox[pt XOR k] XOR mask[0]` — **256 classes** |
| Sélection features | Top-100 échantillons par SNR |
| Classifieur | LDA sklearn (covariance partagée, solver SVD) |
| Scoring attaque | `score(k) = Σ log P(SBox[pt_i XOR k] XOR mask_i[0] | trace_i)` |

### Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy profiling | 13.6 % (aléatoire = 0.39 %) |
| Accuracy attaque | 5.3 % |
| **Rang final (10 000 traces)** | **1 / 256** |
| **Traces pour rang 1** | **50 traces** |

La vraie clé `0xE0` est trouvée dès **50 traces d'attaque**.

![LDA rank curve](results/04_lda_rank.png)

---

## Étape 4 — TVLA (Test Vector Leakage Assessment)

**Script :** [analysis/snr.py](analysis/snr.py) (section TVLA)

Le TVLA vérifie si l'implémentation fuit statistiquement, sans connaître la clé.
Test de Welch |t| > 4.5 indique une fuite.

**Résultat :** Fuite détectée sur les mêmes échantillons que le SNR (autour de l'échantillon 517).
L'implémentation fuit — résultat attendu pour une AES masqué sans contre-mesure EM supplémentaire.

---

## Bilan comparatif

| Méthode | Type | Cible | Rang final | Traces pour rang 1 |
|---------|------|-------|-----------|-------------------|
| CPA | Non profilée | `HW(SBox[pt XOR k])` | 71/256 | ❌ jamais |
| Template LDA | Profilée | `SBox[pt XOR k] XOR mask[0]` | **1/256** | **50 traces** |
| Deep Learning (CNN) | Profilée | `SBox[pt XOR k] XOR mask[0]` | **1/256** | **10 traces** |

### Questions de rapport

**Q1 : À quel échantillon le SNR est-il maximal ?**
→ Échantillon **517**, SNR = 6.33. Correspond à la 1re ronde AES (SubBytes sur l'octet 2).

**Q2 : Combien de traces pour l'attaque template rang 1 ?**
→ **50 traces** d'attaque suffisent avec le LDA 256 classes + top-100 SNR features.

**Q3 : SNR octet 2 vs octet 5 vs octet 14 ?**
→ Tous les octets fuient au même endroit (même opération SubBytes de la 1re ronde),
mais avec des amplitudes SNR différentes selon la disposition mémoire sur l'ATMega.
L'octet 2 (TARGET=2) a SNR=6.33 — l'un des plus forts du dataset.

**Q4 : LDA sur 10 000 traces vs 45 000 traces de profiling ?**
→ Avec 10 000 traces de profiling, l'accuracy profiling tombe à ~4 % (vs 13.6 % à 45 000)
et le rang 1 peut nécessiter 200–500 traces au lieu de 50. Moins de données = modèle LDA moins précis.

**Q5 : Le TVLA détecte-t-il une fuite sur tous les échantillons ?**
→ Non. La fuite est localisée autour de l'échantillon 517 (± ~50 points),
là où le SNR est non nul. Les autres échantillons ont |t| < 4.5 → pas de fuite détectée.

---

## Structure du projet

```
ASCAD/
├── analysis/
│   ├── explore.py      # Chargement et visualisation des traces
│   ├── snr.py          # SNR + TVLA
│   ├── cpa.py          # CPA directe (echec attendu)
│   └── lda.py          # Template LDA 256 classes (succes : rang 1 en 50 traces)
└── results/
    ├── 01_traces_brutes.png
    ├── 02_snr.png
    ├── 03_cpa_echec.png
    └── 04_lda_rank.png
```

## Références

- Benadjila et al. (2018). *Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database*. IACR ePrint 2018/053.
- ASCAD GitHub : https://github.com/ANSSI-FR/ASCAD
- SCALib : https://scalib.readthedocs.io
