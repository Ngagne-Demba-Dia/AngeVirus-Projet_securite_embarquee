"""
01_indexer.py — Construire l'index de tous les fichiers CSV avec labels.
Label encodé dans le nom du dossier : TrojanDisabled=0, TrojanEnabled=1, TrojanTriggered=2
Output: ../results/index.parquet
"""
import re
from pathlib import Path
import pandas as pd
import yaml

LABEL_MAP = {"TrojanDisabled": 0, "TrojanEnabled": 1, "TrojanTriggered": 2}

_DATASET_CANDIDATES = [
    Path("E:/DATA/hardware_trojan"),
    Path("/mnt/e/DATA/hardware_trojan"),
]


def build_index(root: Path) -> pd.DataFrame:
    rows = []
    for bm_top in sorted(root.iterdir()):
        if not bm_top.is_dir():
            continue
        bm_match = re.match(r"(AES-T\d+)", bm_top.name)
        if not bm_match:
            continue
        benchmark = bm_match.group(1)

        # Structure : AES-TXXX_power_Temp25C / AES-TXXX_power_Temp25C / AES-TXXX+Condition_N / AES-TXXX+Condition_N / Sample_*.csv
        inner = bm_top / bm_top.name
        if not inner.exists():
            inner = bm_top  # fallback si pas de double-nesting

        for cond_top in sorted(inner.iterdir()):
            if not cond_top.is_dir():
                continue
            cond_match = re.search(r"\+(Trojan\w+)_(\d)", cond_top.name)
            if not cond_match:
                continue
            condition = cond_match.group(1)
            method = int(cond_match.group(2))
            label = LABEL_MAP.get(condition, -1)
            if label == -1:
                continue

            # Le dossier de CSV a le même nom que son parent
            csv_dir = cond_top / cond_top.name
            if not csv_dir.exists():
                csv_dir = cond_top

            n_before = len(rows)
            for csv_file in sorted(csv_dir.glob("Sample_*.csv")):
                rows.append({
                    "path": str(csv_file),
                    "benchmark": benchmark,
                    "condition": condition,
                    "method": method,
                    "label": label,
                })
            print(f"  {cond_top.name}: {len(rows)-n_before} traces")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    with open("../configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    root = next((p for p in _DATASET_CANDIDATES if p.exists()), None)
    if root is None:
        raise FileNotFoundError(f"Dataset non trouve. Chemin attendu : {_DATASET_CANDIDATES[0]}")

    print(f"Indexing {root} ...")
    df = build_index(root)

    out = Path("../results/index.parquet")
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nIndex total : {len(df):,} traces")
    summary = df.groupby(["benchmark", "condition"])["path"].count().reset_index()
    summary.columns = ["benchmark", "condition", "n_traces"]
    print(summary.to_string(index=False))
    print(f"\nSauvegarde : {out}")
