import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

SEED = 74
FLASH_CUTOFF_K = 333.15
STD_CUTOFF_K = 5.0
TAG_COLUMNS = ["is_silicon", "is_metallic", "is_tin", "is_acid"]

RAW_PATH = Path("data/raw/integrated_dataset_public.csv")
CLEANED_PATH = Path("data/processed/cleaned.csv")
REPORT_PATH = Path("data/processed/cleaning_report.txt")


def parse_smiles(df, report):
    RDLogger.DisableLog("rdApp.*")
    parsed = df["smiles"].map(Chem.MolFromSmiles)
    bad = parsed.isna()
    report.append(f"rows in raw file: {len(df)}")
    report.append(f"unparseable SMILES dropped: {int(bad.sum())} rows, {df.loc[bad, 'smiles'].nunique()} unique SMILES")
    return df.loc[~bad].reset_index(drop=True)


def merge_duplicates(df, report):
    grouped = df.groupby("smiles", sort=True)
    stats = grouped["flashpoint"].agg(["mean", "std", "count"])
    stats["std"] = stats["std"].fillna(0.0)
    conflicting = stats["std"] > STD_CUTOFF_K
    report.append(f"unique SMILES: {len(stats)}")
    report.append(f"unique SMILES with duplicate measurements: {int((stats['count'] > 1).sum())}")
    report.append(f"dropped for conflicting measurements (std > {STD_CUTOFF_K} K): {int(conflicting.sum())}")
    report.append(f"unique compounds kept: {int((~conflicting).sum())}")
    first = grouped[["compound"]].first()
    tags = grouped[TAG_COLUMNS].max()
    merged = first.join(tags).join(stats[["mean", "count"]].rename(columns={"mean": "flashpoint", "count": "n_measurements"}))
    merged = merged.loc[~conflicting].reset_index()
    return merged[["compound", "smiles", "flashpoint", "n_measurements"] + TAG_COLUMNS]


def add_label(df, report):
    df = df.copy()
    df["flammable"] = (df["flashpoint"] <= FLASH_CUTOFF_K).astype(int)
    pos = int(df["flammable"].sum())
    report.append(f"flammable (flash point <= {FLASH_CUTOFF_K} K): {pos} ({pos / len(df):.1%})")
    report.append(f"not flammable: {len(df) - pos} ({(len(df) - pos) / len(df):.1%})")
    for tag in TAG_COLUMNS:
        sub = df[df[tag] == 1]
        report.append(f"{tag}: {len(sub)} compounds, {sub['flammable'].mean():.1%} flammable")
    return df


def main():
    report = []
    raw = pd.read_csv(RAW_PATH)
    parsed = parse_smiles(raw, report)
    merged = merge_duplicates(parsed, report)
    labelled = add_label(merged, report)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(CLEANED_PATH, index=False)
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    sys.exit(main())
