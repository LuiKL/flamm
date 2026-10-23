import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

SEED = 74
FLASH_CUTOFF_K = 333.15
STD_CUTOFF_K = 5.0
SIMILARITY_CUTOFF = 0.7
FP_RADIUS = 2
FP_BITS = 2048
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TAG_COLUMNS = ["is_silicon", "is_metallic", "is_tin", "is_acid"]
ELEMENTS = ["C", "O", "N", "F", "Cl", "Br", "I"]
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "MolLogP": Descriptors.MolLogP,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
}
FEATURE_COLUMNS = list(DESCRIPTORS) + [f"n{e}" for e in ELEMENTS]

RAW_PATH = Path("data/raw/integrated_dataset_public.csv")
OUT_DIR = Path("data/processed")
REPORT_PATH = OUT_DIR / "cleaning_report.txt"


def parse_smiles(df, report):
    RDLogger.DisableLog("rdApp.*")
    mols = df["smiles"].map(Chem.MolFromSmiles)
    bad = mols.isna()
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


def featurize(df, mols):
    rows = []
    for mol in mols:
        counts = {e: 0 for e in ELEMENTS}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in counts:
                counts[atom.GetSymbol()] += 1
        row = {name: fn(mol) for name, fn in DESCRIPTORS.items()}
        row.update({f"n{e}": counts[e] for e in ELEMENTS})
        rows.append(row)
    features = pd.DataFrame(rows, index=df.index)[FEATURE_COLUMNS]
    return pd.concat([df, features], axis=1)


def connectivity_key(mol):
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        key = ""
    if key:
        return key[:14]
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    return Chem.MolToSmiles(flat)


def fingerprints(mols):
    gen = AllChem.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return [gen.GetFingerprint(m) for m in mols]


def butina_clusters(fps):
    n = len(fps)
    neighbours = []
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        idx = np.flatnonzero(sims >= SIMILARITY_CUTOFF)
        neighbours.append(idx[idx != i])
    order = sorted(range(n), key=lambda i: (-len(neighbours[i]), i))
    labels = np.full(n, -1, dtype=int)
    next_id = 0
    for i in order:
        if labels[i] != -1:
            continue
        labels[i] = next_id
        free = neighbours[i][labels[neighbours[i]] == -1]
        labels[free] = next_id
        next_id += 1
    return labels


def merge_by_key(labels, keys):
    parent = {c: c for c in np.unique(labels)}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    first_seen = {}
    for lab, key in zip(labels, keys):
        if key in first_seen:
            ra, rb = find(first_seen[key]), find(lab)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        else:
            first_seen[key] = lab
    roots = np.array([find(c) for c in labels])
    _, merged = np.unique(roots, return_inverse=True)
    return merged


def stratum_matrix(df):
    pos = df["flammable"].to_numpy()
    cols = {
        "total": np.ones(len(df)),
        "pos": pos,
        "neg": 1 - pos,
        "acid": df["is_acid"].to_numpy(),
        "metallic": df["is_metallic"].to_numpy(),
        "silicon": df["is_silicon"].to_numpy(),
        "tin": df["is_tin"].to_numpy(),
        "acid_pos": df["is_acid"].to_numpy() * pos,
        "metallic_pos": df["is_metallic"].to_numpy() * pos,
        "silicon_pos": df["is_silicon"].to_numpy() * pos,
    }
    return pd.DataFrame(cols, index=df.index).astype(float)


def assign_clusters(df, clusters):
    strata = stratum_matrix(df)
    totals = strata.sum().to_numpy()
    names = list(SPLIT_FRACTIONS)
    targets = np.array([SPLIT_FRACTIONS[s] * totals for s in names])
    current = np.zeros_like(targets)
    cluster_vec = strata.groupby(clusters).sum()
    rng = np.random.default_rng(SEED)
    tiebreak = rng.permutation(len(cluster_vec))
    sizes = cluster_vec["total"].to_numpy()
    order = sorted(range(len(cluster_vec)), key=lambda k: (-sizes[k], tiebreak[k]))
    cluster_ids = cluster_vec.index.to_numpy()
    assignment = {}
    for k in order:
        vec = cluster_vec.iloc[k].to_numpy()
        deficit = (targets - current) / np.maximum(targets, 1.0)
        scores = (deficit * vec).sum(axis=1)
        best = int(np.argmax(scores))
        current[best] += vec
        assignment[cluster_ids[k]] = names[best]
    return pd.Series(clusters, index=df.index).map(assignment)


def nearest_train_similarity(fps, split):
    train_fps = [fps[i] for i in np.flatnonzero(split.to_numpy() == "train")]
    result = {}
    for name in ("val", "test"):
        idx = np.flatnonzero(split.to_numpy() == name)
        result[name] = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in idx])
    return result


def describe_splits(df, split, report):
    report.append("")
    report.append("split summary")
    for name in SPLIT_FRACTIONS:
        sub = df[split == name]
        line = f"{name}: {len(sub)} ({len(sub) / len(df):.1%}), flammable {sub['flammable'].mean():.1%}"
        for tag in ("is_acid", "is_metallic", "is_silicon", "is_tin"):
            line += f", {tag[3:]} {int(sub[tag].sum())}"
        report.append(line)


def main():
    report = []
    raw = pd.read_csv(RAW_PATH)
    parsed = parse_smiles(raw, report)
    merged = merge_duplicates(parsed, report)
    df = add_label(merged, report)
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    df = featurize(df, mols)

    fps = fingerprints(mols)
    labels = butina_clusters(fps)
    keys = [connectivity_key(m) for m in mols]
    clusters = merge_by_key(labels, keys)
    df["cluster_id"] = clusters
    sizes = pd.Series(clusters).value_counts()
    report.append("")
    report.append(f"Butina clusters (Tanimoto >= {SIMILARITY_CUTOFF}, Morgan r={FP_RADIUS}, {FP_BITS} bits), merged by InChIKey connectivity: {len(sizes)}")
    report.append(f"singleton clusters: {int((sizes == 1).sum())}; largest cluster: {int(sizes.max())}; clusters > 50: {int((sizes > 50).sum())}")

    split = assign_clusters(df, clusters)
    describe_splits(df, split, report)
    sims = nearest_train_similarity(fps, split)
    for name, vals in sims.items():
        report.append(
            f"{name} -> nearest train neighbour Tanimoto: max {vals.max():.3f}, "
            f"mean {vals.mean():.3f}, share >= {SIMILARITY_CUTOFF}: {(vals >= SIMILARITY_CUTOFF).mean():.2%}"
        )
    overlap = set(df.loc[split == "train", "cluster_id"]) & (set(df.loc[split == "val", "cluster_id"]) | set(df.loc[split == "test", "cluster_id"]))
    report.append(f"clusters shared across splits: {len(overlap)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FRACTIONS:
        df[split == name].to_csv(OUT_DIR / f"{name}.csv", index=False)
    report.append(f"environment: rdkit {__import__('rdkit').__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    sys.exit(main())
