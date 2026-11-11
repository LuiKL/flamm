import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FEATURES, SEED, chosen_params, load_splits, metrics

REPEATS = 5


def fingerprints(smiles):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def near_share(train_smiles, val_smiles):
    train_fps = fingerprints(train_smiles)
    best = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in fingerprints(val_smiles)]
    return float(np.mean(np.array(best) >= 0.7)), float(np.mean(best))


def fit_eval(train, val, params):
    scaler = StandardScaler().fit(train[FEATURES])
    model = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], class_weight=params["class_weight"], random_state=SEED)
    model.fit(scaler.transform(train[FEATURES]), train["flammable"])
    return metrics(val["flammable"], model.predict(scaler.transform(val[FEATURES])))


def main():
    splits = load_splits()
    params = chosen_params("svm_rbf")
    rows = []
    share, mean_sim = near_share(splits["train"]["smiles"], splits["val"]["smiles"])
    rows.append({"split": "cluster (ours)", "seed": SEED, **fit_eval(splits["train"], splits["val"], params), "val_share_neighbour_ge_0.7": share, "val_mean_nearest_sim": mean_sim})
    allrows = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    for i in range(REPEATS):
        seed = SEED + i
        train, rest = train_test_split(allrows, test_size=0.30, stratify=allrows["flammable"], random_state=seed)
        val, _ = train_test_split(rest, test_size=0.5, stratify=rest["flammable"], random_state=seed)
        share, mean_sim = near_share(train["smiles"], val["smiles"])
        rows.append({"split": "random", "seed": seed, **fit_eval(train, val, params), "val_share_neighbour_ge_0.7": share, "val_mean_nearest_sim": mean_sim})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/leakage_check.csv", index=False)
    print(out[["split", "seed", "recall", "precision", "f1", "val_share_neighbour_ge_0.7", "val_mean_nearest_sim"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
