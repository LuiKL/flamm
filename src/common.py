import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

SEED = 74
DATA_DIR = "data/processed"
FEATURES = [
    "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors", "TPSA", "MolLogP",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "nC", "nO", "nN", "nF", "nCl", "nBr", "nI",
]


def load_splits():
    return {name: pd.read_csv(f"{DATA_DIR}/{name}.csv") for name in ("train", "val", "test")}


def scaled(splits, features=FEATURES):
    scaler = StandardScaler().fit(splits["train"][features])
    return {name: scaler.transform(df[features]) for name, df in splits.items()}


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": (tp + tn) / len(y_true),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def chosen_params(model, path="results/chosen_settings_general.csv"):
    row = pd.read_csv(path).set_index("model").loc[model]
    parsed = dict(part.split("=", 1) for part in row["settings"].split(", "))
    cw = parsed.get("class_weight")
    out = {"class_weight": None if cw == "None" else cw}
    for key in ("C", "gamma"):
        if key in parsed:
            out[key] = float(parsed[key])
    return out
