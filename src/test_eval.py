import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import FEATURES, SEED, chosen_params, load_splits, metrics

RESULTS_DIR = Path("results")
OUT_GENERAL = RESULTS_DIR / "test_results_general.csv"
OUT_SPECIALIST = RESULTS_DIR / "test_specialist_vs_general.csv"
OUT_PREDICTIONS = RESULTS_DIR / "test_predictions.csv"
MW_CUTOFFS = [150, 175, 200]


def parse_settings(text):
    if "=" not in text:
        return {}
    parts = dict(p.split("=", 1) for p in text.split(", "))
    out = {"class_weight": None if parts.get("class_weight") == "None" else parts.get("class_weight")}
    for key in ("C", "gamma", "alpha"):
        if key in parts:
            out[key] = float(parts[key])
    if "penalty" in parts:
        out["penalty"] = None if parts["penalty"] == "None" else parts["penalty"]
    return out


def build(model, p):
    if model == "majority":
        return DummyClassifier(strategy="constant", constant=0)
    if model == "logreg":
        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=5000, random_state=SEED)
    if model == "perceptron":
        return Perceptron(alpha=p["alpha"], penalty=p["penalty"], class_weight=p["class_weight"], max_iter=2000, random_state=SEED)
    if model == "svm_linear":
        return LinearSVC(C=p["C"], class_weight=p["class_weight"], max_iter=20000, random_state=SEED)
    return SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight=p["class_weight"], random_state=SEED)


def subset_masks(df):
    masks = {"all": pd.Series(True, index=df.index), "acid": df["is_acid"] == 1, "metallic": df["is_metallic"] == 1,
             "silicon": df["is_silicon"] == 1, "tin": df["is_tin"] == 1}
    for cutoff in MW_CUTOFFS:
        masks[f"mw>{cutoff}"] = df["MolWt"] > cutoff
    return masks


def main():
    if OUT_GENERAL.exists():
        print("test results already exist; the test set is evaluated once. Delete the files deliberately to rerun.")
        return 1
    splits = load_splits()
    train, test = splits["train"], splits["test"]
    chosen = pd.read_csv(RESULTS_DIR / "chosen_settings_general.csv").set_index("model")["settings"].map(parse_settings)
    scaler = StandardScaler().fit(train[FEATURES])
    X_train, X_test = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
    masks = subset_masks(test)

    rows, preds = [], test[["compound", "smiles", "flashpoint", "flammable", "is_acid", "is_metallic", "is_silicon", "is_tin", "MolWt", "n_measurements"]].copy()
    general_pred = {}
    for model in ["majority", "logreg", "perceptron", "svm_linear", "svm_rbf"]:
        est = build(model, chosen[model]).fit(X_train, train["flammable"])
        pred = est.predict(X_test)
        general_pred[model] = pred
        preds[f"pred_{model}"] = pred
        for name, mask in masks.items():
            m = mask.to_numpy()
            if m.sum() == 0:
                continue
            rows.append({"model": model, "settings": chosen_text(model), "subset": name, "n": int(m.sum()),
                         "n_flammable": int(test.loc[mask, "flammable"].sum()), **metrics(test.loc[mask, "flammable"], pred[m])})

    single = pd.read_csv(RESULTS_DIR / "val_single_feature.csv").iloc[0]["feature"]
    sc1 = StandardScaler().fit(train[[single]])
    lr = chosen["logreg"]
    est = LogisticRegression(C=lr["C"], class_weight=lr["class_weight"], max_iter=5000, random_state=SEED).fit(sc1.transform(train[[single]]), train["flammable"])
    pred1 = est.predict(sc1.transform(test[[single]]))
    preds["pred_single_feature"] = pred1
    for name, mask in masks.items():
        m = mask.to_numpy()
        if m.sum():
            rows.append({"model": "single_feature", "settings": single, "subset": name, "n": int(m.sum()),
                         "n_flammable": int(test.loc[mask, "flammable"].sum()), **metrics(test.loc[mask, "flammable"], pred1[m])})
    pd.DataFrame(rows).to_csv(OUT_GENERAL, index=False)

    val_cmp = pd.read_csv(RESULTS_DIR / "val_specialist_vs_general.csv")
    train_masks = subset_masks(train)
    comp = []
    for _, r in val_cmp.iterrows():
        name, model = r["subset"], r["model"]
        tr = train[train_masks[name]]
        sc = StandardScaler().fit(tr[FEATURES])
        est = build(model, parse_settings(r["specialist_settings"])).fit(sc.transform(tr[FEATURES]), tr["flammable"])
        mask = masks[name]
        te = test[mask]
        spec = metrics(te["flammable"], est.predict(sc.transform(te[FEATURES])))
        gen_pred = general_pred[model][mask.to_numpy()]
        gen = metrics(te["flammable"], gen_pred)
        comp.append({"subset": name, "model": model, "n_train": len(tr), "n_test": len(te), "test_flammable": int(te["flammable"].sum()),
                     **{f"spec_{k}": spec[k] for k in ("recall", "precision", "f1", "tn", "fp", "fn", "tp")},
                     **{f"gen_{k}": gen[k] for k in ("recall", "precision", "f1", "tn", "fp", "fn", "tp")},
                     "f1_gain": spec["f1"] - gen["f1"], "recall_gain": spec["recall"] - gen["recall"]})
    pd.DataFrame(comp).to_csv(OUT_SPECIALIST, index=False)
    preds.to_csv(OUT_PREDICTIONS, index=False)
    show = pd.DataFrame(rows).query("subset=='all'")[["model", "recall", "precision", "f1", "tn", "fp", "fn", "tp"]]
    print(show.round(3).to_string(index=False))
    print(pd.DataFrame(comp)[["subset", "model", "n_test", "test_flammable", "spec_recall", "gen_recall", "spec_f1", "gen_f1", "f1_gain"]].round(3).to_string(index=False))


def chosen_text(model):
    return pd.read_csv(RESULTS_DIR / "chosen_settings_general.csv").set_index("model").loc[model, "settings"]


if __name__ == "__main__":
    sys.exit(main())
