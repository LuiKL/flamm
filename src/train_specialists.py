import itertools
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FEATURES, SEED, chosen_params, load_splits, metrics

RESULTS_DIR = Path("results")
MW_CUTOFFS = [150, 175, 200]
CLASS_WEIGHTS = [None, "balanced"]


def subset_masks(df):
    masks = {
        "acid": df["is_acid"] == 1,
        "metallic": df["is_metallic"] == 1,
        "silicon": df["is_silicon"] == 1,
    }
    for cutoff in MW_CUTOFFS:
        masks[f"mw>{cutoff}"] = df["MolWt"] > cutoff
    return masks


def candidates():
    for c, cw in itertools.product([0.001, 0.01, 0.1, 1, 10, 100, 1000], CLASS_WEIGHTS):
        yield "logreg", f"C={c}, class_weight={cw}", lambda c=c, cw=cw: LogisticRegression(C=c, class_weight=cw, max_iter=5000, random_state=SEED)
    for c, gamma, cw in itertools.product([0.1, 1, 10, 100], [0.01, 0.1, 1], CLASS_WEIGHTS):
        yield "svm_rbf", f"C={c}, gamma={gamma}, class_weight={cw}", lambda c=c, gamma=gamma, cw=cw: SVC(kernel="rbf", C=c, gamma=gamma, class_weight=cw, random_state=SEED)


def general_models():
    lr = chosen_params("logreg")
    sv = chosen_params("svm_rbf")
    return {
        "logreg": LogisticRegression(C=lr["C"], class_weight=lr["class_weight"], max_iter=5000, random_state=SEED),
        "svm_rbf": SVC(kernel="rbf", C=sv["C"], gamma=sv["gamma"], class_weight=sv["class_weight"], random_state=SEED),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    general_scaler = StandardScaler().fit(train[FEATURES])
    Xg_train, Xg_val = general_scaler.transform(train[FEATURES]), general_scaler.transform(val[FEATURES])
    general = general_models()
    for model in general.values():
        model.fit(Xg_train, train["flammable"])

    train_masks, val_masks = subset_masks(train), subset_masks(val)
    tried, compare = [], []
    for name in train_masks:
        tr, va = train[train_masks[name]], val[val_masks[name]]
        scaler = StandardScaler().fit(tr[FEATURES])
        X_tr, X_va = scaler.transform(tr[FEATURES]), scaler.transform(va[FEATURES])
        info = {"subset": name, "n_train": len(tr), "n_val": len(va), "val_pos": int(va["flammable"].sum())}
        best = {}
        for model_name, settings, make in candidates():
            est = make().fit(X_tr, tr["flammable"])
            row = {**info, "model": model_name, "settings": settings, **metrics(va["flammable"], est.predict(X_va))}
            tried.append(row)
            if model_name not in best or row["f1"] > best[model_name]["f1"]:
                best[model_name] = row
            print(name, model_name, settings, f"F1={row['f1']:.3f} recall={row['recall']:.3f}", flush=True)
        for model_name, row in best.items():
            gen = metrics(va["flammable"], general[model_name].predict(Xg_val[val_masks[name].to_numpy()]))
            compare.append({**info, "model": model_name, "specialist_settings": row["settings"],
                            **{f"spec_{k}": row[k] for k in ("recall", "precision", "f1")},
                            **{f"gen_{k}": gen[k] for k in ("recall", "precision", "f1")},
                            "f1_gain": row["f1"] - gen["f1"], "recall_gain": row["recall"] - gen["recall"]})
    pd.DataFrame(tried).to_csv(RESULTS_DIR / "val_results_specialists.csv", index=False)
    comparison = pd.DataFrame(compare)
    comparison.to_csv(RESULTS_DIR / "val_specialist_vs_general.csv", index=False)
    show = ["subset", "model", "n_train", "n_val", "val_pos", "spec_recall", "spec_f1", "gen_recall", "gen_f1", "f1_gain"]
    print(comparison[show].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
