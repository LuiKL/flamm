import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FEATURES, SEED, chosen_params, load_splits

RESULTS_DIR = Path("results")
CUTOFF_K = 333.15
MODEL = "pred_svm_rbf"


def error_review():
    df = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    df["error"] = (df[MODEL] != df["flammable"]).astype(int)
    df["kind"] = np.where(df["error"] == 0, "correct", np.where(df["flammable"] == 1, "false_negative", "false_positive"))
    df["dist_to_cutoff_K"] = (df["flashpoint"] - CUTOFF_K).abs()
    df["margin_bin"] = pd.cut(df["dist_to_cutoff_K"], [-0.01, 5, 10, 20, 40, 1000], labels=["0-5", "5-10", "10-20", "20-40", ">40"])
    print("errors:", df["kind"].value_counts().to_dict())
    print("\nerror rate by distance of true flash point from the 333.15 K cutoff")
    print(df.groupby("margin_bin", observed=True)["error"].agg(["mean", "sum", "count"]).round(3).to_string())
    print("\nerror rate by subgroup")
    for tag in ("is_acid", "is_metallic", "is_silicon", "is_tin"):
        sub = df[df[tag] == 1]
        print(tag, len(sub), f"error {sub['error'].mean():.3f}", sub["kind"].value_counts().to_dict())
    other = df[(df[["is_acid", "is_metallic", "is_silicon", "is_tin"]].sum(axis=1) == 0)]
    print("none of the tags", len(other), f"error {other['error'].mean():.3f}")
    print("\nerror rate by number of repeat measurements")
    df["repeat"] = np.where(df["n_measurements"] > 1, "averaged", "single")
    print(df.groupby("repeat")["error"].agg(["mean", "sum", "count"]).round(3).to_string())
    errors = df[df["error"] == 1].sort_values("dist_to_cutoff_K", ascending=False)
    errors[["compound", "smiles", "flashpoint", "flammable", MODEL, "kind", "dist_to_cutoff_K", "is_acid", "is_metallic", "is_silicon", "is_tin", "n_measurements"]].to_csv(RESULTS_DIR / "test_errors_svm_rbf.csv", index=False)
    print("\nlargest errors (true flash point furthest from the cutoff)")
    print(errors[["compound", "flashpoint", "kind", "is_acid", "is_metallic", "is_silicon"]].head(15).to_string(index=False))


def importance():
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    scaler = StandardScaler().fit(train[FEATURES])
    X_train, X_val = scaler.transform(train[FEATURES]), scaler.transform(val[FEATURES])
    lr_p, sv_p = chosen_params("logreg"), chosen_params("svm_rbf")
    lr = LogisticRegression(C=lr_p["C"], class_weight=lr_p["class_weight"], max_iter=5000, random_state=SEED).fit(X_train, train["flammable"])
    sv = SVC(kernel="rbf", C=sv_p["C"], gamma=sv_p["gamma"], class_weight=sv_p["class_weight"], random_state=SEED).fit(X_train, train["flammable"])
    perm = permutation_importance(sv, X_val, val["flammable"], scoring="f1", n_repeats=10, random_state=SEED)
    out = pd.DataFrame({"feature": FEATURES, "logreg_coef": lr.coef_[0], "svm_perm_f1_drop": perm.importances_mean, "svm_perm_std": perm.importances_std})
    out["logreg_abs_coef"] = out["logreg_coef"].abs()
    out = out.sort_values("svm_perm_f1_drop", ascending=False)
    out.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print("\nfeature importance (permutation on validation set; coefficients on standardized features)")
    print(out[["feature", "logreg_coef", "svm_perm_f1_drop", "svm_perm_std"]].round(3).to_string(index=False))


def main():
    error_review()
    importance()


if __name__ == "__main__":
    sys.exit(main())
