import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC, LinearSVC

from common import FEATURES, SEED, load_splits, metrics, scaled

RESULTS_DIR = Path("results")
CLASS_WEIGHTS = [None, "balanced"]


def run(model_name, settings, estimator, X, y, Xv, yv):
    start = time.time()
    estimator.fit(X, y)
    row = {"model": model_name, "settings": settings, **metrics(yv, estimator.predict(Xv))}
    row["seconds"] = round(time.time() - start, 1)
    print(model_name, settings, f"F1={row['f1']:.3f} recall={row['recall']:.3f}", flush=True)
    return row


def majority(X, y, Xv, yv):
    return [run("majority", "always not flammable", DummyClassifier(strategy="constant", constant=0), X, y, Xv, yv)]


def logreg(X, y, Xv, yv):
    rows = []
    for c, cw in itertools.product([0.001, 0.01, 0.1, 1, 10, 100, 1000], CLASS_WEIGHTS):
        est = LogisticRegression(C=c, class_weight=cw, max_iter=5000, random_state=SEED)
        rows.append(run("logreg", f"C={c}, class_weight={cw}", est, X, y, Xv, yv))
    return rows


def perceptron(X, y, Xv, yv):
    rows = []
    for alpha, penalty, cw in itertools.product([1e-5, 1e-4, 1e-3, 1e-2], [None, "l2", "l1", "elasticnet"], CLASS_WEIGHTS):
        est = Perceptron(alpha=alpha, penalty=penalty, class_weight=cw, max_iter=2000, random_state=SEED)
        rows.append(run("perceptron", f"alpha={alpha}, penalty={penalty}, class_weight={cw}", est, X, y, Xv, yv))
    return rows


def svm_linear(X, y, Xv, yv):
    rows = []
    for c, cw in itertools.product([0.001, 0.01, 0.1, 1, 10], CLASS_WEIGHTS):
        est = LinearSVC(C=c, class_weight=cw, max_iter=20000, random_state=SEED)
        rows.append(run("svm_linear", f"C={c}, class_weight={cw}", est, X, y, Xv, yv))
    return rows


def svm_rbf(X, y, Xv, yv):
    rows = []
    for c, gamma, cw in itertools.product([0.1, 1, 10, 100], [0.001, 0.01, 0.1, 1], CLASS_WEIGHTS):
        est = SVC(kernel="rbf", C=c, gamma=gamma, class_weight=cw, random_state=SEED)
        rows.append(run("svm_rbf", f"C={c}, gamma={gamma}, class_weight={cw}", est, X, y, Xv, yv))
    return rows


def f1_for(make, splits, features):
    X = scaled(splits, features)
    est = make()
    est.fit(X["train"], splits["train"]["flammable"])
    return metrics(splits["val"]["flammable"], est.predict(X["val"]))


def feature_scaling(splits, chosen):
    makers = {
        "logreg": lambda: LogisticRegression(C=chosen["logreg"]["C"], class_weight=chosen["logreg"]["cw"], max_iter=5000, random_state=SEED),
        "svm_rbf": lambda: SVC(kernel="rbf", C=chosen["svm_rbf"]["C"], gamma=chosen["svm_rbf"]["gamma"], class_weight=chosen["svm_rbf"]["cw"], random_state=SEED),
    }
    single = [{"feature": f, **f1_for(makers["logreg"], splits, [f])} for f in FEATURES]
    single_df = pd.DataFrame(single).sort_values("f1", ascending=False)
    selected = [single_df.iloc[0]["feature"]]
    remaining = [f for f in FEATURES if f not in selected]
    order = [selected[0]]
    while remaining:
        scores = {f: f1_for(makers["logreg"], splits, selected + [f])["f1"] for f in remaining}
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.remove(best)
        order.append(best)
    rows = []
    for k in range(1, len(order) + 1):
        feats = order[:k]
        for name, make in makers.items():
            rows.append({"model": name, "n_features": k, "added": feats[-1], **f1_for(make, splits, feats)})
        print("features", k, feats[-1], flush=True)
    return single_df, pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    splits = load_splits()
    X = scaled(splits)
    y, yv = splits["train"]["flammable"], splits["val"]["flammable"]
    rows = []
    for fn in (majority, logreg, perceptron, svm_linear, svm_rbf):
        rows += fn(X["train"], y, X["val"], yv)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "val_results_general.csv", index=False)

    best = results.loc[results.groupby("model")["f1"].idxmax()].sort_values("f1", ascending=False)
    best.to_csv(RESULTS_DIR / "chosen_settings_general.csv", index=False)
    print(best[["model", "settings", "recall", "precision", "f1"]].to_string(index=False))

    def parse(model, key):
        s = best.loc[best["model"] == model, "settings"].iloc[0]
        return dict(p.split("=") for p in s.split(", "))[key]

    chosen = {}
    for m in ("logreg", "svm_rbf"):
        cw = parse(m, "class_weight")
        chosen[m] = {"C": float(parse(m, "C")), "cw": None if cw == "None" else cw}
    chosen["svm_rbf"]["gamma"] = float(parse("svm_rbf", "gamma"))
    single_df, curve = feature_scaling(splits, chosen)
    single_df.to_csv(RESULTS_DIR / "val_single_feature.csv", index=False)
    curve.to_csv(RESULTS_DIR / "val_feature_scaling.csv", index=False)
    print(single_df[["feature", "recall", "precision", "f1"]].to_string(index=False))
    print(curve.pivot(index="n_features", columns="model", values="f1").round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
