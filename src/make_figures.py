import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT = Path("figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
SUBSET_LABELS = {"acid": "Acid", "metallic": "Metallic", "silicon": "Silicon", "mw>150": "MW > 150", "mw>175": "MW > 175", "mw>200": "MW > 200"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scaling_curve():
    d = pd.read_csv(RESULTS / "val_feature_scaling.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for model, color, label in (("svm_rbf", ORANGE, "RBF SVM"), ("logreg", BLUE, "Logistic regression")):
        s = d[d["model"] == model].sort_values("n_features")
        ax.plot(s["n_features"], s["f1"], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    lr = d[d["model"] == "logreg"].set_index("n_features")["f1"]
    sv = d[d["model"] == "svm_rbf"].set_index("n_features")["f1"]
    ax.annotate(f"1 feature (HeavyAtomCount): {lr[1]:.3f}", (1, lr[1]), xytext=(2.2, 0.62), color=INK2, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRID))
    ax.annotate(f"peak {lr.max():.3f}\n(14 features)", (int(lr.idxmax()), lr.max()), xytext=(10.6, 0.915), color=BLUE, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRID))
    ax.annotate(f"17 features: {lr[17]:.3f}", (17, lr[17]), xytext=(13.6, 0.79), color=BLUE, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRID))
    ax.annotate(f"{sv[17]:.3f}", (17, sv[17]), xytext=(17.35, sv[17] + 0.004), color=ORANGE, fontsize=8.5)
    ax.set_xlim(0.5, 18.3)
    ax.set_ylim(0.6, 0.95)
    ax.set_xticks(range(1, 18, 2))
    ax.set_xlabel("Number of features (added greedily by validation F1)")
    ax.set_ylabel("Validation F1")
    ax.set_title("F1 as features are added")
    ax.legend(loc="lower right")
    save(fig, "feature_scaling_curve.png")


def subgroup_bars():
    d = pd.read_csv(RESULTS / "test_subgroup_recall.csv")
    d = d[(d["model"] == "svm_rbf") & (d["group"] != "tin")].copy()
    d["f1"] = 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"])
    order = ["no_tag", "silicon", "metallic", "acid"]
    names = {"no_tag": "No tag", "silicon": "Silicon", "metallic": "Metallic", "acid": "Acid"}
    d = d.set_index("group").loc[order]
    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for off, col, color, label in ((-w / 2, "recall", BLUE, "Recall"), (w / 2, "f1", ORANGE, "F1")):
        bars = ax.bar(x + off, d[col], w - 0.04, color=color, label=label)
        for b, v in zip(bars, d[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{names[g]}\nn={int(d.loc[g, 'n'])} ({int(d.loc[g, 'n_flammable'])} flam.)" for g in order], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test score (flammable class)")
    ax.set_title("General RBF SVM by compound group")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    save(fig, "subgroup_recall_f1.png")


def specialist_bars():
    d = pd.read_csv(RESULTS / "test_specialist_vs_general.csv")
    d = d[d["model"] == "svm_rbf"].reset_index(drop=True)
    x = np.arange(len(d))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), sharey=True)
    for ax, metric, title in ((axes[0], "recall", "Recall"), (axes[1], "f1", "F1")):
        for off, prefix, color, label in ((-w / 2, "gen", GRAY, "General model"), (w / 2, "spec", BLUE, "Specialist")):
            vals = d[f"{prefix}_{metric}"]
            bars = ax.bar(x + off, vals, w - 0.04, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{SUBSET_LABELS[s]}\n{int(n)} ({int(p)})" for s, n, p in zip(d["subset"], d["n_test"], d["test_flammable"])], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test score on the subset")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8.5)
    fig.text(0.5, -0.08, "Subset: test compounds (flammable compounds). All models are RBF SVMs scored on the same test subset.", ha="center", fontsize=8.5, color=INK2)
    save(fig, "specialist_vs_general.png")


def importance_bars():
    d = pd.read_csv(RESULTS / "feature_importance.csv").sort_values("svm_perm_f1_drop")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bars = ax.barh(d["feature"], d["svm_perm_f1_drop"], color=BLUE, height=0.62, xerr=d["svm_perm_std"], error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2))
    for b, v, sd in zip(bars, d["svm_perm_f1_drop"], d["svm_perm_std"]):
        ax.text(v + sd + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, d["svm_perm_f1_drop"].max() * 1.22)
    ax.set_xlabel("Drop in validation F1 when the feature is shuffled")
    ax.set_title("Permutation importance, RBF SVM")
    ax.grid(axis="y", visible=False)
    save(fig, "feature_importance.png")


def main():
    scaling_curve()
    subgroup_bars()
    specialist_bars()
    importance_bars()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    sys.exit(main())
