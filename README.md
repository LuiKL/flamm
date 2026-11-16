# Flammability classifier audit

CS 74 term project (Track 2, ML audit). Question: can a classifier reliably tell whether a chemical is flammable, and for which kinds of chemicals does it fail or need its own specialist model?

Task: binary classification from RDKit features. Flammable = flash point <= 333.15 K (60 C, the GHS cutoff).

Data: Sun et al. 2020, "Assessing Graph-Based Deep Learning Models for Predicting Flash Point", Molecular Informatics 39(6), 1900101 (https://doi.org/10.1002/minf.201900101). File `integrated_dataset_public.csv` from https://github.com/uw-cmg/MoleProp (MIT License).

## Setup

```
conda env create -f environment.yml
conda activate flamm
```

Run every script from the project root. Do not use `python -I`; the scripts import `src/common.py`.

## Pipeline

Run in this order.

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `src/prepare_data.py` | Merges duplicate measurements (drops a compound if the std of its flash points is above 5 K), labels, computes 17 RDKit features, clusters similar molecules (Butina, Tanimoto >= 0.7) and splits 70/15/15 by whole clusters, stratified by label and subgroup | `data/processed/{train,val,test}.csv`, `cleaning_report.txt` |
| 2 | `src/train_general.py` | Tunes majority, logistic regression, perceptron, linear SVM and RBF SVM on validation; single-feature baseline and feature-scaling curve | `results/val_results_general.csv`, `chosen_settings_general.csv`, `val_single_feature.csv`, `val_feature_scaling.csv` |
| 3 | `src/train_specialists.py` | Trains specialists on one subset each (acid, metallic, silicon, MW > 150/175/200) and compares them with the general models on validation | `results/val_results_specialists.csv`, `val_specialist_vs_general.csv` |
| 4 | `src/leakage_check.py` | Compares the cluster split with 5 random splits (near-duplicate share and RBF SVM F1) | `results/leakage_check.csv` |
| 5 | `src/test_eval.py` | Evaluates all models once on the test set, using the settings chosen on validation. Refuses to run again if its outputs exist | `results/test_results_general.csv`, `test_specialist_vs_general.csv`, `test_predictions.csv` |
| 6 | `src/audit.py` | Subgroup recall, error review, feature importance | `results/test_subgroup_recall.csv`, `test_errors_svm_rbf.csv`, `feature_importance.csv` |
| 7 | `src/make_figures.py` | Report figures | `figures/*.png` |

`src/common.py` holds the shared loading, scaling (StandardScaler fitted on train only) and metric code. All random seeds are 74.

## Rules followed

- Model selection: best F1 on the validation set. Recall, precision and the confusion matrix are reported for every setting tried.
- Recall on the flammable class is the main metric for the audit.
- The test set is used once, after all settings were fixed. Permutation importance is computed on the validation set.
- The `is_silicon`, `is_metallic`, `is_tin` and `is_acid` columns are audit tags only and are never used as model features. `is_acid` was set from the compound name, not the structure.

## Layout

```
data/raw/          downloaded CSV
data/processed/    cleaned, featurized, split CSVs and cleaning report
src/               scripts
results/           validation and test result tables
figures/           report figures
```
