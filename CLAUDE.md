# CS 74 Project: Flammability Classifier Audit

Dartmouth CS 74 (machine learning) term project, Track 2 (ML audit).
Team: Lui Kamoshita-Langan and Timothy Walters. Final report due Nov 20, 2026.

## Research question
Can a classifier reliably tell whether a chemical is flammable, and for which kinds of chemicals does it fail or need its own specialist model?

## Task
Binary classification. Positive class = flammable = flash point at or below 60 °C (333.15 K, the GHS cutoff). Negative class = not flammable.

## Dataset
- Source: Sun et al. 2020, "Assessing Graph-Based Deep Learning Models for Predicting Flash Point," Molecular Informatics 39(6), 1900101. https://doi.org/10.1002/minf.201900101
- File: `data_engineering/integrated_dataset_public.csv` in https://github.com/uw-cmg/MoleProp (MIT License). Download it into `data/raw/`.
- 16,366 rows, 9 columns: compound, flashpoint (Kelvin), pure substance, smiles, source, is_silicon, is_metallic, is_tin, is_acid.
- The is_* columns are subgroup tags for the audit. Do not use them as model features. is_acid was set by checking the compound name for "acid", not the structure.

## Cleaning (follows Sun et al.)
1. Group rows by SMILES.
2. If a compound's duplicate flash points have standard deviation above 5 K, drop that compound entirely.
3. Otherwise keep one row with the mean flash point.
4. Expected result: about 10,062 unique compounds. 552 dropped for conflicting measurements.
5. Label: flammable = 1 if flash point <= 333.15 K, else 0. Expected balance: about 32% flammable, 68% not.

## Features
Generate with RDKit from SMILES (about 15):
MolWt, HeavyAtomCount, NumHDonors, NumHAcceptors, TPSA, MolLogP, NumRotatableBonds, RingCount, NumAromaticRings, FractionCSP3, and counts of C, O, N, and halogens (F, Cl, Br, I).
Drop any molecule RDKit cannot parse and report how many.
Scale features with StandardScaler fitted on the training set only.

## Split
- 70/15/15 train/validation/test, stratified so the label and each subgroup (acid, metallic, silicon) appear in all three.
- Fixed random seed so both teammates use identical splits.
- Save `train.csv`, `val.csv`, `test.csv` to `data/processed/` and commit them so both teammates share the same files.
- Do not evaluate anything on the test set until Week 4. All tuning uses validation.

## Models
- Majority-class baseline (always predict not flammable).
- General models trained on all compound types: logistic regression, perceptron, SVM.
- Specialist models (Week 3), each trained only on one class: acids, metallic, silicon, and molecules above a molecular weight cutoff (about 150 to 200 g/mol). Tin has only 44 compounds, too few to train on.
- Specialists and general models are compared on the same test-set subset for that class.

## Model selection rule (shared by both teammates)
Recall on the flammable class is the main metric, but do not select on recall alone, since predicting everything flammable gives recall 1.0. Select the setting with the best F1 on the validation set. Also report recall, precision, and the confusion matrix for every setting tried, including poor ones.

## This week (Week 2: Oct 23 to 29)
First, Week 1 setup, which was not done yet: download, clean, label, featurize, split.
Then:
- Lui: logistic regression (tune C, with and without class_weight="balanced") and SVM (linear and RBF kernels, tune C and gamma).
- Tim: perceptron (tune alpha/penalty, class_weight) and the majority baseline.
Output: validation results table (CSV) for every setting tried, and the chosen settings per model.

## Later weeks
- Week 3: specialist models.
- Week 4: run all models once on the test set. Audit: subgroup recall and F1, general vs. specialist, review of mislabeled compounds, feature importance (model coefficients plus permutation importance).
- Week 5: report (5 to 7 pages), demo, cleanup.

## Suggested layout
```
data/raw/          downloaded CSV
data/processed/    cleaned, featurized, split CSVs
src/               scripts (prepare_data.py, train_logreg.py, train_svm.py, ...)
results/           validation and test results tables
```

## Code style
- Python, pandas, RDKit, scikit-learn.
- No comments in code.
- Fix random seeds everywhere for reproducibility.
