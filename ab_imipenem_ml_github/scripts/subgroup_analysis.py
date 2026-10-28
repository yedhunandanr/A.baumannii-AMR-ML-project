#!/usr/bin/env python3

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os


"""
Subgroup analysis: carbapenemase-positive vs non-enzymatic resistant isolates.

Compares performance of all 5 feature sets on:
    Group A: Carbapenemase-positive resistant isolates (OXA-23, OXA-58, NDM etc.)
    Group B: Non-enzymatic resistant isolates (no detected carbapenemase)
    Group C: Susceptible isolates (used as negative class in both comparisons)

Hypothesis:
    - Gene-based models should perform well on Group A (genes present)
    - Gene-based models should struggle on Group B (no target genes)
    - Distance models may partially capture Group B if they belong to resistant lineages

Output:
    subgroup_results.tsv

Usage:
    python subgroup_analysis.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
FEATURE_MATRIX  = str(PROJECT_ROOT / "outputs/final_feature_matrix.tsv")
SPLIT_FILE      = str(PROJECT_ROOT / "outputs/ml_train_test_split.tsv")
NON_ENZ_FILE    = str(PROJECT_ROOT / "outputs/non_enzymatic_resistant.tsv")
OUTPUT          = str(PROJECT_ROOT / "outputs/subgroup_results.tsv")

RANDOM_STATE = 42

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FEATURE_MATRIX, sep="\t")
df["genome_id"] = df["genome_id"].astype(str)
df["ST"] = df["ST"].astype(str)

split = pd.read_csv(SPLIT_FILE, sep="\t")
split["genome_id"] = split["genome_id"].astype(str)
test_ids = set(split[split["split"] == "test"]["genome_id"].tolist())
train_ids = set(split[split["split"] == "train"]["genome_id"].tolist())

ne = pd.read_csv(NON_ENZ_FILE, sep="\t")
ne["genome_id"] = ne["genome_id"].astype(str)
non_enzymatic_ids = set(ne["genome_id"].tolist())

meta_cols = ["genome_id", "measurement_value", "phenotype",
             "ST", "distance_universal", "distance_lineage"]
amr_cols = [c for c in df.columns if c not in meta_cols]

# Carbapenemase genes
carb_genes = ["blaOXA-23", "blaOXA-58", "blaOXA-24", "blaOXA-402",
              "blaOXA-407", "blaOXA-235", "blaOXA-237", "blaNDM",
              "blaNDM-1", "blaVIM", "blaVIM-2"]
carb_genes = [g for g in carb_genes if g in df.columns]

# Label subgroups
df["any_carbapenemase"] = df[carb_genes].sum(axis=1) > 0

df["subgroup"] = "susceptible"
df.loc[(df["phenotype"] == 1) & (df["any_carbapenemase"]), "subgroup"] = "carb_positive"
df.loc[(df["phenotype"] == 1) & (~df["any_carbapenemase"]), "subgroup"] = "non_enzymatic"

print(f"\nSubgroup counts (full dataset):")
print(df["subgroup"].value_counts().to_string())
print(f"\nNon-enzymatic IDs from earlier analysis: {len(non_enzymatic_ids)}")

# ----------------------------------------------------------------
# Feature sets
# ----------------------------------------------------------------
lin_mask = df["distance_lineage"].notna()

FEATURE_SETS = {
    "1_universal_distance": {
        "mask": pd.Series([True] * len(df)),
        "features": ["distance_universal"],
    },
    "2_lineage_distance": {
        "mask": lin_mask,
        "features": ["distance_lineage"],
    },
    "3_amr_genes": {
        "mask": pd.Series([True] * len(df)),
        "features": amr_cols,
    },
    "4_universal_plus_amr": {
        "mask": pd.Series([True] * len(df)),
        "features": ["distance_universal"] + amr_cols,
    },
    "5_lineage_plus_amr": {
        "mask": lin_mask,
        "features": ["distance_lineage"] + amr_cols,
    },
}

# ----------------------------------------------------------------
# XGBoost model
# ----------------------------------------------------------------
def make_xgb(scale_pos_weight):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )

def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return {"AUC": np.nan, "F1": np.nan,
                "Accuracy": np.nan, "MCC": np.nan}
    return {
        "AUC":      round(roc_auc_score(y_true, y_prob), 4),
        "F1":       round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "MCC":      round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ----------------------------------------------------------------
# Subgroup evaluation
# ----------------------------------------------------------------
results = []

for fs_name, fs_config in FEATURE_SETS.items():
    features = fs_config["features"]
    mask     = fs_config["mask"].values if hasattr(fs_config["mask"], "values") \
               else np.array(fs_config["mask"])

    data = df[mask].copy().reset_index(drop=True)
    print(f"\n{'='*60}")
    print(f"Feature set: {fs_name} ({len(features)} features, {len(data)} genomes)")

    # ----------------------------------------------------------------
    # Train on training set (all resistant vs all susceptible)
    # ----------------------------------------------------------------
    train_data = data[data["genome_id"].isin(train_ids)].copy()
    test_data  = data[data["genome_id"].isin(test_ids)].copy()

    X_train = train_data[features].fillna(0).values
    y_train = train_data["phenotype"].values
    scale   = (y_train == 0).sum() / max(y_train.sum(), 1)

    model = make_xgb(scale)
    model.fit(X_train, y_train)

    # ----------------------------------------------------------------
    # Evaluate on test set — split by subgroup
    # ----------------------------------------------------------------
    for subgroup_label, subgroup_name in [
        ("carb_positive", "Carbapenemase-positive resistant"),
        ("non_enzymatic", "Non-enzymatic resistant"),
    ]:
        # Test: subgroup resistant + all susceptible
        sub_test = test_data[
            (test_data["subgroup"] == subgroup_label) |
            (test_data["subgroup"] == "susceptible")
        ].copy()

        if len(sub_test) == 0:
            print(f"  {subgroup_name}: no test samples")
            continue

        n_pos = (sub_test["phenotype"] == 1).sum()
        n_neg = (sub_test["phenotype"] == 0).sum()
        print(f"\n  {subgroup_name}: {n_pos} resistant, {n_neg} susceptible")

        if n_pos == 0 or n_neg == 0:
            print(f"  Skipping — need both classes")
            continue

        X_sub = sub_test[features].fillna(0).values
        y_sub = sub_test["phenotype"].values

        y_pred = model.predict(X_sub)
        y_prob = model.predict_proba(X_sub)[:, 1]

        metrics = compute_metrics(y_sub, y_pred, y_prob)
        print(f"  AUC={metrics['AUC']}, F1={metrics['F1']}, MCC={metrics['MCC']}")

        results.append({
            "feature_set": fs_name,
            "subgroup":    subgroup_label,
            "subgroup_name": subgroup_name,
            "n_resistant": int(n_pos),
            "n_susceptible": int(n_neg),
            **metrics
        })

    # ----------------------------------------------------------------
    # Also evaluate on all resistant vs susceptible (baseline)
    # ----------------------------------------------------------------
    n_pos = (test_data["phenotype"] == 1).sum()
    n_neg = (test_data["phenotype"] == 0).sum()
    X_test = test_data[features].fillna(0).values
    y_test = test_data["phenotype"].values
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print(f"\n  All resistant vs susceptible: AUC={metrics['AUC']}, "
          f"F1={metrics['F1']}, MCC={metrics['MCC']}")
    results.append({
        "feature_set": fs_name,
        "subgroup":    "all_resistant",
        "subgroup_name": "All resistant vs susceptible",
        "n_resistant": int(n_pos),
        "n_susceptible": int(n_neg),
        **metrics
    })

# ----------------------------------------------------------------
# Save and display results
# ----------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT, sep="\t", index=False)

print(f"\n\n{'='*70}")
print("SUBGROUP ANALYSIS SUMMARY")
print("="*70)

summary = results_df.pivot_table(
    index="feature_set",
    columns="subgroup",
    values="AUC"
).round(4)

print(summary.to_string())

print(f"\n\nKey comparison — AUC on carbapenemase-positive vs non-enzymatic:")
print(f"{'Feature Set':<35} {'Carb+':>8} {'Non-Enz':>10} {'All':>8} {'Gap':>8}")
print("-"*65)
for fs in results_df["feature_set"].unique():
    fs_data = results_df[results_df["feature_set"] == fs]
    carb = fs_data[fs_data["subgroup"]=="carb_positive"]["AUC"].values
    nenz = fs_data[fs_data["subgroup"]=="non_enzymatic"]["AUC"].values
    alla = fs_data[fs_data["subgroup"]=="all_resistant"]["AUC"].values
    if len(carb) and len(nenz) and len(alla):
        gap = carb[0] - nenz[0]
        print(f"{fs:<35} {carb[0]:>8.4f} {nenz[0]:>10.4f} "
              f"{alla[0]:>8.4f} {gap:>+8.4f}")

print(f"\nResults saved to: {OUTPUT}")
