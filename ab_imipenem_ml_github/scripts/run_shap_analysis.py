#!/usr/bin/env python3
"""
SHAP analysis for imipenem resistance prediction models.

Runs SHAP on XGBoost models for all 5 feature sets.
Key output: feature importance showing relative contribution of
distance features vs AMR gene features.

Outputs:
    shap_values_fs{N}.tsv     — raw SHAP values per genome per feature
    shap_importance_fs{N}.tsv — mean |SHAP| per feature (importance ranking)
    shap_summary.tsv          — top features across all feature sets

Usage:
    python shap_analysis.py
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore")

import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
FEATURE_MATRIX = str(PROJECT_ROOT / "outputs/final_feature_matrix.tsv")
SPLIT_FILE     = str(PROJECT_ROOT / "outputs/ml_train_test_split.tsv")
OUT_DIR        = str(PROJECT_ROOT / "outputs/shap")

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FEATURE_MATRIX, sep="\t")
df["genome_id"] = df["genome_id"].astype(str)
df["ST"] = df["ST"].astype(str)

split = pd.read_csv(SPLIT_FILE, sep="\t")
split["genome_id"] = split["genome_id"].astype(str)

train_ids = set(split[split["split"] == "train"]["genome_id"].tolist())
test_ids  = set(split[split["split"] == "test"]["genome_id"].tolist())

meta_cols = ["genome_id", "measurement_value", "phenotype",
             "ST", "distance_universal", "distance_lineage"]
amr_cols = [c for c in df.columns if c not in meta_cols]

# Split data
train_df = df[df["genome_id"].isin(train_ids)].copy()
test_df  = df[df["genome_id"].isin(test_ids)].copy()

# Lineage subsets
train_lin = train_df[train_df["distance_lineage"].notna()].copy()
test_lin  = test_df[test_df["distance_lineage"].notna()].copy()

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Train lineage subset: {len(train_lin)}, Test lineage subset: {len(test_lin)}")

# ----------------------------------------------------------------
# Define feature sets
# ----------------------------------------------------------------
FEATURE_SETS = {
    "fs1_universal_distance": {
        "train": train_df,
        "test":  test_df,
        "features": ["distance_universal"],
        "description": "Universal distance only"
    },
    "fs2_lineage_distance": {
        "train": train_lin,
        "test":  test_lin,
        "features": ["distance_lineage"],
        "description": "Lineage distance only"
    },
    "fs3_amr_genes": {
        "train": train_df,
        "test":  test_df,
        "features": amr_cols,
        "description": "AMR genes only"
    },
    "fs4_universal_plus_amr": {
        "train": train_df,
        "test":  test_df,
        "features": ["distance_universal"] + amr_cols,
        "description": "Universal distance + AMR genes"
    },
    "fs5_lineage_plus_amr": {
        "train": train_lin,
        "test":  test_lin,
        "features": ["distance_lineage"] + amr_cols,
        "description": "Lineage distance + AMR genes"
    },
}

# ----------------------------------------------------------------
# XGBoost parameters (use same as best from tuning if available)
# ----------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "verbosity": 0
}

# ----------------------------------------------------------------
# Run SHAP for each feature set
# ----------------------------------------------------------------
all_importance = []

for fs_name, fs_config in FEATURE_SETS.items():
    print(f"\n{'='*60}")
    print(f"SHAP Analysis: {fs_name}")
    print(f"  {fs_config['description']}")

    features = fs_config["features"]
    train    = fs_config["train"]
    test     = fs_config["test"]

    X_train = train[features].fillna(0)
    y_train = train["phenotype"]
    X_test  = test[features].fillna(0)
    y_test  = test["phenotype"]

    # Scale pos weight for class imbalance
    scale = (y_train == 0).sum() / y_train.sum()

    # Train XGBoost
    print(f"  Training XGBoost (scale_pos_weight={scale:.2f})...")
    model = XGBClassifier(scale_pos_weight=scale, **XGB_PARAMS)
    model.fit(X_train, y_train)

    # Test AUC
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Test AUC: {auc:.4f}")

    # SHAP values
    print(f"  Computing SHAP values for {len(X_test)} test samples...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.insert(0, "genome_id", test[["genome_id"]].values.flatten())
    shap_df.insert(1, "phenotype", y_test.values)
    shap_df.to_csv(f"{OUT_DIR}/shap_values_{fs_name}.tsv", sep="\t", index=False)

    # Feature importance — mean absolute SHAP value
    importance = pd.DataFrame({
        "feature":    features,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap":     shap_values.mean(axis=0),
        "feature_set":   fs_name,
        "description":   fs_config["description"],
        "test_auc":      auc
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(f"{OUT_DIR}/shap_importance_{fs_name}.tsv",
                      sep="\t", index=False)

    all_importance.append(importance)

    # Print top 20 features
    print(f"\n  Top 20 features by mean |SHAP|:")
    print(f"  {'Feature':<35} {'Mean |SHAP|':>12} {'Mean SHAP':>12}")
    print(f"  {'-'*60}")
    for _, row in importance.head(20).iterrows():
        direction = "→ R" if row["mean_shap"] > 0 else "→ S"
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:>12.4f} "
              f"{row['mean_shap']:>12.4f} {direction}")

# ----------------------------------------------------------------
# Combined importance summary
# ----------------------------------------------------------------
print(f"\n{'='*60}")
print("COMBINED SHAP IMPORTANCE SUMMARY")
print("="*60)

combined = pd.concat(all_importance, ignore_index=True)
combined.to_csv(f"{OUT_DIR}/shap_importance_all.tsv", sep="\t", index=False)

# Focus on Feature Set 5 — the key combined model
fs5 = all_importance[4]  # fs5_lineage_plus_amr
print(f"\nFeature Set 5 (Lineage + AMR) — Top 30 features:")
print(f"{'Feature':<35} {'Mean |SHAP|':>12} {'Direction':>10}")
print("-"*60)
for _, row in fs5.head(30).iterrows():
    direction = "→ Resistant" if row["mean_shap"] > 0 else "→ Susceptible"
    print(f"{row['feature']:<35} {row['mean_abs_shap']:>12.4f}   {direction}")

# Distance feature rank in combined model
print(f"\nDistance feature rank in Feature Set 4 (Universal + AMR):")
fs4 = all_importance[3]
dist_rank = fs4[fs4["feature"] == "distance_universal"].index[0]
dist_row  = fs4[fs4["feature"] == "distance_universal"].iloc[0]
print(f"  distance_universal: rank {fs4.index.get_loc(dist_rank)+1} of {len(fs4)}, "
      f"mean |SHAP| = {dist_row['mean_abs_shap']:.4f}")

print(f"\nDistance feature rank in Feature Set 5 (Lineage + AMR):")
dist_rank5 = fs5[fs5["feature"] == "distance_lineage"].index[0]
dist_row5  = fs5[fs5["feature"] == "distance_lineage"].iloc[0]
print(f"  distance_lineage: rank {fs5.index.get_loc(dist_rank5)+1} of {len(fs5)}, "
      f"mean |SHAP| = {dist_row5['mean_abs_shap']:.4f}")

print(f"\nAll SHAP outputs saved to: {OUT_DIR}/")
