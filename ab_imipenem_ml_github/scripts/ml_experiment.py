#!/usr/bin/env python3

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os


"""
ML Experiment: Decomposing imipenem resistance prediction accuracy
into clonal background vs causal mechanism signal.

Five feature sets x Two evaluation strategies = 10 comparisons

Feature sets:
    1. Universal distance only
    2. Lineage distance only (577 genomes with lineage-matched reference)
    3. AMR genes only
    4. Universal distance + AMR genes
    5. Lineage distance + AMR genes (577 genomes)

Evaluation strategies:
    A. Random 5-fold cross-validation (standard)
    B. Clade-held-out cross-validation (ST-based)

Models: Random Forest, XGBoost, Logistic Regression

Output: ml_results.tsv, ml_results_summary.tsv

Usage:
    python ml_experiments.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, matthews_corrcoef)
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("XGBoost not installed — skipping XGB models")
    HAS_XGB = False

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
FEATURE_MATRIX = str(PROJECT_ROOT / "outputs/final_feature_matrix.tsv")
OUTPUT_RESULTS = str(PROJECT_ROOT / "outputs/ml_results.tsv")
OUTPUT_SUMMARY = str(PROJECT_ROOT / "outputs/ml_results_summary.tsv")

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
print("Loading feature matrix...")
df = pd.read_csv(FEATURE_MATRIX, sep="\t")
df["genome_id"] = df["genome_id"].astype(str)
df["ST"] = df["ST"].astype(str)

# AMR gene columns (everything after the first 6 metadata columns)
meta_cols = ["genome_id", "measurement_value", "phenotype",
             "ST", "distance_universal", "distance_lineage"]
amr_cols = [c for c in df.columns if c not in meta_cols]

print(f"Total genomes: {len(df)}")
print(f"Resistant: {df['phenotype'].sum()}")
print(f"Susceptible: {(df['phenotype']==0).sum()}")
print(f"AMR gene features: {len(amr_cols)}")

# ----------------------------------------------------------------
# Define feature sets
# ----------------------------------------------------------------
# Full dataset (770 genomes)
df_full = df.copy()

# Lineage subset (577 genomes with lineage distance)
df_lin = df[df["distance_lineage"].notna()].copy()

FEATURE_SETS = {
    "1_universal_distance_only": {
        "data": df_full,
        "features": ["distance_universal"],
        "description": "Universal phylogenetic distance only"
    },
    "2_lineage_distance_only": {
        "data": df_lin,
        "features": ["distance_lineage"],
        "description": "Lineage-matched distance only (ST1/2/3/25/32)"
    },
    "3_amr_genes_only": {
        "data": df_full,
        "features": amr_cols,
        "description": "AMR gene presence/absence only"
    },
    "4_universal_distance_plus_amr": {
        "data": df_full,
        "features": ["distance_universal"] + amr_cols,
        "description": "Universal distance + AMR genes"
    },
    "5_lineage_distance_plus_amr": {
        "data": df_lin,
        "features": ["distance_lineage"] + amr_cols,
        "description": "Lineage distance + AMR genes (ST1/2/3/25/32)"
    },
}

# ----------------------------------------------------------------
# Define models
# ----------------------------------------------------------------
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),
}

if HAS_XGB:
    MODELS["XGBoost"] = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=(df["phenotype"]==0).sum() / df["phenotype"].sum(),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )

# ----------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ----------------------------------------------------------------
# Cross-validation strategies
# ----------------------------------------------------------------
def random_cv(X, y, model, n_splits=5):
    """Standard stratified k-fold CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return compute_metrics(y, y_pred, y_prob)


def clade_held_out_cv(X, y, sts, model):
    """
    Clade-held-out CV: hold out each ST as test set in turn.
    Only STs with both R and S isolates and >= 5 isolates are used.
    """
    unique_sts = [st for st in sts.unique()
                  if st != "-"
                  and (sts == st).sum() >= 5
                  and y[sts == st].nunique() == 2]

    if len(unique_sts) < 2:
        print("  WARNING: Not enough valid clades for clade-held-out CV")
        return None

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for st in unique_sts:
        test_mask  = (sts == st).values
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(y_train.unique()) < 2:
            continue

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())
        except Exception as e:
            print(f"  WARNING: Failed for ST{st}: {e}")
            continue

    if len(all_y_true) == 0:
        return None

    return compute_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob)
    )

# ----------------------------------------------------------------
# Run all experiments
# ----------------------------------------------------------------
print("\nRunning experiments...")
print("=" * 70)

results = []

for fs_name, fs_config in FEATURE_SETS.items():
    data     = fs_config["data"].copy()
    features = fs_config["features"]
    desc     = fs_config["description"]

    X = data[features].fillna(0).values
    y = data["phenotype"]
    sts = data["ST"]

    print(f"\nFeature Set: {fs_name}")
    print(f"  {desc}")
    print(f"  Genomes: {len(data)}, Features: {len(features)}")
    print(f"  Resistant: {y.sum()}, Susceptible: {(y==0).sum()}")

    for model_name, model in MODELS.items():
        print(f"\n  Model: {model_name}")

        # Random CV
        print(f"    Running random 5-fold CV...")
        try:
            metrics_random = random_cv(X, y, model)
            print(f"    AUC={metrics_random['AUC']}, F1={metrics_random['F1']}, "
                  f"MCC={metrics_random['MCC']}")
            results.append({
                "feature_set":    fs_name,
                "description":    desc,
                "model":          model_name,
                "cv_strategy":    "random_5fold",
                "n_genomes":      len(data),
                "n_features":     len(features),
                **metrics_random
            })
        except Exception as e:
            print(f"    ERROR: {e}")

        # Clade-held-out CV
        print(f"    Running clade-held-out CV...")
        try:
            metrics_clade = clade_held_out_cv(
                pd.DataFrame(X, columns=features if len(features) > 1
                             else features),
                y.reset_index(drop=True),
                sts.reset_index(drop=True),
                model
            )
            if metrics_clade:
                print(f"    AUC={metrics_clade['AUC']}, F1={metrics_clade['F1']}, "
                      f"MCC={metrics_clade['MCC']}")
                results.append({
                    "feature_set":    fs_name,
                    "description":    desc,
                    "model":          model_name,
                    "cv_strategy":    "clade_held_out",
                    "n_genomes":      len(data),
                    "n_features":     len(features),
                    **metrics_clade
                })
        except Exception as e:
            print(f"    ERROR: {e}")

# ----------------------------------------------------------------
# Save results
# ----------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_RESULTS, sep="\t", index=False)
print(f"\n\nAll results saved to: {OUTPUT_RESULTS}")

# ----------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

summary = results_df.pivot_table(
    index=["feature_set", "model"],
    columns="cv_strategy",
    values=["AUC", "F1", "MCC"]
).round(4)

print(summary.to_string())
summary.to_csv(OUTPUT_SUMMARY, sep="\t")
print(f"\nSummary saved to: {OUTPUT_SUMMARY}")

# ----------------------------------------------------------------
# Key comparison — the main finding
# ----------------------------------------------------------------
print("\n" + "=" * 70)
print("KEY FINDING: Random CV vs Clade-Held-Out Performance Drop")
print("=" * 70)

for fs_name in FEATURE_SETS.keys():
    fs_results = results_df[results_df["feature_set"] == fs_name]
    for model_name in MODELS.keys():
        m_results = fs_results[fs_results["model"] == model_name]
        random = m_results[m_results["cv_strategy"] == "random_5fold"]["AUC"].values
        clade  = m_results[m_results["cv_strategy"] == "clade_held_out"]["AUC"].values
        if len(random) > 0 and len(clade) > 0:
            drop = random[0] - clade[0]
            print(f"  {fs_name} | {model_name:20s} | "
                  f"Random AUC: {random[0]:.4f} | "
                  f"Clade AUC: {clade[0]:.4f} | "
                  f"Drop: {drop:+.4f}")
