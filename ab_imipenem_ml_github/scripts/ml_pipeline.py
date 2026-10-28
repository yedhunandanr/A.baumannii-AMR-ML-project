#!/usr/bin/env python3

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os


"""
Full ML Pipeline for imipenem resistance prediction decomposition study.
 
Steps:
    1. Stratified train/test split (80/20) by ST and phenotype
    2. Hyperparameter tuning via RandomizedSearchCV on training set
    3. Random 5-fold CV on training set
    4. Clade-held-out CV on training set
    5. Final evaluation on held-out test set
 
Five feature sets:
    1. Universal distance only
    2. Lineage distance only
    3. AMR genes only
    4. Universal distance + AMR genes
    5. Lineage distance + AMR genes
 
Output files:
    ml_train_test_split.tsv   — which genomes are in train/test
    ml_best_params.tsv        — tuned hyperparameters per feature set
    ml_results_full.tsv       — all CV and test results
    ml_results_summary.tsv    — summary table
 
Usage:
    python ml_full.py
"""
 
import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score,
                              matthews_corrcoef, confusion_matrix)
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
FEATURE_MATRIX  = str(PROJECT_ROOT / "outputs/final_feature_matrix.tsv")
OUT_SPLIT       = str(PROJECT_ROOT / "outputs/ml_train_test_split.tsv")
OUT_PARAMS      = str(PROJECT_ROOT / "outputs/ml_best_params.tsv")
OUT_RESULTS     = str(PROJECT_ROOT / "outputs/ml_results_full.tsv")
OUT_SUMMARY     = str(PROJECT_ROOT / "outputs/ml_results_summary.tsv")
 
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_CV_FOLDS   = 5
N_ITER_TUNE  = 30  # RandomizedSearchCV iterations
 
# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
print("Loading feature matrix...")
df = pd.read_csv(FEATURE_MATRIX, sep="\t")
df["genome_id"] = df["genome_id"].astype(str)
df["ST"] = df["ST"].astype(str)
df = df.reset_index(drop=True)
 
meta_cols = ["genome_id", "measurement_value", "phenotype",
             "ST", "distance_universal", "distance_lineage"]
amr_cols = [c for c in df.columns if c not in meta_cols]
 
print(f"Total genomes: {len(df)}")
print(f"Resistant: {df['phenotype'].sum()}")
print(f"Susceptible: {(df['phenotype']==0).sum()}")
print(f"AMR gene features: {len(amr_cols)}")
print(f"Unique STs: {df['ST'].nunique()}")
 
# ----------------------------------------------------------------
# Stratified train/test split by ST + phenotype
# ----------------------------------------------------------------
print("\nCreating stratified train/test split...")
 
# Create stratification label combining ST and phenotype
df["strat_label"] = df["ST"] + "_" + df["phenotype"].astype(str)
 
# For STs with very few samples combine into "other" group
strat_counts = df["strat_label"].value_counts()
rare_labels = strat_counts[strat_counts < 2].index
df["strat_label_safe"] = df["strat_label"].apply(
    lambda x: x if x not in rare_labels else "other_" + x.split("_")[-1]
)
 
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["strat_label_safe"]
)
 
train_df = df.loc[train_idx].copy().reset_index(drop=True)
test_df  = df.loc[test_idx].copy().reset_index(drop=True)
 
print(f"Training set: {len(train_df)} genomes "
      f"(R={train_df['phenotype'].sum()}, S={(train_df['phenotype']==0).sum()})")
print(f"Test set:     {len(test_df)} genomes "
      f"(R={test_df['phenotype'].sum()}, S={(test_df['phenotype']==0).sum()})")
 
# Save split
split_record = pd.concat([
    train_df[["genome_id","ST","phenotype"]].assign(split="train"),
    test_df[["genome_id","ST","phenotype"]].assign(split="test")
])
split_record.to_csv(OUT_SPLIT, sep="\t", index=False)
print(f"Split saved to: {OUT_SPLIT}")
 
# ----------------------------------------------------------------
# Define feature sets
# ----------------------------------------------------------------
# Full dataset splits
train_full = train_df
test_full  = test_df
 
# Lineage subset (only genomes with lineage distance)
train_lin = train_df[train_df["distance_lineage"].notna()].copy()
test_lin  = test_df[test_df["distance_lineage"].notna()].copy()
 
print(f"\nLineage subset - Train: {len(train_lin)}, Test: {len(test_lin)}")
 
FEATURE_SETS = {
    "1_universal_distance_only": {
        "train": train_full,
        "test":  test_full,
        "features": ["distance_universal"],
        "description": "Universal phylogenetic distance only"
    },
    "2_lineage_distance_only": {
        "train": train_lin,
        "test":  test_lin,
        "features": ["distance_lineage"],
        "description": "Lineage-matched distance only (ST1/2/3/25/32)"
    },
    "3_amr_genes_only": {
        "train": train_full,
        "test":  test_full,
        "features": amr_cols,
        "description": "AMR gene presence/absence only"
    },
    "4_universal_distance_plus_amr": {
        "train": train_full,
        "test":  test_full,
        "features": ["distance_universal"] + amr_cols,
        "description": "Universal distance + AMR genes"
    },
    "5_lineage_distance_plus_amr": {
        "train": train_lin,
        "test":  test_lin,
        "features": ["distance_lineage"] + amr_cols,
        "description": "Lineage distance + AMR genes (ST1/2/3/25/32)"
    },
}
 
# ----------------------------------------------------------------
# Define models and hyperparameter search spaces
# ----------------------------------------------------------------
def make_models():
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 10, 20, 30],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2"],
            }
        },
        "LogisticRegression": {
            "model": Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "saga"]
            }
        },
    }
 
    if HAS_XGB:
        models["XGBoost"] = {
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0
            ),
            "params": {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        }
 
    return models
# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
        "Specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
 
# ----------------------------------------------------------------
# CV strategies
# ----------------------------------------------------------------
def random_cv(X, y, model, n_splits=N_CV_FOLDS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(model, X, y, cv=cv,
                               method="predict_proba")[:, 1]
    return compute_metrics(y, y_pred, y_prob)
 
 
def clade_held_out_cv(X_df, y, sts, model):
    valid_sts = [
        st for st in sts.unique()
        if st != "-"
        and (sts == st).sum() >= 5
        and y[sts == st].nunique() == 2
    ]
 
    if len(valid_sts) < 2:
        print("  Not enough valid clades")
        return None
 
    print(f"  Clades used for held-out CV: {sorted(valid_sts)}")
    all_true, all_pred, all_prob = [], [], []
 
    for st in valid_sts:
        test_mask  = (sts == st).values
        train_mask = ~test_mask
        X_train = X_df.values[train_mask]
        X_test  = X_df.values[test_mask]
        y_train = y.values[train_mask]
        y_test  = y.values[test_mask]
 
        if len(np.unique(y_train)) < 2:
            continue
        try:
            model.fit(X_train, y_train)
            all_true.extend(y_test.tolist())
            all_pred.extend(model.predict(X_test).tolist())
            all_prob.extend(model.predict_proba(X_test)[:, 1].tolist())
        except Exception as e:
            print(f"  WARNING ST{st}: {e}")
 
    if not all_true:
        return None
    return compute_metrics(np.array(all_true),
                           np.array(all_pred),
                           np.array(all_prob))
 
# ----------------------------------------------------------------
# Main experiment loop
# ----------------------------------------------------------------
all_results = []
best_params_log = []
 
for fs_name, fs_config in FEATURE_SETS.items():
    train_data = fs_config["train"]
    test_data  = fs_config["test"]
    features   = fs_config["features"]
    desc       = fs_config["description"]
 
    X_train = train_data[features].fillna(0)
    y_train = train_data["phenotype"]
    sts_train = train_data["ST"].reset_index(drop=True)
 
    X_test  = test_data[features].fillna(0)
    y_test  = test_data["phenotype"]
 
    print(f"\n{'='*70}")
    print(f"Feature Set: {fs_name}")
    print(f"  {desc}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, "
          f"Features: {len(features)}")
 
    models = make_models()
 
    for model_name, model_config in models.items():
        model    = model_config["model"]
        params   = model_config["params"]
 
        print(f"\n  Model: {model_name}")
 
        # ----------------------------------------------------
        # 1. Hyperparameter tuning on training set
        # ----------------------------------------------------
        print(f"    Tuning hyperparameters ({N_ITER_TUNE} iterations)...")
        cv_tune = StratifiedKFold(n_splits=3, shuffle=True,
                                  random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            model, params,
            n_iter=N_ITER_TUNE,
            cv=cv_tune,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"    Best params: {search.best_params_}")
        print(f"    Best CV AUC (tuning): {search.best_score_:.4f}")
 
        best_params_log.append({
            "feature_set": fs_name,
            "model": model_name,
            "best_params": json.dumps(search.best_params_),
            "tuning_auc": round(search.best_score_, 4)
        })
 
        # ----------------------------------------------------
        # 2. Random 5-fold CV on training set
        # ----------------------------------------------------
        print(f"    Random {N_CV_FOLDS}-fold CV...")
        try:
            metrics_random = random_cv(X_train.values, y_train, best_model)
            print(f"    AUC={metrics_random['AUC']}, "
                  f"F1={metrics_random['F1']}, "
                  f"MCC={metrics_random['MCC']}")
            all_results.append({
                "feature_set": fs_name,
                "description": desc,
                "model": model_name,
                "cv_strategy": "random_5fold",
                "n_genomes": len(X_train),
                "n_features": len(features),
                **metrics_random
            })
        except Exception as e:
            print(f"    ERROR: {e}")
 
        # ----------------------------------------------------
        # 3. Clade-held-out CV on training set
        # ----------------------------------------------------
        print(f"    Clade-held-out CV...")
        try:
            metrics_clade = clade_held_out_cv(
                X_train.reset_index(drop=True),
                y_train.reset_index(drop=True),
                sts_train,
                best_model
            )
            if metrics_clade:
                print(f"    AUC={metrics_clade['AUC']}, "
                      f"F1={metrics_clade['F1']}, "
                      f"MCC={metrics_clade['MCC']}")
                all_results.append({
                    "feature_set": fs_name,
                    "description": desc,
                    "model": model_name,
                    "cv_strategy": "clade_held_out",
                    "n_genomes": len(X_train),
                    "n_features": len(features),
                    **metrics_clade
                })
        except Exception as e:
            print(f"    ERROR: {e}")
 
        # ----------------------------------------------------
        # 4. Final evaluation on held-out test set
        # ----------------------------------------------------
        print(f"    Final test set evaluation...")
        try:
            best_model.fit(X_train.values, y_train.values)
            y_pred_test = best_model.predict(X_test.values)
            y_prob_test = best_model.predict_proba(X_test.values)[:, 1]
            metrics_test = compute_metrics(y_test.values,
                                           y_pred_test,
                                           y_prob_test)
            print(f"    AUC={metrics_test['AUC']}, "
                  f"F1={metrics_test['F1']}, "
                  f"MCC={metrics_test['MCC']}")
            all_results.append({
                "feature_set": fs_name,
                "description": desc,
                "model": model_name,
                "cv_strategy": "held_out_test",
                "n_genomes": len(X_test),
                "n_features": len(features),
                **metrics_test
            })
        except Exception as e:
            print(f"    ERROR: {e}")
 
# ----------------------------------------------------------------
# Save all results
# ----------------------------------------------------------------
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_RESULTS, sep="\t", index=False)
 
params_df = pd.DataFrame(best_params_log)
params_df.to_csv(OUT_PARAMS, sep="\t", index=False)
 
# ----------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------
print("\n" + "="*70)
print("RESULTS SUMMARY — AUC by feature set, model, CV strategy")
print("="*70)
 
summary = results_df.pivot_table(
    index=["feature_set", "model"],
    columns="cv_strategy",
    values="AUC"
).round(4)
print(summary.to_string())
summary.to_csv(OUT_SUMMARY, sep="\t")
 
# ----------------------------------------------------------------
# Key finding
# ----------------------------------------------------------------
print("\n" + "="*70)
print("KEY FINDING: Performance drop Random CV → Clade-Held-Out → Test Set")
print("="*70)
 
for fs_name in FEATURE_SETS.keys():
    fs_res = results_df[results_df["feature_set"] == fs_name]
    for model_name in ([m for m in ["RandomForest","LogisticRegression","XGBoost"]
                        if m in results_df["model"].unique()]):
        m_res = fs_res[fs_res["model"] == model_name]
        r_auc = m_res[m_res["cv_strategy"]=="random_5fold"]["AUC"].values
        c_auc = m_res[m_res["cv_strategy"]=="clade_held_out"]["AUC"].values
        t_auc = m_res[m_res["cv_strategy"]=="held_out_test"]["AUC"].values
        if len(r_auc) and len(c_auc) and len(t_auc):
            print(f"  {fs_name[:35]:<35} | {model_name:<20} | "
                  f"Random: {r_auc[0]:.4f} | "
                  f"Clade: {c_auc[0]:.4f} | "
                  f"Test: {t_auc[0]:.4f} | "
                  f"Drop(R→C): {r_auc[0]-c_auc[0]:+.4f}")
 
print(f"\nAll results saved to: {OUT_RESULTS}")
print(f"Best params saved to: {OUT_PARAMS}")
print(f"Summary saved to:     {OUT_SUMMARY}")
 
