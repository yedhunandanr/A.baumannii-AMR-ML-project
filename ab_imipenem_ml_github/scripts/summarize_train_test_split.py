#!/usr/bin/env python3

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os


"""
Summarize train/test split distribution.
Shows which genomes went to training vs test set,
broken down by ST and phenotype.

Usage:
    python split_summary.py
"""

import pandas as pd

SPLIT_FILE = str(PROJECT_ROOT / "outputs/ml_train_test_split.tsv")

df = pd.read_csv(SPLIT_FILE, sep="\t")
df["genome_id"] = df["genome_id"].astype(str)
df["ST"] = df["ST"].astype(str)

train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]

# ----------------------------------------------------------------
# Overall summary
# ----------------------------------------------------------------
print("=" * 60)
print("OVERALL SPLIT SUMMARY")
print("=" * 60)
print(f"Total genomes:      {len(df)}")
print(f"Training set:       {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"Test set:           {len(test)} ({len(test)/len(df)*100:.1f}%)")
print()
print(f"Training - Resistant:   {train['phenotype'].sum()}")
print(f"Training - Susceptible: {(train['phenotype']==0).sum()}")
print(f"Test     - Resistant:   {test['phenotype'].sum()}")
print(f"Test     - Susceptible: {(test['phenotype']==0).sum()}")

# ----------------------------------------------------------------
# ST breakdown
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("ST DISTRIBUTION IN TRAIN VS TEST")
print("=" * 60)

st_summary = df.groupby(["ST", "split", "phenotype"]).size().unstack(
    ["split", "phenotype"], fill_value=0)

# Flatten column names
st_summary.columns = [f"{s}_{p}" for s, p in st_summary.columns]
st_summary = st_summary.rename(columns={
    "train_1": "Train_R",
    "train_0": "Train_S",
    "test_1":  "Test_R",
    "test_0":  "Test_S",
})
st_summary["Train_Total"] = st_summary.get("Train_R", 0) + st_summary.get("Train_S", 0)
st_summary["Test_Total"]  = st_summary.get("Test_R", 0)  + st_summary.get("Test_S", 0)
st_summary = st_summary.sort_values("Train_Total", ascending=False)

print(f"\n{'ST':<10} {'Train_R':>10} {'Train_S':>10} {'Train_Tot':>10} "
      f"{'Test_R':>10} {'Test_S':>10} {'Test_Tot':>10}")
print("-" * 70)
for st, row in st_summary.iterrows():
    tr = int(row.get("Train_R", 0))
    ts = int(row.get("Train_S", 0))
    tt = int(row.get("Train_Total", 0))
    er = int(row.get("Test_R", 0))
    es = int(row.get("Test_S", 0))
    et = int(row.get("Test_Total", 0))
    print(f"{st:<10} {tr:>10} {ts:>10} {tt:>10} {er:>10} {es:>10} {et:>10}")

# ----------------------------------------------------------------
# Save detailed genome list
# ----------------------------------------------------------------
out_train = str(PROJECT_ROOT / "outputs/train_genomes.tsv")
out_test  = str(PROJECT_ROOT / "outputs/test_genomes.tsv")

train.to_csv(out_train, sep="\t", index=False)
test.to_csv(out_test,  sep="\t", index=False)

print(f"\nTraining genome list saved to: {out_train}")
print(f"Test genome list saved to:     {out_test}")
