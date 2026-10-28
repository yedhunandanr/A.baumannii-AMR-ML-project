#!/usr/bin/env python3

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os
import pandas as pd



matrix = pd.read_csv(str(PROJECT_ROOT / "outputs/amr_feature_matrix.tsv"), sep="\t", index_col=0)
res = pd.read_csv(str(PROJECT_ROOT / "dataset/mic_values/finalres_qc.tsv"), sep="\t")
mlst = pd.read_csv(str(PROJECT_ROOT / "outputs/mlst_results.tsv"),
                   sep="\t", header=None,
                   names=["file","scheme","ST","a1","a2","a3","a4","a5","a6","a7"])
mlst["genome_id"] = mlst["file"].str.extract(r'(\d+\.\d+)\.fasta')

matrix.index = matrix.index.astype(str)
res["genome_id"] = res["genome_id"].astype(str)
mlst["genome_id"] = mlst["genome_id"].astype(str)

resistant_ids = set(res["genome_id"].tolist())

carb_genes = ["blaOXA-23","blaOXA-58","blaOXA-24","blaOXA-402",
              "blaOXA-407","blaOXA-235","blaOXA-237","blaNDM",
              "blaNDM-1","blaVIM","blaVIM-2"]
carb_genes = [g for g in carb_genes if g in matrix.columns]

res_matrix = matrix[matrix.index.isin(resistant_ids)].copy()
res_matrix["any_carbapenemase"] = res_matrix[carb_genes].sum(axis=1) > 0

# Get non-enzymatic resistant isolates
non_enzymatic_ids = res_matrix[~res_matrix["any_carbapenemase"]].index.tolist()

# Build summary dataframe
ne_df = pd.DataFrame({"genome_id": non_enzymatic_ids})

# Add MIC
ne_df = ne_df.merge(
    res[["genome_id", "measurement_value", "measurement_sign"]],
    on="genome_id", how="left")

# Add ST
ne_df = ne_df.merge(
    mlst[["genome_id", "ST"]],
    on="genome_id", how="left")

# Add intrinsic OXA-51-like genes (not in acquired carb list)
oxa51_like = [c for c in matrix.columns if "OXA-" in c and c not in carb_genes]
for gene in oxa51_like:
    ne_df[gene] = ne_df["genome_id"].map(matrix[gene])

print(f"Non-enzymatic resistant isolates: {len(ne_df)}")

print(f"\nST distribution:")
print(ne_df["ST"].value_counts().to_string())

print(f"\nMIC distribution (mg/L):")
print(ne_df["measurement_value"].value_counts().sort_index().to_string())

print(f"\nIntrinsic OXA genes present:")
for gene in oxa51_like:
    count = int(ne_df[gene].sum())
    if count > 0:
        print(f"  {gene:<30} {count}")

# Save
ne_df.to_csv(str(PROJECT_ROOT / "outputs/non_enzymatic_resistant.tsv"),
             sep="\t", index=False)
print(f"\nSaved to non_enzymatic_resistant.tsv")
