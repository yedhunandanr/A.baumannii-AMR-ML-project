#!/usr/bin/env python3
"""
Build binary AMR gene presence/absence matrix from AMRFinder output.
Filters to only study genomes (excludes references and non-study genomes).
Focuses on AMR genes relevant to carbapenem resistance and general resistance.

Output: amr_feature_matrix.tsv
  Rows: genome IDs
  Columns: AMR gene symbols (binary 0/1)

Usage:
    python build_amr_matrix.py
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import os
import pandas as pd



# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
AMRFINDER_DIR = str(PROJECT_ROOT / "outputs/amrfinder")
GOOD_GENOMES  = str(PROJECT_ROOT / "outputs/good_genomes.txt")
INTERNAL_REFS = {"470.8084", "470.7633"}
OUTPUT_FILE   = str(PROJECT_ROOT / "outputs/amr_feature_matrix.tsv")

# ----------------------------------------------------------------
# Load study genome IDs
# ----------------------------------------------------------------
good = set(open(GOOD_GENOMES).read().splitlines())
good -= INTERNAL_REFS
print(f"Study genomes: {len(good)}")

# ----------------------------------------------------------------
# Parse all AMRFinder TSV files
# ----------------------------------------------------------------
records = []
missing = []

for genome_id in sorted(good):
    tsv_path = os.path.join(AMRFINDER_DIR, f"{genome_id}.tsv")
    if not os.path.exists(tsv_path):
        missing.append(genome_id)
        continue

    df = pd.read_csv(tsv_path, sep="\t")

    # Keep only AMR type (exclude STRESS, VIRULENCE for now)
    df = df[df["Type"] == "AMR"]

    # Get unique gene symbols for this genome
    genes = df["Element symbol"].dropna().unique().tolist()
    records.append({"genome_id": genome_id, "genes": genes})

print(f"Parsed: {len(records)} genomes")
if missing:
    print(f"Missing AMRFinder output for {len(missing)} genomes: {missing[:5]}...")

# ----------------------------------------------------------------
# Build presence/absence matrix
# ----------------------------------------------------------------
# Get all unique genes across dataset
all_genes = sorted(set(g for r in records for g in r["genes"]))
print(f"Total unique AMR gene symbols detected: {len(all_genes)}")

# Build matrix
rows = []
for r in records:
    gene_set = set(r["genes"])
    row = {"genome_id": r["genome_id"]}
    for gene in all_genes:
        row[gene] = 1 if gene in gene_set else 0
    rows.append(row)

matrix = pd.DataFrame(rows).set_index("genome_id")
print(f"Matrix shape: {matrix.shape} (genomes x genes)")

# ----------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------
print("\nTop 30 most prevalent AMR genes:")
prevalence = matrix.sum().sort_values(ascending=False)
for gene, count in prevalence.head(30).items():
    pct = count / len(matrix) * 100
    print(f"  {gene:<30} {count:>4} ({pct:.1f}%)")

# Carbapenem-relevant genes specifically
carb_genes = [g for g in all_genes if any(x in g.upper() for x in
              ["OXA-23", "OXA-24", "OXA-40", "OXA-58", "NDM", "IMP", "VIM",
               "KPC", "OXA-143", "OXA-235", "ISAba"])]
print(f"\nCarbapenem-relevant genes found: {len(carb_genes)}")
for gene in carb_genes:
    count = matrix[gene].sum()
    pct = count / len(matrix) * 100
    print(f"  {gene:<30} {count:>4} ({pct:.1f}%)")

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------
matrix.to_csv(OUTPUT_FILE, sep="\t")
print(f"\nFeature matrix saved to: {OUTPUT_FILE}")
print(f"Shape: {matrix.shape[0]} genomes x {matrix.shape[1]} genes")
