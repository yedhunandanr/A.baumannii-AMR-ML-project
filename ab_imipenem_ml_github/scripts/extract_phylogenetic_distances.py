#!/usr/bin/env python3

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os


"""
Extract patristic distances from IQ-TREE2 phylogenies.
Handles snippy-core naming convention where reference is named 'Reference'.

Output: phylogenetic_distances.tsv
"""

import pandas as pd
from ete3 import Tree

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
IQTREE_DIR = str(PROJECT_ROOT / "outputs/iqtree")
MLST_FILE  = str(PROJECT_ROOT / "outputs/mlst_results.tsv")
OUTPUT     = str(PROJECT_ROOT / "outputs/phylogenetic_distances.tsv")

# snippy-core renames reference to 'Reference' in all trees
REFERENCE_NAME = "Reference"

# Internal references to exclude from study set
INTERNAL_REFS = {"470.8084", "470.7633"}

# Lineage trees
LINEAGE_STS = {"1", "2", "3", "25", "32"}

# ----------------------------------------------------------------
# Load MLST
# ----------------------------------------------------------------
mlst = pd.read_csv(MLST_FILE, sep="\t", header=None,
                   names=["file","scheme","ST","a1","a2","a3","a4","a5","a6","a7"])
mlst["genome_id"] = mlst["file"].str.extract(r'(\d+\.\d+)\.fasta')
mlst["genome_id"] = mlst["genome_id"].astype(str)
mlst["ST"] = mlst["ST"].astype(str)
st_lookup = dict(zip(mlst["genome_id"], mlst["ST"]))

# ----------------------------------------------------------------
# Helper
# ----------------------------------------------------------------
def get_distance(tree, name1, name2):
    try:
        return tree.get_distance(name1, name2)
    except Exception:
        return None

# ----------------------------------------------------------------
# Load universal tree
# ----------------------------------------------------------------
print("Loading universal tree...")
universal_tree = Tree(f"{IQTREE_DIR}/universal.treefile")
universal_leaves = set(universal_tree.get_leaf_names())
print(f"  Leaves: {len(universal_leaves)}")
print(f"  Reference present: {REFERENCE_NAME in universal_leaves}")

# Get study genomes from universal tree
study_genomes = [l for l in universal_leaves
                 if l != REFERENCE_NAME
                 and l not in INTERNAL_REFS]
print(f"  Study genomes: {len(study_genomes)}")

# ----------------------------------------------------------------
# Load lineage trees
# ----------------------------------------------------------------
lineage_trees = {}
for st in LINEAGE_STS:
    treefile = f"{IQTREE_DIR}/ST{st}.treefile"
    try:
        t = Tree(treefile)
        lineage_trees[st] = t
        leaves = set(t.get_leaf_names())
        print(f"  ST{st} tree: {len(leaves)} leaves, "
              f"Reference present: {REFERENCE_NAME in leaves}")
    except Exception as e:
        print(f"  WARNING: Could not load ST{st} tree: {e}")

# ----------------------------------------------------------------
# Extract distances
# ----------------------------------------------------------------
print("\nExtracting distances...")
results = []

for i, genome_id in enumerate(study_genomes):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(study_genomes)}...")

    st = st_lookup.get(genome_id, "-")

    # Universal distance — distance to Reference in universal tree
    dist_universal = get_distance(universal_tree, genome_id, REFERENCE_NAME)

    # Lineage distance — distance to Reference in ST-specific tree
    dist_lineage = None
    if st in lineage_trees:
        st_leaves = set(lineage_trees[st].get_leaf_names())
        if genome_id in st_leaves:
            dist_lineage = get_distance(lineage_trees[st], genome_id, REFERENCE_NAME)

    results.append({
        "genome_id":          genome_id,
        "ST":                 st,
        "distance_universal": dist_universal,
        "distance_lineage":   dist_lineage,
    })

df = pd.DataFrame(results)

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print(f"\nDistance extraction complete:")
print(f"  Total genomes: {len(df)}")
print(f"  With universal distance: {df['distance_universal'].notna().sum()}")
print(f"  With lineage distance:   {df['distance_lineage'].notna().sum()}")

print(f"\nUniversal distance stats:")
print(df["distance_universal"].describe().to_string())

print(f"\nLineage distance stats:")
print(df["distance_lineage"].describe().to_string())

print(f"\nMean universal distance by ST (top 10):")
print(df.groupby("ST")["distance_universal"].mean().sort_values(
    ascending=False).head(10).to_string())

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------
df.to_csv(OUTPUT, sep="\t", index=False)
print(f"\nSaved to: {OUTPUT}")
