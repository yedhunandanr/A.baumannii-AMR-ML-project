#!/usr/bin/env python3
"""
Figure 3 — Main Results: AUC Comparison
Grouped bar chart comparing all 5 feature sets across 3 evaluation strategies
for all 3 models. Shows performance drop from random CV to clade-held-out to test set.

Output: figure3_ml_results.pdf, figure3_ml_results.png
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

OUTPUT_PDF = str(PROJECT_ROOT / "outputs/figures/figure3_ml_results.pdf")
OUTPUT_PNG = str(PROJECT_ROOT / "outputs/figures/figure3_ml_results.png")
os.makedirs(str(PROJECT_ROOT / "outputs/figures"), exist_ok=True)

# ----------------------------------------------------------------
# Load results
# ----------------------------------------------------------------
df = pd.read_csv(str(PROJECT_ROOT / "outputs/ml_results_full.tsv"), sep="\t")

# Clean feature set labels
fs_labels = {
    "1_universal_distance_only":       "FS1\nUniversal\ndistance",
    "2_lineage_distance_only":         "FS2\nLineage\ndistance",
    "3_amr_genes_only":                "FS3\nAMR\ngenes",
    "4_universal_distance_plus_amr":   "FS4\nUniversal\n+ AMR",
    "5_lineage_distance_plus_amr":     "FS5\nLineage\n+ AMR",
}

cv_labels = {
    "random_5fold":   "Random 5-fold CV",
    "clade_held_out": "Clade-held-out CV",
    "held_out_test":  "Held-out test set",
}

models = ["RandomForest", "LogisticRegression", "XGBoost"]
model_labels = {
    "RandomForest":       "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "XGBoost":            "XGBoost",
}

cv_strategies = ["random_5fold", "clade_held_out", "held_out_test"]

# Colors for CV strategies — colorblind friendly
CV_COLORS = {
    "random_5fold":   "#2166ac",  # blue
    "clade_held_out": "#d7191c",  # red
    "held_out_test":  "#1a9641",  # green
}

feature_sets = list(fs_labels.keys())

# ----------------------------------------------------------------
# Figure — 3 rows (one per model), 1 column
# ----------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=False)
fig.patch.set_facecolor("white")

for ax_idx, model in enumerate(models):
    ax = axes[ax_idx]
    model_data = df[df["model"] == model]

    n_fs   = len(feature_sets)
    n_cv   = len(cv_strategies)
    width  = 0.22
    gap    = 0.08
    group_w = n_cv * width + gap

    x_centers = np.arange(n_fs) * (group_w + 0.3)

    for cv_idx, cv in enumerate(cv_strategies):
        cv_data = model_data[model_data["cv_strategy"] == cv]
        aucs = []
        for fs in feature_sets:
            row = cv_data[cv_data["feature_set"] == fs]["AUC"]
            aucs.append(row.values[0] if len(row) > 0 else np.nan)

        x_pos = x_centers + (cv_idx - 1) * width

        bars = ax.bar(
            x_pos, aucs,
            width=width,
            color=CV_COLORS[cv],
            alpha=0.85,
            label=cv_labels[cv],
            edgecolor="white",
            linewidth=0.8,
            zorder=3
        )

        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{auc:.3f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#333333",
                    rotation=90
                )

    # Formatting
    ax.set_xticks(x_centers)
    ax.set_xticklabels([fs_labels[fs] for fs in feature_sets],
                        fontsize=9.5, ha="center")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("AUC (ROC)", fontsize=10)
    ax.set_title(f"{model_labels[model]}",
                 fontsize=11, fontweight="bold", loc="left", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.5, color="gray", linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=1)
    ax.text(x_centers[-1] + 0.35, 0.505, "Random",
            fontsize=7.5, color="gray", va="bottom")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_xlim(-0.4, x_centers[-1] + 0.5)

    if ax_idx == 0:
        ax.legend(fontsize=9, loc="upper right",
                  framealpha=0.9, ncol=3)

    # Add drop annotations — arrows showing collapse from random to clade
    for fs_idx, fs in enumerate(feature_sets):
        random_row = model_data[
            (model_data["cv_strategy"] == "random_5fold") &
            (model_data["feature_set"] == fs)]["AUC"]
        clade_row = model_data[
            (model_data["cv_strategy"] == "clade_held_out") &
            (model_data["feature_set"] == fs)]["AUC"]

        if len(random_row) > 0 and len(clade_row) > 0:
            r_auc = random_row.values[0]
            c_auc = clade_row.values[0]
            drop  = r_auc - c_auc

            if drop > 0.2:  # Only annotate large drops
                x_r = x_centers[fs_idx] + (-1) * width
                x_c = x_centers[fs_idx] + 0 * width
                ax.annotate(
                    f"↓{drop:.2f}",
                    xy=(x_c, c_auc + 0.02),
                    fontsize=7,
                    color="#d7191c",
                    ha="center",
                    fontweight="bold"
                )

# ----------------------------------------------------------------
# Feature set descriptions below x-axis of last panel
# ----------------------------------------------------------------
desc_text = (
    "FS1: Universal phylogenetic distance from ATCC 19606 (n=770)  |  "
    "FS2: Lineage-matched distance from ST-specific susceptible reference (n=577)  |  "
    "FS3: AMR gene presence/absence (228 genes, n=770)\n"
    "FS4: Universal distance + AMR genes (n=770)  |  "
    "FS5: Lineage-matched distance + AMR genes (n=577)"
)
fig.text(0.5, -0.02, desc_text, ha="center", va="top",
         fontsize=8, color="#555555",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                   edgecolor="#cccccc", alpha=0.9))

# ----------------------------------------------------------------
# Overall title
# ----------------------------------------------------------------
fig.suptitle(
    "Imipenem Resistance Prediction Performance Across Feature Sets and Evaluation Strategies",
    fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout()
plt.subplots_adjust(hspace=0.45)
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", format="png")
plt.close()

print(f"Figure 3 saved:")
print(f"  PDF: {OUTPUT_PDF}")
print(f"  PNG: {OUTPUT_PNG}")
