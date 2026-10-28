#!/usr/bin/env python3
"""
Figure 4 — SHAP Feature Importance
Horizontal bar chart showing top 20 features from Feature Set 4
(Universal distance + AMR genes), with distance_universal highlighted.
Also shows comparison of distance feature rank across feature sets 4 and 5.

Output: figure4_shap.pdf, figure4_shap.png
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

OUTPUT_PDF = str(PROJECT_ROOT / "outputs/figures/figure4_shap.pdf")
OUTPUT_PNG = str(PROJECT_ROOT / "outputs/figures/figure4_shap.png")
os.makedirs(str(PROJECT_ROOT / "outputs/figures"), exist_ok=True)

# ----------------------------------------------------------------
# Load SHAP importance data
# ----------------------------------------------------------------
fs4 = pd.read_csv(str(PROJECT_ROOT / "outputs/shap/shap_importance_fs4_universal_plus_amr.tsv"), sep="\t")
fs5 = pd.read_csv(str(PROJECT_ROOT / "outputs/shap/shap_importance_fs5_lineage_plus_amr.tsv"), sep="\t")

# Top 20 features for FS4
top20_fs4 = fs4.head(20).copy()
top20_fs4 = top20_fs4.sort_values("mean_abs_shap", ascending=True)

# Top 20 for FS5
top20_fs5 = fs5.head(20).copy()
top20_fs5 = top20_fs5.sort_values("mean_abs_shap", ascending=True)

# ----------------------------------------------------------------
# Color scheme
# ----------------------------------------------------------------
def get_color(feature, mean_shap):
    if feature in ["distance_universal", "distance_lineage"]:
        return "#ff7f00"  # orange — distance features highlighted
    elif "OXA-23" in feature or "OXA-58" in feature or "NDM" in feature:
        return "#d7191c"  # red — major carbapenemases
    elif feature.startswith("bla"):
        return "#e7298a"  # pink — other resistance genes
    else:
        return "#2166ac"  # blue — other AMR genes

# ----------------------------------------------------------------
# Figure — 2 panels side by side
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.patch.set_facecolor("white")

for ax, data, title, dist_col in [
    (axes[0], top20_fs4, "A  Feature Set 4: Universal distance + AMR genes", "distance_universal"),
    (axes[1], top20_fs5, "B  Feature Set 5: Lineage distance + AMR genes", "distance_lineage"),
]:
    features = data["feature"].tolist()
    values   = data["mean_abs_shap"].values
    colors   = [get_color(f, data[data["feature"]==f]["mean_shap"].values[0])
                for f in features]

    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, values, color=colors, alpha=0.88,
                   edgecolor="white", linewidth=0.8, height=0.7)

    # Highlight distance bar with thicker edge
    for i, (bar, feat) in enumerate(zip(bars, features)):
        if feat in ["distance_universal", "distance_lineage"]:
            bar.set_edgecolor("#ff7f00")
            bar.set_linewidth(2.5)
            # Add rank annotation
            rank = len(fs4) - fs4[fs4["feature"] == feat].index[0] \
                   if feat == "distance_universal" \
                   else len(fs5) - fs5[fs5["feature"] == feat].index[0]
            ax.text(
                values[i] + 0.02, i,
                f"Rank {fs4[fs4['feature']==feat].index[0]+1 if feat=='distance_universal' else fs5[fs5['feature']==feat].index[0]+1}"
                f" of {len(fs4)}",
                va="center", ha="left",
                fontsize=8, color="#ff7f00", fontweight="bold"
            )

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=7.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.35)
    ax.grid(axis="x", alpha=0.3, zorder=0)

    # Color distance feature label bold
    for label, feat in zip(ax.get_yticklabels(), features):
        if feat in ["distance_universal", "distance_lineage"]:
            label.set_fontweight("bold")
            label.set_color("#ff7f00")

# ----------------------------------------------------------------
# Shared legend
# ----------------------------------------------------------------
legend_patches = [
    mpatches.Patch(color="#ff7f00", label="Phylogenetic distance feature"),
    mpatches.Patch(color="#d7191c", label="Major carbapenemase gene"),
    mpatches.Patch(color="#e7298a", label="Other β-lactamase gene"),
    mpatches.Patch(color="#2166ac", label="Other AMR gene / mutation"),
]
fig.legend(handles=legend_patches, loc="lower center",
           ncol=4, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

# ----------------------------------------------------------------
# Title
# ----------------------------------------------------------------
fig.suptitle(
    "SHAP Feature Importance — XGBoost Models\n"
    "Top 20 features by mean absolute SHAP value",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", format="png")
plt.close()

print(f"Figure 4 saved:")
print(f"  PDF: {OUTPUT_PDF}")
print(f"  PNG: {OUTPUT_PNG}")
