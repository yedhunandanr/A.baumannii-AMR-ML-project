#!/usr/bin/env python3
"""
Figure 5 — Subgroup Analysis
Grouped bar chart comparing AUC for carbapenemase-positive vs
non-enzymatic resistant isolates across all 5 feature sets.

Output: figure5_subgroup.pdf, figure5_subgroup.png
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

OUTPUT_PDF = str(PROJECT_ROOT / "outputs/figures/figure5_subgroup.pdf")
OUTPUT_PNG = str(PROJECT_ROOT / "outputs/figures/figure5_subgroup.png")
os.makedirs(str(PROJECT_ROOT / "outputs/figures"), exist_ok=True)

# ----------------------------------------------------------------
# Load subgroup results
# ----------------------------------------------------------------
df = pd.read_csv(str(PROJECT_ROOT / "outputs/subgroup_results.tsv"), sep="\t")

# Feature set labels
fs_labels = {
    "1_universal_distance":      "FS1\nUniversal\ndistance",
    "2_lineage_distance":        "FS2\nLineage\ndistance",
    "3_amr_genes":               "FS3\nAMR\ngenes",
    "4_universal_plus_amr":      "FS4\nUniversal\n+ AMR",
    "5_lineage_plus_amr":        "FS5\nLineage\n+ AMR",
}

feature_sets = list(fs_labels.keys())

# Colors
COL_ALL   = "#555555"   # gray — all resistant
COL_CARB  = "#d7191c"   # red — carbapenemase positive
COL_NENZ  = "#2166ac"   # blue — non-enzymatic
COL_GAP   = "#ff7f00"   # orange — gap indicator

# ----------------------------------------------------------------
# Figure setup — two panels
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                          gridspec_kw={"width_ratios": [3, 1]})
fig.patch.set_facecolor("white")

# ----------------------------------------------------------------
# Panel A — Grouped bar chart
# ----------------------------------------------------------------
ax1 = axes[0]

n_fs   = len(feature_sets)
width  = 0.22
gap    = 0.06
group_w = 3 * width + gap

x_centers = np.arange(n_fs) * (group_w + 0.25)

subgroups = [
    ("all_resistant",  "All resistant",              COL_ALL),
    ("carb_positive",  "Carbapenemase-positive",      COL_CARB),
    ("non_enzymatic",  "Non-enzymatic resistant",     COL_NENZ),
]

for sg_idx, (sg_key, sg_label, sg_color) in enumerate(subgroups):
    sg_data = df[df["subgroup"] == sg_key]
    aucs = []
    for fs in feature_sets:
        row = sg_data[sg_data["feature_set"] == fs]["AUC"]
        aucs.append(row.values[0] if len(row) > 0 else np.nan)

    x_pos = x_centers + (sg_idx - 1) * width

    bars = ax1.bar(
        x_pos, aucs,
        width=width,
        color=sg_color,
        alpha=0.85,
        label=sg_label,
        edgecolor="white",
        linewidth=0.8,
        zorder=3
    )

    # Value labels
    for bar, auc in zip(bars, aucs):
        if not np.isnan(auc):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{auc:.3f}",
                ha="center", va="bottom",
                fontsize=7, color="#333333",
                rotation=90
            )

# Add gap annotations between carb+ and non-enzymatic
for fs_idx, fs in enumerate(feature_sets):
    carb_row = df[(df["feature_set"] == fs) & (df["subgroup"] == "carb_positive")]["AUC"]
    nenz_row = df[(df["feature_set"] == fs) & (df["subgroup"] == "non_enzymatic")]["AUC"]

    if len(carb_row) > 0 and len(nenz_row) > 0:
        c_auc = carb_row.values[0]
        n_auc = nenz_row.values[0]
        gap_val = c_auc - n_auc
        x_mid = x_centers[fs_idx] + 0.5 * width

        # Draw bracket
        y_top = max(c_auc, n_auc) + 0.08
        ax1.annotate(
            f"Gap\n{gap_val:+.3f}",
            xy=(x_mid, y_top),
            fontsize=7.5,
            color=COL_GAP,
            ha="center",
            fontweight="bold"
        )
        ax1.plot(
            [x_centers[fs_idx], x_centers[fs_idx] + width],
            [y_top - 0.01, y_top - 0.01],
            color=COL_GAP, linewidth=1.2, alpha=0.7
        )

ax1.set_xticks(x_centers)
ax1.set_xticklabels([fs_labels[fs] for fs in feature_sets], fontsize=10)
ax1.set_ylim(0.0, 1.20)
ax1.set_ylabel("AUC (ROC)", fontsize=11)
ax1.set_title("A  Model performance by resistance mechanism subgroup",
              fontsize=11, fontweight="bold", loc="left", pad=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.axhline(0.5, color="gray", linewidth=1.0,
            linestyle="--", alpha=0.5, zorder=1)
ax1.text(x_centers[-1] + 0.35, 0.505, "Random",
         fontsize=8, color="gray", va="bottom")
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax1.set_xlim(-0.4, x_centers[-1] + 0.55)

# ----------------------------------------------------------------
# Panel B — Gap comparison (dot plot)
# ----------------------------------------------------------------
ax2 = axes[1]

gaps = []
fs_names = []
for fs in feature_sets:
    carb_row = df[(df["feature_set"] == fs) & (df["subgroup"] == "carb_positive")]["AUC"]
    nenz_row = df[(df["feature_set"] == fs) & (df["subgroup"] == "non_enzymatic")]["AUC"]
    if len(carb_row) > 0 and len(nenz_row) > 0:
        gaps.append(carb_row.values[0] - nenz_row.values[0])
        fs_names.append(fs_labels[fs].replace("\n", " "))

y_pos = np.arange(len(gaps))

# Color bars by gap size
gap_colors = [COL_CARB if g > 0.18 else COL_NENZ for g in gaps]

bars2 = ax2.barh(y_pos, gaps, color=gap_colors, alpha=0.85,
                  edgecolor="white", height=0.55, zorder=3)

for bar, gap_val in zip(bars2, gaps):
    ax2.text(
        gap_val + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{gap_val:.3f}",
        va="center", ha="left",
        fontsize=9, color="#333333", fontweight="bold"
    )

ax2.set_yticks(y_pos)
ax2.set_yticklabels(fs_names, fontsize=9)
ax2.set_xlabel("AUC gap\n(Carb+ minus Non-enzymatic)", fontsize=10)
ax2.set_title("B  Performance gap\nbetween subgroups",
              fontsize=11, fontweight="bold", loc="left", pad=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlim(0, max(gaps) * 1.35)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.grid(axis="x", alpha=0.3, zorder=0)

# Annotate smallest gap
min_gap_idx = gaps.index(min(gaps))
ax2.annotate(
    "Smallest gap\n(best for\nnon-enzymatic)",
    xy=(gaps[min_gap_idx], min_gap_idx),
    xytext=(gaps[min_gap_idx] + 0.04, min_gap_idx + 0.8),
    fontsize=7.5, color=COL_NENZ,
    arrowprops=dict(arrowstyle="->", color=COL_NENZ, lw=1.2)
)

# ----------------------------------------------------------------
# Subgroup sizes note
# ----------------------------------------------------------------
fig.text(
    0.5, -0.04,
    "Carbapenemase-positive: isolates with blaOXA-23, blaOXA-58, blaOXA-24, blaNDM-1, or blaVIM detected (n=262 resistant, test set n≈52)\n"
    "Non-enzymatic resistant: resistant isolates with no detected acquired carbapenemase (n=51 total, test set n≈10)\n"
    "All models trained on full training set (resistant vs susceptible); evaluated on test set stratified by resistance mechanism",
    ha="center", va="top", fontsize=8, color="#555555",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
              edgecolor="#cccccc", alpha=0.9)
)

# ----------------------------------------------------------------
# Title
# ----------------------------------------------------------------
fig.suptitle(
    "Subgroup Analysis: Carbapenemase-Positive vs Non-Enzymatic Resistant Isolates\n"
    "XGBoost model evaluated on held-out test set",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.subplots_adjust(wspace=0.45)
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", format="png")
plt.close()

print(f"Figure 5 saved:")
print(f"  PDF: {OUTPUT_PDF}")
print(f"  PNG: {OUTPUT_PNG}")
