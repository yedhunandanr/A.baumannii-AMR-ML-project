#!/usr/bin/env python3
"""
Figure 2 — Lineage and Resistance Distribution
Shows resistant vs susceptible counts for each major ST,
highlighting the mixed resistance distribution within lineages.

Output: figure2_st_distribution.pdf, figure2_st_distribution.png
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

OUTPUT_PDF = str(PROJECT_ROOT / "outputs/figures/figure2_st_distribution.pdf")
OUTPUT_PNG = str(PROJECT_ROOT / "outputs/figures/figure2_st_distribution.png")
os.makedirs(str(PROJECT_ROOT / "outputs/figures"), exist_ok=True)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
df = pd.read_csv(str(PROJECT_ROOT / "outputs/final_feature_matrix.tsv"), sep="\t")
df["ST"] = df["ST"].astype(str)

# Get ST counts by phenotype
st_counts = df.groupby(["ST", "phenotype"]).size().unstack(fill_value=0)
st_counts.columns = ["Susceptible", "Resistant"]
st_counts["Total"] = st_counts.sum(axis=1)
st_counts = st_counts.sort_values("Total", ascending=False)

# Keep STs with >= 5 isolates for readability
st_plot = st_counts[st_counts["Total"] >= 5].copy()

# Label rare STs as "Other"
other = st_counts[st_counts["Total"] < 5]
if len(other) > 0:
    other_row = pd.DataFrame({
        "Susceptible": [other["Susceptible"].sum()],
        "Resistant":   [other["Resistant"].sum()],
        "Total":       [other["Total"].sum()]
    }, index=["Other (<5)"])
    st_plot = pd.concat([st_plot, other_row])

print(f"STs plotted: {len(st_plot)}")
print(st_plot.to_string())

# ----------------------------------------------------------------
# Figure setup — two panels
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                          gridspec_kw={"width_ratios": [2, 1]})
fig.patch.set_facecolor("white")

# Color palette — colorblind friendly
COL_R = "#d7191c"   # red — resistant
COL_S = "#2166ac"   # blue — susceptible

# ----------------------------------------------------------------
# Panel A — Stacked horizontal bar chart
# ----------------------------------------------------------------
ax1 = axes[0]

sts    = st_plot.index.tolist()
resist = st_plot["Resistant"].values
suscept = st_plot["Susceptible"].values
y_pos  = np.arange(len(sts))

bars_s = ax1.barh(y_pos, suscept, color=COL_S, alpha=0.85,
                   label="Susceptible", height=0.7)
bars_r = ax1.barh(y_pos, resist, left=suscept, color=COL_R,
                   alpha=0.85, label="Resistant", height=0.7)

# Add count labels inside bars
for i, (r, s) in enumerate(zip(resist, suscept)):
    if s >= 5:
        ax1.text(s / 2, i, str(int(s)), ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")
    if r >= 5:
        ax1.text(s + r / 2, i, str(int(r)), ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"ST{st}" if st != "Other (<5)" and st != "-"
                      else st for st in sts], fontsize=9)
ax1.set_xlabel("Number of isolates", fontsize=11)
ax1.set_title("A  Isolate distribution by sequence type",
              fontsize=12, fontweight="bold", loc="left", pad=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=10, loc="lower right",
           framealpha=0.9)

# Highlight major STs
for i, st in enumerate(sts):
    if st in ["2", "3", "32", "1", "25"]:
        ax1.get_yticklabels()[i].set_fontweight("bold")
        ax1.get_yticklabels()[i].set_color("#333333")

# Add vertical line at 50% for visual reference
ax1.axvline(0, color="black", linewidth=0.8, alpha=0.3)

# ----------------------------------------------------------------
# Panel B — Resistance fraction per ST (dot plot)
# ----------------------------------------------------------------
ax2 = axes[1]

# Calculate resistance fraction
st_plot["R_fraction"] = st_plot["Resistant"] / st_plot["Total"]

# Color dots by whether predominantly R, S, or mixed
dot_colors = []
for _, row in st_plot.iterrows():
    frac = row["R_fraction"]
    if frac >= 0.7:
        dot_colors.append(COL_R)
    elif frac <= 0.3:
        dot_colors.append(COL_S)
    else:
        dot_colors.append("#756bb1")  # purple — mixed

scatter = ax2.scatter(
    st_plot["R_fraction"], y_pos,
    c=dot_colors, s=st_plot["Total"] * 2.5,
    alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3
)

# Reference line at 0.5
ax2.axvline(0.5, color="gray", linewidth=1.2,
            linestyle="--", alpha=0.7, label="50% threshold")

ax2.set_yticks(y_pos)
ax2.set_yticklabels([])
ax2.set_xlim(-0.05, 1.05)
ax2.set_xlabel("Fraction resistant", fontsize=11)
ax2.set_title("B  Resistance fraction per ST\n(dot size ∝ total isolates)",
              fontsize=12, fontweight="bold", loc="left", pad=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

# Legend for dot colors
leg_patches = [
    mpatches.Patch(color=COL_R,    label="Predominantly resistant (>70%)"),
    mpatches.Patch(color="#756bb1", label="Mixed (30–70%)"),
    mpatches.Patch(color=COL_S,    label="Predominantly susceptible (<30%)"),
]
ax2.legend(handles=leg_patches, fontsize=8, loc="lower right",
           framealpha=0.9)

# Annotate key STs
key_sts = {"2": "ST2", "3": "ST3", "32": "ST32", "1": "ST1", "25": "ST25"}
for i, st in enumerate(sts):
    if st in key_sts:
        frac = st_plot.loc[st, "R_fraction"]
        ax2.annotate(
            key_sts[st],
            xy=(frac, i),
            xytext=(frac + 0.08 if frac < 0.5 else frac - 0.08, i),
            fontsize=8, fontweight="bold", color="#333333",
            va="center",
            ha="left" if frac < 0.5 else "right"
        )

# ----------------------------------------------------------------
# Overall title and summary stats
# ----------------------------------------------------------------
total_r = df["phenotype"].sum()
total_s = (df["phenotype"] == 0).sum()
n_sts   = df["ST"].nunique()

fig.suptitle(
    f"Sequence Type Distribution of A. baumannii Isolates\n"
    f"n = {len(df)} total ({total_r} resistant, {total_s} susceptible) | "
    f"{n_sts} unique sequence types (Pasteur scheme)",
    fontsize=12, y=1.01, fontweight="bold"
)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", format="png")
plt.close()

print(f"Figure 2 saved:")
print(f"  PDF: {OUTPUT_PDF}")
print(f"  PNG: {OUTPUT_PNG}")
