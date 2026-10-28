#!/usr/bin/env python3
"""
Figure 1 — Dataset Curation Flowchart
CONSORT-style flowchart showing filtering steps from raw BV-BRC download
to final analysis dataset.

Output: figure1_curation_flowchart.pdf, figure1_curation_flowchart.png
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_PDF = str(PROJECT_ROOT / "outputs/figures/figure1_curation_flowchart.pdf")
OUTPUT_PNG = str(PROJECT_ROOT / "outputs/figures/figure1_curation_flowchart.png")

import os
os.makedirs(str(PROJECT_ROOT / "outputs/figures"), exist_ok=True)

# ----------------------------------------------------------------
# Data for flowchart boxes
# ----------------------------------------------------------------
steps = [
    {
        "label": "BV-BRC AMR Phenotype Database\nImipenem + Laboratory Method",
        "resistant": 964,
        "susceptible": 928,
        "color": "#2166ac",
        "removed": None,
    },
    {
        "label": "After mg/L units filter\n(removed disk diffusion & missing values)",
        "resistant": 363,
        "susceptible": 567,
        "color": "#2166ac",
        "removed": "Removed: 601 R, 361 S\n(disk diffusion, missing MIC values)",
    },
    {
        "label": "After EUCAST 2023 breakpoints\n(MIC ≥8 resistant, MIC ≤2 susceptible)",
        "resistant": 362,
        "susceptible": 514,
        "color": "#2166ac",
        "removed": "Removed: 1 R, 53 S\n(gray zone MIC=4, label mismatches)",
    },
    {
        "label": "After genome assembly download\n(BV-BRC FASTA sequences)",
        "resistant": 338,
        "susceptible": 485,
        "color": "#2166ac",
        "removed": "Removed: 24 R, 29 S\n(no assembly available in BV-BRC)",
    },
    {
        "label": "After assembly quality control\n(QUAST: N50 ≥10 kb, contigs ≤500,\nlength 3.5–4.5 Mb, GC 38–41%)",
        "resistant": 330,
        "susceptible": 475,
        "color": "#2166ac",
        "removed": "Removed: 8 R, 10 S\n(18 total failed QC thresholds)",
    },
    {
        "label": "After species verification\n(removed non-A. baumannii)",
        "resistant": 313,
        "susceptible": 462,
        "color": "#2166ac",
        "removed": "Removed: 17 R, 13 S\n(30 Acinetobacter pittii, taxon 48296)",
    },
    {
        "label": "After phylogenetic outlier removal\n(distance >3× IQR from ATCC 19606)",
        "resistant": 313,
        "susceptible": 459,
        "color": "#2166ac",
        "removed": "Removed: 0 R, 3 S\n(1 Acinetobacter sp., 2 extreme outliers)",
    },
    {
        "label": "Final Analysis Dataset\nA. baumannii — Imipenem",
        "resistant": 313,
        "susceptible": 457,
        "color": "#1a9641",
        "removed": None,
    },
]

# ----------------------------------------------------------------
# Figure setup
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 22))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(steps) * 3.2 + 1)
ax.axis("off")

fig.patch.set_facecolor("white")

BOX_W      = 6.0
BOX_H      = 1.8
BOX_X      = 2.0
REMOVE_X   = 7.2
REMOVE_W   = 2.6
REMOVE_H   = 1.0

def draw_box(ax, x, y, w, h, text, color, fontsize=9.5, bold=False):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.92,
        zorder=2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, color="white",
        fontweight=weight,
        wrap=True,
        zorder=3
    )

def draw_arrow(ax, x, y_start, y_end):
    ax.annotate(
        "", xy=(x, y_end), xytext=(x, y_start),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#555555",
            lw=1.8,
        ),
        zorder=1
    )

def draw_remove_box(ax, x, y, w, h, text):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor="#d7191c",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        zorder=2
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=7.8, color="white",
        zorder=3
    )

def draw_remove_arrow(ax, box_x, box_y, box_h, remove_x, remove_y, remove_h):
    mid_y = box_y + box_h / 2
    ax.annotate(
        "", xy=(remove_x, remove_y + remove_h / 2),
        xytext=(box_x + BOX_W, mid_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#d7191c",
            lw=1.5,
            connectionstyle="arc3,rad=0.0"
        ),
        zorder=1
    )

# ----------------------------------------------------------------
# Draw flowchart
# ----------------------------------------------------------------
total = len(steps)
y_positions = []

for i, step in enumerate(steps):
    y = (total - 1 - i) * 3.2 + 0.8
    y_positions.append(y)

    # Count text
    n_total = step["resistant"] + step["susceptible"]
    count_text = (f"n = {n_total:,}\n"
                  f"Resistant: {step['resistant']:,} | "
                  f"Susceptible: {step['susceptible']:,}")

    # Full label
    full_label = f"{step['label']}\n{count_text}"

    # Main box
    is_last = (i == total - 1)
    draw_box(ax, BOX_X, y, BOX_W, BOX_H,
             full_label, step["color"],
             fontsize=8.5, bold=is_last)

    # Arrow down (except last)
    if i < total - 1:
        next_y = (total - 2 - i) * 3.2 + 0.8
        draw_arrow(ax, BOX_X + BOX_W / 2,
                   y - 0.05, next_y + BOX_H + 0.05)

    # Removal box (if applicable)
    if step["removed"]:
        remove_y = y + (BOX_H - REMOVE_H) / 2
        draw_remove_arrow(ax, BOX_X, y, BOX_H,
                          REMOVE_X, remove_y, REMOVE_H)
        draw_remove_box(ax, REMOVE_X, remove_y,
                        REMOVE_W, REMOVE_H, step["removed"])

# ----------------------------------------------------------------
# Title and legend
# ----------------------------------------------------------------
ax.text(5, total * 3.2 + 0.5,
        "Dataset Curation Flowchart",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#333333")

retained_patch = mpatches.Patch(color="#2166ac", alpha=0.92, label="Retained")
final_patch    = mpatches.Patch(color="#1a9641", alpha=0.92, label="Final dataset")
removed_patch  = mpatches.Patch(color="#d7191c", alpha=0.85, label="Excluded")

ax.legend(
    handles=[retained_patch, final_patch, removed_patch],
    loc="lower left",
    bbox_to_anchor=(0.01, 0.01),
    fontsize=9,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", format="png")
plt.close()

print(f"Figure 1 saved:")
print(f"  PDF: {OUTPUT_PDF}")
print(f"  PNG: {OUTPUT_PNG}")
