#!/usr/bin/env python3
"""
Parse QUAST report.tsv and filter assemblies by quality thresholds.

Thresholds for A. baumannii:
  - Total length: 3.5 Mb - 4.5 Mb
  - N50: >= 10,000 bp
  - # contigs: <= 500
  - GC%: 38.0 - 41.0%

Usage:
    python filter_quast.py --input quast_report/report.tsv --output good_genomes.txt
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import argparse
import pandas as pd
import os



# ----------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------
MIN_LENGTH   = 3_500_000
MAX_LENGTH   = 4_500_000
MIN_N50      = 10_000
MAX_CONTIGS  = 500
MIN_GC       = 38.0
MAX_GC       = 41.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to QUAST report.tsv")
    parser.add_argument("--output", "-o", default="good_genomes.txt",
                        help="Output file: one genome ID per line")
    args = parser.parse_args()

    # QUAST report.tsv has metrics as rows and assemblies as columns
    df = pd.read_csv(args.input, sep="\t", index_col=0)

    print(f"Total assemblies in QUAST report: {df.shape[1]}")

    # Extract rows we need
    total_length = df.loc["Total length"].astype(float)
    n50          = df.loc["N50"].astype(float)
    contigs      = df.loc["# contigs"].astype(float)
    gc           = df.loc["GC (%)"].astype(float)

    # Apply filters
    pass_length  = (total_length >= MIN_LENGTH) & (total_length <= MAX_LENGTH)
    pass_n50     = n50 >= MIN_N50
    pass_contigs = contigs <= MAX_CONTIGS
    pass_gc      = (gc >= MIN_GC) & (gc <= MAX_GC)

    pass_all = pass_length & pass_n50 & pass_contigs & pass_gc

    # Summary
    print(f"\nFilter results:")
    print(f"  Failed total length ({MIN_LENGTH/1e6:.1f}-{MAX_LENGTH/1e6:.1f} Mb): "
          f"{(~pass_length).sum()}")
    print(f"  Failed N50 (< {MIN_N50:,} bp):          {(~pass_n50).sum()}")
    print(f"  Failed contig count (> {MAX_CONTIGS}):      {(~pass_contigs).sum()}")
    print(f"  Failed GC% ({MIN_GC}-{MAX_GC}%):         {(~pass_gc).sum()}")
    print(f"\n  PASSED all filters: {pass_all.sum()}")
    print(f"  FAILED any filter:  {(~pass_all).sum()}")

    good = df.columns[pass_all].tolist()
    bad  = df.columns[~pass_all].tolist()

    # Print bad ones with reason
    if bad:
        print(f"\nFailed assemblies:")
        for g in bad:
            reasons = []
            tl = total_length[g]
            if not (MIN_LENGTH <= tl <= MAX_LENGTH):
                reasons.append(f"length={tl/1e6:.2f}Mb")
            if n50[g] < MIN_N50:
                reasons.append(f"N50={n50[g]:.0f}")
            if contigs[g] > MAX_CONTIGS:
                reasons.append(f"contigs={contigs[g]:.0f}")
            if not (MIN_GC <= gc[g] <= MAX_GC):
                reasons.append(f"GC={gc[g]:.2f}%")
            print(f"  {g}: {', '.join(reasons)}")

    # Save good genome IDs
    with open(args.output, "w") as f:
        for g in good:
            f.write(str(g) + "\n")

    # Save bad genome IDs separately
    bad_out = args.output.replace(".txt", "_failed.txt")
    with open(bad_out, "w") as f:
        for g in bad:
            f.write(str(g) + "\n")

    print(f"\nGood genome IDs saved to: {args.output}")
    print(f"Failed genome IDs saved to: {bad_out}")


if __name__ == "__main__":
    main()
