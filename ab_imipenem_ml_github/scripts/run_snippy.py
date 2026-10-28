#!/usr/bin/env python3
"""
Run Snippy for each genome against the appropriate reference:
  - Universal reference (ATCC 19606) -> all 805 genomes
  - Lineage-matched references -> only their respective ST isolates

Usage:
    python run_snippy.py --threads 8
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import os
import subprocess
import pandas as pd
import argparse



# ----------------------------------------------------------------
# Paths — edit if needed
# ----------------------------------------------------------------
ASSEMBLY_DIR  = str(PROJECT_ROOT / "dataset/assemblies")
REFERENCE_DIR = str(PROJECT_ROOT / "dataset/references")
OUTPUT_DIR    = str(PROJECT_ROOT / "outputs/snippy")
MLST_FILE     = str(PROJECT_ROOT / "outputs/mlst_results.tsv")
GOOD_GENOMES  = str(PROJECT_ROOT / "outputs/good_genomes.txt")

# Reference files
REFERENCES = {
    "universal": f"{REFERENCE_DIR}/AP022836.1.fasta",   # ATCC 19606 ST52
    "ST1":       f"{REFERENCE_DIR}/CP178354.2.fasta",
    "ST2":       f"{REFERENCE_DIR}/CP032055.1.fasta",   # RUH134
    "ST3":       f"{REFERENCE_DIR}/470.8084.fasta",
    "ST25":      f"{REFERENCE_DIR}/CP098521.1.fasta",
    "ST32":      f"{REFERENCE_DIR}/470.7633.fasta",
}

# Internal references to exclude from analysis
INTERNAL_REFS = {"470.8084", "470.7633"}

# STs with lineage-matched references
LINEAGE_MATCHED_STS = {"1", "2", "3", "25", "32"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", "-t", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    # Load good genome IDs
    good = set(open(GOOD_GENOMES).read().splitlines())

    # Remove internal reference genomes from analysis set
    good -= INTERNAL_REFS
    print(f"Study genomes after excluding internal references: {len(good)}")

    # Load MLST results
    mlst = pd.read_csv(MLST_FILE, sep="\t", header=None,
                       names=["file","scheme","ST","a1","a2","a3","a4","a5","a6","a7"])
    mlst["genome_id"] = mlst["file"].str.extract(r'(\d+\.\d+)\.fasta')
    mlst = mlst[mlst["genome_id"].isin(good)]

    # Build ST lookup
    st_lookup = dict(zip(mlst["genome_id"], mlst["ST"].astype(str)))

    # Create output directories
    for ref_name in REFERENCES:
        os.makedirs(f"{OUTPUT_DIR}/{ref_name}", exist_ok=True)

    # ----------------------------------------------------------------
    # Build job list
    # ----------------------------------------------------------------
    jobs = []  # (genome_id, ref_name, ref_fasta, outdir)

    for genome_id in good:
        fasta = f"{ASSEMBLY_DIR}/{genome_id}.fasta"
        if not os.path.exists(fasta):
            print(f"WARNING: Assembly not found for {genome_id}, skipping")
            continue

        st = st_lookup.get(genome_id, "-")

        # Universal reference — all genomes
        jobs.append((genome_id, "universal",
                     REFERENCES["universal"],
                     f"{OUTPUT_DIR}/universal/{genome_id}"))

        # Lineage-matched reference — only matching ST
        if st in LINEAGE_MATCHED_STS:
            ref_key = f"ST{st}"
            jobs.append((genome_id, ref_key,
                         REFERENCES[ref_key],
                         f"{OUTPUT_DIR}/{ref_key}/{genome_id}"))

    print(f"Total Snippy jobs to run: {len(jobs)}")
    print(f"  Universal: {sum(1 for j in jobs if j[1] == 'universal')}")
    for st in LINEAGE_MATCHED_STS:
        count = sum(1 for j in jobs if j[1] == f"ST{st}")
        print(f"  ST{st}: {count}")

    # ----------------------------------------------------------------
    # Run Snippy jobs
    # ----------------------------------------------------------------
    completed = 0
    skipped   = 0
    failed    = 0

    for genome_id, ref_name, ref_fasta, outdir in jobs:
        fasta = f"{ASSEMBLY_DIR}/{genome_id}.fasta"

        # Skip if already done
        if os.path.exists(f"{outdir}/snps.vcf"):
            skipped += 1
            continue

        cmd = [
            "snippy",
            "--ref",     ref_fasta,
            "--ctgs",    fasta,
            "--outdir",  outdir,
            "--cpus",    str(args.threads),
            "--quiet",
            "--force",
        ]

        if args.dry_run:
            print(" ".join(cmd))
            completed += 1
            continue

        try:
            subprocess.run(cmd, check=True)
            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed} completed, {skipped} skipped, {failed} failed")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Snippy failed for {genome_id} vs {ref_name}: {e}")
            failed += 1

    print(f"\nDone. Completed: {completed}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
