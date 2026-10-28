#!/usr/bin/env python3
"""
Clean BV-BRC AMR phenotype data for imipenem resistance study.
Removes:
  - Rows with blank/null measurement values
  - Rows where measurement units are not mg/L (e.g. mm disk diffusion)
  - Rows failing EUCAST 2023 imipenem breakpoints (gray zone MIC=4, label mismatches)
  - Duplicate genome entries (keeps first occurrence)

EUCAST 2023 imipenem breakpoints for A. baumannii:
  Susceptible : MIC <= 2 mg/L
  Resistant   : MIC >= 8 mg/L
  Gray zone   : MIC 4 mg/L → EXCLUDED

Usage:
    python clean_bvbrc_amr.py --input <your_downloaded_file> --output cleaned_amr.tsv

Author: Generated for CRAB imipenem resistance study
"""

import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


import argparse
import pandas as pd
import sys
import os




def load_file(filepath: str) -> pd.DataFrame:
    """Load TSV or CSV based on file extension."""
    if filepath.endswith(".tsv") or filepath.endswith(".txt"):
        df = pd.read_csv(filepath, sep="\t", low_memory=False)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False)
    else:
        try:
            df = pd.read_csv(filepath, sep="\t", low_memory=False)
            if df.shape[1] == 1:
                raise ValueError("Only one column parsed — trying CSV")
        except Exception:
            df = pd.read_csv(filepath, low_memory=False)
    return df


def print_column_names(df: pd.DataFrame):
    """Print column names to help user verify correct columns."""
    print("\nColumns found in your file:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] {col}")
    print()


def clean_amr_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:

    original_n = len(df)
    if verbose:
        print(f"Starting with {original_n} rows")
        print_column_names(df)

    # ----------------------------------------------------------------
    # 1. Standardize column names
    # ----------------------------------------------------------------
    column_map = {
        # Measurement value
        "Measurement Value": "measurement_value",
        "measurement_value": "measurement_value",
        "MeasurementValue": "measurement_value",
        "value": "measurement_value",

        # Measurement units — BV-BRC uses singular "Measurement Unit"
        "Measurement Units": "measurement_units",
        "Measurement Unit": "measurement_units",
        "measurement_units": "measurement_units",
        "measurement_unit": "measurement_units",
        "MeasurementUnits": "measurement_units",
        "units": "measurement_units",

        # Genome ID
        "Genome ID": "genome_id",
        "genome_id": "genome_id",
        "GenomeID": "genome_id",

        # Genome name
        "Genome Name": "genome_name",
        "genome_name": "genome_name",

        # Resistant phenotype
        "Resistant Phenotype": "resistant_phenotype",
        "resistant_phenotype": "resistant_phenotype",
        "ResistantPhenotype": "resistant_phenotype",

        # Antibiotic
        "Antibiotic": "antibiotic",
        "antibiotic": "antibiotic",

        # Measurement sign — BV-BRC calls this just "Measurement"
        "Measurement": "measurement_sign",
        "Measurement Sign": "measurement_sign",
        "measurement_sign": "measurement_sign",

        # Lab typing method
        "Laboratory Typing Method": "lab_typing_method",
        "laboratory_typing_method": "lab_typing_method",

        # Evidence
        "Evidence": "evidence",
        "evidence": "evidence",
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Check required columns exist
    required = ["measurement_value", "measurement_units"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Could not find these required columns: {missing}")
        print("Please check your column names above and update the column_map in this script.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Remove rows with blank / null measurement values
    # ----------------------------------------------------------------
    before = len(df)
    df["measurement_value"] = df["measurement_value"].astype(str).str.strip()
    df = df[
        df["measurement_value"].notna() &
        (df["measurement_value"] != "") &
        (df["measurement_value"].str.lower() != "nan") &
        (df["measurement_value"].str.lower() != "none") &
        (df["measurement_value"].str.lower() != "na")
    ]
    removed_blank = before - len(df)
    if verbose:
        print(f"Removed {removed_blank} rows with blank/null measurement values")

    # ----------------------------------------------------------------
    # 3. Keep only mg/L units
    # ----------------------------------------------------------------
    before = len(df)
    df["measurement_units"] = df["measurement_units"].astype(str).str.strip()
    df = df[df["measurement_units"].str.lower() == "mg/l"]
    removed_units = before - len(df)
    if verbose:
        print(f"Removed {removed_units} rows with non-mg/L units (disk diffusion etc.)")

    # ----------------------------------------------------------------
    # 4. Ensure measurement value is numeric
    # ----------------------------------------------------------------
    before = len(df)
    df["measurement_value"] = pd.to_numeric(df["measurement_value"], errors="coerce")
    df = df[df["measurement_value"].notna()]
    removed_nonnumeric = before - len(df)
    if verbose:
        print(f"Removed {removed_nonnumeric} rows where measurement value was not numeric")

    # ----------------------------------------------------------------
    # 5. Enforce EUCAST 2023 imipenem breakpoints
    #
    #    Susceptible : MIC <= 2 mg/L  → keep
    #    Gray zone   : MIC  = 4 mg/L  → REMOVE (ambiguous)
    #    Resistant   : MIC >= 8 mg/L  → keep
    #
    #    Also catches label/MIC mismatches:
    #      - Isolates labeled Susceptible but MIC >= 4 (old breakpoint artifacts)
    #      - Isolates labeled Resistant but MIC <= 4 (low-level ambiguous)
    # ----------------------------------------------------------------
    before = len(df)

    if "resistant_phenotype" in df.columns:
        susceptible_mask = (
            (df["resistant_phenotype"].str.strip().str.lower() == "susceptible") &
            (df["measurement_value"] <= 2.0)
        )
        resistant_mask = (
            (df["resistant_phenotype"].str.strip().str.lower() == "resistant") &
            (df["measurement_value"] >= 8.0)
        )
        df = df[susceptible_mask | resistant_mask].copy()
        removed_eucast = before - len(df)
        if verbose:
            print(f"Removed {removed_eucast} rows failing EUCAST 2023 breakpoints "
                  f"(gray zone MIC=4, or label/MIC mismatch)")
    else:
        if verbose:
            print("Skipping EUCAST filter — resistant_phenotype column not found")

    # ----------------------------------------------------------------
    # 6. Remove duplicates — same genome, same antibiotic
    # ----------------------------------------------------------------
    before = len(df)
    if "genome_id" in df.columns and "antibiotic" in df.columns:
        df = df.drop_duplicates(subset=["genome_id", "antibiotic"], keep="first")
        removed_dups = before - len(df)
        if verbose:
            print(f"Removed {removed_dups} duplicate genome-antibiotic entries")
    else:
        if verbose:
            print("Skipping deduplication — genome_id or antibiotic column not found")

    # ----------------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------------
    if verbose:
        print(f"\n{'='*50}")
        print(f"FINAL DATASET: {len(df)} rows retained from {original_n} original")
        print(f"{'='*50}")

        if "resistant_phenotype" in df.columns:
            print("\nPhenotype distribution:")
            print(df["resistant_phenotype"].value_counts().to_string())

        if "measurement_value" in df.columns:
            print("\nMIC value distribution (mg/L):")
            print(df["measurement_value"].value_counts().sort_index().to_string())

        if "lab_typing_method" in df.columns:
            print("\nLab typing method distribution:")
            print(df["lab_typing_method"].value_counts().to_string())

        # Sanity check — flag any remaining mismatches
        if "resistant_phenotype" in df.columns:
            s_high = df[
                (df["resistant_phenotype"].str.lower() == "susceptible") &
                (df["measurement_value"] > 2)
            ]
            r_low = df[
                (df["resistant_phenotype"].str.lower() == "resistant") &
                (df["measurement_value"] < 8)
            ]
            if len(s_high) > 0 or len(r_low) > 0:
                print(f"\nWARNING: {len(s_high)} susceptible isolates with MIC > 2 remain")
                print(f"WARNING: {len(r_low)} resistant isolates with MIC < 8 remain")
            else:
                print("\nSanity check passed — no label/MIC mismatches in final dataset")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Clean BV-BRC AMR phenotype data for imipenem resistance study"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to downloaded BV-BRC AMR phenotype file (TSV or CSV)"
    )
    parser.add_argument(
        "--output", "-o",
        default="cleaned_amr.tsv",
        help="Output file path (default: cleaned_amr.tsv)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = load_file(args.input)

    df_clean = clean_amr_data(df, verbose=not args.quiet)

    df_clean.to_csv(args.output, sep="\t", index=False)
    print(f"\nCleaned data saved to: {args.output}")


if __name__ == "__main__":
    main()
