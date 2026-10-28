
import os
from pathlib import Path

# Override this environment variable if your project directory is elsewhere.
PROJECT_ROOT = Path(os.environ.get("SAMR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

import os
import pandas as pd



assembly_dir = str(PROJECT_ROOT / "dataset/assemblies")

available = set()
for f in os.listdir(assembly_dir):
    if f.endswith(".fasta"):
        genome_id = f.replace(".fasta", "")
        available.add(genome_id)

print(f"Assemblies available: {len(available)}")

for label, path in [
    ("resistant", str(PROJECT_ROOT / "dataset/mic_values/finalres.tsv")),
    ("susceptible", str(PROJECT_ROOT / "dataset/mic_values/finalsus.tsv"))
]:
    df = pd.read_csv(path, sep="\t")

    # handle both possible column name casings
    if "genome_id" in df.columns:
        id_col = "genome_id"
    elif "Genome ID" in df.columns:
        id_col = "Genome ID"
    else:
        print(f"ERROR: no genome ID column found in {path}")
        print(f"Columns present: {list(df.columns)}")
        continue

    df[id_col] = df[id_col].astype(str)
    filtered = df[df[id_col].isin(available)]
    out = path.replace(".tsv", "_final.tsv")
    filtered.to_csv(out, sep="\t", index=False)
    print(f"{label}: {len(df)} -> {len(filtered)} after matching assemblies")

print("\nDone — final files saved as finalres_final.tsv and finalsus_final.tsv")
