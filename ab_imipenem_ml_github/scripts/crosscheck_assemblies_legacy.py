
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
    ("resistant", str(PROJECT_ROOT / "outputs/finalres.tsv")),
    ("susceptible", str(PROJECT_ROOT / "outputs/finalsus.tsv"))
]:
    df = pd.read_csv(path, sep="\t")
    df["Genome ID"] = df["Genome ID"].astype(str)
    filtered = df[df["Genome ID"].isin(available)]
    out = path.replace(".tsv", "_final.tsv")
    filtered.to_csv(out, sep="\t", index=False)
    print(f"{label}: {len(df)} -> {len(filtered)} after matching assemblies")
