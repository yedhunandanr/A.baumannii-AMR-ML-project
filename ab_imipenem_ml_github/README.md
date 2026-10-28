# *Acinetobacter baumannii* Imipenem Resistance: Comparative Genomics and Machine Learning

A research code repository for investigating how antimicrobial-resistance (AMR) gene features and genomic relatedness contribute to prediction of imipenem resistance in *Acinetobacter baumannii*.

## Research question

How well do AMR gene features and phylogenetic-distance features predict imipenem resistance, and how does model performance change when evaluation accounts for lineage or clade structure?

## What this repository contains

- **Phenotype curation** scripts for quality-checking BV-BRC antimicrobial susceptibility records and preparing analysis labels.
- **Genome QC and comparative genomics** scripts for assembly filtering, Snippy-based variant analysis, Gubbins recombination filtering, and IQ-TREE phylogenies.
- **Feature engineering** scripts for AMR gene presence/absence matrices and phylogenetic-distance features.
- **Machine-learning experiments** comparing alternative feature sets and evaluation strategies.
- **Interpretability and subgroup analysis** scripts using SHAP and resistance-mechanism groupings.
- **Figure-generation** scripts for dataset curation, lineage distribution, model performance, feature importance, and subgroup analyses.

The scripts are organized by purpose in [`scripts/`](scripts/). Some scripts are alternative or legacy versions; check each script's docstring before choosing which to run.

## Feature-set design

The ML scripts describe comparisons involving these feature groups:

1. Universal phylogenetic distance only.
2. Lineage-matched phylogenetic distance only.
3. AMR gene features only.
4. Universal distance plus AMR gene features.
5. Lineage-matched distance plus AMR gene features.

The project explores random cross-validation, lineage/clade-aware evaluation, and held-out test evaluation. Consult the individual experiment script and saved experiment configuration before interpreting any comparison; not every script necessarily implements every evaluation strategy.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── methodology.md
├── results/
│   ├── figures/
│   └── metrics/
└── scripts/
    └── analysis and pipeline scripts
```

## Data availability and privacy

Raw genome assemblies, phenotype tables, intermediate files, and full model outputs are **not included** in this repository. Add only data that you are permitted to redistribute. See [`data/README.md`](data/README.md) for guidance on documenting source data and retrieval.

Before publishing, check that any public data-release terms, sample metadata, and institutional or project restrictions allow redistribution. Avoid committing private sample information, access credentials, local logs, or large generated files.

## Requirements

- Python 3.10 or newer is recommended.
- Python packages listed in [`requirements.txt`](requirements.txt).
- External command-line tools for the corresponding workflow steps may include Snippy, Gubbins, IQ-TREE 2, AMRFinderPlus, QUAST, and MLST. Install only the tools required for the scripts you intend to run, and record their versions for reproducibility.

## Setup

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <REPOSITORY-DIRECTORY>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux, the external bioinformatics tools are usually installed separately using their official installation instructions or a suitable package/environment manager.

## Project directory conventions

The scripts use the repository root as their default base directory. Arrange your local working data and outputs using the paths expected by the selected script, or edit its input/output arguments where available. You can override the root directory for scripts that use the shared path configuration:

```bash
export SAMR_PROJECT_ROOT="/path/to/your/project"
```

Not every script exposes command-line options for every input and output yet. Read the script's usage text and docstring before running it. Some scripts expect intermediate files produced by earlier steps.

## Reproducibility notes

- Record the source database, retrieval date, accession identifiers, phenotype units, breakpoint guideline/version, and inclusion/exclusion criteria.
- Keep genome identifiers consistent across phenotype, assembly, AMR annotation, MLST, and distance tables.
- Prevent data leakage: fit preprocessing and feature selection only within training folds where appropriate; ensure related isolates are handled consistently with the intended evaluation design.
- Report class balance, sensitivity, specificity, ROC-AUC, confidence intervals where available, and the exact split/validation method.
- Compare model performance with simple baselines and document the random seeds and hyperparameters.
- Treat SHAP importance as model explanation, not proof of a causal biological mechanism.

## Status

This repository contains analysis scripts. It does not make a claim about a particular model's accuracy or clinical utility. Add validated summary metrics and final figures only after checking them against the actual experiment outputs.

## Citation

If you use or build on this code, cite the associated project report, poster, or publication when available. Add a citation here once a stable DOI or public report link exists.
