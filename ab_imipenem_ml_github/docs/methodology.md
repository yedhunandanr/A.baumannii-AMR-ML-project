# Methodology overview

This document is a starting outline. Confirm each item against the final analysis and scripts before treating it as the definitive protocol.

## 1. Phenotype curation
Clean antimicrobial susceptibility records, standardize measurement units, resolve duplicate genome entries, and apply the selected imipenem breakpoint guideline. Preserve an audit table of excluded records and reasons.

## 2. Genome quality control and lineage assignment
Filter assemblies using documented quality thresholds. Record genome identifiers and quality metrics. Assign sequence types or lineages using the exact tool and database versions used in the study.

## 3. Comparative genomics
Run variant calling/core-genome alignment against the selected reference(s), filter recombination where appropriate, infer phylogenies, and extract phylogenetic distances. Record reference accession(s), software versions, parameters, and alignment inclusion criteria.

## 4. AMR features
Generate a binary presence/absence matrix from the selected AMR annotation tool. Document the annotation database/version and gene filtering rules. Gene detection alone should not be interpreted as a direct measurement of phenotype.

## 5. Machine learning
Compare the five feature sets described in the README. Document split strategy, stratification variables, cross-validation scheme, hyperparameter search space, random seeds, missing-data handling, and all metrics. Any clade-aware split should be described precisely enough to reproduce.

## 6. Interpretation and subgroup analysis
Use SHAP and subgroup comparisons as descriptive/model-interpretation analyses. Report subgroup sizes and uncertainty, and avoid causal claims based solely on feature importance.
